"""Judge: classify each row's docstring as accurate/hallucinated/outdated/partial/missing.

Reads data/apis.jsonl plus optional data/doctest_results.jsonl as execution
evidence. Writes data/predictions.jsonl atomically after each row, so a crash
mid-run does not lose completed work. Supports --skip-existing for incremental
resume and --names/--limit for iterating on the prompt without re-judging all rows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types

VALID_LABELS = {"accurate", "hallucinated", "outdated", "partial", "missing"}

SYSTEM_PROMPT = """You are a documentation auditor. You receive a function signature, a docstring, a list of structural findings produced by deterministic code (trust them as ground truth), and an optional execution-evidence line. Classify the docstring into EXACTLY ONE category.

HARD RULES (these override everything below):
1. Arguments whose names start with `_` are private. NEVER cite them as defects. NEVER mention them in your rationale. The structural checker already excludes them.
2. Type descriptions in the docstring that differ from the signature's static type annotation are NOT defects. If the signature says `int` and the docstring says "int or tuple of ints", or the signature says `int | None` and the docstring says only "int", do NOT flag it. PyTorch signatures are conservative static annotations; docstrings describe runtime types. The structural checker does not flag this and neither should you.
3. If the Structural Findings list is "none", you may NOT invent structural concerns. You may only flag prose-level issues such as math contradictions, inverted descriptions, wrong return types stated in prose, or factually impossible behavior claims. "The docstring is incomplete" is NOT a valid prose-level issue.
4. If you cite execution evidence in your rationale, quote the exact text from the "Execution evidence" line that was provided in the prompt. Do NOT paraphrase, do NOT infer requirements that were not stated, do NOT invent error messages. SMOKE_PASSED and PASSED only prove callability; they do NOT prove prose claims are accurate or inaccurate.

Categories:
- accurate: no structural findings AND no prose-level inaccuracies.
- hallucinated: a serious wrong claim. A default_mismatch where the documented value is a clearly fabricated number swapped for the real one (e.g. signature default 0.5, docstring claims 0.1), a fictional_arg finding, contradictory math or definitions in the prose (e.g. a variable defined two different ways in one Shape block), inverted behavior descriptions ("during training" stated as "during evaluation"), or a return type stated in prose that contradicts the signature.
- outdated: an argument name in the docstring is absent from the signature, suggesting the API was renamed but the docs lag.
- partial: a structural finding is present but isolated in an otherwise correct doc. In particular, a default_mismatch where the documented value is a value the runtime kernel actually uses even though the static signature differs (e.g. signature shows None but documented default is 0 or False because the C++ kernel substitutes that) = partial.
- missing: empty_docstring finding is present (no docstring at all).

How to use findings:
- Empty findings AND no prose-level lies = accurate. This is the default outcome; do not strain to find something wrong.
- default_mismatch with a believable runtime value (0, False, None, common identity defaults): partial.
- default_mismatch with a fabricated-looking number swap (specific float to different specific float): hallucinated.
- fictional_arg: hallucinated.
- undocumented_arg with severity "low": does NOT affect classification.
- undocumented_arg with severity "medium" on its own: partial.

Output JSON only:
{"classification": "<one of: accurate, hallucinated, outdated, partial, missing>", "rationale": "<one sentence citing the specific finding or prose claim>"}
"""

PREFERRED_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]


def find_working_model(client) -> str:
    print("Detecting available models...", end=" ", flush=True, file=sys.stderr)
    try:
        available = {m.name.split("/")[-1] for m in client.models.list()}
        for candidate in PREFERRED_MODELS:
            if candidate in available:
                print(f"using {candidate}", file=sys.stderr)
                return candidate
        for name in sorted(available):
            if "flash" in name and "embedding" not in name and "live" not in name:
                print(f"using fallback {name}", file=sys.stderr)
                return name
    except Exception as e:
        print(f"list failed ({e})", file=sys.stderr)
    print(f"defaulting to {PREFERRED_MODELS[0]}", file=sys.stderr)
    return PREFERRED_MODELS[0]


def build_prompt(record: dict, execution_res: dict | None, findings: dict | None) -> str:
    signature = record["signature"]
    docstring = record.get("docstring") or "(no docstring)"
    parts = [f"Signature:\n{signature}", "", f"Docstring:\n{docstring}"]

    if findings is not None:
        finding_list = findings.get("findings", [])
        if finding_list:
            lines = ["Structural findings (deterministic, trust as ground truth):"]
            for f in finding_list:
                check = f.get("check", "?")
                evidence = f.get("evidence", "")
                severity = f" [severity={f['severity']}]" if f.get("severity") else ""
                lines.append(f"  - {check}{severity}: {evidence}")
            parts.extend(["", "\n".join(lines)])
        else:
            parts.extend(["", "Structural findings: none."])

    if execution_res:
        status = execution_res.get("status")
        ret_type = execution_res.get("return_type")
        ret_info = f" (returned: {ret_type})" if ret_type else ""
        if status == "passed":
            line = (
                f"Execution evidence: PASSED — "
                f"{execution_res.get('total_tests', '?')} doctest(s) ran cleanly{ret_info}"
            )
        elif status == "smoke_passed":
            line = f"Execution evidence: SMOKE_PASSED — callable with a dummy tensor{ret_info}"
        elif status == "failed":
            line = (
                "Execution evidence: FAILED\n"
                f"Output:\n{execution_res.get('output', '')}"
            )
        elif status == "skipped":
            line = f"Execution evidence: SKIPPED — {execution_res.get('reason', '')}"
        else:
            line = f"Execution evidence: {status}"
        parts.extend(["", line])

    return "\n".join(parts)


def judge_row(
    client,
    model_name: str,
    record: dict,
    execution_res: dict | None,
    findings: dict | None,
    max_retries: int = 3,
) -> dict:
    prompt = build_prompt(record, execution_res, findings)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        temperature=0.0,
    )

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            raw_text = response.text or ""
            try:
                prediction = json.loads(raw_text)
            except json.JSONDecodeError as je:
                return {
                    "api_id": record["api_id"],
                    "error": f"json decode: {je}",
                    "raw_response": raw_text[:300],
                }

            label = str(prediction.get("classification", "")).lower().strip()
            valid = label in VALID_LABELS
            result: dict = {
                "api_id": record["api_id"],
                "prediction": label if valid else "invalid",
                "rationale": prediction.get("rationale"),
                "model": model_name,
            }
            if not valid:
                result["raw_label"] = label
            return result
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    return {"api_id": record["api_id"], "error": f"after {max_retries} attempts: {last_error}"}


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def load_existing_predictions(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                p = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "api_id" in p:
                out[p["api_id"]] = p
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/apis.jsonl"))
    parser.add_argument("--results", type=Path, default=Path("data/doctest_results.jsonl"))
    parser.add_argument("--findings", type=Path, default=Path("data/static_findings.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/predictions.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--names", type=str, default=None,
                        help="Comma-separated allowlist of bare function names")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Do not re-judge rows that already have a non-error prediction")
    parser.add_argument("--model", type=str, default=None,
                        help="Override auto-detected model name")
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("error: GOOGLE_API_KEY not set", file=sys.stderr)
        return 1

    client = genai.Client(api_key=api_key)
    model_name = args.model or find_working_model(client)

    records: list[dict] = []
    with args.data.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    results_by_api: dict[str, dict] = {}
    if args.results.exists():
        with args.results.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    results_by_api[r["api_id"]] = r
                except json.JSONDecodeError:
                    continue

    findings_by_api: dict[str, dict] = {}
    if args.findings.exists():
        with args.findings.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    fr = json.loads(line)
                    if "api_id" in fr:
                        findings_by_api[fr["api_id"]] = fr
                except json.JSONDecodeError:
                    continue
    else:
        print(f"warning: no findings at {args.findings}; run static_checks.py first",
              file=sys.stderr)

    if args.names:
        allow = {n.strip() for n in args.names.split(",") if n.strip()}
        records = [r for r in records if r["name"] in allow]
    if args.limit is not None:
        records = records[: args.limit]

    existing = load_existing_predictions(args.output)

    if args.skip_existing:
        before = len(records)
        records_to_judge = [
            r for r in records
            if r["api_id"] not in existing or "error" in existing[r["api_id"]]
        ]
        skipped = before - len(records_to_judge)
        if skipped:
            print(f"Skipping {skipped} already-judged row(s); judging {len(records_to_judge)}.",
                  file=sys.stderr)
    else:
        records_to_judge = records

    predictions_by_api: dict[str, dict] = dict(existing)

    n = len(records_to_judge)
    print(f"Judging {n} record(s) with {model_name}...", file=sys.stderr)

    for i, rec in enumerate(records_to_judge, start=1):
        print(f"[{i}/{n}] {rec['api_id']}...", end=" ", flush=True, file=sys.stderr)
        pred = judge_row(
            client,
            model_name,
            rec,
            results_by_api.get(rec["api_id"]),
            findings_by_api.get(rec["api_id"]),
        )
        predictions_by_api[rec["api_id"]] = pred
        if "prediction" in pred:
            tag = pred["prediction"]
            if tag == "invalid":
                tag = f"invalid (raw={pred.get('raw_label')!r})"
            print(tag, file=sys.stderr)
        else:
            print(f"FAILED: {pred.get('error')}", file=sys.stderr)
        atomic_write_jsonl(args.output, list(predictions_by_api.values()))

    print(f"\nWrote {len(predictions_by_api)} prediction(s) to {args.output}", file=sys.stderr)

    counts: dict[str, int] = {}
    for p in predictions_by_api.values():
        key = p.get("prediction") if "prediction" in p else "error"
        counts[key] = counts.get(key, 0) + 1
    print("Predictions by label:", file=sys.stderr)
    for label in ("accurate", "hallucinated", "outdated", "partial", "missing", "invalid", "error"):
        if label in counts:
            print(f"  {label:<14} {counts[label]}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
