"""Deterministic structural checks against extracted PyTorch API rows.

Parses the Args section of each docstring and compares it against the
signature's parameters. Emits findings the LLM judge can trust as ground
truth: default-value mismatches, fictional argument names, undocumented
public args, and empty docstrings.

Output: one JSON record per api_id to data/static_findings.jsonl. The judge
consumes these so it does not have to redo structural reasoning.

Usage:
    python static_checks.py
    python static_checks.py --data data/apis.jsonl --output data/static_findings.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ARGS_SECTION_HEADERS = {"Args:", "Arguments:", "Parameters:"}
OTHER_SECTION_NAMES = {
    "Returns", "Return", "Yields", "Raises", "Example", "Examples",
    "Note", "Notes", "Warning", "Warnings", "See Also", "References",
    "Shape", "Attributes",
}

ENTRY_RE = re.compile(r"^(\w+)\s*(?:\(([^)]+)\))?\s*:\s*(.*)$")

DEFAULT_RE = re.compile(
    r"Default:\s*"
    r"(?:"
    r"``([^`]+)``"
    r"|`([^`]+)`"
    r"|\"([^\"]+)\""
    r"|'([^']+)'"
    r"|(None|True|False)\b"
    r"|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    r"|(\S+)"
    r")"
)


def parse_args_section(docstring: str) -> list[dict]:
    if not docstring:
        return []
    lines = docstring.split("\n")

    start = None
    args_header_indent = None
    for i, line in enumerate(lines):
        if line.strip() in ARGS_SECTION_HEADERS:
            start = i + 1
            args_header_indent = len(line) - len(line.lstrip())
            break
    if start is None:
        return []

    entry_indent = None
    for line in lines[start:]:
        if line.strip():
            entry_indent = len(line) - len(line.lstrip())
            break
    if entry_indent is None or entry_indent <= args_header_indent:
        return []

    entries: list[dict] = []
    current: dict | None = None

    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            continue
        line_indent = len(line) - len(line.lstrip())

        if line_indent <= args_header_indent and stripped.endswith(":"):
            name = stripped[:-1]
            if name in OTHER_SECTION_NAMES or stripped in ARGS_SECTION_HEADERS:
                break

        if line_indent == entry_indent:
            m = ENTRY_RE.match(stripped)
            if m:
                if current:
                    entries.append(current)
                current = {
                    "name": m.group(1),
                    "type": m.group(2),
                    "raw_description": m.group(3) or "",
                }
                continue

        if line_indent > entry_indent and current is not None:
            current["raw_description"] += " " + stripped

    if current:
        entries.append(current)

    for entry in entries:
        entry["default"] = extract_default(entry["raw_description"])

    return entries


RST_ROLE_RE = re.compile(r":\w+:`([^`]+)`")


def extract_default(description: str) -> str | None:
    if not description:
        return None
    # Strip RST inline roles so :math:`0.0` becomes 0.0 before matching.
    cleaned = RST_ROLE_RE.sub(r"\1", description)
    m = DEFAULT_RE.search(cleaned)
    if not m:
        return None
    for group in m.groups():
        if group is not None:
            return group
    return None


def normalize_default(s: str | None) -> str | None:
    if s is None:
        return None
    val = s.strip().rstrip(".,")
    for left, right in (("``", "``"), ("`", "`"), ('"', '"'), ("'", "'")):
        if val.startswith(left) and val.endswith(right) and len(val) > len(left) + len(right):
            val = val[len(left):-len(right)]
            break
    val = val.strip().rstrip(".,")
    if val.lower() in {"true", "false"}:
        return val.capitalize()
    if val.lower() == "none":
        return "None"
    try:
        f = float(val)
        return str(int(f)) if f == int(f) else repr(f)
    except (ValueError, TypeError):
        pass
    return val


def check_row(record: dict, doctest_status: str | None) -> dict:
    findings: list[dict] = []
    sig_params = {p["name"]: p for p in record.get("parameters", [])}
    documented = parse_args_section(record.get("docstring") or "")
    documented_names = [e["name"] for e in documented]
    documented_set = set(documented_names)
    sig_names = set(sig_params.keys())
    public_sig_names = {n for n in sig_names if not n.startswith("_")}

    if not (record.get("docstring") or "").strip():
        findings.append({
            "check": "empty_docstring",
            "evidence": "docstring is empty or absent",
        })

    for name in sorted(documented_set - sig_names):
        findings.append({
            "check": "fictional_arg",
            "arg": name,
            "evidence": f"docstring documents '{name}' but signature has no such parameter",
        })

    # Only flag undocumented args when the docstring HAS an Args section but
    # missed some entries. Docstrings without any Args section at all are
    # PyTorch's common redirect pattern ("See :class:`nn.X` for details") and
    # the flood of "undocumented" findings on every arg is just noise.
    if documented:
        for name in sorted(public_sig_names - documented_set):
            findings.append({
                "check": "undocumented_arg",
                "arg": name,
                "evidence": f"Args section is present but does not document parameter '{name}'",
                "severity": "low" if name in {"input", "self"} else "medium",
            })

    for entry in documented:
        name = entry["name"]
        if name not in sig_params:
            continue
        sig_default = normalize_default(sig_params[name]["default"])
        doc_default = normalize_default(entry.get("default"))
        if doc_default is None:
            continue
        if sig_default is None:
            findings.append({
                "check": "default_for_required_arg",
                "arg": name,
                "documented": doc_default,
                "evidence": (
                    f"docstring claims Default: {doc_default} but signature "
                    f"shows '{name}' is required"
                ),
            })
        elif doc_default != sig_default:
            findings.append({
                "check": "default_mismatch",
                "arg": name,
                "documented": doc_default,
                "actual": sig_default,
                "evidence": (
                    f"docstring says Default: {doc_default}; "
                    f"signature default is {sig_default}"
                ),
            })

    return {
        "api_id": record["api_id"],
        "findings": findings,
        "documented_args": documented_names,
        "signature_args": [n for n in sig_names if not n.startswith("_")],
        "has_args_section": bool(documented),
        "doctest_status": doctest_status,
    }


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/apis.jsonl"))
    parser.add_argument("--doctests", type=Path, default=Path("data/doctest_results.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/static_findings.jsonl"))
    args = parser.parse_args()

    records = load_jsonl(args.data)
    if not records:
        print(f"error: no records in {args.data}", file=sys.stderr)
        return 1

    doctest_status_by_api: dict[str, str] = {}
    for r in load_jsonl(args.doctests):
        if "api_id" in r:
            doctest_status_by_api[r["api_id"]] = r.get("status")

    results = [check_row(rec, doctest_status_by_api.get(rec["api_id"])) for rec in records]
    atomic_write_jsonl(args.output, results)

    finding_counts: dict[str, int] = {}
    rows_with_findings = 0
    for r in results:
        if r["findings"]:
            rows_with_findings += 1
        for f in r["findings"]:
            finding_counts[f["check"]] = finding_counts.get(f["check"], 0) + 1

    print(f"Wrote {len(results)} rows to {args.output}", file=sys.stderr)
    print(f"Rows with at least one finding: {rows_with_findings}/{len(results)}", file=sys.stderr)
    for check_name, count in sorted(finding_counts.items()):
        print(f"  {check_name:<28} {count}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
