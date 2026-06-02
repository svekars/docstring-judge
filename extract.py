"""Extract API signatures and docstrings from a local PyTorch checkout.

Walks the AST of a single module file and emits one JSONL record per public
module-level function. On re-extraction over an existing output, the human
label fields (`label`, `label_rationale`, `labeled_by`, `labeled_at`) are
preserved per `api_id`.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SKIP_DECORATORS = {"overload", "typing.overload"}


def resolve_module_file(pytorch_root: Path, module: str) -> Path:
    rel = module.replace(".", "/")
    candidate = pytorch_root / f"{rel}.py"
    if candidate.exists():
        return candidate
    candidate = pytorch_root / rel / "__init__.py"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Could not find source for module {module!r} under {pytorch_root}"
    )


def get_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def src(node: ast.AST | None, source: str) -> str | None:
    if node is None:
        return None
    return ast.get_source_segment(source, node)


def decorator_name(d: ast.expr, source: str) -> str:
    target: ast.AST = d.func if isinstance(d, ast.Call) else d
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        parts: list[str] = []
        cur: ast.AST = target
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))
    return src(target, source) or "<unknown>"


def extract_parameters(args: ast.arguments, source: str) -> list[dict[str, Any]]:
    params: list[dict[str, Any]] = []
    posonly = args.posonlyargs
    regular = args.args
    all_positional = posonly + regular
    n_pos = len(all_positional)
    n_def = len(args.defaults)

    for i, arg in enumerate(all_positional):
        default_idx = i - (n_pos - n_def)
        default = src(args.defaults[default_idx], source) if default_idx >= 0 else None
        kind = "POSITIONAL_ONLY" if i < len(posonly) else "POSITIONAL_OR_KEYWORD"
        params.append({
            "name": arg.arg,
            "annotation": src(arg.annotation, source),
            "default": default,
            "kind": kind,
        })

    if args.vararg:
        params.append({
            "name": args.vararg.arg,
            "annotation": src(args.vararg.annotation, source),
            "default": None,
            "kind": "VAR_POSITIONAL",
        })

    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        params.append({
            "name": arg.arg,
            "annotation": src(arg.annotation, source),
            "default": src(default, source) if default is not None else None,
            "kind": "KEYWORD_ONLY",
        })

    if args.kwarg:
        params.append({
            "name": args.kwarg.arg,
            "annotation": src(args.kwarg.annotation, source),
            "default": None,
            "kind": "VAR_KEYWORD",
        })

    return params


def render_signature(
    name: str,
    parameters: list[dict[str, Any]],
    return_annotation: str | None,
) -> str:
    parts: list[str] = []
    posonly_indices = [i for i, p in enumerate(parameters) if p["kind"] == "POSITIONAL_ONLY"]
    last_posonly = posonly_indices[-1] if posonly_indices else -1
    has_vararg = any(p["kind"] == "VAR_POSITIONAL" for p in parameters)
    has_kwonly = any(p["kind"] == "KEYWORD_ONLY" for p in parameters)
    need_star_separator = has_kwonly and not has_vararg
    star_separator_inserted = False

    for i, p in enumerate(parameters):
        if need_star_separator and not star_separator_inserted and p["kind"] == "KEYWORD_ONLY":
            parts.append("*")
            star_separator_inserted = True

        prefix = ""
        if p["kind"] == "VAR_POSITIONAL":
            prefix = "*"
        elif p["kind"] == "VAR_KEYWORD":
            prefix = "**"

        s = prefix + p["name"]
        if p["annotation"]:
            s += f": {p['annotation']}"
        if p["default"] is not None:
            sep = " = " if p["annotation"] else "="
            s += f"{sep}{p['default']}"
        parts.append(s)

        if i == last_posonly:
            parts.append("/")

    rendered = f"{name}({', '.join(parts)})"
    if return_annotation:
        rendered += f" -> {return_annotation}"
    return rendered


def extract_functions(
    module: str,
    module_file: Path,
    source: str,
    include_private: bool,
    include_overloads: bool,
) -> list[dict[str, Any]]:
    tree = ast.parse(source, filename=str(module_file))
    found: dict[str, dict[str, Any]] = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not include_private and node.name.startswith("_"):
            continue
        decorators = [decorator_name(d, source) for d in node.decorator_list]
        if not include_overloads and any(d in SKIP_DECORATORS for d in decorators):
            continue

        params = extract_parameters(node.args, source)
        return_annotation = src(node.returns, source)
        signature = render_signature(node.name, params, return_annotation)
        docstring = ast.get_docstring(node, clean=False)

        found[node.name] = {
            "api_id": f"{module}.{node.name}",
            "module": module,
            "name": node.name,
            "kind": "function",
            "signature": signature,
            "parameters": params,
            "return_annotation": return_annotation,
            "decorators": decorators,
            "docstring": docstring,
            "source_line": node.lineno,
        }

    return list(found.values())


def load_existing_metadata(path: Path) -> dict[str, dict[str, Any]]:
    """Read existing rows so we can preserve hand-labeled and hand-corrupted fields."""
    if not path.exists():
        return {}
    metadata: dict[str, dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            api_id = rec.get("api_id")
            if not api_id:
                continue
            metadata[api_id] = {
                "label": rec.get("label"),
                "label_rationale": rec.get("label_rationale"),
                "labeled_by": rec.get("labeled_by"),
                "labeled_at": rec.get("labeled_at"),
                "corrupted": rec.get("corrupted"),
                "corruption_note": rec.get("corruption_note"),
                "original_docstring": rec.get("original_docstring"),
                "corrupted_at": rec.get("corrupted_at"),
                "_docstring_if_corrupted": (
                    rec.get("docstring") if rec.get("corrupted") else None
                ),
            }
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pytorch-root", type=Path, required=True,
                        help="Path to local PyTorch checkout")
    parser.add_argument("--module", required=True,
                        help="Dotted module path, e.g. torch.nn.functional")
    parser.add_argument("--output", type=Path, default=Path("data/apis.jsonl"))
    parser.add_argument("--limit", type=int, default=None,
                        help="Keep first N functions in source order")
    parser.add_argument("--names", type=str, default=None,
                        help="Comma-separated allowlist of function names")
    parser.add_argument("--include-private", action="store_true")
    parser.add_argument("--include-overloads", action="store_true")
    args = parser.parse_args()

    pytorch_root = args.pytorch_root.expanduser().resolve()
    if not pytorch_root.exists():
        print(f"error: --pytorch-root {pytorch_root} does not exist", file=sys.stderr)
        return 1

    module_file = resolve_module_file(pytorch_root, args.module)
    source = module_file.read_text()
    commit = get_commit(pytorch_root)
    extracted_at = datetime.now(timezone.utc).isoformat()

    records = extract_functions(
        module=args.module,
        module_file=module_file,
        source=source,
        include_private=args.include_private,
        include_overloads=args.include_overloads,
    )

    if args.names:
        allow = {n.strip() for n in args.names.split(",") if n.strip()}
        records = [r for r in records if r["name"] in allow]

    if args.limit is not None:
        records = records[: args.limit]

    rel_source = module_file.relative_to(pytorch_root)
    existing = load_existing_metadata(args.output)

    for r in records:
        r["source_file"] = str(rel_source)
        r["pytorch_commit"] = commit
        r["extracted_at"] = extracted_at
        prior = existing.get(r["api_id"], {})

        r["label"] = prior.get("label")
        r["label_rationale"] = prior.get("label_rationale")
        r["labeled_by"] = prior.get("labeled_by")
        r["labeled_at"] = prior.get("labeled_at")

        r["corrupted"] = bool(prior.get("corrupted"))
        r["corruption_note"] = prior.get("corruption_note")
        r["original_docstring"] = prior.get("original_docstring")
        r["corrupted_at"] = prior.get("corrupted_at")
        if r["corrupted"] and prior.get("_docstring_if_corrupted") is not None:
            r["docstring"] = prior["_docstring_if_corrupted"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {args.output}", file=sys.stderr)
    preserved_labels = sum(1 for r in records if r["label"] is not None)
    preserved_corruptions = sum(1 for r in records if r["corrupted"])
    if preserved_labels:
        print(f"Preserved {preserved_labels} existing labels.", file=sys.stderr)
    if preserved_corruptions:
        print(f"Preserved {preserved_corruptions} corrupted docstring(s).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
