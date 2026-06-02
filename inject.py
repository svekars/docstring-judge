"""Inject deliberate corruptions into docstrings so the eval set has a hallucinated/outdated minority.

PyTorch's public docstrings are mostly accurate, so an unaltered ground-truth set
gives the judge nothing to fail on. This tool replaces literal text in a row's
`docstring` field and records what was changed (and the upstream original) so
the corruption is reversible.

Use the literal `swap` form on a substring you read out of the docstring in the
labeling UI. Keep `--note` descriptive: it will end up as the rationale text
when you label the row.

Operations:
    swap     - replace literal text in a docstring
    revert   - restore docstring to the upstream original
    list     - show currently corrupted rows

Examples:
    python inject.py swap torch.nn.functional.dropout \\
        --find "Default: 0.5" --replace "Default: 0.1" \\
        --note "Changed p default in Args section from 0.5 to 0.1"

    python inject.py swap torch.nn.functional.relu \\
        --find "rectified linear unit" --replace "smoothed linear unit" \\
        --note "Inverted the activation name in the description"

    python inject.py revert torch.nn.functional.dropout
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"error: data file not found: {path}")
    records: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_records(path: Path, records: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def find_row(records: list[dict], api_id: str) -> dict:
    for r in records:
        if r["api_id"] == api_id:
            return r
    raise SystemExit(f"error: no row with api_id {api_id!r}")


def cmd_swap(args: argparse.Namespace) -> int:
    path: Path = args.data
    records = load_records(path)
    r = find_row(records, args.api_id)

    current = r.get("docstring") or ""
    if args.find not in current:
        print(
            f"error: --find string not present in docstring of {args.api_id}",
            file=sys.stderr,
        )
        print("\n--- current docstring ---", file=sys.stderr)
        print(current, file=sys.stderr)
        return 1

    occurrences = current.count(args.find)
    if occurrences > 1 and not args.allow_multiple:
        print(
            f"error: --find string appears {occurrences} times. "
            "Use --allow-multiple to replace all, or refine the --find string.",
            file=sys.stderr,
        )
        return 1

    if not r.get("corrupted"):
        r["original_docstring"] = r.get("docstring")

    r["docstring"] = current.replace(args.find, args.replace)
    r["corrupted"] = True
    existing_note = r.get("corruption_note") or ""
    if args.replace_note or not existing_note:
        r["corruption_note"] = args.note
    else:
        r["corruption_note"] = existing_note + " | " + args.note
    r["corrupted_at"] = datetime.now(timezone.utc).isoformat()

    save_records(path, records)
    print(f"Corrupted {args.api_id}: {args.note}")
    return 0


def cmd_revert(args: argparse.Namespace) -> int:
    path: Path = args.data
    records = load_records(path)
    r = find_row(records, args.api_id)

    if not r.get("corrupted"):
        print(f"{args.api_id} is not corrupted; nothing to revert.")
        return 0

    r["docstring"] = r.get("original_docstring")
    r["corrupted"] = False
    r["corruption_note"] = None
    r["original_docstring"] = None
    r["corrupted_at"] = None

    if args.clear_label:
        r["label"] = None
        r["label_rationale"] = None
        r["labeled_by"] = None
        r["labeled_at"] = None

    save_records(path, records)
    print(f"Reverted {args.api_id}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    path: Path = args.data
    records = load_records(path)
    corrupted = [r for r in records if r.get("corrupted")]
    if not corrupted:
        print("No corrupted rows.")
        return 0

    print(f"{len(corrupted)} corrupted row(s):")
    for r in corrupted:
        print(f"  {r['api_id']}")
        print(f"    note:  {r.get('corruption_note')}")
        print(f"    label: {r.get('label')}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=Path("data/apis.jsonl"))
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_swap = sub.add_parser("swap", help="Replace literal text in a docstring")
    p_swap.add_argument("api_id")
    p_swap.add_argument("--find", required=True, help="Literal substring to find")
    p_swap.add_argument("--replace", required=True, help="Replacement text")
    p_swap.add_argument("--note", required=True, help="Short description of the corruption")
    p_swap.add_argument("--allow-multiple", action="store_true",
                        help="Replace every occurrence rather than failing on >1")
    p_swap.add_argument("--replace-note", action="store_true",
                        help="Overwrite the existing corruption_note instead of appending")
    p_swap.set_defaults(func=cmd_swap)

    p_revert = sub.add_parser("revert", help="Restore docstring to upstream original")
    p_revert.add_argument("api_id")
    p_revert.add_argument("--clear-label", action="store_true",
                          help="Also clear the human label fields")
    p_revert.set_defaults(func=cmd_revert)

    p_list = sub.add_parser("list", help="Show currently corrupted rows")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
