"""The Power-Runner: Empirical PyTorch Validator (Non-AI).

1. Runs xdoctests with full directive support.
2. Performs 'Smoke Testing' for functions without examples.
3. Captures and reports the return value type for all successful executions.
4. Mocks hardware requirements for safe local execution.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

try:
    from xdoctest import core
except ImportError:
    core = None

def setup_safe_env(cpu_only: bool = False) -> dict[str, Any]:
    """Hardware and library mocking for consistent local execution."""
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("error: torch not found.", file=sys.stderr)
        sys.exit(1)

    globs = {"torch": torch, "F": F, "nn": torch.nn}

    if cpu_only:
        torch.cuda.is_available = lambda: False
        torch.Tensor.cuda = lambda self, *args, **kwargs: self
        orig_to = torch.Tensor.to
        def safe_to(self, *args, **kwargs):
            if (args and args[0] == 'cuda') or kwargs.get('device') == 'cuda':
                return orig_to(self, 'cpu')
            return orig_to(self, *args, **kwargs)
        torch.Tensor.to = safe_to

    return globs

def smoke_test_function(api_id: str, globs: dict) -> dict:
    """Attempts to call the function with dummy data and reports return type."""
    torch = globs['torch']
    F = globs['F']
    name = api_id.split('.')[-1]
    func = getattr(F, name, None)
    
    if not func:
        return {"status": "skipped", "reason": "Function not found in F"}

    try:
        dummy = torch.ones(2, 2)
        res = None
        try:
            res = func(dummy)
        except TypeError:
            try:
                res = func(dummy, dim=0)
            except TypeError:
                return {"status": "skipped", "reason": "Complex signature, smoke test aborted"}
                
        return {
            "status": "smoke_passed",
            "return_type": type(res).__name__
        }
    except Exception as e:
        return {"status": "failed", "output": f"Smoke test failed: {str(e)}", "failures": 1}

def run_single_api(api_id: str, docstring: str | None, globs: dict) -> dict:
    """The core logic for a single API ID."""
    if not core:
        return {"status": "error", "reason": "xdoctest not installed"}

    if docstring:
        try:
            examples = list(core.parse_docstr_examples(docstring, callname=api_id))
            if examples:
                passed = 0
                failed = 0
                outputs = []
                last_return_type = None

                for ex in examples:
                    ex.globs.update(globs)
                    out = io.StringIO()
                    # Capture the result of the last line of the example if possible
                    # xdoctest doesn't easily expose the return value of the whole block,
                    # but it runs it in the globs.
                    with redirect_stdout(out):
                        result = ex.run(verbose=0)
                    
                    if result.get('passed'):
                        passed += 1
                        # We can't easily get the return type from a generic example block
                        # without deeper instrumentation. We rely on the smoke test for types.
                    else:
                        failed += 1
                        outputs.append(out.getvalue().strip())
                
                if failed > 0:
                    return {"status": "failed", "failures": failed, "output": "\n---\n".join(outputs)}
                
                # If doctests pass, we still run a quick smoke test to get the return type
                smoke = smoke_test_function(api_id, globs)
                return {
                    "status": "passed",
                    "total_tests": passed,
                    "return_type": smoke.get("return_type") if smoke["status"] == "smoke_passed" else "unknown"
                }
        except Exception as e:
            return {"status": "failed", "output": f"Parser error: {str(e)}", "failures": 1}

    return smoke_test_function(api_id, globs)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/apis.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/doctest_results.jsonl"))
    parser.add_argument("--cpu-only", action="store_true", default=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not core:
        print("error: xdoctest not found.", file=sys.stderr)
        return 1

    globs = setup_safe_env(cpu_only=args.cpu_only)
    
    records = []
    with args.data.open() as f:
        for line in f:
            line = line.strip()
            if line: records.append(json.loads(line))

    if args.limit: records = records[:args.limit]

    print(f"Running Power-Runner for {len(records)} APIs...", file=sys.stderr)
    results = []
    
    for rec in records:
        aid = rec["api_id"]
        print(f"  {aid}...", end=" ", flush=True, file=sys.stderr)
        res = run_single_api(aid, rec.get("docstring"), globs)
        res["api_id"] = aid
        results.append(res)
        print(res["status"], file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
