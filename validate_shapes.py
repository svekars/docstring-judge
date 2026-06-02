"""Empirical shape validator for PyTorch docstrings.

Uses Claude to interpret "Shape" sections in docstrings, generates a Python
test case with dummy tensors, and executes it to verify if the documented
shapes match reality.

Saves results to data/shape_results.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None

SYSTEM_PROMPT = """You are an expert PyTorch test engineer.
Your task is to extract shape constraints from a docstring and write a Python validation test.

You will be given a function signature and its docstring.
You must return a JSON object with a single field 'test_code' which is a string of Python code.

The 'test_code' must:
1. Define all necessary input tensors with appropriate shapes and dtypes.
2. Call the function using the provided signature.
3. Assert that the resulting tensor's shape matches the description in the docstring.
4. Use 'torch' and 'F' (which will be provided in the environment).

Example Output:
{
  "test_code": "input = torch.randn(2, 4)\nweight = torch.randn(10, 3)\noutput = F.embedding(input, weight)\nassert output.shape == (2, 4, 3)"
}

Keep the test code as simple and robust as possible. Use small dimensions (e.g., 2, 3, 4) to avoid memory issues.
"""

def setup_safe_env(cpu_only: bool = False) -> dict[str, Any]:
    if torch is None:
        print("error: torch not found.", file=sys.stderr)
        sys.exit(1)

    globs = {"torch": torch, "F": F, "nn": torch.nn}
    if cpu_only:
        torch.cuda.is_available = lambda: False
        def safe_cuda(self, *args, **kwargs): return self
        torch.Tensor.cuda = safe_cuda
    return globs

def run_test_code(test_code: str, globs: dict) -> dict:
    try:
        # We use a clean local dict for each execution but keep the globals
        locs = {}
        exec(test_code, globs, locs)
        return {"status": "passed"}
    except AssertionError as e:
        return {"status": "failed", "reason": "Assertion failed: The actual shape did not match the documented shape."}
    except Exception as e:
        return {"status": "error", "reason": f"Execution error: {str(e)}"}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/apis.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/shape_results.jsonl"))
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if anthropic is None:
        print("error: anthropic package not found.", file=sys.stderr)
        return 1
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("error: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1

    client = anthropic.Anthropic(api_key=api_key)
    globs = setup_safe_env(cpu_only=args.cpu_only)

    records = []
    with args.data.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    
    if args.limit:
        records = records[:args.limit]

    print(f"Validating shapes for {len(records)} records...", file=sys.stderr)
    results = []

    for i, rec in enumerate(records):
        api_id = rec["api_id"]
        docstring = rec.get("docstring")
        if not docstring or "Shape" not in docstring and "Output" not in docstring:
            results.append({"api_id": api_id, "status": "skipped", "reason": "No shape section found"})
            continue

        print(f"[{i+1}/{len(records)}] {api_id}...", end=" ", flush=True, file=sys.stderr)
        
        # 1. Ask Claude for the test code
        prompt = f"Signature: {rec['signature']}\n\nDocstring:\n{docstring}"
        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            raw_json = response.content[0].text
            test_data = json.loads(raw_json)
            test_code = test_data["test_code"]
            
            # 2. Execute the test code
            outcome = run_test_code(test_code, globs)
            outcome["api_id"] = api_id
            outcome["test_code"] = test_code
            results.append(outcome)
            print(outcome["status"], file=sys.stderr)
            
        except Exception as e:
            print("failed (llm/parse error)", file=sys.stderr)
            results.append({"api_id": api_id, "status": "error", "reason": f"LLM/Parse error: {str(e)}"})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    print(f"\nWrote shape results to {args.output}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
