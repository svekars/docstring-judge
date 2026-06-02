# pytorch-automated-docs

Tooling to evaluate and (eventually) auto-update PyTorch API documentation.

## Status

v0: extractor and dataset schema.

The first artifact is a hand-labeled ground-truth dataset of public API entries from a local PyTorch checkout. `extract.py` reads a module via AST and emits one JSON object per public function. A human then fills in the `label` and `label_rationale` fields. The labeled rows are the anchor for everything downstream: the LLM-as-judge, the RAG generator, and the meta-eval that validates the judge.

## Workflow

End-to-end, from a fresh checkout. Each tool has sensible defaults pointing at `data/`, so most commands take no arguments.

```sh
source .venv/bin/activate
export GEMINI_API_KEY=...   # needed only for step 6
```

### 1. Extract API rows from PyTorch source

```sh
python extract.py \
    --pytorch-root ../pytorch \
    --module torch.nn.functional \
    --output data/apis.jsonl \
    --limit 20
```

Walks the module via AST, writes one JSON object per public function to `data/apis.jsonl`. Re-running preserves any existing hand labels (rows are matched by `api_id`). See [Running the extractor](#running-the-extractor) for all flags.

### 2. Inject corruptions (optional, recommended)

Pick a few rows to deliberately break so the eval has something to fail on.

```sh
python inject.py swap torch.nn.functional.dropout \
    --find "Default: 0.5" --replace "Default: 0.1" \
    --note "Changed p default in Args section from 0.5 to 0.1"
```

`inject.py list` shows current corruptions, `inject.py revert <api_id>` undoes one. See [Injecting corruptions](#injecting-corruptions) for the rationale and target eval-set composition.

### 3. Hand-label

```sh
streamlit run label_app.py
```

Opens http://localhost:8501. Walk through each row, pick a label (`accurate`, `hallucinated`, `outdated`, `partial`, `missing`), write a one-line rationale, click "Save and next." Writes back to `data/apis.jsonl` atomically. Aim for at least 2 examples per label class. See [Hand-labeling](#hand-labeling) for sidebar options and corrupted-row behavior.

### 4. Run deterministic static checks

```sh
python static_checks.py
```

Reads `data/apis.jsonl`, writes `data/static_findings.jsonl` with structured findings (`default_mismatch`, `fictional_arg`, `undocumented_arg`, `empty_docstring`). No API key, runs in seconds. These findings become the structural ground truth the LLM judge consumes in step 6.

### 5. Run doctests

```sh
python run_doctests.py --cpu-only
```

Executes the `>>>` blocks in each docstring via `xdoctest`. Writes `data/doctest_results.jsonl` with pass/fail/smoke-passed/skipped status per row. Use `--limit N` to test on a subset first. The judge consumes this only as evidence of callability, not as proof of prose correctness.

### 6. Run the LLM judge

```sh
python judge.py --skip-existing
```

Reads `apis.jsonl + static_findings.jsonl + doctest_results.jsonl`, sends each row to Gemini, writes `data/predictions.jsonl` with a classification (`accurate`, `hallucinated`, `outdated`, `partial`, `missing`) and rationale. Atomic per-row write, so a crash mid-run does not lose completed work.

Useful flags:

- `--skip-existing`: resume without re-judging rows that already have a non-error prediction.
- `--names a,b,c`: restrict to specific function names while iterating on the prompt.
- `--limit N`: judge only the first N rows.
- `--model <name>`: override the auto-detected model.

### 7. Meta-eval against the hand labels

```sh
python meta_eval.py
```

Compares `data/predictions.jsonl` to the hand labels in `data/apis.jsonl`. Prints simple agreement, Cohen's kappa, per-class precision/recall/F1, a confusion matrix, a corrupted-vs-natural subset breakdown so synthetic and real rows are not pooled, and a list of disagreements with both rationales side by side. This is the report you read to decide whether to trust the judge.

### Inner loop

Once everything is in place, the iteration loop is: tighten the judge prompt, re-run `python judge.py` (delete or rename `data/predictions.jsonl` if you want a full re-judge instead of `--skip-existing`), then `python meta_eval.py`, read the disagreements, repeat. Static checks and doctests only need to re-run if the dataset itself changes.

## Dataset schema

`data/apis.jsonl`, one JSON object per line.

| Field | Description |
|---|---|
| `api_id` | Dotted import path. Unique key. Example: `torch.nn.functional.relu`. |
| `module` | Dotted module path. |
| `name` | Bare function name. |
| `kind` | `function` (v0 covers module-level functions only). |
| `signature` | Reconstructed signature string. |
| `parameters` | List of `{name, annotation, default, kind}`. `kind` is one of `POSITIONAL_ONLY`, `POSITIONAL_OR_KEYWORD`, `VAR_POSITIONAL`, `KEYWORD_ONLY`, `VAR_KEYWORD`. `annotation` and `default` are source strings or null. |
| `return_annotation` | Source string or null. |
| `decorators` | List of decorator names in dotted form. |
| `docstring` | Raw docstring with original indentation, or null. |
| `source_file` | Path relative to the PyTorch root. |
| `source_line` | Line of the `def` statement. |
| `pytorch_commit` | SHA of the PyTorch checkout at extraction time. |
| `extracted_at` | ISO 8601 UTC timestamp. |
| `label` | Human label. One of `accurate`, `hallucinated`, `outdated`, `partial`, `missing`, or null. |
| `label_rationale` | Short text explanation from the labeler. |
| `labeled_by` | Identifier of the labeler. |
| `labeled_at` | ISO 8601 UTC timestamp. |
| `corrupted` | `true` if the docstring was deliberately altered for the eval set. |
| `corruption_note` | What was changed, in plain text. |
| `original_docstring` | Upstream docstring before corruption, so the row can be reverted. |
| `corrupted_at` | ISO 8601 UTC timestamp of the corruption. |

## Label definitions

These are the categories you assign by hand. Pick one per row.

- `accurate`: docstring matches the signature and observable behavior. The judge should agree.
- `hallucinated`: docstring asserts something that is not and was never true. Wrong arg name, wrong default, wrong return type, fictional behavior. The judge should catch this.
- `outdated`: docstring describes a prior version. The claim was true once but is no longer. Requires knowing the history; mark these conservatively and explain in the rationale which version the docstring fits.
- `partial`: a mix. Use rationale to spell out which sentences are accurate and which are not.
- `missing`: no docstring at all. Set this when `docstring` is null and the function is public.

The categories matter because hallucinated and outdated need different judges later. Keep them distinct.

## Running the extractor

```
python extract.py \
    --pytorch-root ../pytorch \
    --module torch.nn.functional \
    --output data/apis.jsonl \
    --limit 20
```

Flags:

- `--names a,b,c`: extract only these named functions (intersected with `--limit`).
- `--limit N`: take the first N functions in source order.
- `--include-private`: include `_`-prefixed names (default skipped).
- `--include-overloads`: include `@overload`-decorated stubs (default skipped).

The extractor merges with an existing output file: rows are matched by `api_id`, and the `label`, `label_rationale`, `labeled_by`, `labeled_at` fields are preserved across re-extraction. Other fields are overwritten with the fresh extraction.

Re-extraction does NOT remove rows for APIs that no longer exist upstream. For v0 the dataset is pinned to one PyTorch commit so this is fine.

## Hand-labeling

`label_app.py` is a Streamlit UI that reads and writes `data/apis.jsonl` in place. Writes are atomic via a tmp file and `os.replace`.

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run label_app.py
```

Streamlit needs Python 3.9+. Override the data path with `LABEL_APP_DATA=path/to/file.jsonl streamlit run label_app.py`.

The UI shows one row at a time: signature, parameters table, raw docstring, source pointer. Pick a label, write a rationale, click "Save and next". The "Skip already-labeled on Next" sidebar toggle is on by default so you don't re-walk rows you have done.

## Injecting corruptions

PyTorch's public docstrings are mostly accurate, so an unaltered ground-truth set gives the judge nothing to fail on. To get a useful eval, you need to inject a minority of deliberate errors.

`inject.py` does this safely: it stashes the upstream original in `original_docstring`, sets `corrupted=true` and `corruption_note`, and can revert. Re-running `extract.py` over a corrupted row preserves the corrupted docstring (it only refreshes signature metadata).

```
python inject.py swap torch.nn.functional.dropout \
    --find "Default: 0.5" --replace "Default: 0.1" \
    --note "Changed p default in Args section from 0.5 to 0.1"

python inject.py list
python inject.py revert torch.nn.functional.dropout
```

The `swap` command is a literal find/replace, so use a substring you can read out of the docstring in the labeling UI. It fails if the substring isn't found, or if it matches more than once (override with `--allow-multiple`).

After corrupting a row, label it in the UI. The label app shows a banner on corrupted rows and exposes the original docstring in a collapsed panel. The corruption note is a good starting point for the label rationale.

Target eval-set composition for ~20 rows: roughly 60% `accurate`, 25% `hallucinated`, 10% `outdated`, 5% `partial`. Exact counts don't matter; what matters is that every label class has at least 2 examples so meta-eval can compute per-class agreement.

## Next steps

1. Hand-label ~20 rows using `label_app.py`.
2. Write `judge.py` using the Anthropic SDK directly. One metric for v0: hallucination detection against the source signature.
3. Meta-eval the judge against the hand labels. Cohen's kappa or simple agreement rate. Target > 0.6 before trusting the judge to grade anything you have not seen.
4. Only then start the RAG generator. Use the judge as the regression test.
