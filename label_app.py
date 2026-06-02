"""Streamlit UI for hand-labeling the eval dataset.

Reads and writes data/apis.jsonl in place. Incorporates results from
run_doctests.py if available.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

DATA_PATH = Path(os.environ.get("LABEL_APP_DATA", "data/apis.jsonl"))
DOCTEST_PATH = Path(os.environ.get("LABEL_APP_DOCTEST", "data/doctest_results.jsonl"))

LABEL_OPTIONS: list[str | None] = [
    None,
    "accurate",
    "hallucinated",
    "outdated",
    "partial",
    "missing",
]
LABEL_DISPLAY = {
    None: "(unlabeled)",
    "accurate": "accurate",
    "hallucinated": "hallucinated",
    "outdated": "outdated",
    "partial": "partial",
    "missing": "missing",
}
LABEL_HELP = (
    "**accurate** — docstring matches signature and behavior.  \n"
    "**hallucinated** — claim that was never true (wrong arg, wrong default, fictional behavior).  \n"
    "**outdated** — was true once, no longer. Note the version in the rationale.  \n"
    "**partial** — mix of accurate and inaccurate. Spell out which is which.  \n"
    "**missing** — no docstring at all."
)


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_doctest_results(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    results = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                results[rec["api_id"]] = rec
            except Exception:
                continue
    return results


def save_records(path: Path, records: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def find_next_unlabeled(records: list[dict], start: int) -> int | None:
    for i in range(start, len(records)):
        if records[i].get("label") is None:
            return i
    return None


def render_sidebar(records: list[dict]) -> tuple[str, bool]:
    n = len(records)
    labeled = sum(1 for r in records if r.get("label") is not None)

    with st.sidebar:
        st.markdown(f"**Data**  \n`{DATA_PATH}`")
        st.markdown(f"**Progress**  \n{labeled} / {n} labeled")
        st.progress(labeled / n if n else 0.0)

        labeler = st.text_input("Labeler", value=os.environ.get("USER", ""))
        skip_labeled = st.checkbox("Skip already-labeled on Next", value=True)

        if st.button("Reload from disk"):
            st.session_state.records = load_records(DATA_PATH)
            st.session_state.doctests = load_doctest_results(DOCTEST_PATH)
            st.rerun()

        st.divider()
        st.caption("Label definitions")
        st.markdown(LABEL_HELP)

    return labeler, skip_labeled


def render_record(rec: dict) -> None:
    if rec.get("corrupted"):
        st.warning(
            f"**Synthetic row.** The docstring below was deliberately altered "
            f"for the eval set. Note: _{rec.get('corruption_note') or '(no note)'}_",
            icon="⚠️",
        )

    # --- Doctest Results ---
    doctest_res = st.session_state.doctests.get(rec["api_id"])
    if doctest_res:
        status = doctest_res["status"]
        if status == "passed":
            st.success(f"**Doctest Passed** ({doctest_res.get('total_tests')} tests)", icon="✅")
        elif status == "failed":
            st.error(f"**Doctest Failed** ({doctest_res.get('failures')} failures)", icon="❌")
            with st.expander("Show failure traceback"):
                st.code(doctest_res.get("output"), language="text")
        elif status == "skipped":
            st.info(f"**Doctest Skipped**: {doctest_res.get('reason')}", icon="ℹ️")

    st.code(rec["signature"], language="python")
    st.caption(
        f"`{rec.get('source_file')}:{rec.get('source_line')}` "
        f"@ commit `{(rec.get('pytorch_commit') or 'unknown')[:12]}`"
    )

    if rec.get("decorators"):
        st.caption("Decorators: " + ", ".join(f"@{d}" for d in rec["decorators"]))

    if rec.get("parameters"):
        st.markdown("**Parameters**")
        st.dataframe(rec["parameters"], use_container_width=True, hide_index=True)

    st.markdown("**Docstring**")
    docstring = rec.get("docstring")
    if docstring:
        st.code(docstring, language="text")
    else:
        st.info("No docstring on this function.")

    if rec.get("corrupted") and rec.get("original_docstring"):
        with st.expander("Show upstream original docstring"):
            st.code(rec["original_docstring"], language="text")


def main() -> None:
    st.set_page_config(page_title="Doc labeler", layout="wide")

    if "records" not in st.session_state:
        st.session_state.records = load_records(DATA_PATH)
        st.session_state.idx = 0
    
    if "doctests" not in st.session_state:
        st.session_state.doctests = load_doctest_results(DOCTEST_PATH)

    records: list[dict] = st.session_state.records
    n = len(records)

    labeler, skip_labeled = render_sidebar(records)

    if n == 0:
        st.error(
            f"No records in `{DATA_PATH}`. Run `extract.py` first, then reload."
        )
        return

    st.session_state.idx = max(0, min(st.session_state.idx, n - 1))
    idx = st.session_state.idx
    rec = records[idx]

    st.title(rec["api_id"])
    nav_cols = st.columns([1, 2, 1])
    with nav_cols[1]:
        new_idx = st.number_input(
            "Row",
            min_value=1,
            max_value=n,
            value=idx + 1,
            label_visibility="collapsed",
        )
        if int(new_idx) - 1 != idx:
            st.session_state.idx = int(new_idx) - 1
            st.rerun()
    st.caption(f"Row {idx + 1} of {n}")

    left, right = st.columns([3, 2])
    with left:
        render_record(rec)

    with right:
        st.markdown("**Label**")
        current_label = rec.get("label")
        try:
            default_idx = LABEL_OPTIONS.index(current_label)
        except ValueError:
            default_idx = 0
        label = st.radio(
            "Label",
            options=LABEL_OPTIONS,
            format_func=lambda v: LABEL_DISPLAY[v],
            index=default_idx,
            key=f"label_{idx}",
            label_visibility="collapsed",
        )
        rationale = st.text_area(
            "Rationale (what is wrong, or why it is correct)",
            value=rec.get("label_rationale") or "",
            height=220,
            key=f"rationale_{idx}",
        )

        st.divider()
        btn_cols = st.columns(3)
        prev_clicked = btn_cols[0].button("Previous", disabled=idx == 0)
        save_clicked = btn_cols[1].button("Save", type="secondary")
        next_clicked = btn_cols[2].button(
            "Save and next", type="primary", disabled=idx == n - 1
        )

        if prev_clicked:
            st.session_state.idx = max(0, idx - 1)
            st.rerun()

        if save_clicked or next_clicked:
            now = datetime.now(timezone.utc).isoformat()
            rec["label"] = label
            rec["label_rationale"] = rationale.strip() or None
            if label is not None:
                rec["labeled_by"] = labeler.strip() or None
                rec["labeled_at"] = now
            else:
                rec["labeled_by"] = None
                rec["labeled_at"] = None
            save_records(DATA_PATH, records)

            if next_clicked:
                if skip_labeled:
                    nxt = find_next_unlabeled(records, idx + 1)
                    st.session_state.idx = nxt if nxt is not None else min(n - 1, idx + 1)
                else:
                    st.session_state.idx = min(n - 1, idx + 1)
            st.toast("Saved", icon="✅")
            st.rerun()


if __name__ == "__main__":
    main()
