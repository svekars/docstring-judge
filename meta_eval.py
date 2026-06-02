"""Meta-evaluation: compare judge predictions against hand labels.

Reports the inclusion/exclusion breakdown, simple agreement, Cohen's kappa,
per-class precision/recall/F1, a confusion matrix, a corrupted-vs-natural
subset breakdown so synthetic and real rows are not pooled, and the list of
disagreements with both rationales side by side.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LABEL_ORDER = ["accurate", "hallucinated", "outdated", "partial", "missing", "invalid"]


def label_sort_key(c: str) -> tuple[int, str]:
    return (LABEL_ORDER.index(c) if c in LABEL_ORDER else 99, c)


def load_jsonl(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    data: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "api_id" in rec:
                data[rec["api_id"]] = rec
    return data


def cohens_kappa(y_true: list[str], y_pred: list[str]) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    classes = sorted(set(y_true) | set(y_pred))
    p_o = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n
    p_e = sum((y_true.count(c) / n) * (y_pred.count(c) / n) for c in classes)
    if p_e >= 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def per_class_prf(y_true: list[str], y_pred: list[str], classes: list[str]) -> dict[str, dict]:
    metrics: dict[str, dict] = {}
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        metrics[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return metrics


def print_confusion(y_true: list[str], y_pred: list[str]) -> None:
    classes = sorted(set(y_true) | set(y_pred), key=label_sort_key)
    col = max(14, max(len(c) for c in classes) + 2)
    print("\nConfusion matrix (rows = hand label, cols = prediction):")
    header = " " * 14 + "".join(f"{c:<{col}}" for c in classes)
    print(header)
    for true_label in classes:
        row = f"{true_label:<14}"
        for pred_label in classes:
            count = sum(1 for t, p in zip(y_true, y_pred) if t == true_label and p == pred_label)
            row += f"{count:<{col}}"
        print(row)


def subset_metrics(name: str, y_true: list[str], y_pred: list[str]) -> None:
    n = len(y_true)
    if n == 0:
        print(f"\n[{name}] no rows")
        return
    matches = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    agreement = matches / n
    if n < 3:
        print(f"\n[{name}] n={n}, agreement={agreement:.1%}, kappa undefined (n<3)")
    else:
        kappa = cohens_kappa(y_true, y_pred)
        print(f"\n[{name}] n={n}, agreement={agreement:.1%}, kappa={kappa:.3f}")


def print_disagreements(rows: list[tuple[str, dict, dict]]) -> None:
    print(f"\nDisagreements ({len(rows)}):")
    if not rows:
        print("  (none)")
        return
    for aid, gt_rec, pred_rec in rows:
        hand = gt_rec.get("label")
        hand_rat = gt_rec.get("label_rationale") or "(no rationale)"
        pred = pred_rec.get("prediction")
        pred_rat = pred_rec.get("rationale") or "(no rationale)"
        marker = " [corrupted]" if gt_rec.get("corrupted") else ""
        print(f"\n  {aid}{marker}")
        print(f"    hand:  {str(hand):<14} {hand_rat}")
        print(f"    judge: {str(pred):<14} {pred_rat}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ground-truth", type=Path, default=Path("data/apis.jsonl"))
    parser.add_argument("--predictions", type=Path, default=Path("data/predictions.jsonl"))
    args = parser.parse_args()

    gt = load_jsonl(args.ground_truth)
    preds = load_jsonl(args.predictions)

    if not gt:
        print(f"error: ground truth {args.ground_truth} empty or missing", file=sys.stderr)
        return 1
    if not preds:
        print(f"error: predictions {args.predictions} empty or missing", file=sys.stderr)
        return 1

    all_ids = set(gt)
    labeled_ids = {aid for aid in all_ids if gt[aid].get("label") is not None}
    unlabeled_ids = all_ids - labeled_ids
    with_pred_ids = {aid for aid in all_ids if aid in preds and "prediction" in preds[aid]}
    pred_error_ids = {aid for aid in all_ids if aid in preds and "error" in preds[aid]}
    predictions_no_label = with_pred_ids - labeled_ids
    included = sorted(labeled_ids & with_pred_ids)

    print("Eval scope")
    print(f"  total api_ids in ground truth:     {len(all_ids)}")
    print(f"  with hand label:                   {len(labeled_ids)}")
    print(f"  unlabeled (excluded):              {len(unlabeled_ids)}")
    print(f"  with non-error prediction:         {len(with_pred_ids)}")
    print(f"  with prediction error (excluded):  {len(pred_error_ids)}")
    print(f"  predicted but unlabeled:           {len(predictions_no_label)}  (label these to expand the eval set)")
    print(f"  included in metrics:               {len(included)}")

    if not included:
        print("\nerror: no rows have both a hand label and a non-error prediction", file=sys.stderr)
        if pred_error_ids:
            print("\nFirst few prediction errors:", file=sys.stderr)
            for aid in list(pred_error_ids)[:3]:
                print(f"  {aid}: {preds[aid].get('error')}", file=sys.stderr)
        return 1

    y_true = [gt[aid]["label"] for aid in included]
    y_pred = [preds[aid]["prediction"] for aid in included]

    print("\nPer-row comparison:")
    print(f"{'api_id':<52} {'hand':<14} {'judge':<14} {'match':<6} {'corrupted'}")
    print("-" * 100)
    for aid in included:
        hand = gt[aid]["label"]
        pred = preds[aid]["prediction"]
        match = "yes" if hand == pred else "no"
        corrupted = "yes" if gt[aid].get("corrupted") else ""
        aid_short = aid if len(aid) <= 50 else aid[:47] + "..."
        print(f"{aid_short:<52} {str(hand):<14} {str(pred):<14} {match:<6} {corrupted}")

    matches = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    agreement = matches / len(included)
    kappa = cohens_kappa(y_true, y_pred)
    print(f"\nOverall: n={len(included)}, agreement={agreement:.1%}, Cohen's kappa={kappa:.3f}")
    if len(included) < 20:
        print("  (n < 20: kappa is noisy; expand the eval set before trusting it)")

    seen_classes = sorted(set(y_true) | set(y_pred), key=label_sort_key)
    prf = per_class_prf(y_true, y_pred, seen_classes)
    print("\nPer-class metrics:")
    print(f"  {'class':<14} {'precision':<11} {'recall':<11} {'f1':<8} {'support':<8}")
    for c in seen_classes:
        m = prf[c]
        print(
            f"  {c:<14} {m['precision']:<11.3f} {m['recall']:<11.3f} "
            f"{m['f1']:<8.3f} {m['support']:<8}"
        )

    print_confusion(y_true, y_pred)

    corrupted_included = [aid for aid in included if gt[aid].get("corrupted")]
    natural_included = [aid for aid in included if not gt[aid].get("corrupted")]
    if corrupted_included:
        subset_metrics(
            "corrupted rows (oracle test: judge should catch these)",
            [gt[aid]["label"] for aid in corrupted_included],
            [preds[aid]["prediction"] for aid in corrupted_included],
        )
    if natural_included:
        subset_metrics(
            "natural rows (real PyTorch docs)",
            [gt[aid]["label"] for aid in natural_included],
            [preds[aid]["prediction"] for aid in natural_included],
        )

    disagreements = [
        (aid, gt[aid], preds[aid])
        for aid in included
        if gt[aid]["label"] != preds[aid]["prediction"]
    ]
    print_disagreements(disagreements)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
