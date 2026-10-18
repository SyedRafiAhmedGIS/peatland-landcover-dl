"""Metrics: overall accuracy, per-class P/R/F1, macro F1, kappa, confusion matrix. Peat reported separately."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, precision_recall_fscore_support

from .labels import CLASS_NAMES, PEAT

LABELS = list(range(7))


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all metrics. Macro F1 averages only classes present in the reference."""
    pr, rc, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    present = sup > 0
    has_peat = bool(sup[PEAT])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1[present].mean()),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "peat_f1": float(f1[PEAT]) if has_peat else None,
        "peat_precision": float(pr[PEAT]) if has_peat else None,
        "peat_recall": float(rc[PEAT]) if has_peat else None,
        "peat_support_px": int(sup[PEAT]),
        "per_class": {
            CLASS_NAMES[i]: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]), "support": int(sup[i])}
            for i in LABELS
        },
    }


def normalised_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Row-normalised confusion matrix (each row sums to 1 where the class is present)."""
    m = confusion_matrix(y_true, y_pred, labels=LABELS).astype(float)
    return m / np.maximum(m.sum(1, keepdims=True), 1)


def write_metrics(results: dict, meta: dict, reports: str | Path) -> None:
    """Write metrics.json plus one file per model."""
    reports = Path(reports)
    reports.mkdir(parents=True, exist_ok=True)
    for name, res in results.items():
        (reports / f"metrics_{name}.json").write_text(json.dumps({**res, "model": name, **meta}, indent=2))
    (reports / "metrics.json").write_text(json.dumps({**meta, "models": results}, indent=2))


def results_table_md(results: dict) -> str:
    """Markdown table for the README, generated from real metrics only."""
    fmt = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
    rows = ["| Model | Overall accuracy | Macro F1 | Peat-bog F1 | Peat precision | Peat recall |", "|---|---|---|---|---|---|"]
    for name, r in results.items():
        rows.append(f"| {name} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | {fmt(r['peat_f1'])} | "
                    f"{fmt(r['peat_precision'])} | {fmt(r['peat_recall'])} |")
    return "\n".join(rows)
