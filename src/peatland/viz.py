"""Figures (matplotlib, headless)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from .evaluate import normalised_confusion  # noqa: E402
from .labels import CLASS_COLORS, CLASS_NAMES, NODATA, PEAT  # noqa: E402

CMAP = ListedColormap(CLASS_COLORS)


def _lab_img(m: np.ndarray, step: int) -> np.ndarray:
    return np.where(m == NODATA, 7, m)[::step, ::step]


def _class_legend(ax, **kw) -> None:
    ax.legend(handles=[mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(7)], **kw)


def plot_study_area(comp_rgb: np.ndarray, lab: np.ndarray, title: str, out: Path, step: int = 3) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.2))
    ax[0].imshow(np.nan_to_num(comp_rgb[::step, ::step]))
    ax[0].set_title("Sentinel-2 composite (true colour)", fontweight="bold")
    ax[1].imshow(_lab_img(lab, step), cmap=CMAP, vmin=0, vmax=7, interpolation="nearest")
    ax[1].set_title(title, fontweight="bold")
    _class_legend(ax[1], loc="lower left", fontsize=8, framealpha=0.9)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_split(tile_split: np.ndarray, seed: int, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7))
    cmap = ListedColormap(["#ffffff", "#1f77b4", "#2ca02c", "#d62728", "#9ca3af"])
    ax.imshow(tile_split + 1, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    ax.set_title(f"Spatial split (cell = one tile) - seed {seed}", fontweight="bold")
    ax.legend(handles=[mpatches.Patch(color=c, label=n) for c, n in
                       zip(["#1f77b4", "#2ca02c", "#d62728", "#9ca3af"], ["Train", "Validation", "Test", "Buffer"], strict=True)],
              loc="lower right")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_results(results: dict, y_true: np.ndarray, preds: dict, out: Path) -> None:
    names = list(preds)
    fig, ax = plt.subplots(1, len(names) + 1, figsize=(6.5 * (len(names) + 1), 5.6))
    for a, nm in zip(ax[:-1], names, strict=True):
        m = normalised_confusion(y_true, preds[nm])
        a.imshow(m, cmap="Blues", vmin=0, vmax=1)
        for i in range(7):
            for j in range(7):
                a.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if m[i, j] > 0.5 else "black")
        a.set_xticks(range(7))
        a.set_yticks(range(7))
        a.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
        a.set_yticklabels(CLASS_NAMES, fontsize=8)
        a.set_xlabel("predicted")
        a.set_ylabel("reference")
        a.set_title(f"{nm} - confusion (row-normalised)", fontweight="bold")
    width = 0.8 / len(names)
    for k, nm in enumerate(names):
        ax[-1].bar(np.arange(7) + (k - (len(names) - 1) / 2) * width,
                   [results[nm]["per_class"][c]["f1"] for c in CLASS_NAMES], width, label=nm)
    ax[-1].set_xticks(range(7))
    ax[-1].set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax[-1].set_ylim(0, 1.18)
    ax[-1].set_ylabel("F1")
    ax[-1].axvspan(PEAT - 0.5, PEAT + 0.5, color="#6d4c41", alpha=0.12)
    ax[-1].set_title("Per-class F1 (peat highlighted)", fontweight="bold")
    ax[-1].legend()
    ax[-1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_test_maps(ref: np.ndarray, preds: dict, test_mask: np.ndarray, best: str, out: Path, step: int = 3) -> None:
    n = len(preds) + 2
    fig, ax = plt.subplots(1, n, figsize=(5.5 * n, 5.8))
    ax[0].imshow(_lab_img(np.where(test_mask, ref, NODATA), step), cmap=CMAP, vmin=0, vmax=7, interpolation="nearest")
    ax[0].set_title("Reference (test blocks only)", fontweight="bold")
    _class_legend(ax[0], loc="lower left", fontsize=7, framealpha=0.9)
    for a, (nm, p) in zip(ax[1:-1], preds.items(), strict=True):
        a.imshow(_lab_img(np.where(test_mask, p, NODATA), step), cmap=CMAP, vmin=0, vmax=7, interpolation="nearest")
        a.set_title(nm, fontweight="bold")
    bm = preds[best]
    agree = np.zeros(ref.shape, np.uint8)
    agree[test_mask] = 1
    agree[test_mask & (ref == PEAT) & (bm == PEAT)] = 2
    agree[test_mask & (ref == PEAT) & (bm != PEAT)] = 3
    agree[test_mask & (ref != PEAT) & (bm == PEAT)] = 4
    ax[-1].imshow(agree[::step, ::step], cmap=ListedColormap(["#ffffff", "#e5e7eb", "#16a34a", "#dc2626", "#f59e0b"]),
                  vmin=0, vmax=4, interpolation="nearest")
    ax[-1].set_title(f"Peat bogs - {best}", fontweight="bold")
    ax[-1].legend(handles=[mpatches.Patch(color=c, label=n_) for c, n_ in
                           zip(["#16a34a", "#dc2626", "#f59e0b"], ["correct peat", "missed peat", "false alarm"], strict=True)],
                  loc="lower left", fontsize=8)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_training(hist: list, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot([h[0] for h in hist], "o-", label="train")
    ax.plot([h[1] for h in hist], "s-", label="validation")
    ax.set_yscale("log")
    ax.set(xlabel="epoch", ylabel="weighted CE loss (log scale)", title="U-Net training curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
