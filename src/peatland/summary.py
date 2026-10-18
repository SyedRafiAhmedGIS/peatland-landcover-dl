"""Honest, computed summary messages shared by the CLI and the Streamlit app (one source of truth)."""

from __future__ import annotations

LIMITATIONS = (
    "CORINE is 100 m with a 25 ha minimum mapping unit (coarse against 10 m imagery); check its reference year against the 2018 imagery. "
    "Summer 2018 was unusually hot and dry in Ireland, so the composite may not be typical. One area, one season, one block split "
    "with few peat blocks (results are noisy). No field validation. The small U-Net is trained from scratch. "
    "This is not a peatland-condition assessment."
)
PROXY_LIMITATION = "Labels are a PROXY (ESA WorldCover: 10 m, no peat class, release year differs from the 2018 imagery)."
CONVERGENCE_KAPPA = 0.10
MIN_PEAT_TEST_PX = 5_000


TIE_MARGIN = 0.02


def _verdict(rf: dict, un: dict) -> str:
    """Compare the two models on macro F1, peat F1 and kappa (not on one metric alone) and say who leads on which."""
    unet_ahead, rf_ahead = [], []
    for key, label in (("macro_f1", "macro F1"), ("peat_f1", "peat F1"), ("kappa", "kappa")):
        a, b = rf.get(key), un.get(key)
        if a is None or b is None:
            continue
        if b - a > TIE_MARGIN:
            unet_ahead.append(label)
        elif a - b > TIE_MARGIN:
            rf_ahead.append(label)
    if unet_ahead and rf_ahead:
        return f"mixed result: U-Net ahead on {', '.join(unet_ahead)}; Random Forest ahead on {', '.join(rf_ahead)}"
    if unet_ahead:
        return f"U-Net outperforms the Random Forest on {', '.join(unet_ahead)}"
    if rf_ahead:
        return f"Random Forest outperforms the U-Net on {', '.join(rf_ahead)}"
    return "no clear difference between the two models"


def headline(results: dict, meta: dict) -> list[tuple[str, str]]:
    """Return [(level, message)] with level in {"error", "warning", "info", "success"}, computed from the metrics only."""
    msgs: list[tuple[str, str]] = []
    if meta.get("label_source") != "corine":
        msgs.append(("warning", "These results use PROXY labels (WorldCover). Peat F1 is NOT a real peat-bog score."))
    if meta.get("mode") == "demo":
        msgs.append(("warning", "DEMO mode - fewer scenes, smaller Random Forest and a shorter U-Net cap for a fast live run. "
                                "Do not quote these numbers as final results; use mode 'full' for the reported run."))
    rf, un = results.get("RandomForest"), results.get("UNet")
    converged = None
    if un is not None:
        converged = un["kappa"] >= CONVERGENCE_KAPPA
        if not converged:
            msgs.append(("error", f"The U-Net did not converge (kappa {un['kappa']:.2f}). Do NOT interpret its scores; re-run "
                                  "(a GPU runtime helps)."))
        if meta.get("unet_time_budget_hit"):
            msgs.append(("warning", "The U-Net hit its safety time cap - this was NOT a fixed-epoch run."))
    peat_px = max((r.get("peat_support_px", 0) for r in results.values()), default=0)
    if peat_px < MIN_PEAT_TEST_PX:
        msgs.append(("warning", f"Only {peat_px:,} peat pixels in the test blocks - peat metrics are very noisy."))
    if rf and un:
        f = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
        if not converged:
            verdict = "the U-Net did NOT converge, so this comparison is invalid"
        else:
            verdict = _verdict(rf, un)
        msgs.append(("info", f"Headline: macro F1 RF {rf['macro_f1']:.3f} vs U-Net {un['macro_f1']:.3f}; peat F1 RF {f(rf['peat_f1'])} vs "
                             f"U-Net {f(un['peat_f1'])}; kappa RF {rf['kappa']:.3f} vs U-Net {un['kappa']:.3f} -> {verdict} on this split "
                             f"(test peat pixels: {peat_px:,}). One split, no confidence interval: "
                             f"gaps under ~{TIE_MARGIN:.2f} are noise."))
    elif rf or un:
        name, r = ("Random Forest", rf) if rf else ("U-Net", un)
        msgs.append(("info", f"{name}: accuracy {r['accuracy']:.3f}, macro F1 {r['macro_f1']:.3f}, kappa {r['kappa']:.3f}."))
    limit = LIMITATIONS if meta.get("label_source") == "corine" else PROXY_LIMITATION + " " + LIMITATIONS
    msgs.append(("info", "Read these as honest limitations: " + limit))
    return msgs
