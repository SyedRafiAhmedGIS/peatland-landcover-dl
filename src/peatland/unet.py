"""Small U-Net (segmentation_models_pytorch, ResNet-18 encoder, trained from scratch)."""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np

from .labels import NODATA
from .tiling import epoch_indices, tile_slices


def _gather(xn: np.ndarray, lab: np.ndarray, tile_split: np.ndarray, code: int, tile: int):
    idx = np.argwhere(tile_split == code)
    if len(idx) == 0:
        return None, None, idx
    xs = np.stack([xn[:, slice_r, slice_c] for slice_r, slice_c in (tile_slices(r, c, tile) for r, c in idx)])
    ys = np.stack([lab[slice_r, slice_c] for slice_r, slice_c in (tile_slices(r, c, tile) for r, c in idx)])
    return xs, ys.astype(np.int64), idx


def class_weights(y: np.ndarray) -> np.ndarray:
    """Square-root inverse-frequency class weights (0 for classes absent from training)."""
    cnt = np.bincount(y[y != NODATA].ravel(), minlength=7).astype(np.float64)
    w = np.where(cnt > 0, (cnt.sum() / (7 * np.maximum(cnt, 1))) ** 0.5, 0.0)
    return w / w[cnt > 0].mean()


def train_unet(xn: np.ndarray, lab: np.ndarray, tile_split: np.ndarray, cfg: dict, model_dir: str | Path, on_epoch=None):
    """Train with early stopping (not before min_epochs) and a safety time cap.

    Returns (full-grid prediction map, history, info). `on_epoch(epoch, train_loss, val_loss, seconds)` is called each epoch.
    """
    import segmentation_models_pytorch as smp
    import torch
    import torch.nn as nn

    seed, tile, ucfg = cfg["seed"], cfg["tiling"]["tile"], cfg["unet"]
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    budget = ucfg["budget_seconds_gpu"] if use_amp else ucfg["budget_seconds_cpu"]

    xtr, ytr, _ = _gather(xn, lab, tile_split, 0, tile)
    xva, yva, _ = _gather(xn, lab, tile_split, 1, tile)
    xte, _, te_tiles = _gather(xn, lab, tile_split, 2, tile)
    if xtr is None or xva is None or xte is None:
        raise ValueError("train, validation and test tiles are all required")
    xtr, ytr, xva, yva = (torch.from_numpy(a) for a in (xtr, ytr, xva, yva))

    crit = nn.CrossEntropyLoss(weight=torch.tensor(class_weights(ytr.numpy()), dtype=torch.float32, device=device),
                               ignore_index=NODATA)
    model = smp.Unet(ucfg["encoder"], encoder_weights=None, in_channels=xn.shape[0], classes=7,
                     decoder_channels=(128, 64, 32, 16, 16)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=ucfg["lr"], weight_decay=1e-4)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except Exception:  # older torch
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    rng = np.random.default_rng(seed)

    def augment(xb, yb):
        k = int(rng.integers(4))
        xb, yb = torch.rot90(xb, k, (2, 3)), torch.rot90(yb, k, (1, 2))
        if rng.random() < 0.5:
            xb, yb = torch.flip(xb, (3,)), torch.flip(yb, (2,))
        return xb, yb

    hist: list[tuple[float, float]] = []
    budget_hit = False
    best, bad, best_state, t0 = float("inf"), 0, None, time.time()
    for epoch in range(1, ucfg["epochs"] + 1):
        model.train()
        perm = torch.from_numpy(epoch_indices(len(xtr), ucfg["tiles_per_epoch"], rng))   # exactly tiles_per_epoch tiles
        tl, nb = 0.0, 0
        for i in range(0, len(perm), ucfg["batch"]):
            ib = perm[i:i + ucfg["batch"]]
            xb, yb = (t.to(device) for t in augment(xtr[ib], ytr[ib]))
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tl, nb = tl + float(loss), nb + 1
            if time.time() - t0 > budget:
                budget_hit = True
                break
        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(xva), 32):
                xb, yb = xva[i:i + 32].to(device), yva[i:i + 32].to(device)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    vl += float(crit(model(xb), yb)) * len(xb)
                vn += len(xb)
        hist.append((tl / max(nb, 1), vl / max(vn, 1)))
        improved = hist[-1][1] < best - 1e-4
        if improved:
            best, bad, best_state = hist[-1][1], 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
        print(f"epoch {epoch:2d} | train {hist[-1][0]:.3f} | val {hist[-1][1]:.3f} | {time.time() - t0:5.0f}s{' *' if improved else ''}")
        if on_epoch:
            on_epoch(epoch, hist[-1][0], hist[-1][1], time.time() - t0)
        if bad >= ucfg["patience"] and epoch >= ucfg["min_epochs"]:
            print("early stopping (validation loss stopped improving)")
            break
        if time.time() - t0 > budget:
            budget_hit = True
            print("time cap reached - this is NOT a fixed-epoch run")
            break
    best_state = best_state or copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    torch.save(best_state, Path(model_dir) / "unet_best.pth")

    pred = np.full(lab.shape, NODATA, np.uint8)
    with torch.no_grad():
        for i in range(0, len(xte), 32):
            out = model(torch.from_numpy(xte[i:i + 32]).to(device)).argmax(1).cpu().numpy().astype(np.uint8)
            for (r, c), p in zip(te_tiles[i:i + 32], out, strict=True):
                pred[tile_slices(r, c, tile)] = p
    info = {"epochs_run": len(hist), "time_budget_hit": budget_hit, "min_epochs": ucfg["min_epochs"], "device": device.type,
            "tiles_per_epoch": int(ucfg["tiles_per_epoch"]), "train_tiles": int(len(xtr))}
    return pred, hist, info
