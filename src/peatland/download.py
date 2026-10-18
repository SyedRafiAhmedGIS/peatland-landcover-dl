"""Download stage: Sentinel-2 median composite (Planetary Computer) and label rasters.

Network libraries are imported inside functions so the rest of the package (and the tests) work offline.
"""

from __future__ import annotations

import calendar
import json
from pathlib import Path

import numpy as np

from . import cache
from .grid import Grid, grid_lonlat, make_grid
from .labels import CORINE_HELP, NODATA, PEAT, find_label_file, remap_corine, remap_worldcover
from .preprocess import scl_valid

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def _catalog():
    import planetary_computer
    import pystac_client

    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def load_corine(path: Path, grid: Grid) -> tuple[np.ndarray, str]:
    """Warp a CORINE GeoTIFF onto the study grid (nearest neighbour) and remap to 7 classes."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    raw = np.zeros(grid.shape, np.int16)
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1), destination=raw, dst_transform=grid.affine, dst_crs=grid.crs,
            resampling=Resampling.nearest, src_nodata=src.nodata, dst_nodata=0,
        )
    return remap_corine(raw)


def load_worldcover(grid: Grid, year: int) -> tuple[np.ndarray, str]:
    """Load ESA WorldCover on the study grid. PROXY labels: no peat class exists."""
    import odc.stac
    from odc.geo.geobox import GeoBox

    items = list(_catalog().search(collections=["esa-worldcover"], bbox=grid_lonlat(grid)).item_collection())
    if not items:
        raise RuntimeError("No WorldCover items found for this area.")

    def item_year(it) -> int:
        s = str(it.properties.get("start_datetime") or it.datetime or "0000")
        return int(s[:4]) if s[:4].isdigit() else 0

    best = min({item_year(i) for i in items}, key=lambda y: abs(y - year))
    gb = GeoBox(grid.shape, grid.affine, grid.crs)
    ds = odc.stac.load([i for i in items if item_year(i) == best], bands=["map"], geobox=gb, resampling="nearest")
    arr = np.asarray(ds["map"].values)
    arr = arr[0] if arr.ndim == 3 else arr
    return remap_worldcover(arr), f"worldcover_{best}"


def date_range(year: int, m0: int, m1: int) -> str:
    """STAC interval from the first day of month m0 to the LAST day of month m1 (28-31 days, leap years handled)."""
    return f"{year}-{m0:02d}-01/{year}-{m1:02d}-{calendar.monthrange(year, m1)[1]:02d}"


def pick_scenes(grid: Grid, cfg: dict) -> tuple[list, dict, int]:
    """Search Sentinel-2 L2A and keep the least cloudy scenes per MGRS tile."""
    img = cfg["imagery"]
    y, m0, m1 = int(img["year"]), int(img["months"][0]), int(img["months"][1])
    found = _catalog().search(
        collections=["sentinel-2-l2a"], bbox=grid_lonlat(grid),
        datetime=date_range(y, m0, m1), query={"eo:cloud_cover": {"lt": img["max_cloud"]}},
    ).item_collection()
    by_tile: dict[str, list] = {}
    for it in found:
        by_tile.setdefault(it.properties.get("s2:mgrs_tile", "?"), []).append(it)
    selected: list = []
    for items in by_tile.values():
        items.sort(key=lambda i: i.properties.get("eo:cloud_cover", 100))
        selected += items[: img["max_scenes_per_tile"]]
    return selected, {t: len(v) for t, v in by_tile.items()}, len(found)


def baseline_offset(baselines: list[str]) -> tuple[float, bool]:
    """Processing baseline >= 04.00 adds a +1000 DN offset to L2A reflectance; it must be removed or NDVI/NDMI are biased.

    Returns (offset to subtract, whether the scenes mix baselines). The offset is only removed if ALL scenes carry it.
    """
    flags = set()
    for b in baselines:
        try:
            flags.add(float(b) >= 4.0)
        except ValueError:
            flags.add(None)
    return (1000.0 if flags == {True} else 0.0), len(flags) > 1


def build_composite(grid: Grid, cfg: dict) -> tuple[np.ndarray, dict]:
    """Per-scene SCL cloud masking, then a median over time.

    Returns ((bands, H, W) float32 DN with NaN = no data, info dict with scenes / baselines / offset).
    """
    import odc.stac
    from dask.diagnostics import ProgressBar
    from odc.geo.geobox import GeoBox

    bands = cfg["imagery"]["bands"]
    scenes, per_tile, n_found = pick_scenes(grid, cfg)
    if not scenes:
        raise RuntimeError("No Sentinel-2 scenes found - widen the dates or the cloud limit in config.yaml.")
    baselines = sorted({str(i.properties.get("s2:processing_baseline", "?")) for i in scenes})
    offset, mixed = baseline_offset(baselines)
    print(f"{n_found} scenes found; using {len(scenes)} clearest ({per_tile})")
    print(f"Processing baselines {baselines} -> reflectance offset removed: {offset:.0f} DN")
    if mixed:
        print("WARNING: scenes mix processing baselines; a per-scene offset correction is NOT applied (note in limitations).")
    gb = GeoBox(grid.shape, grid.affine, grid.crs)
    ds = odc.stac.load(
        scenes, bands=[*bands, "SCL"], geobox=gb, groupby="solar_day", chunks={"x": 1024, "y": 1024},
        resampling={**{b: "bilinear" for b in bands}, "SCL": "nearest"}, fail_on_error=False,
    )
    refl = ds[bands].to_array("band").astype("float32")
    masked = refl.where(scl_valid(ds["SCL"]) & (refl > 0)) - offset
    with ProgressBar():
        comp = masked.chunk({"time": -1, "band": 1}).median("time", skipna=True).compute().values
    info = {"n_scenes": len(scenes), "baselines": baselines, "offset": offset, "mixed_baselines": mixed}
    return comp.astype("float32"), info


def run_download(cfg: dict, label_file: str | None = None, allow_proxy: bool = False) -> None:
    """Choose the study area (needs enough peat labels), build the composite and save everything to data/interim."""
    interim = Path(cfg["paths"]["interim"])
    interim.mkdir(parents=True, exist_ok=True)
    img, tiling = cfg["imagery"], cfg["tiling"]

    path = find_label_file(label_file or cfg["labels"]["corine_path"]) if cfg["labels"]["source"] == "corine" else None
    use_corine = path is not None
    if cfg["labels"]["source"] == "corine" and not use_corine:
        if not allow_proxy:
            raise SystemExit(CORINE_HELP)
        print("WARNING: no CORINE file - using WorldCover PROXY labels (no peat class).")

    best = None
    for area in cfg["study_areas"]:
        grid = make_grid(tuple(area["bbox"]), area.get("fraction", 1.0), img["resolution"], img["epsg"], tiling["block"])
        lab, scheme = load_corine(path, grid) if use_corine else load_worldcover(grid, img["year"])
        n_peat = int((lab == PEAT).sum())
        print(f"{area['name']}: grid {grid.shape}, {scheme}, peat labels {n_peat:,} px")
        cand = {"area": area["name"], "grid": grid, "lab": lab, "scheme": scheme, "n_peat": n_peat}
        if best is None or n_peat > best["n_peat"]:
            best = cand
        if n_peat >= cfg["labels"]["min_peat_px"]:
            best = cand
            break
    if best is None or best["n_peat"] == 0:
        raise SystemExit("No peat-bog pixels in any study area - check the label file. Stopping instead of inventing data.")

    g = best["grid"]
    area_cfg = next(a for a in cfg["study_areas"] if a["name"] == best["area"])
    key = cache.cache_key(area_cfg["bbox"], g.shape, list(g.affine)[:6], img)
    cpath = cache.cache_path(interim, key)
    cached = cpath.exists()
    if cached:
        comp, info = cache.load_composite(cpath)
        print(f"Composite loaded from cache ({info['n_scenes']} scenes). Delete {interim}/ to rebuild it.")
    else:
        comp, info = build_composite(g, cfg)
        comp = cache.save_composite(cpath, comp, info)
    np.save(interim / "labels_raw.npy", best["lab"].astype(np.uint8))
    meta = {
        "area": best["area"], "grid": best["grid"].to_dict(), "label_scheme": best["scheme"],
        "label_source": "corine" if use_corine else best["scheme"], "peat_label_px": best["n_peat"],
        "coverage": float(np.isfinite(comp).all(0).mean()), "nodata": NODATA,
        "composite_file": str(cpath), "composite_cached": cached, **{f"s2_{k}": v for k, v in info.items()},
    }
    (interim / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Composite {comp.shape} ready; labels saved to {interim}/")
