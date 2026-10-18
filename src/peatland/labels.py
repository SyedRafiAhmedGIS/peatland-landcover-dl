"""Class scheme and label remapping (CORINE and ESA WorldCover -> 7 classes)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

NODATA = 255
PEAT = 4
CLASS_NAMES = [
    "Artificial", "Agriculture", "Forest", "Open semi-natural", "Peat bogs", "Other wetlands", "Water",
]
CLASS_COLORS = ["#d62728", "#ffd966", "#2e7d32", "#a8d08d", "#6d4c41", "#26a69a", "#1e88e5", "#ffffff"]

# The 44 CORINE classes in legend order. Copernicus GeoTIFFs store either the index (1-44) or the 3-digit code.
CLC_CODES = [
    111, 112, 121, 122, 123, 124, 131, 132, 133, 141, 142, 211, 212, 213, 221, 222, 223, 231, 241, 242, 243,
    244, 311, 312, 313, 321, 322, 323, 324, 331, 332, 333, 334, 335, 411, 412, 421, 422, 423, 511, 512, 521,
    522, 523,
]


def corine_code_to_class(code: int) -> int:
    """Map a 3-digit CORINE code to the 7-class scheme (255 = not used)."""
    if 111 <= code <= 142:
        return 0
    if 211 <= code <= 244:
        return 1
    if code in (311, 312, 313):
        return 2
    if code in (321, 322, 324, 331, 332, 333, 334, 335):
        return 3
    if code == 412:
        return 4
    if code in (411, 421, 422, 423):
        return 5
    if 511 <= code <= 523:
        return 6
    return NODATA


CODE_LUT = np.full(1024, NODATA, np.uint8)
for _c in CLC_CODES:
    CODE_LUT[_c] = corine_code_to_class(_c)
IDX_LUT = np.full(256, NODATA, np.uint8)
for _i, _c in enumerate(CLC_CODES, start=1):
    IDX_LUT[_i] = CODE_LUT[_c]

# WorldCover has NO peat class. Herbaceous wetland (90) is only a PROXY for peat bogs.
WC_LUT = np.full(256, NODATA, np.uint8)
for _k, _v in {50: 0, 40: 1, 10: 2, 20: 3, 30: 3, 60: 3, 100: 3, 90: 4, 95: 5, 80: 6}.items():
    WC_LUT[_k] = _v


def remap_corine(raw: np.ndarray) -> tuple[np.ndarray, str]:
    """Remap a raw CORINE raster (index 1-44 or 3-digit codes) to the 7 classes. Returns (labels, scheme)."""
    values = np.unique(raw[raw > 0])
    n_code = int(np.isin(values, CLC_CODES).sum())
    n_idx = int(np.isin(values, np.arange(1, 45)).sum())
    if n_idx > n_code:
        return IDX_LUT[np.clip(raw, 0, 255)].astype(np.uint8), "index(1-44)"
    return CODE_LUT[np.clip(raw, 0, 1023)].astype(np.uint8), "code(3-digit)"


def remap_worldcover(raw: np.ndarray) -> np.ndarray:
    """Remap an ESA WorldCover raster to the 7 classes (peat = herbaceous wetland PROXY)."""
    return WC_LUT[raw.astype(np.uint8)]


def find_label_file(path: str | Path) -> Path | None:
    """Locate a CORINE GeoTIFF from a file, a zip (possibly nested) or a folder. Returns None if absent."""
    import zipfile

    def resolve(p: Path, depth: int = 0) -> Path | None:
        if p.suffix.lower() in (".tif", ".tiff"):
            return p
        if p.suffix.lower() == ".zip" and depth < 4:
            out = Path("data/raw") / p.stem
            out.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(p) as zf:
                zf.extractall(out)
            tifs = [t for t in out.rglob("*") if t.suffix.lower() in (".tif", ".tiff") and "ovr" not in t.name.lower()]
            if tifs:
                tifs.sort(key=lambda t: ("clc" not in t.name.lower(), -t.stat().st_size))
                return tifs[0]
            for inner in out.rglob("*.zip"):
                found = resolve(inner, depth + 1)
                if found:
                    return found
        return None

    p = Path(path)
    if p.is_file():
        return resolve(p)
    if p.is_dir():
        candidates = sorted(p.rglob("*"), key=lambda f: ("clc" not in f.name.lower() and "u20" not in f.name.lower(), f.name))
        for f in candidates:
            if f.suffix.lower() in (".tif", ".tiff", ".zip"):
                found = resolve(f)
                if found:
                    return found
    return None


CORINE_HELP = """
CORINE Land Cover file not found. This is a manual step (free Copernicus login):
  1. Open https://land.copernicus.eu/en/products/corine-land-cover
  2. Download 'CORINE Land Cover 2018 (raster 100 m)' (clip to Ireland to keep it small).
  3. Save the .zip or .tif inside data/raw/ and run the command again.
The pipeline never invents labels. To try the (proxy) WorldCover labels instead, pass --allow-proxy.
"""
