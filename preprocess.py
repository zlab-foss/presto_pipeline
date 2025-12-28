from __future__ import annotations

import re
import numpy as np
import torch
from pathlib import Path
from typing import Sequence, Dict, List, Tuple
from collections import OrderedDict

from utils.utils import align_and_stack_tifs
from utils.preprocess_utils import interpolate_nan_temporal, _gather_cube
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
#  Fixed Z-Score (mu/std) Values  (MATCH TRAIN)
# ============================================================
FIXED_ZSCORE: Dict[str, Dict[str, Tuple[float, float]]] = {
    "sentinel": {
        "red": (0.0, 1000.0),
        "green": (0.0, 1000.0),
        "blue": (0.0, 1000.0),
        "nir": (0.0, 1000.0),
        "swir1": (0.0, 1000.0),
        "swir2": (0.0, 1000.0),
        "red_edge1": (0.0, 1000.0),
        "red_edge2": (0.0, 1000.0),
        "red_edge3": (0.0, 1000.0),
        "red_edge4": (0.0, 1000.0),
        "coastal": (0.0, 1000.0),
        "water_vapor": (0.0, 1000.0),
        "vv": (0.0, 0.1),
        "vh": (0.0, 0.1),
        "t2m": (0.0, 20.0),
        "tp": (0.0, 0.05),
    },
    "landsat": {
        "blue": (0.0, 0.2),
        "green": (0.0, 0.2),
        "red": (0.0, 0.2),
        "nir": (0.0, 0.2),
        "swir1": (0.0, 0.2),
        "swir2": (0.0, 0.2),
        "t2m": (0.0, 20.0),
        "tp": (0.0, 0.05),
    },
}


# ============================================================
#  Band / group configuration
# ============================================================
_MONTH_BAND_RE = re.compile(r"^(?:\d+_)?M(\d{2})_(.+)$")

BANDS_GROUPS_IDX: Dict[str, List[int]] = OrderedDict(
    [
        ("S1", [0, 1]),
        ("S2_RGB", [2, 3, 4]),
        ("S2_Red_Edge", [5, 6, 7]),
        ("S2_NIR_10m", [8]),
        ("S2_NIR_20m", [9]),
        ("S2_SWIR", [10, 11]),
        ("ERA5", [12, 13]),
        ("SRTM", [14, 15]),
        ("NDVI", [16]),
    ]
)

GROUP_AVAILABLE: Dict[str, bool] = OrderedDict(
    [
        ("S1", True),
        ("S2_RGB", True),
        ("S2_Red_Edge", True),
        ("S2_NIR_10m", True),
        ("S2_NIR_20m", True),
        ("S2_SWIR", True),
        ("ERA5", False),
        ("SRTM", False),
        ("NDVI", True),
    ]
)

GROUP_BANDS_ORDER: Dict[str, List[str]] = OrderedDict(
    [
        ("S1", ["vv", "vh"]),
        ("S2_RGB", ["red", "green", "blue"]),
        ("S2_Red_Edge", ["red_edge1", "red_edge2", "red_edge3"]),
        ("S2_NIR_10m", ["nir"]),
        ("S2_NIR_20m", ["red_edge4"]),
        ("S2_SWIR", ["swir1", "swir2"]),
        ("ERA5", ["t2m", "tp"]),
        ("SRTM", ["elevation", "slope"]),
        ("NDVI", []),
    ]
)


# ============================================================
#  Your clipping function (UNCHANGED)
# ============================================================
S2_RGBNIR = ["red", "green", "blue", "nir"]
S2_FULL = [
    "coastal", "blue", "green", "red", "red_edge1", "red_edge2", "red_edge3",
    "nir", "red_edge4", "water_vapor", "swir1", "swir2",
]
S1_VV_VH = ["vv", "vh"]
LSAT_BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]
ERA5_BANDS = ["t2m", "tp"]

S2_ALL_BANDS = set(S2_FULL)
S2_RGBNIR_SET = set(S2_RGBNIR)
S1_SET = set(S1_VV_VH)
LSAT_SET = set(LSAT_BANDS)
ERA5_SET = set(ERA5_BANDS)

def _clip_invalid_to_nan(cube: np.ndarray, sensor_type: str, band: str) -> np.ndarray:
    """
    cube: (12,H,W) numpy array
    Sets out-of-range values to NaN (float32) BEFORE interpolation/zscore.
    """
    st = sensor_type.lower()
    b = band.lower()
    cube = cube.astype(np.float32, copy=False)

    invalid = None
    if st == "sentinel":
        if b in S1_SET:
            invalid = (cube < 0.0) | (cube > 2.0)
        elif b in S2_ALL_BANDS or b in S2_RGBNIR_SET:
            invalid = (cube < 0.0) | (cube > 10000.0)
        elif b == "t2m":
            invalid = (cube < -20.0) | (cube > 60.0)
        elif b == "tp":
            invalid = (cube < 0.0)

    elif st == "landsat":
        if b in LSAT_SET:
            invalid = (cube < -0.2) | (cube > 1.0)
        elif b == "t2m":
            invalid = (cube < -20.0) | (cube > 60.0)
        elif b == "tp":
            invalid = (cube < 0.0)

    if invalid is not None and np.any(invalid):
        cube[invalid] = np.nan
    return cube


# ============================================================
#  Internals: parse band names
# ============================================================
def _parse_month_band_indices(descs: List[str]) -> Tuple[Dict[Tuple[int, str], int], List[str]]:
    mapping: Dict[Tuple[int, str], int] = {}
    names_seen: List[str] = []

    for idx, name in enumerate(descs):
        if name is None:
            continue
        m = _MONTH_BAND_RE.match(name)
        if not m:
            continue
        month_s, band_name = m.groups()
        try:
            month_idx = int(month_s) - 1
        except Exception:
            continue
        if not (0 <= month_idx < 12):
            continue

        band_name = band_name.lower()
        key = (month_idx, band_name)
        if key in mapping:
            raise ValueError(f"Duplicate band for month {month_idx + 1}, '{band_name}' in descs.")
        mapping[key] = idx
        names_seen.append(band_name)

    return mapping, sorted(set(names_seen))


# ============================================================
#  Main builder (train-matching)
# ============================================================
class PrestoTensorBuilder:
    """
    Builds inference tensors to match training:
      - structural-only mask
      - clip invalid -> NaN before interpolation
      - interpolate along time
      - fixed zscore normalization
      - final: nonfinite -> 0 WITHOUT changing mask
      - keeps self.all_nan_pixels (NIR >=90% missing rule)
    """

    def __init__(
        self,
        group_flags: Dict[str, bool] | None = None,
        batch_size: int = 2048,
        sensor_type: str = "sentinel",
    ):
        self.batch_size = int(batch_size)
        self.sensor_type = sensor_type.lower()
        if self.sensor_type not in ["sentinel", "landsat"]:
            raise ValueError(f"sensor_type must be 'sentinel' or 'landsat', got '{sensor_type}'")

        base = OrderedDict(GROUP_AVAILABLE)
        if group_flags is not None:
            for k, v in group_flags.items():
                if k in base:
                    base[k] = bool(v)
        self.group_flags = base

        # channel index -> list of groups that use that channel
        ch2groups: Dict[int, List[str]] = {}
        for gname, chs in BANDS_GROUPS_IDX.items():
            for ch in chs:
                ch2groups.setdefault(ch, []).append(gname)
        self.channel_to_groups = ch2groups

        self.shape = [None, None]
        self.all_nan_pixels: torch.Tensor | None = None
        self.bad_pixels: torch.Tensor | None = None

    def build_from_tifs(self, paths: Sequence[Path], ref_index: int = 0) -> DataLoader:
        stacked, lat, lon, desc_lists = align_and_stack_tifs(paths, ref_index=ref_index)
        C, H, W = stacked.shape
        N = H * W
        self.shape = [H, W]

        flat_descs: List[str] = []
        for dl in desc_lists:
            flat_descs.extend(dl)

        index_map, _band_names = _parse_month_band_indices(flat_descs)

        x = torch.zeros((N, 12, 17), dtype=torch.float32)
        mask = torch.zeros_like(x)  # STRUCTURAL ONLY
        dw = torch.full((N, 12), 9, dtype=torch.long)

        red_cube: np.ndarray | None = None
        nir_cube: np.ndarray | None = None

        # Fill groups
        for grp_name, ch_idx_list in BANDS_GROUPS_IDX.items():
            if grp_name == "NDVI":
                continue

            enabled = self.group_flags.get(grp_name, GROUP_AVAILABLE.get(grp_name, False))
            band_list = GROUP_BANDS_ORDER.get(grp_name, [])

            if not enabled:
                for ch in ch_idx_list:
                    mask[..., ch] = 1.0
                continue

            if grp_name == "SRTM":
                # Not provided in your inference -> structurally mask (match train: SRTM absent)
                for ch in ch_idx_list:
                    mask[..., ch] = 1.0
                continue

            # Strict monthly bands for enabled groups
            for bi, ch in enumerate(ch_idx_list):
                if bi >= len(band_list):
                    break
                bname = band_list[bi]

                # If missing months -> structural mask (DO NOT raise)
                if not all((m, bname) in index_map for m in range(12)):
                    mask[..., ch] = 1.0
                    continue

                cube = _gather_cube(stacked, index_map, bname)  # (12,H,W)
                cube = _clip_invalid_to_nan(cube, sensor_type=self.sensor_type, band=bname)

                if bname == "red":
                    red_cube = cube
                if bname == "nir":
                    nir_cube = cube

                x[..., ch] = torch.from_numpy(cube.reshape(12, N).transpose(1, 0))

        # NDVI (ch 16)
        ndvi_enabled = self.group_flags.get("NDVI", GROUP_AVAILABLE.get("NDVI", True))
        if ndvi_enabled and (red_cube is not None) and (nir_cube is not None):
            ndvi = self._compute_ndvi(red_cube, nir_cube)
            x[..., 16] = torch.from_numpy(ndvi.reshape(12, N).transpose(1, 0))
        else:
            mask[..., 16] = 1.0

        # ======================================================
        # KEEP your NIR ≥90% missing rule (all_nan_pixels)
        # ======================================================
        if nir_cube is not None:
            T = nir_cube.shape[0]  # usually 12
            nir_valid = np.isfinite(nir_cube)
            valid_count = nir_valid.sum(axis=0)
            valid_fraction = valid_count.astype(np.float32) / float(T)

            bad_nir_mask_hw = valid_fraction <= 0.1
            self.all_nan_pixels = torch.from_numpy(bad_nir_mask_hw.reshape(-1)).bool()
        else:
            self.all_nan_pixels = torch.zeros(N, dtype=torch.bool)

        # Structural mask -> zero before interpolation
        x[mask == 1] = 0.0

        # Interpolate temporally
        x = interpolate_nan_temporal(x)

        # Fixed z-score scaling (match train)
        self._zscore_with_fixed_values_inplace(x, mask)

        # Final cleanup: nonfinite -> 0 (DO NOT modify mask)
        self.bad_pixels = self._final_nonfinite_to_zero_keep_mask(x)
        # ---- print percent of bad pixels ----
        n_bad = int(self.bad_pixels.sum().item()) if isinstance(self.bad_pixels, torch.Tensor) else int(np.sum(self.bad_pixels))
        pct_bad = 100.0 * (n_bad / max(N, 1))
        print(f"⚠️ bad_pixels: {n_bad:,}/{N:,} ({pct_bad:.2f}%)")
        

        # lat/lon
        latlons = torch.from_numpy(
            np.stack([lat.reshape(-1), lon.reshape(-1)], axis=-1).astype(np.float32)
        )

        labels = torch.zeros((N,), dtype=torch.long)
        ds = TensorDataset(x, dw, latlons, mask, labels)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False)

    @staticmethod
    def _compute_ndvi(red_cube: np.ndarray, nir_cube: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        red = red_cube.astype(np.float32, copy=False)
        nir = nir_cube.astype(np.float32, copy=False)
        denom = nir + red
        with np.errstate(invalid="ignore", divide="ignore"):
            ndvi = (nir - red) / (denom + eps)
        ndvi = np.where(np.abs(denom) < eps, np.nan, ndvi)
        ndvi = np.clip(ndvi, -1.0, 1.0)
        return ndvi.astype(np.float32, copy=False)

    def _zscore_with_fixed_values_inplace(self, x: torch.Tensor, mask: torch.Tensor) -> None:
        st = self.sensor_type.lower()
        params = FIXED_ZSCORE.get(st, None)
        if params is None:
            raise ValueError(f"Unknown sensor_type for FIXED_ZSCORE: {st}")

        channel_to_band = {
            0: "vv", 1: "vh",
            2: "red", 3: "green", 4: "blue",
            5: "red_edge1", 6: "red_edge2", 7: "red_edge3",
            8: "nir", 9: "red_edge4",
            10: "swir1", 11: "swir2",
            12: "t2m", 13: "tp",
        }

        for ch, band_name in channel_to_band.items():
            if band_name not in params:
                continue

            groups = self.channel_to_groups.get(ch, [])
            if not any(self.group_flags.get(g, GROUP_AVAILABLE.get(g, False)) for g in groups):
                continue

            mu, std = params[band_name]
            std = float(std) if float(std) != 0.0 else 1.0

            valid = (mask[..., ch] == 0)
            if valid.any():
                x_ch = x[..., ch]
                x_ch[valid].sub_(float(mu)).div_(std)

    @staticmethod
    def _final_nonfinite_to_zero_keep_mask(x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        bad = (~torch.isfinite(x.view(N, -1))).any(dim=1)
        if bad.any():
            x[~torch.isfinite(x)] = 0.0
        return bad
