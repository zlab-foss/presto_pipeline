from __future__ import annotations

import re
import numpy as np
import torch
from pathlib import Path
from typing import Sequence, Dict, List, Tuple, Optional
from torch.utils.data import DataLoader, TensorDataset

from utils.utils import align_and_stack_tifs
from utils.preprocess_utils import interpolate_nan_temporal, _gather_cube


# ============================================================
#  FIXED_ZSCORE: (min, max)  (MATCH PrestoProcessor)
# ============================================================
FIXED_ZSCORE: Dict[str, Dict[str, Tuple[float, float]]] = {
    "sentinel": {
        "red": (0.0, 10000.0),
        "green": (0.0, 10000.0),
        "blue": (0.0, 10000.0),
        "nir": (0.0, 10000.0),
        "vv": (0.0, 0.1),
        "vh": (0.0, 0.1),
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

# band desc pattern: "M01_red" or "0_M01_red"
_MONTH_BAND_RE = re.compile(r"^(?:\d+_)?M(\d{2})_(.+)$", re.IGNORECASE)

# Presto structural mask behavior (match your PrestoProcessor "structural-only" variant)
_ALWAYS_MASK = {5, 6, 7, 9, 14, 15}     # always inactive
_L8_ONLY = {10, 11, 12, 13}             # swir + era5 slots


def _parse_month_band_indices(descs: List[str]) -> Dict[Tuple[int, str], int]:
    """
    Return mapping: (month_idx 0..11, band_name_lower) -> band_index_in_stacked
    """
    mapping: Dict[Tuple[int, str], int] = {}
    for idx, name in enumerate(descs):
        if name is None:
            continue
        m = _MONTH_BAND_RE.match(str(name))
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
            raise ValueError(f"Duplicate band for month {month_idx+1} '{band_name}' in descriptions.")
        mapping[key] = idx
    return mapping


class PrestoTensorBuilder:
    """
    Builds inference tensors that MATCH your PrestoProcessor behavior:

    - x: (N,12,17)
    - mask: STRUCTURAL ONLY (constant per pixel)
    - interpolate NaNs temporally
    - fixed z-score with (min,max)->(mean,std) conversion
    - NDVI computed from red/nir when 'ndvi' band is missing
    - final: replace non-finite with 0 WITHOUT changing mask
    """

    def __init__(self, *, batch_size: int = 2048, sensor_type: str = "sentinel"):
        self.batch_size = int(batch_size)
        self.sensor_type = sensor_type.lower()
        if self.sensor_type not in ("sentinel", "landsat"):
            raise ValueError("sensor_type must be 'sentinel' or 'landsat'")
        self.shape: list[Optional[int]] = [None, None]

    @staticmethod
    def _minmax_to_meanstd(minv: float, maxv: float) -> Tuple[float, float]:
        mean = 0.5 * (minv + maxv)
        std = 0.5 * (maxv - minv)
        if std <= 0:
            std = 1.0
        return mean, std

    @staticmethod
    def _compute_ndvi(red_cube: np.ndarray, nir_cube: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        red_cube/nir_cube: (12,H,W) float32 with NaNs allowed
        returns NDVI: (12,H,W) float32 with NaNs
        """
        red = red_cube.astype(np.float32, copy=False)
        nir = nir_cube.astype(np.float32, copy=False)
        denom = nir + red
        with np.errstate(invalid="ignore", divide="ignore"):
            ndvi = (nir - red) / (denom + eps)
        # where denom ~0 => NaN
        ndvi = np.where(np.abs(denom) < eps, np.nan, ndvi)
        # clamp like typical NDVI
        ndvi = np.clip(ndvi, -1.0, 1.0)
        return ndvi.astype(np.float32, copy=False)

    def _apply_fixed_zscore(self, x: torch.Tensor, mask: torch.Tensor) -> None:
        """
        In-place z-score on active channels ONLY, using PrestoProcessor logic:
          FIXED_ZSCORE stores (min,max) -> convert to (mean,std)
        """
        params = FIXED_ZSCORE[self.sensor_type]

        ch_to_var = {
            0: "vv", 1: "vh",
            2: "red", 3: "green", 4: "blue",
            8: "nir",
            10: "swir1", 11: "swir2",
            12: "t2m", 13: "tp",
            # 16 is NDVI => DO NOT scale
        }

        for ch, var in ch_to_var.items():
            if var not in params:
                continue

            minv, maxv = params[var]
            mean, std = self._minmax_to_meanstd(float(minv), float(maxv))

            valid = (mask[..., ch] == 0)
            if valid.any():
                xch = x[..., ch]
                xch[valid] = (xch[valid] - mean) / std

    def _structural_mask(self, N: int, T: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros((N, T, 17), dtype=torch.float32, device=device)

        for ch in _ALWAYS_MASK:
            mask[:, :, ch] = 1.0

        if self.sensor_type == "sentinel":
            # sentinel has no SWIR/ERA5 slots
            for ch in _L8_ONLY:
                mask[:, :, ch] = 1.0
        else:
            # landsat has no S1 slots
            mask[:, :, 0] = 1.0
            mask[:, :, 1] = 1.0

        return mask

    def build_from_tifs(self, paths: Sequence[Path], ref_index: int = 0) -> DataLoader:
        """
        Returns DataLoader yielding (x, dw, latlons, mask, labels)

        latlons order = (lon, lat)
        """
        stacked, lat, lon, desc_lists = align_and_stack_tifs(paths, ref_index=ref_index)
        C, H, W = stacked.shape
        N = H * W
        self.shape = [H, W]

        # flatten all band descriptions across stacked inputs
        flat_descs: List[str] = []
        for dl in desc_lists:
            flat_descs.extend(dl)

        index_map = _parse_month_band_indices(flat_descs)

        # x init NaN (like PrestoProcessor before its final cleanup)
        x = torch.full((N, 12, 17), float("nan"), dtype=torch.float32)
        mask = self._structural_mask(N, 12, x.device)
        dw = torch.full((N, 12), 9, dtype=torch.long)

        # Keep cubes for NDVI computation if needed
        red_cube: Optional[np.ndarray] = None
        nir_cube: Optional[np.ndarray] = None

        def has_all_months(band: str) -> bool:
            b = band.lower()
            return all((m, b) in index_map for m in range(12))

        def fill_channel_from_band(dst_ch: int, band_name: str) -> Optional[np.ndarray]:
            """
            Fill x[:, :, dst_ch] from band_name across all months if available.
            Returns the (12,H,W) cube if filled, else None.
            """
            band = band_name.lower()
            if not has_all_months(band):
                return None
            cube = _gather_cube(stacked, index_map, band).astype(np.float32, copy=False)  # (12,H,W)
            x[:, :, dst_ch] = torch.from_numpy(cube.reshape(12, N).T)                    # (N,12)
            return cube

        # -------- Fill channels --------
        if self.sensor_type == "sentinel":
            fill_channel_from_band(0, "vv")
            fill_channel_from_band(1, "vh")
            red_cube = fill_channel_from_band(2, "red")
            fill_channel_from_band(3, "green")
            fill_channel_from_band(4, "blue")
            nir_cube = fill_channel_from_band(8, "nir")

            # NDVI: use band if exists else compute
            ndvi_cube = fill_channel_from_band(16, "ndvi")
            if ndvi_cube is None and (red_cube is not None) and (nir_cube is not None):
                ndvi_cube = self._compute_ndvi(red_cube, nir_cube)
                x[:, :, 16] = torch.from_numpy(ndvi_cube.reshape(12, N).T)

        else:
            # landsat layout uses same RGB+NIR slots + swir/era5 slots
            red_cube = fill_channel_from_band(2, "red")
            fill_channel_from_band(3, "green")
            fill_channel_from_band(4, "blue")
            nir_cube = fill_channel_from_band(8, "nir")
            fill_channel_from_band(10, "swir1")
            fill_channel_from_band(11, "swir2")
            fill_channel_from_band(12, "t2m")
            fill_channel_from_band(13, "tp")

            ndvi_cube = fill_channel_from_band(16, "ndvi")
            if ndvi_cube is None and (red_cube is not None) and (nir_cube is not None):
                ndvi_cube = self._compute_ndvi(red_cube, nir_cube)
                x[:, :, 16] = torch.from_numpy(ndvi_cube.reshape(12, N).T)

        # Structural-only masked channels set to 0 before interpolation
        x[mask == 1] = 0.0

        # Interpolate temporally (fills NaNs in active channels)
        x = interpolate_nan_temporal(x)

        # Fixed z-score scaling (active channels only)
        self._apply_fixed_zscore(x, mask)

        # Final cleanup: non-finite -> 0 (do NOT modify mask)
        bad = ~torch.isfinite(x)
        if bad.any():
            x[bad] = 0.0

        # latlons (lon,lat)
        latlons = torch.from_numpy(
            np.stack([lon.reshape(-1), lat.reshape(-1)], axis=-1).astype(np.float32)
        )

        labels = torch.zeros((N,), dtype=torch.long)
        ds = TensorDataset(x, dw, latlons, mask, labels)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False)
