from __future__ import annotations
import re
import numpy as np
import torch
from pathlib import Path
from typing import Sequence, Dict, List, Tuple
from collections import OrderedDict

from utils.utils import align_and_stack_tifs
from utils.preprocess_utils import interpolate_nan_temporal, _gather_cube

# ============================================================
#  Band / group configuration
# ============================================================

# Regex for band descriptions like "M01_red", "M12_vv", "0_M03_elevation", ...
# We allow an optional numeric prefix ("0_", "1_", ...) before "M##".
_MONTH_BAND_RE = re.compile(r"^(?:\d+_)?M(\d{2})_(.+)$")

BANDS_GROUPS_IDX: Dict[str, List[int]] = OrderedDict(
    [
        ("S1", [0, 1]),         # vv, vh
        ("S2_RGB", [2, 3, 4]),  # red, green, blue
        ("S2_Red_Edge", [5, 6, 7]),
        ("S2_NIR_10m", [8]),    # nir
        ("S2_NIR_20m", [9]),    # red_edge4 (proxy)
        ("S2_SWIR", [10, 11]),  # swir1, swir2
        ("ERA5", [12, 13]),     # t2m, tp
        ("SRTM", [14, 15]),     # elevation, slope
        ("NDVI", [16]),         # computed
    ]
)

# Default availability of each group.
# The user can override with `group_flags` in the constructor.
GROUP_AVAILABLE: Dict[str, bool] = OrderedDict(
    [
        ("S1", True),
        ("S2_RGB", True),
        ("S2_Red_Edge", True),
        ("S2_NIR_10m", True),
        ("S2_NIR_20m", True),
        ("S2_SWIR", True),
        ("ERA5", False),   # off by default until wired
        ("SRTM", False),   # off by default until SRTM is provided
        ("NDVI", True),
    ]
)

# Mapping from logical group → physical band names expected in the TIF
# (suffix after "M##_" in band descriptions, or plain "elevation" for static SRTM).
GROUP_BANDS_ORDER: Dict[str, List[str]] = OrderedDict(
    [
        ("S1", ["vv", "vh"]),
        ("S2_RGB", ["red", "green", "blue"]),
        ("S2_Red_Edge", ["red_edge1", "red_edge2", "red_edge3"]),
        ("S2_NIR_10m", ["nir"]),
        ("S2_NIR_20m", ["red_edge4"]),
        ("S2_SWIR", ["swir1", "swir2"]),
        ("ERA5", ["t2m", "tp"]),
        ("SRTM", ["elevation", "slope"]),  # slope computed from elevation
        ("NDVI", []),                      # computed from red + nir
    ]
)


# ---------------------------------------------------------------------
#  Main builder class
# ---------------------------------------------------------------------
class PrestoTensorBuilder:
    """
    Builder for Presto-ready tensors from monthly GeoTIFF stacks.

    This class:
      * Uses `align_and_stack_tifs` to align multiple GeoTIFFs to a common grid.
      * Expects band descriptions like:
            'M01_red', 'M02_vv', 'M03_t2m', 'M01_t2m', ...
        for monthly data, and e.g.:
            'elevation'
        for static SRTM DEM.
      * Allows enabling/disabling logical groups via `group_flags`:
            S1, S2_RGB, S2_Red_Edge, S2_NIR_10m, S2_NIR_20m,
            S2_SWIR, ERA5, SRTM, NDVI.
      * Requirements:
          - Any *monthly* band used (e.g. 'red', 'nir', 'vv', 't2m')
            must exist for all 12 months as "M01_xxx" .. "M12_xxx".
          - SRTM elevation can be either static:
                "elevation"
            or monthly "M01_elevation" .. "M12_elevation".
            In both cases:
              - slope is computed from elevation
              - both elevation and slope are min–max normalized
                (per AOI) to [0, 1]
              - then repeated for 12 months.

    Output layout (Presto channels):
      - 0–1   : S1 (vv, vh)
      - 2–4   : S2_RGB (red, green, blue)
      - 5–7   : S2_Red_Edge (red_edge1..3)
      - 8     : S2_NIR_10m (nir)
      - 9     : S2_NIR_20m (red_edge4)
      - 10–11 : S2_SWIR (swir1, swir2)
      - 12–13 : ERA5 (t2m, tp)
      - 14–15 : SRTM (elevation_norm, slope_norm)
      - 16    : NDVI (computed from red/nir)

    Pipeline:
      1. Align & merge TIFs → (C, H, W) + band descriptions.
      2. Build monthly cubes for bands required by enabled groups.
      3. Assemble `x` in shape (N, 12, 17).
      4. Compute NDVI if enabled.
      5. For SRTM:
         - compute slope from elevation
         - min–max normalize (elev, slope) to [0, 1].
      6. Mark structurally unavailable channels in `mask`.
      7. Interpolate NaNs along the time axis (per pixel + channel).
      8. Apply q10–q90 scaling for selected temporal groups (S1, optical, ERA5).
      9. Mask remaining NaN/Inf and zero them.
     10. Return x, mask, dw in shapes:
         - x    : (12, 17, N)
         - mask : (12, 17, N)
         - dw   : (12, N)
    """

    def __init__(
        self,
        group_flags: Dict[str, bool] | None = None,
        q_low: float = 10.0,
        q_high: float = 90.0,
    ):
        """
        Parameters
        ----------
        group_flags : dict[str,bool] or None
            Optional overrides for group availability, e.g.:
              {
                "S1": True,
                "S2_RGB": False,
                "ERA5": True,
                "SRTM": True,
                "NDVI": True,
              }
            If None, defaults to GROUP_AVAILABLE.
        q_low, q_high : float
            Percentile bounds for robust scaling per channel for
            *temporal* groups (S1, S2, ERA5).
        """
        self.q_low = q_low
        self.q_high = q_high

        # Merge default availability with user overrides
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

        # Groups that will use percentile scaling (q10–q90).
        # SRTM is EXCLUDED (it uses custom min–max instead).
        self.groups_to_scale = {
            "S1",
            "S2_RGB",
            "S2_Red_Edge",
            "S2_NIR_10m",
            "S2_NIR_20m",
            "S2_SWIR",
            "ERA5",
        }

        # Store last (H,W) for debugging / visualization
        self.shape = [None, None]

    # -----------------------------------------------------------------
    #  Public entrypoint
    # -----------------------------------------------------------------
    def build_from_tifs(
        self,
        paths: Sequence[Path],
        ref_index: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build Presto tensors from a list of aligned monthly GeoTIFF stacks.

        Parameters
        ----------
        paths : sequence of Path
            List of GeoTIFF files. Each file must have band descriptions
            like:
                "M01_red", "M02_vv", ..., "M12_t2m"
            for monthly data. For SRTM:
                - Either a static band "elevation"
                - Or monthly bands "M01_elevation" .. "M12_elevation".
        ref_index : int
            Index into `paths` specifying which file defines the reference
            grid (CRS, transform, H, W) for alignment.

        Returns
        -------
        x_12_17_N : torch.Tensor
            Shape (12, 17, N), float32. Presto inputs.
        mask_12_17_N : torch.Tensor
            Shape (12, 17, N), float32. 1 = masked/unavailable, 0 = valid.
        dw_12_N : torch.Tensor
            Shape (12, N), long. Placeholder DynamicWorld (all 9).
        """
        # 1) Align & stack TIFs to common grid
        stacked, lat, lon, desc_lists = align_and_stack_tifs(paths, ref_index=ref_index)
        C, H, W = stacked.shape
        N = H * W
        self.shape = [H, W]

        # 2) Flatten band descriptions
        flat_descs: List[str] = []
        for dl in desc_lists:
            flat_descs.extend(dl)

        # 3) Parse month-band indices (only monthly "M##_" style bands)
        index_map, band_names = _parse_month_band_indices(flat_descs)

        # 4) Allocate outputs (N, 12, 17) then permute later
        x = torch.zeros((N, 12, 17), dtype=torch.float32)
        mask = torch.zeros_like(x)
        dw = torch.full((N, 12), 9, dtype=torch.long)  # Dynamic World placeholder

        red_cube: np.ndarray | None = None
        nir_cube: np.ndarray | None = None

        # 5) Fill channels group-by-group (S1, S2_RGB, ERA5, SRTM, ...)
        for grp_name, ch_idx_list in BANDS_GROUPS_IDX.items():
            if grp_name == "NDVI":
                continue

            enabled = self.group_flags.get(grp_name, GROUP_AVAILABLE.get(grp_name, False))
            band_list = GROUP_BANDS_ORDER.get(grp_name, [])

            # If group disabled → mask entire group
            if not enabled:
                for ch in ch_idx_list:
                    mask[..., ch] = 1.0
                continue

            # ---------- SRTM special handling: elevation + slope ----------
            if grp_name == "SRTM":
                # Expected logical channels: [elevation, slope] → [14,15]
                if len(ch_idx_list) != 2 or band_list != ["elevation", "slope"]:
                    raise ValueError(
                        f"SRTM group expects exactly ['elevation','slope'], got {band_list}"
                    )

                # 1) Get elevation cube (12,H,W) from either static or monthly bands
                elev_cube = self._get_srtm_elevation_cube(
                    stacked=stacked,
                    flat_descs=flat_descs,
                    index_map=index_map,
                )  # (12,H,W)

                # 2) Compute slope cube from elevation
                slope_cube = self._compute_slope_from_elevation(elev_cube)  # (12,H,W)

                # 3) Min–max normalize elevation and slope to [0,1]
                elev_norm, slope_norm = self._minmax_normalize_srtm(elev_cube, slope_cube)
    
                
                # 4) Write to x
                elev_flat = elev_norm.reshape(12, N).transpose(1, 0)   # (N,12)
                slope_flat = slope_norm.reshape(12, N).transpose(1, 0) # (N,12)

                x[..., ch_idx_list[0]] = torch.from_numpy(elev_flat)
                x[..., ch_idx_list[1]] = torch.from_numpy(slope_flat)
                # do NOT percentile-scale later (SRTM not in groups_to_scale)
                continue

            # ---------- All other groups: strict monthly ----------
            for bi, ch in enumerate(ch_idx_list):
                if bi >= len(band_list):
                    break
                bname = band_list[bi]

                # Require monthly coverage for each band
                if not all((m, bname) in index_map for m in range(12)):
                    raise ValueError(
                        f"Group '{grp_name}' enabled but band '{bname}' "
                        "not available for all 12 months."
                    )

                cube = _gather_cube(stacked, index_map, bname)  # (12,H,W)

                if bname == "red":
                    red_cube = cube
                if bname == "nir":
                    nir_cube = cube

                flat = cube.reshape(12, N).transpose(1, 0)  # (N,12)
                x[..., ch] = torch.from_numpy(flat)

        # 6) NDVI group (channel 16)
        ndvi_enabled = self.group_flags.get("NDVI", GROUP_AVAILABLE.get("NDVI", True))
        if ndvi_enabled:
            if red_cube is None or nir_cube is None:
                raise ValueError("NDVI enabled but 'red' or 'nir' band not found for all months.")
            ndvi = self._compute_ndvi(red_cube, nir_cube)     # (12,H,W)
            ndvi_flat = ndvi.reshape(12, N).transpose(1, 0)   # (N,12)
            x[..., 16] = torch.from_numpy(ndvi_flat)
        else:
            mask[..., 16] = 1.0

        # 7) Mask groups that are structurally unavailable based on *effective* flags
        for grp_name, ch_idx_list in BANDS_GROUPS_IDX.items():
            enabled = self.group_flags.get(grp_name, GROUP_AVAILABLE.get(grp_name, False))
            if not enabled:
                for ch in ch_idx_list:
                    mask[..., ch] = 1.0

        # 8) Ensure structurally masked positions contain no NaNs (so interpolation ignores them)
        x[mask == 1] = 0.0

        # 9) Interpolate NaNs along time axis (per pixel+channel)
        x = interpolate_nan_temporal(x)

        # 10) Percentile scaling (q_low–q_high) for selected temporal groups
        self._scale_percentiles_inplace(x, mask)

        # 11) Final NaN/Inf cleanup → zero + mask=1
        self._final_nan_mask_inplace(x, mask)

        # 12) Reorder to (12,17,N)
        x_12_17_N = x.permute(1, 2, 0)
        mask_12_17_N = mask.permute(1, 2, 0)
        dw_12_N = dw.permute(1, 0)

        return x_12_17_N, mask_12_17_N, dw_12_N

    # -----------------------------------------------------------------
    #  Helpers
    # -----------------------------------------------------------------
    @staticmethod
    def _compute_ndvi(
        red_cube: np.ndarray,
        nir_cube: np.ndarray,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """
        Compute NDVI per month from red and nir monthly cubes.

        Parameters
        ----------
        red_cube : np.ndarray
            (12, H, W) red reflectance.
        nir_cube : np.ndarray
            (12, H, W) near-infrared reflectance.
        eps : float
            Small epsilon added to denominator for numerical stability.

        Returns
        -------
        ndvi : np.ndarray
            (12, H, W) NDVI in [-1, 1], NaN where invalid.
        """
        red = red_cube.astype(np.float32, copy=False)
        nir = nir_cube.astype(np.float32, copy=False)
        denom = nir + red

        with np.errstate(invalid="ignore", divide="ignore"):
            ndvi = (nir - red) / (denom + eps)

        ndvi = np.where(np.abs(denom) < eps, np.nan, ndvi)
        ndvi = np.clip(ndvi, -1.0, 1.0)
        return ndvi.astype(np.float32, copy=False)

    def _scale_percentiles_inplace(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Apply q_low–q_high percentile scaling per channel for selected groups.

        Parameters
        ----------
        x : torch.Tensor
            (N, 12, 17) data tensor.
        mask : torch.Tensor
            (N, 12, 17) mask tensor. Masked values (1) are ignored.

        Notes
        -----
        - Only channels belonging to groups in `self.groups_to_scale`
          (and enabled in `group_flags`) are scaled.
        - SRTM is *not* percentile-normalized; it is handled separately
          via `_minmax_normalize_srtm`.
        """
        N, T, C = x.shape
        x_np = x.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()

        for ch in range(C):
            groups = self.channel_to_groups.get(ch, [])
            # scale only if any scaled group uses this channel and is enabled
            if not any(
                self.group_flags.get(g, GROUP_AVAILABLE.get(g, False)) and g in self.groups_to_scale
                for g in groups
            ):
                continue

            valid = (mask_np[..., ch] == 0) & np.isfinite(x_np[..., ch])
            if not np.any(valid):
                continue

            vals = x_np[..., ch][valid]
            q_lo = np.percentile(vals, self.q_low)
            q_hi = np.percentile(vals, self.q_high)

            if not np.isfinite(q_lo) or not np.isfinite(q_hi) or q_hi <= q_lo:
                continue

            ch_tensor = x[..., ch]
            ch_tensor.sub_(float(q_lo)).div_(float(q_hi - q_lo))
            ch_tensor.clamp_(0.0, 1.0)

    @staticmethod
    def _final_nan_mask_inplace(x: torch.Tensor, mask: torch.Tensor):
        """
        Any remaining NaN/Inf in x are set to 0 and marked as masked.

        Parameters
        ----------
        x : torch.Tensor
            (N, 12, 17) data tensor.
        mask : torch.Tensor
            (N, 12, 17) mask tensor.
        """
        nan_mask = torch.isnan(x) | torch.isinf(x)
        if nan_mask.any():
            x[nan_mask] = 0.0
            mask[nan_mask] = 1.0

    @staticmethod
    def _compute_slope_from_elevation(elev_cube: np.ndarray) -> np.ndarray:
        """
        Compute slope (in degrees) from an elevation cube.

        Parameters
        ----------
        elev_cube : np.ndarray
            (12, H, W) elevation values (can be repeated static DEM).

        Returns
        -------
        slope_cube : np.ndarray
            (12, H, W) slope in degrees, NaN where elevation is NaN.

        Notes
        -----
        - Uses unit spacing in x,y (pixel units). Physical scaling
          is not critical because we later normalize slope with
          min–max to [0,1] over the AOI.
        """
        if elev_cube.ndim != 3 or elev_cube.shape[0] != 12:
            raise ValueError(
                f"Expected elev_cube shape (12,H,W), got {elev_cube.shape}"
            )

        T, H, W = elev_cube.shape
        slope_cube = np.empty_like(elev_cube, dtype=np.float32)

        for t in range(T):
            elev = elev_cube[t].astype(np.float32, copy=False)
            elev_nan = np.isnan(elev)
            elev_filled = np.nan_to_num(elev, nan=0.0)

            gy, gx = np.gradient(elev_filled)  # assume unit spacing
            slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
            slope_deg = np.degrees(slope_rad)

            slope_deg[elev_nan] = np.nan
            slope_cube[t] = slope_deg

        return slope_cube

    @staticmethod
    def _minmax_normalize_srtm(
        elev_cube: np.ndarray,
        slope_cube: np.ndarray,
        eps: float = 1e-6,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Min–max normalize SRTM elevation and slope to [0,1].

        Parameters
        ----------
        elev_cube : np.ndarray
            (12, H, W) elevation values.
        slope_cube : np.ndarray
            (12, H, W) slope values in degrees.
        eps : float
            Small epsilon to avoid division by zero.

        Returns
        -------
        elev_norm : np.ndarray
            (12, H, W) normalized elevation ∈ [0,1].
        slope_norm : np.ndarray
            (12, H, W) normalized slope ∈ [0,1].

        Notes
        -----
        - If the data is truly static across time, min/max will be
          identical for all months, which is fine.
        - NaNs remain NaN.
        """
        elev = elev_cube.astype(np.float32, copy=False)
        slope = slope_cube.astype(np.float32, copy=False)

        elev_min = np.nanmin(elev)
        elev_max = np.nanmax(elev)
        slope_min = np.nanmin(slope)
        slope_max = np.nanmax(slope)

        elev_range = np.maximum(elev_max - elev_min, eps)
        slope_range = np.maximum(slope_max - slope_min, eps)

        elev_norm = (elev - elev_min) / elev_range
        slope_norm = (slope - slope_min) / slope_range

        return elev_norm, slope_norm

    @staticmethod
    def _get_srtm_elevation_cube(
        stacked: np.ndarray,
        flat_descs: List[str],
        index_map: Dict[Tuple[int, str], int],
    ) -> np.ndarray:
        """
        Retrieve SRTM elevation cube (12,H,W) from stacked data.

        Logic:
        -------
        1. Try to find a static band named "elevation" (no 'M##_').
           If found → replicate to 12 months.
        2. Otherwise, require monthly bands "M01_elevation" .. "M12_elevation"
           (already parsed into index_map).

        Parameters
        ----------
        stacked : np.ndarray
            (C, H, W) stacked bands from align_and_stack_tifs.
        flat_descs : list[str]
            Band descriptions, same length as C.
        index_map : dict[(month_idx, band_name), int]
            Result of `_parse_month_band_indices`.

        Returns
        -------
        elev_cube : np.ndarray
            (12, H, W) elevation per month (may be static repeated).
        """
        # 1) Try static "elevation"
        static_idx = None
        for i, d in enumerate(flat_descs):
            if d is None:
                continue
            d_lower = d.lower()
            # Ignore monthly-style names here
            if _MONTH_BAND_RE.match(d):
                continue
            if d_lower == "elevation":
                static_idx = i
                break

        if static_idx is not None:
            tile = stacked[static_idx].astype(np.float32, copy=False)  # (H,W)
            return np.repeat(tile[None, ...], 12, axis=0)              # (12,H,W)

        # 2) Fallback: require monthly M##_elevation
        if not all((m, "elevation") in index_map for m in range(12)):
            raise ValueError(
                "SRTM elevation enabled but not found as static 'elevation' or monthly 'M##_elevation'."
            )

        return _gather_cube(stacked, index_map, "elevation")  # (12,H,W)


# ---------------------------------------------------------------------
#  Internals: parse band names and gather monthly cubes
# ---------------------------------------------------------------------
def _parse_month_band_indices(
    descs: List[str],
) -> Tuple[Dict[Tuple[int, str], int], List[str]]:
    """
    Parse band descriptions like 'M01_red' into a mapping:
      (month_idx, band_name) -> channel index

    Parameters
    ----------
    descs : list[str]
        Band descriptions from the stacked GeoTIFFs.

    Returns
    -------
    mapping : dict[(int,str), int]
        (month_idx, band_name) -> channel index in stacked array.
        month_idx is 0-based (0..11).
    names_seen : list[str]
        Unique band names encountered (e.g. ['blue','green','red',...]).
    """
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
