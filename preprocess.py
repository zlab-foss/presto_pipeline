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
#  Fixed Percentile and Min-Max Values
# ============================================================

FIXED_PERCENTILES = {
    "s2s1_percentiles": {
        "vv": {"q10": 0.017242368496954444, "q90": 0.10406609922647475},
        "vh": {"q10": 0.0015369368134997786, "q90": 0.018560633063316342},
        "red": {"q10": 559.5, "q90": 2486.0},
        "green": {"q10": 631.9500000000044, "q90": 1902.0},
        "blue": {"q10": 338.0, "q90": 1400.0},
        "nir": {"q10": 1804.0, "q90": 4040.0},
        "coastal": {"q10": 243.5, "q90": 1056.0},
        "red_edge1": {"q10": 1107.5, "q90": 2775.0},
        "red_edge2": {"q10": 1589.0, "q90": 3456.0},
        "red_edge3": {"q10": 1729.0, "q90": 3840.0},
        "red_edge4": {"q10": 1895.0, "q90": 4071.0},
        "water_vapor": {"q10": 1966.0, "q90": 4106.5},
        "swir1": {"q10": 1864.0, "q90": 3868.0},
        "swir2": {"q10": 1157.0, "q90": 3213.0}
    },
    "l8_percentiles": {
        "blue": {"q10": 0.027947500348091125, "q90": 0.10626749694347382},
        "green": {"q10": 0.05594249814748764, "q90": 0.15601499378681183},
        "red": {"q10": 0.03740749880671501, "q90": 0.2044149935245514},
        "nir": {"q10": 0.1866225004196167, "q90": 0.45777249336242676},
        "swir1": {"q10": 0.14746250212192535, "q90": 0.3486250042915344},
        "swir2": {"q10": 0.07128749787807465, "q90": 0.26359501481056213}
    },
    "era5_minmax": {
        "t2m": {"min": -12.109452247619629, "max": 40.341400146484375},
        "tp": {"min": 2.4831748305587098e-05, "max": 0.4156825542449951}
    }
}

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
      8. Apply FIXED percentile scaling for S1/S2 and min-max scaling for ERA5.
      9. Mask remaining NaN/Inf and zero them.
     10. Return x, mask, dw in shapes:
         - x    : (12, 17, N)
         - mask : (12, 17, N)
         - dw   : (12, N)
    """

    def __init__(
        self,
        group_flags: Dict[str, bool] | None = None,
        batch_size: int = 2048,
        sensor_type: str = "sentinel",
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
        batch_size : int
            Batch size for the output DataLoader.
        sensor_type : str
            Either "sentinel" or "landsat". Determines which percentile values to use
            for optical bands. Default is "sentinel".
        """
        self.batch_size = int(batch_size)
        self.sensor_type = sensor_type.lower()
        
        if self.sensor_type not in ["sentinel", "landsat"]:
            raise ValueError(f"sensor_type must be 'sentinel' or 'landsat', got '{sensor_type}'")

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

        # Store last (H,W) for debugging / visualization
        self.shape = [None, None]
        self.all_nan_pixels: torch.Tensor | None = None  # (N,) bool

    # -----------------------------------------------------------------
    #  Public entrypoint
    # -----------------------------------------------------------------
    def build_from_tifs(
        self,
        paths: Sequence[Path],
        ref_index: int = 0,
    ) -> DataLoader:
        """
        Build Presto DataLoader from a list of aligned monthly GeoTIFF stacks.

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
        DataLoader
            PyTorch DataLoader with batches of (x, dw, latlons, mask, labels)
            where:
                x: (batch, 12, 17) - Presto inputs
                dw: (batch, 12) - DynamicWorld placeholder
                latlons: (batch, 2) - lat/lon coordinates
                mask: (batch, 12, 17) - validity mask
                labels: (batch,) - dummy labels (zeros)
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
            
        
        # ======================================================
        # NIR-based "all_nan_pixels":
        # pixel is True if in NIR band 90% or more of months are NaN
        # (i.e. valid_fraction <= 0.1)
        # ======================================================
        if nir_cube is not None:
            # nir_cube: (12, H, W)
            T = nir_cube.shape[0]           # usually 12
            nir_valid = np.isfinite(nir_cube)    # True where NIR is finite
            valid_count = nir_valid.sum(axis=0)  # (H, W)
            valid_fraction = valid_count.astype(np.float32) / float(T)

            # "bad NIR" pixels: ≥90% NaN → valid_fraction <= 0.1
            bad_nir_mask_hw = valid_fraction <= 0.1   # (H, W) bool

            # flatten to (N,)
            self.all_nan_pixels = torch.from_numpy(
                bad_nir_mask_hw.reshape(-1)
            ).bool()
        else:
            # if no NIR available, default to no pixel being flagged
            self.all_nan_pixels = torch.zeros(N, dtype=torch.bool)


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

        # 10) Apply FIXED percentile scaling for S1/S2 and min-max for ERA5
        self._scale_with_fixed_values_inplace(x, mask)

        # 11) Final NaN/Inf cleanup → zero + mask=1
        self._final_nan_mask_inplace(x, mask)

        # 12) Create lat/lon coordinates (N, 2)
        latlons = torch.from_numpy(
            np.stack([lat.reshape(-1), lon.reshape(-1)], axis=-1).astype(np.float32)
        )

        # 13) Dummy labels (not used during inference)
        labels = torch.zeros((N,), dtype=torch.long)

        # 14) Create dataset
        # All tensors must have N as first dimension
        # x: (N, 12, 17)
        # dw: (N, 12)
        # latlons: (N, 2)
        # mask: (N, 12, 17)
        # labels: (N,)
        
        print(f"📊 Tensor shapes before DataLoader:")
        print(f"   x: {x.shape}")
        print(f"   dw: {dw.shape}")
        print(f"   latlons: {latlons.shape}")
        print(f"   mask: {mask.shape}")
        print(f"   labels: {labels.shape}")
        
        ds = TensorDataset(x, dw, latlons, mask, labels)
        loader = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
        )

        return loader

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

    def _scale_with_fixed_values_inplace(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Apply FIXED percentile scaling for S1/S2 bands and min-max scaling for ERA5.

        Parameters
        ----------
        x : torch.Tensor
            (N, 12, 17) data tensor.
        mask : torch.Tensor
            (N, 12, 17) mask tensor. Masked values (1) are ignored.

        Notes
        -----
        - S1 bands use sentinel s2s1_percentiles.
        - Optical bands (RGB, NIR, SWIR) use percentiles based on sensor_type:
          * "sentinel" -> s2s1_percentiles
          * "landsat" -> l8_percentiles
        - Sentinel-specific bands (red_edge1-4, coastal, water_vapor) ONLY scaled 
          if sensor_type is "sentinel" AND the band is enabled.
        - ERA5 bands use fixed min/max scaling.
        - SRTM is already normalized separately.
        - NDVI is normalized from [-1, 1] to [0, 1].
        """
        # Define channel to band name mapping
        channel_to_band = {
            0: "vv", 1: "vh",
            2: "red", 3: "green", 4: "blue",
            5: "red_edge1", 6: "red_edge2", 7: "red_edge3",
            8: "nir", 9: "red_edge4",
            10: "swir1", 11: "swir2",
            12: "t2m", 13: "tp",
            # 14, 15: SRTM (already normalized)
            # 16: NDVI (will be normalized separately)
        }
        
        # Bands that exist in both Sentinel and Landsat
        common_optical_bands = {"red", "green", "blue", "nir", "swir1", "swir2"}
        
        # Bands that only exist in Sentinel-2
        sentinel_only_bands = {"red_edge1", "red_edge2", "red_edge3", "red_edge4", 
                               "coastal", "water_vapor"}
        
        # Select the appropriate percentile dictionary based on sensor type
        if self.sensor_type == "sentinel":
            optical_percentiles = FIXED_PERCENTILES["s2s1_percentiles"]
        else:  # landsat
            optical_percentiles = FIXED_PERCENTILES["l8_percentiles"]

        for ch, band_name in channel_to_band.items():
            groups = self.channel_to_groups.get(ch, [])
            
            # Check if channel is enabled
            if not any(self.group_flags.get(g, GROUP_AVAILABLE.get(g, False)) for g in groups):
                continue

            ch_tensor = x[..., ch]
            
            # ERA5 special handling: min-max scaling
            if band_name in ["t2m", "tp"]:
                era5_minmax = FIXED_PERCENTILES["era5_minmax"]
                min_val = era5_minmax[band_name]["min"]
                max_val = era5_minmax[band_name]["max"]
                
                ch_tensor.sub_(float(min_val)).div_(float(max_val - min_val))
                ch_tensor.clamp_(0.0, 1.0)
                
            # S1 bands: always use sentinel percentiles
            elif band_name in ["vv", "vh"]:
                q10 = FIXED_PERCENTILES["s2s1_percentiles"][band_name]["q10"]
                q90 = FIXED_PERCENTILES["s2s1_percentiles"][band_name]["q90"]
                
                ch_tensor.sub_(float(q10)).div_(float(q90 - q10))
                ch_tensor.clamp_(0.0, 1.0)
            
            # Sentinel-only bands: only scale if sensor_type is sentinel
            elif band_name in sentinel_only_bands:
                if self.sensor_type == "sentinel" and band_name in FIXED_PERCENTILES["s2s1_percentiles"]:
                    q10 = FIXED_PERCENTILES["s2s1_percentiles"][band_name]["q10"]
                    q90 = FIXED_PERCENTILES["s2s1_percentiles"][band_name]["q90"]
                    
                    ch_tensor.sub_(float(q10)).div_(float(q90 - q10))
                    ch_tensor.clamp_(0.0, 1.0)
                # else: leave untouched (will remain as-is or masked)
            
            # Common optical bands: use sensor-specific percentiles
            elif band_name in common_optical_bands:
                if band_name in optical_percentiles:
                    q10 = optical_percentiles[band_name]["q10"]
                    q90 = optical_percentiles[band_name]["q90"]
                    
                    ch_tensor.sub_(float(q10)).div_(float(q90 - q10))
                    ch_tensor.clamp_(0.0, 1.0)
        
        # Handle NDVI (channel 16) - normalize from [-1, 1] to [0, 1]
        if self.group_flags.get("NDVI", GROUP_AVAILABLE.get("NDVI", True)):
            ndvi_tensor = x[..., 16]
            # NDVI is already in [-1, 1], normalize to [0, 1]
            ndvi_tensor.add_(1.0).div_(2.0)
            ndvi_tensor.clamp_(0.0, 1.0)

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