from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


from geo_utils import _read_tif, align_mask_to_tile
from preprocess import reshape_168_to_month_band, build_presto_inputs_from_cube




@torch.no_grad()
def inference_loader(
    tif_path: str | Path,
    mask_path: str | Path | None,
    *,
    batch_size: int = 4096,
    num_workers: int = 0,
    pin_memory: bool = False,
    device: Optional[str | torch.device] = None,
) -> Tuple[DataLoader, Dict[str, Any]]:
    """
    Build a DataLoader for inference on a single tile.

    Returns
    -------
    loader : DataLoader
        Yields batches as (X, DW, LL, MK, Y) to match PrestoClassifier._batch_unpack.
        - X  : [B, 12, 17] float32
        - DW : [B, 12]     long   (all 9 = masked class)
        - LL : [B, 2]      float32 (lat, lon in degrees)
        - MK : [B, 12, 17] float32 (structural mask already set in builder)
        - Y  : [B]         long (dummy zeros; ignored by predict/predict_proba)
    meta : dict
        {
          "shape_hw": (H, W),                # full tile size
          "path": Path,                      # input .tif path
          "per_month_order": list[str],      # band ordering
          "flat_indices": np.ndarray[int],   # indices used for inference
          "mask_path": Path | None,          # mask path (if any)
          "mask_value": int | None,          # value used for selection (4)
        }
    """
    tif_path = Path(tif_path)

    # ---- read (C,H,W) + band descs + per-pixel lat/lon in degrees ----
    arr, nodata, lat, lon, descs = _read_tif(str(tif_path), as_mask=False)  # arr: (168,H,W)

    # drop last band if it's something like QA / extra
    arr = arr[:-1, :, :]
    descs = descs[:-1]

    C, H, W = arr.shape
    if C != 168:
        raise RuntimeError(f"Expected 168 bands, got {C} at {tif_path}")

    # ---- reshape to (12,14,H,W) and build Presto inputs ----
    month_bands, per_month_order = reshape_168_to_month_band(arr, descs)
    # x: (N,12,17), mk: (N,12,17), dw: (N,12)
    x, mk, dw = build_presto_inputs_from_cube(month_bands, per_month_order)

    # 🔹 fill NaNs in X over time (per pixel, per feature)
    x = torch.where(torch.isfinite(x), x, torch.tensor(float('nan'), dtype=x.dtype, device=x.device))
    x = interpolate_nan_temporal(x)

    # ---- lat/lon to (N,2) ----
    N = x.shape[0]
    latlons = torch.from_numpy(
        np.stack([lat.reshape(-1), lon.reshape(-1)], axis=-1).astype(np.float32)
    )

    # ---- dummy labels (not used during inference) ----
    labels = torch.zeros((N,), dtype=torch.long)

    # ============================================================
    # Optional external mask: keep only pixels where mask == 4
    # ============================================================
    MASK_VALUE = 4
    if mask_path is not None:
        mask_path = Path(mask_path)

        # 🔹 align mask to tile grid using geolocation
        mask_arr = align_mask_to_tile(mask_path, tif_path)  # (H, W) in tile grid

        # sanity check vs (H, W) from arr
        if mask_arr.shape != (H, W):
            raise RuntimeError(
                f"Aligned mask shape {mask_arr.shape} still does not match tile shape {(H, W)} "
                f"for {mask_path}"
            )

        flat_mask = mask_arr.reshape(-1)
        flat_indices = np.where(flat_mask == MASK_VALUE)[0]

        if flat_indices.size == 0:
            raise RuntimeError(
                f"No pixels with value {MASK_VALUE} found in aligned mask {mask_path}"
            )
    else:
        # no mask: use all pixels
        flat_indices = np.arange(N, dtype=np.int64)


    # ---- apply selection to all tensors (keep order consistent) ----
    idx_t = torch.from_numpy(flat_indices).long()

    x = x[idx_t]          # (B, 12, 17)
    mk = mk[idx_t]        # (B, 12, 17)
    dw = dw[idx_t]        # (B, 12)
    latlons = latlons[idx_t]  # (B, 2)
    labels = labels[idx_t]    # (B,)

    # ---- optional move to device (keeps DataLoader simple) ----
    if device is not None:
        dev = torch.device(device)
        x = x.to(dev)
        mk = mk.to(dev)
        dw = dw.to(dev)
        latlons = latlons.to(dev)
        labels = labels.to(dev)

    # ---- dataset / loader ----
    # tuple aligns with _batch_unpack: (X, DW, LL, MK, Y)
    ds = TensorDataset(x, dw, latlons, mk, labels)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    meta: Dict[str, Any] = {
        "shape_hw": (H, W),
        "path": tif_path,
        "per_month_order": per_month_order,
        "flat_indices": flat_indices,     # <-- for reconstructing full map
        "mask_path": mask_path,
        "mask_value": MASK_VALUE if mask_path is not None else None,
    }

    return loader, meta




def interpolate_nan_temporal(x: torch.Tensor) -> torch.Tensor:
    """
    Interpolate NaN values in X along the time dimension using vectorized linear interpolation.
    
    Parameters
    ----------
    x : torch.Tensor
        Shape (N, T, F) where N=pixels, T=time_steps (months), F=features
        
    Returns
    -------
    x_interp : torch.Tensor
        Same shape as input with NaN values interpolated along time axis
    """
    N, T, F = x.shape
    x_interp = x.clone()
    
    # Reshape to (N*F, T) for vectorized processing
    x_flat = x_interp.view(N * F, T)
    
    # Find which series have NaNs (avoid processing series without NaNs)
    has_nan = torch.isnan(x_flat).any(dim=1)
    series_with_nan = torch.where(has_nan)[0]
    
    if len(series_with_nan) == 0:
        return x_interp
    
    # Process only series with NaNs
    x_subset = x_flat[series_with_nan]  # (M, T) where M = number of series with NaNs
    
    # Create time indices
    time_idx = torch.arange(T, dtype=x.dtype, device=x.device)
    
    # Identify NaN positions
    nan_mask = torch.isnan(x_subset)  # (M, T)
    
    # For each series, find valid (non-NaN) values
    valid_mask = ~nan_mask
    
    # Forward fill: propagate last valid value forward
    x_ffill = x_subset.clone()
    for t in range(1, T):
        # Where current is NaN, use previous value
        needs_fill = nan_mask[:, t]
        x_ffill[needs_fill, t] = x_ffill[needs_fill, t - 1]
    
    # Backward fill: propagate next valid value backward
    x_bfill = x_subset.clone()
    for t in range(T - 2, -1, -1):
        # Where current is NaN, use next value
        needs_fill = nan_mask[:, t]
        x_bfill[needs_fill, t] = x_bfill[needs_fill, t + 1]
    
    # Find first and last valid indices for each series
    valid_indices = valid_mask.float() * time_idx.unsqueeze(0)  # (M, T)
    valid_indices[~valid_mask] = float('inf')
    first_valid = valid_indices.min(dim=1, keepdim=True)[0]  # (M, 1)
    
    valid_indices_rev = valid_mask.float() * time_idx.unsqueeze(0)
    valid_indices_rev[~valid_mask] = float('-inf')
    last_valid = valid_indices_rev.max(dim=1, keepdim=True)[0]  # (M, 1)
    
    # For each NaN position, determine if it's between valid values
    time_grid = time_idx.unsqueeze(0).expand(x_subset.shape[0], -1)  # (M, T)
    is_between = (time_grid > first_valid) & (time_grid < last_valid) & nan_mask
    is_before = (time_grid < first_valid) & nan_mask
    is_after = (time_grid > last_valid) & nan_mask
    
    # Linear interpolation for NaNs between valid values
    # Find nearest valid neighbors using cummax and flip tricks
    # Left neighbor (last valid value before each position)
    valid_cummax = torch.where(valid_mask, time_idx.unsqueeze(0), 
                                torch.tensor(-1, dtype=time_idx.dtype, device=x.device))
    valid_cummax = torch.cummax(valid_cummax, dim=1)[0]  # (M, T)
    
    # Right neighbor (first valid value after each position)
    valid_cummin = torch.where(valid_mask, time_idx.unsqueeze(0),
                                torch.tensor(T, dtype=time_idx.dtype, device=x.device))
    valid_cummin = torch.flip(
        torch.cummin(torch.flip(valid_cummin, dims=[1]), dim=1)[0],
        dims=[1]
    )  # (M, T)
    
    # Get values at left and right neighbors
    left_times = valid_cummax.clamp(0, T - 1).long()
    right_times = valid_cummin.clamp(0, T - 1).long()
    
    left_vals = torch.gather(x_subset, 1, left_times)
    right_vals = torch.gather(x_subset, 1, right_times)
    
    # Compute interpolation weights
    time_diffs = (right_times - left_times).clamp(min=1)  # Avoid division by zero
    weights = (time_grid - left_times).float() / time_diffs.float()
    
    # Linear interpolation
    x_interp_vals = left_vals + weights * (right_vals - left_vals)
    
    # Combine: use interpolation for between, forward fill for before, backward fill for after
    result = x_subset.clone()
    result = torch.where(is_between, x_interp_vals, result)
    result = torch.where(is_before, x_bfill, result)
    result = torch.where(is_after, x_ffill, result)
    
    # Put back into original tensor
    x_flat[series_with_nan] = result
    
    return x_flat.view(N, T, F)



