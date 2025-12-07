import numpy as np
import torch
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------
#  Temporal NaN interpolation
# ---------------------------------------------------------------------
def interpolate_nan_temporal(x: torch.Tensor) -> torch.Tensor:
    """
    Interpolate NaN values in x along the time dimension using vectorized
    linear interpolation.

    Parameters
    ----------
    x : torch.Tensor
        Shape (N, T, F) where:
          N = number of pixels
          T = time steps (months)
          F = features (channels)

    Returns
    -------
    x_interp : torch.Tensor
        Same shape as input, with NaN values interpolated along the time axis.
        - NaNs fully surrounded by valid values → linear interpolation.
        - NaNs before the first valid time      → backward fill from first valid.
        - NaNs after the last valid time       → forward fill from last valid.
        - Time series that are all-NaN remain NaN.
    """
    N, T, F = x.shape
    x_interp = x.clone()

    # Reshape to (N*F, T) for vectorized temporal interpolation
    x_flat = x_interp.view(N * F, T)

    # Only process series that actually contain NaNs
    has_nan = torch.isnan(x_flat).any(dim=1)
    series_with_nan = torch.where(has_nan)[0]

    if len(series_with_nan) == 0:
        return x_interp

    # (M, T), where M = number of series with NaNs
    x_subset = x_flat[series_with_nan]

    # Time index (0..T-1)
    time_idx = torch.arange(T, dtype=x.dtype, device=x.device)

    # Mask of NaN positions
    nan_mask = torch.isnan(x_subset)           # (M, T)
    valid_mask = ~nan_mask                     # (M, T)

    # ------------------ Forward fill ------------------
    x_ffill = x_subset.clone()
    for t in range(1, T):
        needs_fill = nan_mask[:, t]
        x_ffill[needs_fill, t] = x_ffill[needs_fill, t - 1]

    # ------------------ Backward fill ------------------
    x_bfill = x_subset.clone()
    for t in range(T - 2, -1, -1):
        needs_fill = nan_mask[:, t]
        x_bfill[needs_fill, t] = x_bfill[needs_fill, t + 1]

    # ------------------ First/last valid indices ------------------
    valid_indices = valid_mask.float() * time_idx.unsqueeze(0)  # (M, T)
    valid_indices[~valid_mask] = float("inf")
    first_valid = valid_indices.min(dim=1, keepdim=True)[0]     # (M, 1)

    valid_indices_rev = valid_mask.float() * time_idx.unsqueeze(0)
    valid_indices_rev[~valid_mask] = float("-inf")
    last_valid = valid_indices_rev.max(dim=1, keepdim=True)[0]  # (M, 1)

    # Position classification: before / between / after valid
    time_grid = time_idx.unsqueeze(0).expand(x_subset.shape[0], -1)  # (M, T)
    is_between = (time_grid > first_valid) & (time_grid < last_valid) & nan_mask
    is_before = (time_grid < first_valid) & nan_mask
    is_after = (time_grid > last_valid) & nan_mask

    # ------------------ Neighbor indices for interpolation ------------------
    # Left neighbor (last valid idx up to t)
    valid_cummax = torch.where(
        valid_mask,
        time_idx.unsqueeze(0),
        torch.tensor(-1, dtype=time_idx.dtype, device=x.device),
    )
    valid_cummax = torch.cummax(valid_cummax, dim=1)[0]  # (M, T)

    # Right neighbor (first valid idx at or after t)
    valid_cummin = torch.where(
        valid_mask,
        time_idx.unsqueeze(0),
        torch.tensor(T, dtype=time_idx.dtype, device=x.device),
    )
    valid_cummin = torch.flip(
        torch.cummin(torch.flip(valid_cummin, dims=[1]), dim=1)[0],
        dims=[1],
    )  # (M, T)

    left_times = valid_cummax.clamp(0, T - 1).long()
    right_times = valid_cummin.clamp(0, T - 1).long()

    left_vals = torch.gather(x_subset, 1, left_times)
    right_vals = torch.gather(x_subset, 1, right_times)

    # Avoid division by zero where left == right
    time_diffs = (right_times - left_times).clamp(min=1).float()
    weights = (time_grid - left_times).float() / time_diffs

    # Linear interpolation between neighbors
    x_interp_vals = left_vals + weights * (right_vals - left_vals)

    # Combine:
    # - between → interpolated
    # - before  → backward filled
    # - after   → forward filled
    result = x_subset.clone()
    result = torch.where(is_between, x_interp_vals, result)
    result = torch.where(is_before, x_bfill, result)
    result = torch.where(is_after, x_ffill, result)

    # Write back to full array
    x_flat[series_with_nan] = result

    return x_flat.view(N, T, F)




def _gather_cube(
    stacked: np.ndarray,
    index_map: Dict[Tuple[int, str], int],
    band_name: str,
) -> np.ndarray:
    """
    Gather a (12, H, W) cube for a given band_name, requiring that
    this band exists for all 12 months.

    Parameters
    ----------
    stacked : np.ndarray
        (C, H, W) stacked array from align_and_stack_tifs.
    index_map : dict
        (month_idx, band_name) -> channel index.
    band_name : str
        Physical band name (lowercase), e.g. 'red', 'nir', 'vv', 'elevation'.

    Returns
    -------
    cube : np.ndarray
        (12, H, W) monthly cube, dtype float32.
    """
    C, H, W = stacked.shape
    cubes = []
    for m in range(12):
        key = (m, band_name)
        if key not in index_map:
            raise ValueError(f"Missing band '{band_name}' for month {m + 1}.")
        ch_idx = index_map[key]
        cubes.append(stacked[ch_idx])
    return np.stack(cubes, axis=0).astype(np.float32, copy=False)

