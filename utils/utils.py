import os
from pathlib import Path
# ----------------------------------------------------------------------
# Fix PROJ / GDAL environment (important for geopandas / raster stuff)
# ----------------------------------------------------------------------
os.environ.pop("PROJ_LIB", None)
os.environ.pop("PROJ_DATA", None)

from pyproj import datadir
_pyproj_dir = datadir.get_data_dir()
os.environ["PROJ_DATA"] = _pyproj_dir
os.environ["PROJ_LIB"] = _pyproj_dir

if "CONDA_PREFIX" in os.environ:
    os.environ.setdefault("GDAL_DATA", f"{os.environ['CONDA_PREFIX']}/share/gdal")



from typing import Sequence, List, Tuple

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.warp import reproject, Resampling


def _read_tif(path: Path, as_mask: bool = False) -> Tuple[np.ndarray, float | None, np.ndarray, np.ndarray, List[str]]:
    """
    Read GeoTIFF and return:
      - arr   : (C, H, W) or (H, W) if as_mask=True
                * FILL_MASK band is dropped if present (by description)
                * nodata and all non-finite values are set to np.nan
      - nodata: original nodata value from file (may be None)
      - lat   : (H, W) latitude grid (EPSG:4326)
      - lon   : (H, W) longitude grid (EPSG:4326)
      - descs : list of band descriptions (after dropping FILL_MASK)
    """
    path = Path(path)

    with rasterio.open(path) as src:
        # read data
        arr = src.read(1) if as_mask else src.read()   # (H, W) or (C, H, W)
        nodata = src.nodata
        transform = src.transform
        src_crs = src.crs

        if src_crs is None:
            raise ValueError(f"{path} has no CRS; cannot compute lat/lon.")

        H, W = src.height, src.width

        # band descriptions (may be None)
        if src.descriptions is not None:
            descs = list(src.descriptions)
        else:
            if arr.ndim == 2:
                descs = [None]
            else:
                descs = [None] * arr.shape[0]

        # ----------------------------
        # Drop FILL_MASK band if present
        # ----------------------------
        if not as_mask and arr.ndim == 3:
            keep_idx = []
            for i, d in enumerate(descs):
                if d is None:
                    keep_idx.append(i)
                else:
                    name = d.strip().upper()
                    if name != "FILL_MASK":
                        keep_idx.append(i)

            if len(keep_idx) < arr.shape[0]:
                arr = arr[keep_idx, ...]
                descs = [descs[i] for i in keep_idx]

        # ensure float for cleaning
        arr = arr.astype("float32", copy=False)

        # ----------------------------
        # Replace nodata with NaN
        # ----------------------------
        if nodata is not None:
            if arr.ndim == 2:
                arr[arr == nodata] = np.nan
            else:
                arr[arr == nodata] = np.nan

        # ----------------------------
        # Replace non-finite (inf, -inf, NaN, etc.) with NaN
        # ----------------------------
        bad = ~np.isfinite(arr)
        if bad.any():
            arr[bad] = np.nan

        # ----------------------------
        # Build lat / lon grids from transform
        # ----------------------------
        rows = np.arange(H)
        cols = np.arange(W)
        cols_grid, rows_grid = np.meshgrid(cols, rows)

        xs, ys = rasterio.transform.xy(transform, rows_grid, cols_grid, offset="center")
        xs = np.asarray(xs)
        ys = np.asarray(ys)

        transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(xs, ys)

    # If as_mask=True, arr is (H, W), descs should be length 1
    if as_mask and arr.ndim == 2 and len(descs) > 1:
        descs = [descs[0]]

    return arr, nodata, lat, lon, descs



def align_and_stack_tifs(
    paths: Sequence[Path],
    ref_index: int = 0,
    resampling: Resampling = Resampling.nearest,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[List[str]]]:
    """
    Align and merge multiple GeoTIFFs to the same grid.

    Parameters
    ----------
    paths : sequence of Path
        List of GeoTIFF paths.
    ref_index : int
        Index of the file in `paths` to use as the reference grid (CRS, transform, H, W).
    resampling : rasterio.warp.Resampling
        Resampling method for reprojection.

    Returns
    -------
    stacked : np.ndarray
        (C_total, H, W) array, all inputs stacked along channels,
        with nodata and non-finite values set to np.nan and without FILL_MASK.
    lat : np.ndarray
        (H, W) latitude grid (EPSG:4326) for the reference grid.
    lon : np.ndarray
        (H, W) longitude grid (EPSG:4326) for the reference grid.
    band_desc_lists : list[list[str]]
        Per-file list of band descriptions AFTER dropping FILL_MASK.
        (i.e., band_desc_lists[i] corresponds to paths[i]).
    """
    if len(paths) == 0:
        raise ValueError("No paths provided to align_and_stack_tifs().")

    # -----------------------------
    # Open reference file
    # -----------------------------
    ref_path = Path(paths[ref_index])
    with rasterio.open(ref_path) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_height = ref.height
        ref_width = ref.width

    # Build lat/lon grid for the reference
    rows = np.arange(ref_height)
    cols = np.arange(ref_width)
    cols_grid, rows_grid = np.meshgrid(cols, rows)

    xs, ys = rasterio.transform.xy(ref_transform, rows_grid, cols_grid, offset="center")
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    transformer = Transformer.from_crs(ref_crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(xs, ys)

    # -----------------------------
    # Reproject each raster to reference grid
    # -----------------------------
    aligned_arrays: List[np.ndarray] = []
    band_desc_lists: List[List[str]] = []

    for p in paths:
        p = Path(p)
        with rasterio.open(p) as src:
            data = src.read()  # (C, H, W) or (1, H, W)
            src_transform = src.transform
            src_crs = src.crs
            nodata = src.nodata
            if src.descriptions is not None:
                descs = list(src.descriptions)
            else:
                descs = [None] * data.shape[0]

        # Drop FILL_MASK band if present
        keep_idx = []
        for i, d in enumerate(descs):
            if d is None:
                keep_idx.append(i)
            else:
                name = d.strip().upper()
                if name != "FILL_MASK":
                    keep_idx.append(i)

        if len(keep_idx) < data.shape[0]:
            data = data[keep_idx, ...]
            descs = [descs[i] for i in keep_idx]

        # Prepare destination array on reference grid
        C = data.shape[0]
        dst = np.full((C, ref_height, ref_width), np.nan, dtype="float32")

        # Reproject each band
        for b in range(C):
            reproject(
                source=data[b].astype("float32"),
                destination=dst[b],
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=resampling,
                src_nodata=nodata,
                dst_nodata=np.nan,
            )

        # Clean non-finite values
        bad = ~np.isfinite(dst)
        if bad.any():
            dst[bad] = np.nan

        aligned_arrays.append(dst)
        band_desc_lists.append(descs)

    # -----------------------------
    # Stack all channels together
    # -----------------------------
    stacked = np.concatenate(aligned_arrays, axis=0)  # (C_total, H, W)

    return stacked, lat, lon, band_desc_lists
