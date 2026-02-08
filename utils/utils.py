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



from typing import Sequence, List, Tuple, Union


import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.warp import reproject, Resampling
import torch


def lonlat_grids_from_transform(transform, src_crs, H: int, W: int):
    """
    Robust (H,W) lon/lat grids using affine transform directly (pixel centers).
    Works regardless of rasterio.transform.xy return shape quirks.
    """
    rows = np.arange(H, dtype=np.float64)
    cols = np.arange(W, dtype=np.float64)
    cols_grid, rows_grid = np.meshgrid(cols, rows)  # (H,W)

    # pixel center
    cc = cols_grid + 0.5
    rr = rows_grid + 0.5

    # Affine: x = a*col + b*row + c ; y = d*col + e*row + f
    a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    xs = a * cc + b * rr + c
    ys = d * cc + e * rr + f

    transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(xs, ys)

    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    return lon, lat




def _read_tif(
    path: Path,
    as_mask: bool = False,
) -> Tuple[np.ndarray, float | None, np.ndarray, np.ndarray, List[str]]:
    """
    Returns:
      arr   : (C,H,W) or (H,W) if as_mask=True
      nodata: src.nodata
      lon   : (H,W) EPSG:4326
      lat   : (H,W) EPSG:4326
      descs : list of band descriptions (after dropping FILL_MASK)
    """
    path = Path(path)

    with rasterio.open(path) as src:
        arr = src.read(1) if as_mask else src.read()
        nodata = src.nodata
        transform = src.transform
        src_crs = src.crs
        H, W = src.height, src.width

        if src_crs is None:
            raise ValueError(f"{path} has no CRS; cannot compute lon/lat.")

        # descriptions
        descs = list(src.descriptions) if src.descriptions is not None else (
            [None] if arr.ndim == 2 else [None] * arr.shape[0]
        )

        # drop FILL_MASK only for non-mask multi-band rasters
        if (not as_mask) and (arr.ndim == 3):
            keep = []
            for i, d in enumerate(descs):
                if d is None:
                    keep.append(i)
                else:
                    if d.strip().upper() != "FILL_MASK":
                        keep.append(i)
            if len(keep) < arr.shape[0]:
                arr = arr[keep, ...]
                descs = [descs[i] for i in keep]

        # float + cleanup
        arr = arr.astype(np.float32, copy=False)

        # nodata -> NaN
        if nodata is not None:
            arr[arr == nodata] = np.nan

        # nonfinite -> NaN
        bad = ~np.isfinite(arr)
        if bad.any():
            arr[bad] = np.nan

        # robust lon/lat grids
        lon, lat = lonlat_grids_from_transform(transform, src_crs, H, W)

    if as_mask and arr.ndim == 2 and len(descs) > 1:
        descs = [descs[0]]

    return arr, nodata, lon, lat, descs



def align_and_stack_tifs(
    paths: Sequence[Path],
    ref_index: int = 0,
    resampling: Resampling = Resampling.nearest,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[List[str]]]:
    """
    Returns:
      stacked : (C_total,H,W) float32 with NaNs
      lon     : (H,W) EPSG:4326
      lat     : (H,W) EPSG:4326
      band_desc_lists : per-file band descriptions after dropping FILL_MASK
    """
    if not paths:
        raise ValueError("No paths provided to align_and_stack_tifs().")

    ref_path = Path(paths[ref_index])
    with rasterio.open(ref_path) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_height = ref.height
        ref_width = ref.width

    if ref_crs is None:
        raise ValueError(f"Reference tif has no CRS: {ref_path}")

    lon, lat = lonlat_grids_from_transform(ref_transform, ref_crs, ref_height, ref_width)

    aligned_arrays: List[np.ndarray] = []
    band_desc_lists: List[List[str]] = []

    for p in paths:
        p = Path(p)
        with rasterio.open(p) as src:
            data = src.read()  # (C,H,W)
            src_transform = src.transform
            src_crs = src.crs
            nodata = src.nodata
            descs = list(src.descriptions) if src.descriptions is not None else [None] * data.shape[0]

        if src_crs is None:
            raise ValueError(f"Source tif has no CRS: {p}")

        # drop FILL_MASK
        keep = []
        for i, d in enumerate(descs):
            if d is None:
                keep.append(i)
            else:
                if d.strip().upper() != "FILL_MASK":
                    keep.append(i)

        if len(keep) < data.shape[0]:
            data = data[keep, ...]
            descs = [descs[i] for i in keep]

        C = data.shape[0]
        dst = np.full((C, ref_height, ref_width), np.nan, dtype=np.float32)

        for b in range(C):
            reproject(
                source=data[b].astype(np.float32, copy=False),
                destination=dst[b],
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=resampling,
                src_nodata=nodata,
                dst_nodata=np.nan,
            )

        bad = ~np.isfinite(dst)
        if bad.any():
            dst[bad] = np.nan

        aligned_arrays.append(dst)
        band_desc_lists.append(descs)

    stacked = np.concatenate(aligned_arrays, axis=0)
    return stacked, lon, lat, band_desc_lists






def align_mask_to_image(
    img_tif: Path,
    mask_tif: Path,
    out_mask_tif: Path,
) -> np.ndarray:
    """
    Reproject and align mask to match image's CRS, resolution, and grid.

    Writes the aligned mask to `out_mask_tif` and returns the aligned
    mask as a 2D NumPy array (H, W).

    This works both for:
    - Sentinel-2 (10 m): ESRI is already 10 m, only CRS/grid alignment.
    - Landsat-8 (30 m): ESRI 10 m is downsampled to 30 m using nearest.
    """
    with rasterio.open(img_tif) as img_src, rasterio.open(mask_tif) as mask_src:
        same_crs = (img_src.crs == mask_src.crs)
        same_size = (
            img_src.width == mask_src.width
            and img_src.height == mask_src.height
        )
        same_origin_scale = (
            np.isclose(img_src.transform.a, mask_src.transform.a)
            and np.isclose(img_src.transform.e, mask_src.transform.e)
            and np.isclose(img_src.transform.c, mask_src.transform.c)
            and np.isclose(img_src.transform.f, mask_src.transform.f)
        )

        # Fast path: already perfectly aligned
        if same_crs and same_size and same_origin_scale:
            print("✓ Mask already perfectly aligned with image")
            aligned_data = mask_src.read(1)
            return aligned_data

        # Check if dimensions are close (within 1–2 pixels) and grids aligned
        dim_diff = abs(img_src.width - mask_src.width) + abs(
            img_src.height - mask_src.height
        )

        if (
            dim_diff <= 2
            and same_crs
            and np.isclose(img_src.transform.a, mask_src.transform.a)
            and np.isclose(img_src.transform.e, mask_src.transform.e)
        ):
            print(
                f"⚠️  Mask has minor dimension mismatch "
                f"(img={img_src.width}x{img_src.height}, "
                f"mask={mask_src.width}x{mask_src.height}) - reprojecting..."
            )
        else:
            print("⚠️  Mask misaligned - reprojecting to match image...")

        # Read mask data
        mask_data = mask_src.read(1)
        mask_dtype = mask_src.dtypes[0]

        # Build profile for the aligned mask (match image grid)
        out_profile = {
            "driver": "GTiff",
            "dtype": mask_dtype,
            "width": img_src.width,
            "height": img_src.height,
            "count": 1,
            "crs": img_src.crs,
            "transform": img_src.transform,
            "compress": "lzw",
            "tiled": False,
        }

        # Set nodata according to dtype
        if mask_dtype == "uint8" or mask_dtype == np.uint8:
            out_profile["nodata"] = 0
        elif mask_dtype == "int8" or mask_dtype == np.int8:
            out_profile["nodata"] = -128
        elif mask_dtype == "uint16" or mask_dtype == np.uint16:
            out_profile["nodata"] = 0
        elif mask_dtype == "int16" or mask_dtype == np.int16:
            out_profile["nodata"] = -32768
        elif mask_dtype == "uint32" or mask_dtype == np.uint32:
            out_profile["nodata"] = 0
        elif mask_dtype == "int32" or mask_dtype == np.int32:
            out_profile["nodata"] = -2147483648
        else:
            # Float types
            out_profile["nodata"] = -9999.0

        # Allocate aligned array with image dimensions
        aligned_data = np.zeros((img_src.height, img_src.width), dtype=mask_dtype)

        # Reproject / resample mask → image grid
        reproject(
            source=mask_data,
            destination=aligned_data,
            src_transform=mask_src.transform,
            src_crs=mask_src.crs,
            dst_transform=img_src.transform,
            dst_crs=img_src.crs,
            resampling=Resampling.nearest,
        )

        # Write aligned mask to disk
        with rasterio.open(out_mask_tif, "w", **out_profile) as dst:
            dst.write(aligned_data, 1)

        print(
            f"✓ Aligned mask saved to: {out_mask_tif} "
            f"(dimensions: {img_src.width}x{img_src.height})"
        )
        return aligned_data


def save_pred_to_tiff(
    out_arr: Union[np.ndarray, torch.Tensor],
    out_tiff: Union[str, Path],
    ref_tiff: Union[str, Path],
    dtype: str = "uint8",
    nodata: int = None,
    compress: str = "lzw",
) -> Path:
    """
    Save prediction array as GeoTIFF aligned to a reference raster.
    ❌ No resizing is performed — shape must exactly match hw_shape.

    Parameters
    ----------
    out_arr : np.ndarray | torch.Tensor
        Predicted map (H, W) or flat [H*W].
    out_tiff : str | Path
        Path to save output GeoTIFF.
    ref_tiff : str | Path
        Reference raster to copy CRS, transform, and metadata from.
    hw_shape : (int, int)
        Expected raster shape (H, W). Must match out_arr shape.
    dtype : str, default="uint8"
        Output raster dtype.
    nodata : int, optional
        No-data pixel value.
    compress : str, default="lzw"
        Compression type (e.g., "lzw", "deflate").
    """

    out_tiff = Path(out_tiff)
    ref_tiff = Path(ref_tiff)
    H, W = out_arr.shape

    # --- Convert to numpy ---
    if isinstance(out_arr, torch.Tensor):
        out_arr = out_arr.detach().cpu().numpy()
    out_arr = np.asarray(out_arr)

    # --- Reshape or validate ---
    if out_arr.ndim == 1:
        if out_arr.size != H * W:
            raise ValueError(f"Flat prediction has {out_arr.size} elements, expected {H*W}.")
        out_arr = out_arr.reshape(H, W)
    elif out_arr.shape != (H, W):
        raise ValueError(
            f"Prediction shape {out_arr.shape} does not match expected {(H, W)}. "
            "Resizing is disabled."
        )

    out_arr = out_arr.astype(dtype, copy=False)

    # --- Load georeference from reference raster ---
    with rasterio.open(ref_tiff) as src:
        profile = src.profile.copy()

    # --- Update output profile ---
    profile.update(
        count=1,
        dtype=dtype,
        compress=compress,
        height=H,
        width=W,
        nodata=nodata,
    )

    # --- Write GeoTIFF ---
    out_tiff.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_tiff, "w", **profile) as dst:
        dst.write(out_arr, 1)

