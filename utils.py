import re
from pathlib import Path
from geo_utils import _list_tifs


from typing import List, Tuple
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import  reproject, Resampling
from tqdm import tqdm
import csv


# ----------------------------------------------------
# Helpers
# ----------------------------------------------------
def _pct_nonfinite_pixels(tif_path: str | Path) -> float:
    """Return percentage of non-finite pixels (NaN/±inf) in a TIFF."""
    tif_path = Path(tif_path)
    with rasterio.open(tif_path) as src:
        arr = src.read()  # (bands, H, W)
    total = arr.size
    if total == 0:
        return 0.0
    nonfinite = np.count_nonzero(~np.isfinite(arr))
    return float(nonfinite) * 100.0 / float(total)


def _make_synthetic_lulc_mask_like(
    input_tif: str | Path,
    out_tif: str | Path,
    value: int = 4,
):
    """
    Create a 1-band LULC mask with all pixels == value,
    aligned to the georeferencing of input_tif.

    IMPORTANT:
    - Do NOT keep the input nodata (often -inf / float),
      because we're writing an int16 mask.
    """
    input_tif = Path(input_tif)
    out_tif = Path(out_tif)

    with rasterio.open(input_tif) as src:
        meta = src.meta.copy()

        # Force a clean integer mask meta
        meta.update(
            count=1,
            dtype="int16",
            nodata=0,  # safe int16 nodata within range
        )

        # Some drivers / older files may carry extra tags that conflict – remove what we don't need
        # (GDAL can ignore extras, but this keeps it clean)
        meta.pop("alph", None)
        meta.pop("mask", None)

        data = np.full((1, src.height, src.width), value, dtype="int16")

        out_tif.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_tif, "w", **meta) as dst:
            dst.write(data)



def _append_metadata_row(csv_path: Path, row: dict, fieldnames: list[str]) -> None:
    """Append one row to ROI-level metadata CSV, write header if file is new."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def natural_key(p: Path):
    """Extract numeric index from filename like roi_123.shp."""
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else 0





def _get_band_descriptions(src) -> list[str]:
    """
    Return band descriptions for all bands in a rasterio dataset.

    If the source TIFF has no band descriptions, we fall back to the
    standard Presto naming scheme:

        M01_coastal, M01_blue, ..., M12_vh, [optional FILL_MASK]

    We do NOT drop any bands here. Dropping the last "FILL_MASK"
    band is already handled inside dataset_loaders._read_tif.
    """
    # Read raw descriptions from the file (may be None / empty / contain None)
    raw_descs = list(src.descriptions) if src.descriptions is not None else []
    band_count = src.count

    # -------------------------------
    # 1) If no descriptions at all → build default names
    # -------------------------------
    if not raw_descs or all(d is None for d in raw_descs):
        if band_count not in (168, 169):
            raise RuntimeError(
                f"Source TIFF has no band descriptions and band_count={band_count}, "
                "expected 168 or 169 for Presto."
            )

        month_ids = [f"M{m:02d}" for m in range(1, 13)]
        per_month_feats = [
            "coastal",
            "blue",
            "green",
            "red",
            "red_edge1",
            "red_edge2",
            "red_edge3",
            "nir",
            "red_edge4",
            "water_vapor",
            "swir1",
            "swir2",
            "vv",
            "vh",
        ]

        descs: list[str] = []
        for m in month_ids:
            for f in per_month_feats:
                descs.append(f"{m}_{f}")

        # If there is an extra band, assume it's the fill mask
        if band_count == 169:
            descs.append("FILL_MASK")

        print("ℹ️ Source TIFF had no band descriptions; using default Presto names.")
        return descs

    # -------------------------------
    # 2) There ARE some descriptions → clean / validate
    # -------------------------------
    if len(raw_descs) != band_count:
        raise RuntimeError(
            f"Band descriptions length ({len(raw_descs)}) "
            f"!= band count ({band_count})."
        )

    cleaned: list[str] = []
    for i, d in enumerate(raw_descs):
        if d is None:
            # Fallback in case a single band has no name
            # Try to follow the same Mxx_xxx scheme if possible.
            month = i // 14 + 1
            idx_in_month = i % 14
            per_month_feats = [
                "coastal",
                "blue",
                "green",
                "red",
                "red_edge1",
                "red_edge2",
                "red_edge3",
                "nir",
                "red_edge4",
                "water_vapor",
                "swir1",
                "swir2",
                "vv",
                "vh",
            ]
            if 1 <= month <= 12 and 0 <= idx_in_month < len(per_month_feats):
                default_name = f"M{month:02d}_{per_month_feats[idx_in_month]}"
            else:
                default_name = f"B{i+1}"
            cleaned.append(default_name)
        else:
            cleaned.append(str(d))

    return cleaned




def align_mask_to_image(
    img_tif: Path,
    mask_tif: Path,
    out_mask_tif: Path
) -> Path:
    """
    Reproject and align mask to match image's CRS, resolution, and grid.
    
    Returns path to aligned mask.
    """
    with rasterio.open(img_tif) as img_src, rasterio.open(mask_tif) as mask_src:
        # Check if already perfectly aligned (same dimensions and transform)
        if (img_src.crs == mask_src.crs and 
            img_src.width == mask_src.width and
            img_src.height == mask_src.height and
            np.isclose(img_src.transform.a, mask_src.transform.a) and
            np.isclose(img_src.transform.e, mask_src.transform.e) and
            np.isclose(img_src.transform.c, mask_src.transform.c) and
            np.isclose(img_src.transform.f, mask_src.transform.f)):
            print("✓ Mask already perfectly aligned with image")
            return mask_tif
        
        # Check if dimensions are close (within 1-2 pixels) and grids aligned
        dim_diff = abs(img_src.width - mask_src.width) + abs(img_src.height - mask_src.height)
        
        if (dim_diff <= 2 and 
            img_src.crs == mask_src.crs and
            np.isclose(img_src.transform.a, mask_src.transform.a) and
            np.isclose(img_src.transform.e, mask_src.transform.e)):
            print(f"⚠️  Mask has minor dimension mismatch (img={img_src.width}x{img_src.height}, "
                  f"mask={mask_src.width}x{mask_src.height}) - reprojecting...")
        else:
            print("⚠️  Mask misaligned - reprojecting to match image...")
        
        # Read mask data
        mask_data = mask_src.read(1)
        mask_dtype = mask_src.dtypes[0]
        
        # Build profile from scratch for the mask (don't copy from image!)
        out_profile = {
            'driver': 'GTiff',
            'dtype': mask_dtype,
            'width': img_src.width,
            'height': img_src.height,
            'count': 1,
            'crs': img_src.crs,
            'transform': img_src.transform,
            'compress': 'lzw',
            'tiled': False
        }
        
        # Set appropriate nodata value for the mask's dtype
        if mask_dtype == 'uint8' or mask_dtype == np.uint8:
            out_profile['nodata'] = 0
        elif mask_dtype == 'int8' or mask_dtype == np.int8:
            out_profile['nodata'] = -128
        elif mask_dtype == 'uint16' or mask_dtype == np.uint16:
            out_profile['nodata'] = 0
        elif mask_dtype == 'int16' or mask_dtype == np.int16:
            out_profile['nodata'] = -32768
        elif mask_dtype == 'uint32' or mask_dtype == np.uint32:
            out_profile['nodata'] = 0
        elif mask_dtype == 'int32' or mask_dtype == np.int32:
            out_profile['nodata'] = -2147483648
        else:
            # Float types
            out_profile['nodata'] = -9999.0
        
        # Create aligned output array with image dimensions
        aligned_data = np.zeros((img_src.height, img_src.width), dtype=mask_dtype)
        
        reproject(
            source=mask_data,
            destination=aligned_data,
            src_transform=mask_src.transform,
            src_crs=mask_src.crs,
            dst_transform=img_src.transform,
            dst_crs=img_src.crs,
            resampling=Resampling.nearest
        )
        
        # Write aligned mask
        with rasterio.open(out_mask_tif, 'w', **out_profile) as dst:
            dst.write(aligned_data, 1)
        
        print(f"✓ Aligned mask saved to: {out_mask_tif} (dimensions: {img_src.width}x{img_src.height})")
        return out_mask_tif


def tile_raster_with_mask(
    img_tif: str | Path,
    mask_tif: str | Path,
    out_dir_img: str | Path,
    out_dir_mask: str | Path,
    tile_size: int = 1024,
    *,
    prefix: str | None = None,
    skip_empty_mask: bool = True,
) -> List[Tuple[Path, Path]]:
    """
    Tile aligned image and mask into tile pairs.
    Automatically handles alignment if needed.
    Preserves band descriptions from the source image.
    """
    img_tif = Path(img_tif)
    mask_tif = Path(mask_tif)
    out_dir_img = Path(out_dir_img)
    out_dir_mask = Path(out_dir_mask)
    
    out_dir_img.mkdir(parents=True, exist_ok=True)
    out_dir_mask.mkdir(parents=True, exist_ok=True)
    
    if prefix is None:
        prefix = img_tif.stem
    
    # Create aligned mask if needed
    aligned_mask_dir = out_dir_mask.parent / "aligned_masks"
    aligned_mask_dir.mkdir(parents=True, exist_ok=True)
    aligned_mask_path = aligned_mask_dir / f"{mask_tif.stem}_aligned.tif"
    
    mask_tif = align_mask_to_image(img_tif, mask_tif, aligned_mask_path)
    
    tile_pairs: List[Tuple[Path, Path]] = []
    
    with rasterio.open(img_tif) as src_img, rasterio.open(mask_tif) as src_mask:
        # Verify alignment after potential reprojection
        if (src_img.width != src_mask.width or 
            src_img.height != src_mask.height):
            raise RuntimeError(
                f"After alignment, dimensions still mismatch: "
                f"img={src_img.width}x{src_img.height}, "
                f"mask={src_mask.width}x{src_mask.height}"
            )
        
        width = src_img.width
        height = src_img.height
        img_bands = src_img.count
        
        img_profile = src_img.profile.copy()
        mask_profile = src_mask.profile.copy()
        
        img_profile.update(tiled=False, blockxsize=None, blockysize=None)
        mask_profile.update(tiled=False, blockxsize=None, blockysize=None, count=1)
        
        # Read band descriptions from source image (validated)
        band_descriptions = _get_band_descriptions(src_img)
        
        # Calculate number of tiles
        n_tiles_x = int(np.ceil(width / tile_size))
        n_tiles_y = int(np.ceil(height / tile_size))
        
        print(f"📐 Image size: {width}x{height}")
        print(f"🧩 Creating {n_tiles_x}x{n_tiles_y} = {n_tiles_x * n_tiles_y} tiles")
        
        tile_idx = 0
        skipped = 0
        for ty in tqdm(range(n_tiles_y), desc="🧩 Tiling rows", unit="row"):
            for tx in range(n_tiles_x):
                # Calculate tile boundaries
                col_off = tx * tile_size
                row_off = ty * tile_size
                w = min(tile_size, width - col_off)
                h = min(tile_size, height - row_off)
                
                if w <= 0 or h <= 0:
                    continue
                
                # Create window for this tile
                win = Window(
                    col_off=col_off,
                    row_off=row_off,
                    width=w,
                    height=h
                )
                
                # Read data
                img_data = src_img.read(window=win)
                mask_data = src_mask.read(1, window=win)
                
                # Skip tiles with no valid mask data at all (all zeros)
                if skip_empty_mask:
                    if not np.any(mask_data > 0):
                        skipped += 1
                        continue
                
                # Calculate transform for this tile
                tile_transform = rasterio.windows.transform(win, src_img.transform)
                
                # Update profiles for tile
                img_tile_profile = img_profile.copy()
                img_tile_profile.update(
                    width=w,
                    height=h,
                    transform=tile_transform,
                    count=img_bands
                )
                
                mask_tile_profile = mask_profile.copy()
                mask_tile_profile.update(
                    width=w,
                    height=h,
                    transform=tile_transform,
                    count=1
                )
                
                # Generate filenames
                img_name = f"{prefix}_tile_{tile_idx:04d}.tif"
                mask_name = f"{prefix}_mask_tile_{tile_idx:04d}.tif"
                
                img_out_path = out_dir_img / img_name
                mask_out_path = out_dir_mask / mask_name
                
                # Write image tile with band descriptions
                with rasterio.open(img_out_path, "w", **img_tile_profile) as dst_img:
                    dst_img.write(img_data)
                    for i, desc in enumerate(band_descriptions, start=1):
                        if desc:
                            dst_img.set_band_description(i, desc)
                
                # Write mask tile
                with rasterio.open(mask_out_path, "w", **mask_tile_profile) as dst_mask:
                    dst_mask.write(mask_data, 1)
                
                tile_pairs.append((img_out_path, mask_out_path))
                tile_idx += 1
        
        print(f"✓ Created {len(tile_pairs)} valid tiles ({skipped} skipped)")
    
    return tile_pairs




def merge_prediction_tiles(
    tile_dir: Path,
    output_path: Path,
    reference_tif: Path,
    tile_size: int = 1024
):
    """
    Merge prediction tiles back into a single GeoTIFF.
    Uses reference image for dimensions and geotransform.
    """
    from geo_utils import save_pred_to_tiff
    
    with rasterio.open(reference_tif) as ref:
        width = ref.width
        height = ref.height
        
        # Initialize full prediction array
        full_pred = np.zeros((height, width), dtype=np.uint8)
        
        # Get all prediction tiles
        pred_tiles = sorted(tile_dir.glob("*_pred.tif"))
        
        print(f"🧩 Merging {len(pred_tiles)} prediction tiles...")
        
        for tile_path in pred_tiles:
            with rasterio.open(tile_path) as tile_src:
                # Get tile's position from its transform
                tile_transform = tile_src.transform
                ref_transform = ref.transform
                
                # Calculate pixel offset
                col_off = int(round((tile_transform.c - ref_transform.c) / ref_transform.a))
                row_off = int(round((tile_transform.f - ref_transform.f) / ref_transform.e))
                
                # Read tile data
                tile_data = tile_src.read(1)
                
                # Place in full array
                h, w = tile_data.shape
                full_pred[row_off:row_off+h, col_off:col_off+w] = tile_data
        
        # Save merged prediction
        save_pred_to_tiff(
            out_arr=full_pred,
            out_tiff=output_path,
            ref_tiff=reference_tif,
            dtype="uint8",
            nodata=0,
            compress="lzw"
        )
        
        print(f"✓ Merged prediction saved to: {output_path}")
        
    return full_pred




def tile_image_only(
    img_tif: Path,
    out_dir_img: Path,
    tile_size: int = 1024,
    prefix: str | None = None,
) -> List[Path]:
    """
    Tile image without mask (for no-mask inference).
    Preserves band descriptions from the source image.
    """
    out_dir_img.mkdir(parents=True, exist_ok=True)
    
    if prefix is None:
        prefix = img_tif.stem
    
    tiles: List[Path] = []
    
    with rasterio.open(img_tif) as src:
        width = src.width
        height = src.height
        img_bands = src.count
        
        profile = src.profile.copy()
        profile.update(tiled=False, blockxsize=None, blockysize=None)
        
        # Read band descriptions (validated)
        band_descriptions = _get_band_descriptions(src)
        
        n_tiles_x = int(np.ceil(width / tile_size))
        n_tiles_y = int(np.ceil(height / tile_size))
        
        tile_idx = 0
        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                col_off = tx * tile_size
                row_off = ty * tile_size
                w = min(tile_size, width - col_off)
                h = min(tile_size, height - row_off)
                
                if w <= 0 or h <= 0:
                    continue
                
                win = Window(col_off=col_off, row_off=row_off, width=w, height=h)
                data = src.read(window=win)
                
                tile_transform = rasterio.windows.transform(win, src.transform)
                
                tile_profile = profile.copy()
                tile_profile.update(
                    width=w,
                    height=h,
                    transform=tile_transform,
                    count=img_bands,
                )
                
                tile_name = f"{prefix}_tile_{tile_idx:04d}.tif"
                tile_path = out_dir_img / tile_name
                
                with rasterio.open(tile_path, "w", **tile_profile) as dst:
                    dst.write(data)
                    for i, desc in enumerate(band_descriptions, start=1):
                        if desc:
                            dst.set_band_description(i, desc)
                
                tiles.append(tile_path)
                tile_idx += 1
    
    return tiles


def extract_idx(path: Path) -> int:
    """
    Extract index from filename patterns like:
    - season2024_feat0.tif  -> 0
    - poly_0.tif            -> 0
    """
    name = path.stem  # no extension
    
    # find any integer at the end of the filename
    m = re.search(r'(\d+)$', name)
    if m is None:
        raise ValueError(f"Cannot extract idx from filename: {path}")
    
    return int(m.group(1))


def pair_by_idx(input_dir, mask_dir):

    
    input_tifs = _list_tifs(input_dir)
    mask_tifs  = _list_tifs(mask_dir)
    
    
    # Index them into dicts
    input_map = {extract_idx(p): p for p in input_tifs}
    mask_map  = {extract_idx(p): p for p in mask_tifs}

    # Intersection of available indices
    common = sorted(set(input_map.keys()) & set(mask_map.keys()))

    if not common:
        raise RuntimeError("No matching indices between input_tifs and mask_tifs")

    # Build paired list
    pairs = [
        {"idx": idx, "input": input_map[idx], "mask": mask_map[idx]}
        for idx in common
    ]
    return pairs

