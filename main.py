

from pathlib import Path
import numpy as np
import torch
import math
from typing import Dict, List, Optional, Tuple

from torch.utils.data import TensorDataset, DataLoader  

from shap_tiler import ShapefileTiler
from data_sources.pysentinel import S1GEEDownloader, S2GEEDownloader
from data_sources.pylandsat import LandsatGEEDownloader
from data_sources.pysatellite import ERA5GEEDownloader, EsriLULCMaskDownloader
from preprocess import PrestoTensorBuilder
from model.presto import PrestoClassifier
from utils.utils import align_mask_to_image, save_pred_to_tiff


def validate_config(configs: Dict) -> None:
    """Validate configuration parameters."""
    required_keys = [
        "asset_path",
        "credentials_path",
        "service_account",
        "year",
        "sensor_type",
        "out_dir",
        "tile_size",
    ]

    for key in required_keys:
        if key not in configs:
            raise ValueError(f"Missing required config key: {key}")

    if configs["sensor_type"].lower() not in ["sentinel", "landsat"]:
        raise ValueError(
            f"Invalid sensor_type: {configs['sensor_type']}. "
            "Must be 'sentinel' or 'landsat'"
        )

    if configs["year"] < 1984:
        raise ValueError(f"Year {configs['year']} is too early for satellite data")

    if configs["sensor_type"].lower() == "sentinel" and configs["year"] < 2017:
        raise ValueError(f"Sentinel-2 not available before 2017. Got year={configs['year']}")


def setup_directories(out_dir: Path) -> None:
    """Create output directory structure."""
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Output directory ready: {out_dir}")
    except Exception as e:
        raise RuntimeError(f"Failed to create output directory: {e}")


def initialize_downloaders(configs: Dict, out_root: Path) -> Tuple:
    """Initialize appropriate downloaders based on sensor type."""
    sensor_type = configs["sensor_type"].lower()

    try:
        if sensor_type == "sentinel":
            dl_optical = S2GEEDownloader(
                credentials_path=configs["credentials_path"],
                service_account=configs["service_account"],
                output_dir=out_root / "s2",
                bands=configs.get("sentinel_bands", ["red", "green", "blue", "nir"]),
            )

            dl_aux = S1GEEDownloader(
                credentials_path=configs["credentials_path"],
                service_account=configs["service_account"],
                output_dir=out_root / "s1",
            )

            print("✅ Initialized Sentinel-2 and Sentinel-1 downloaders")

        else:  # landsat
            dl_optical = LandsatGEEDownloader(
                credentials_path=configs["credentials_path"],
                service_account=configs["service_account"],
                output_dir=out_root / "landsat",
            )

            dl_aux = ERA5GEEDownloader(
                credentials_path=configs["credentials_path"],
                service_account=configs["service_account"],
                output_dir=out_root / "era5",
            )

            print("✅ Initialized Landsat and ERA5 downloaders")

        return dl_optical, dl_aux

    except Exception as e:
        raise RuntimeError(f"Failed to initialize downloaders: {e}")


def initialize_tiler(configs: Dict) -> ShapefileTiler:
    """Initialize shapefile tiler."""
    try:
        tiler = ShapefileTiler(
            shp_path=configs["asset_path"],
            max_pixels=configs["tile_size"],
            temp_dir="./TEMP/",
        )
        print(f"✅ Initialized tiler for: {configs['asset_path']}")
        return tiler
    except Exception as e:
        raise RuntimeError(f"Failed to initialize tiler: {e}")


def get_tile_statistics(tiler: ShapefileTiler) -> Dict:
    """Get statistics about tiles without consuming the iterator."""
    try:
        print("📊 Analyzing ROI and calculating tile information...")

        # Count polygons
        num_polygons = len(tiler.gdf_orig)

        # Estimate tiles (this is approximate)
        tile_info = {"num_polygons": num_polygons, "tiles": []}

        for poly_idx, (idx, row) in enumerate(tiler.gdf_metric.iterrows()):
            geom_metric = row.geometry

            if geom_metric is None or geom_metric.is_empty:
                continue

            bounds = geom_metric.bounds
            minx, miny, maxx, maxy = bounds
            width_m = maxx - minx
            height_m = maxy - miny

            # Calculate number of tiles needed
            nx = max(1, math.ceil(width_m / tiler.max_size_m))
            ny = max(1, math.ceil(height_m / tiler.max_size_m))

            tile_info["tiles"].append(
                {
                    "poly_idx": poly_idx,
                    "width_m": width_m,
                    "height_m": height_m,
                    "nx": nx,
                    "ny": ny,
                    "total": nx * ny,
                }
            )

        return tile_info

    except Exception as e:
        print(f"⚠️  Warning: Could not calculate tile statistics: {e}")
        return {"num_polygons": 0, "tiles": []}


def create_presto_builder(sensor_type: str) -> PrestoTensorBuilder:
    """Create PrestoTensorBuilder with appropriate band configuration."""
    try:
        sensor_type = sensor_type.lower()

        if sensor_type == "sentinel":
            builder = PrestoTensorBuilder(
                group_flags={
                    "S1": True,
                    "S2_RGB": True,
                    "S2_Red_Edge": False,
                    "S2_NIR_10m": True,
                    "S2_NIR_20m": False,
                    "S2_SWIR": False,
                    "ERA5": False,
                    "SRTM": False,
                    "NDVI": True,
                },
            )
        else:  # landsat
            builder = PrestoTensorBuilder(
                group_flags={
                    "S1": False,
                    "S2_RGB": True,   # using RGB-like stack from Landsat
                    "S2_Red_Edge": False,
                    "S2_NIR_10m": True,
                    "S2_NIR_20m": False,
                    "S2_SWIR": True,
                    "ERA5": True,
                    "SRTM": False,
                    "NDVI": True,
                },
            )

        print(f"✅ Created PrestoTensorBuilder for {sensor_type}")
        return builder

    except Exception as e:
        raise RuntimeError(f"Failed to create PrestoTensorBuilder: {e}")


def load_model(configs: Dict) -> PrestoClassifier:
    """Load the appropriate Presto model based on configuration."""
    sensor_type = configs["sensor_type"].lower()

    try:
        # Determine device
        want_cuda = configs.get("device", "cpu") == "cuda"
        device = torch.device("cuda" if (want_cuda and torch.cuda.is_available()) else "cpu")

        # Select model path
        if sensor_type == "sentinel":
            sentinel_bands = configs.get("sentinel_bands", ["red", "green", "blue", "nir"])
            if set(sentinel_bands) == {"red", "green", "blue", "nir"}:
                model_path = "./weights/irrigation/Presto_S2RGBNIR_S1.pth"
            else:
                model_path = "./weights/irrigation/Presto_S2Full_S1.pth"
        else:  # landsat
            model_path = "./weights/irrigation/Presto_L8_ERA5.pth"

        clf = PrestoClassifier.load(model_path, device=device)
        print(f"✅ Loaded model from: {model_path}")
        print(f"🖥️  Using device: {device}")

        return clf

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model file not found: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


def create_lulc_mask(
    configs: Dict,
    item: Dict,
    out_name: str,
    out_root: Path,
    data_shape: Tuple[int, int],
    path_optical: Path,
) -> Optional[np.ndarray]:
    """
    Create land use/land cover mask based on configuration.

    ESRI behavior:
    - ESRI LULC is downloaded at native 10m.
    - `align_mask_to_image` resamples it onto the optical grid:
        * Sentinel-2: 10m (no resolution change)
        * Landsat-8: 30m (downsample 10m → 30m)
    - Output mask is binary, with 1 for cropland (class 4), else 0.
    """
    landuse_method = configs.get("landuse_method", "skip")
    year = configs["year"]

    try:
        if landuse_method == "ESRI" and 2017 <= year <= 2024:
            print("📥 Downloading ESRI LULC data...")

            dl_lulc = EsriLULCMaskDownloader(
                credentials_path=configs["credentials_path"],
                service_account=configs["service_account"],
                output_dir=out_root / "esri_lulc",
            )

            dl_lulc.download_from_shapefile(
                shp_path=item["shp_path"],
                out_tif=out_name,
                year=year,
            )

            path_lulc = (out_root / "esri_lulc" / out_name).resolve()

            # Align LULC mask to optical image grid
            mask_data = align_mask_to_image(path_optical, path_lulc, path_lulc)

            # Create binary mask: 1 for cropland (class 4), 0 otherwise
            lulc_mask = (mask_data == 4).astype(np.uint8)

            # Safety check on shape
            H, W = data_shape
            if lulc_mask.shape != (H, W):
                raise RuntimeError(
                    f"LULC mask shape {lulc_mask.shape} does not match data shape {(H, W)} "
                    "(check align_mask_to_image implementation)."
                )

            num_cropland = int(lulc_mask.sum())
            print(f"✅ Created ESRI LULC mask ({num_cropland} cropland pixels)")
            return lulc_mask

        elif landuse_method == "presto":
            raise NotImplementedError("Presto-based LULC classification not yet implemented")

        else:  # "skip" or unknown -> use all pixels
            print("⏭️  Skipping LULC mask (processing all pixels)")
            H, W = data_shape
            return np.ones((H, W), dtype=np.uint8)

    except NotImplementedError:
        raise
    except Exception as e:
        print(f"⚠️  Warning: Failed to create LULC mask: {e}")
        print("⏭️  Proceeding without LULC mask")
        H, W = data_shape
        return np.ones((H, W), dtype=np.uint8)


def build_masked_dataloader(
    data_loader: DataLoader,
    lulc_mask: np.ndarray,
) -> Tuple[Optional[DataLoader], np.ndarray]:
    """
    Restrict the DataLoader to only pixels where lulc_mask == 1.

    Assumes underlying dataset is a TensorDataset with:
        x, dw, latlons, mask, labels
    each of shape:
        x:      (N, 12, 17)
        dw:     (N, 12)
        latlons:(N, 2)
        mask:   (N, 12, 17)
        labels: (N,)

    Returns:
        new_loader (or None if no valid pixels),
        valid_idx (flat indices into the full H*W raster).
    """
    flat_mask = lulc_mask.ravel().astype(bool)
    valid_idx = np.where(flat_mask)[0]
    num_valid = int(valid_idx.size)

    if num_valid == 0:
        # No cropland pixels
        return None, valid_idx

    if not hasattr(data_loader, "dataset"):
        print("⚠️ DataLoader has no 'dataset' attribute; skipping masking.")
        return data_loader, valid_idx

    ds = data_loader.dataset
    if not isinstance(ds, TensorDataset):
        print("⚠️ Underlying dataset is not a TensorDataset; skipping masking.")
        return data_loader, valid_idx

    tensors = ds.tensors
    idx_tensor = torch.as_tensor(valid_idx, dtype=torch.long)

    masked_tensors = tuple(t.index_select(0, idx_tensor) for t in tensors)

    batch_size = getattr(data_loader, "batch_size", None)
    if batch_size is None:
        # Fallback: keep everything in a single batch if not specified
        batch_size = num_valid

    new_ds = TensorDataset(*masked_tensors)
    new_loader = DataLoader(
        new_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    return new_loader, valid_idx


def process_tile(
    item: Dict,
    configs: Dict,
    dl_optical,
    dl_aux,
    builder: PrestoTensorBuilder,
    clf: PrestoClassifier,
    out_root: Path,
) -> bool:
    """Process a single tile: download, build tensor, predict, save."""
    poly_idx = item["poly_idx"]
    tile_idx = item["tile_idx"]
    year = configs["year"]
    sensor_type = configs["sensor_type"].lower()

    # Create tile tag for filename
    if tile_idx is None:
        tile_tag = "full"
    else:
        ty, tx = tile_idx
        tile_tag = f"t{ty:03d}_{tx:03d}"

    out_name = f"{year}_p{poly_idx:03d}_{tile_tag}.tif"

    try:
        print(f"\n{'=' * 60}")
        print(f"Processing: {out_name}")
        print(f"{'=' * 60}")

        # ===== 1. Download imagery =====
        print("📥 Downloading optical imagery...")
        dl_optical.download_from_shapefile(
            shp_path=item["shp_path"],
            out_tif=out_name,
            season_year=year,
        )

        print("📥 Downloading auxiliary data...")
        dl_aux.download_from_shapefile(
            shp_path=item["shp_path"],
            out_tif=out_name,
            season_year=year,
        )

        # Get full paths
        if sensor_type == "sentinel":
            path_optical = (out_root / "s2" / out_name).resolve()
            path_aux = (out_root / "s1" / out_name).resolve()
        else:
            path_optical = (out_root / "landsat" / out_name).resolve()
            path_aux = (out_root / "era5" / out_name).resolve()

        # ===== 2. Build Presto tensor =====
        print("🔨 Building Presto tensor...")
        paths_for_builder = [path_optical, path_aux]

        data_loader = builder.build_from_tifs(
            paths=paths_for_builder,
            ref_index=0,
        )

        data_shape = builder.shape  # (H, W)
        H, W = data_shape
        print(f"📐 Data shape: {data_shape}")

        # ===== 3. Create LULC mask =====
        lulc_mask = create_lulc_mask(
            configs, item, out_name, out_root, data_shape, path_optical
        )

        # Ensure mask shape is right
        if lulc_mask is not None and lulc_mask.shape != (H, W):
            raise RuntimeError(
                f"LULC mask shape {lulc_mask.shape} does not match data shape {(H, W)}"
            )

        # ===== 4. Apply LULC mask BEFORE inference =====
        # Only pixels with lulc_mask == 1 (cropland) are sent to the model.
        full_pred = np.zeros((H * W,), dtype=np.uint8)  # default: 0 (nodata/non-crop)

        if lulc_mask is not None:
            flat_mask = lulc_mask.ravel().astype(bool)
            num_valid = int(flat_mask.sum())

            if num_valid == 0:
                # -> No cropland in this tile: skip inference and save all-zero raster
                print(
                    "⚠️ No cropland (LULC class 4) pixels in this tile. "
                    "Skipping irrigation inference and saving all-zero raster."
                )

                full_pred = full_pred.reshape(H, W)
                output_dir = out_root / "predictions"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / out_name

                save_pred_to_tiff(
                    out_arr=full_pred,
                    out_tiff=str(output_path),
                    ref_tiff=str(path_optical),
                    dtype="uint8",
                    nodata=0,
                    compress="lzw",
                )

                print(f"✅ Saved empty (all-zero) prediction: {output_path}")
                return True

            # Mask DataLoader down to only cropland pixels
            print(f"🔍 Filtering to {num_valid} cropland pixels for irrigation inference")
            data_loader, valid_idx = build_masked_dataloader(data_loader, lulc_mask)
        else:
            # No mask: all pixels are considered valid
            print("ℹ️ No LULC mask in use; running inference on all pixels")
            valid_idx = np.arange(H * W, dtype=np.int64)

        # Sanity: if masked dataloader ended up None (no valid pixels)
        if data_loader is None or len(valid_idx) == 0:
            print(
                "⚠️ No valid pixels after masking. "
                "Saving all-zero raster and skipping inference."
            )
            full_pred = full_pred.reshape(H, W)
            output_dir = out_root / "predictions"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / out_name

            save_pred_to_tiff(
                out_arr=full_pred,
                out_tiff=str(output_path),
                ref_tiff=str(path_optical),
                dtype="uint8",
                nodata=0,
                compress="lzw",
            )
            print(f"✅ Saved empty (all-zero) prediction: {output_path}")
            return True

        # ===== 5. Run prediction on VALID pixels only =====
        print("🤖 Running irrigation classification...")
        pred = clf.predict(data_loader)

        # Convert to numpy
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        else:
            pred = np.asarray(pred)

        # Apply label shift (0->1, 1->2, etc.)
        pred = pred + 1

        # ===== 6. Rebuild full image =====
        if pred.shape[0] != valid_idx.shape[0]:
            raise RuntimeError(
                f"Prediction length {pred.shape[0]} does not match "
                f"number of valid pixels {valid_idx.shape[0]}"
            )

        full_pred[valid_idx] = pred.astype(np.uint8)
        full_pred = full_pred.reshape(H, W)

        # ===== 7. Save result =====
        output_dir = out_root / "predictions"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / out_name

        print("💾 Saving prediction...")
        save_pred_to_tiff(
            out_arr=full_pred,
            out_tiff=str(output_path),
            ref_tiff=str(path_optical),
            dtype="uint8",
            nodata=0,
            compress="lzw",
        )

        print(f"✅ Successfully saved: {output_path}")
        return True

    except Exception as e:
        print(f"❌ Error processing {out_name}: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_pipeline(configs: Dict) -> None:
    """Main pipeline execution function."""
    print("\n" + "=" * 60)
    print("🚀 Starting Presto Irrigation Classification Pipeline")
    print("=" * 60 + "\n")

    try:
        # Validate configuration
        print("🔍 Validating configuration...")
        validate_config(configs)

        # Setup directories
        out_root = Path(configs["out_dir"])
        setup_directories(out_root)

        # Initialize components
        dl_optical, dl_aux = initialize_downloaders(configs, out_root)
        tiler = initialize_tiler(configs)

        # Get tile statistics
        tile_stats = get_tile_statistics(tiler)
        estimated_total = sum(t["total"] for t in tile_stats["tiles"])

        print(f"\n📊 Tile Statistics:")
        print(f"   • Number of polygons: {tile_stats['num_polygons']}")
        print(f"   • Estimated total tiles: {estimated_total}")
        for t in tile_stats["tiles"]:
            if t["total"] > 1:
                print(
                    f"   • Polygon {t['poly_idx']}: {t['nx']}×{t['ny']} = "
                    f"{t['total']} tiles ({t['width_m']/1000:.1f}×{t['height_m']/1000:.1f} km)"
                )

        builder = create_presto_builder(configs["sensor_type"].lower())
        clf = load_model(configs)

        # Count total tiles first
        print("\n📊 Counting total tiles...")
        tile_list = list(tiler)
        total_tiles = len(tile_list)
        print(f"✅ Found {total_tiles} tiles to process\n")

        # Process tiles
        print("\n🔄 Starting tile processing...\n")

        success_count = 0
        fail_count = 0

        for idx, item in enumerate(tile_list, 1):
            print(f"\n📍 Tile {idx}/{total_tiles}")

            success = process_tile(
                item, configs, dl_optical, dl_aux, builder, clf, out_root
            )

            if success:
                success_count += 1
            else:
                fail_count += 1

        # Summary
        print("\n" + "=" * 60)
        print("🏁 Pipeline Complete!")
        print("=" * 60)
        print(f"✅ Successful tiles: {success_count}")
        print(f"❌ Failed tiles: {fail_count}")
        print(f"📁 Output directory: {out_root / 'predictions'}")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        raise


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    # User Configuration
    configs = {
        "asset_path": "./ROI/karkheh.shp",
        "credentials_path": "./credentials/earthengine_credentials.json",
        "service_account": "fanapanomaly@fanapanomaly.iam.gserviceaccount.com",
        "year": 2024,
        "sensor_type": "sentinel",  # "sentinel" or "landsat"
        "sentinel_bands": ["red", "green", "blue", "nir"],
        "out_dir": "./data/karkheh",
        "tile_size": 4026,
        "landuse_method": "ESRI",  # "ESRI", "presto", or "skip"
        "device": "cuda",  # "cuda" or "cpu"
    }

    try:
        run_pipeline(configs)
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        exit(1)
