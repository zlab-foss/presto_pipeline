from pathlib import Path
import torch
import numpy as np
import rasterio  # <-- added

from dataset_loaders import inference_loader
from model.presto import PrestoClassifier
from geo_utils import save_pred_to_tiff
from utils import tile_image_only, merge_prediction_tiles, tile_raster_with_mask


def run_inference_single(config: dict):
    """
    Run inference on a *single* GeoTIFF tile.

    Expects:
      - config["input_path"]  : path to tile .tif
      - config["output_path"] : where to write prediction .tif
      - config["model_path"]  : pretrained Presto weights
      - config.get("mask_path") may be None
    """
    want_cuda = (config.get("device", "cpu") == "cuda")
    device = torch.device("cuda" if (want_cuda and torch.cuda.is_available()) else "cpu")

    # --- Build inference DataLoader ---
    loader, meta = inference_loader(
        config["input_path"],
        mask_path=config.get("mask_path"),
        batch_size=config.get("batch_size", 2048),
        device=device,
    )
    
    try:
        loader, meta = inference_loader(
            config["input_path"],
            mask_path=config.get("mask_path", None),
            batch_size=config.get("batch_size", 2048),
            device=device,
        )
    
    except RuntimeError as e:
        msg = str(e)
    
        # Case: mask has no cropland pixels (value = 4)
        if "No pixels with value 4" in msg:
    
            # Read tile shape from image tile
            with rasterio.open(config["input_path"]) as src:
                H, W = src.height, src.width
                profile = src.profile
    
            # Full-zero prediction tile
            full_pred = np.zeros((H, W), dtype=np.uint8)
    
            # Output path
            tile_pred_path = config["output_path"],
    
            # Save fully zero TIFF
            save_pred_to_tiff(
                out_arr=full_pred,
                out_tiff=str(tile_pred_path),
                ref_tiff=str(config["input_path"]),
                dtype="uint8",
                nodata=0,
                compress="lzw",
            )
    
            return   full_pred
    
    

    # --- Load model ---
    clf = PrestoClassifier.load(config["model_path"], device=device)
    print(f"✅ Loaded model from: {config['model_path']}")
    print(f"🖥️  Using device: {clf.device}")

    # --- Predict ---
    pred = clf.predict(loader)

    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    else:
        pred = np.asarray(pred)

    # label shift
    pred = pred + 1

    # --- Rebuild full image ---
    H, W = meta["shape_hw"]
    idx = meta["flat_indices"]  # 1D indices in row-major order

    full_pred = np.full((H * W,), fill_value=0, dtype=np.uint8)  # nodata = 0
    full_pred[idx] = pred
    full_pred = full_pred.reshape(H, W)

    # --- Save to GeoTIFF ---
    save_pred_to_tiff(
        out_arr=full_pred,
        out_tiff=config["output_path"],
        ref_tiff=config["input_path"],
        dtype="uint8",
        nodata=0,
        compress="lzw",
    )

    print(f"💾 Saved prediction to {config['output_path']}")
    return full_pred

def run_inference_big_tif(config: dict):
    """
    Automatically:
      - if image is smaller than or equal to tile_size → run single inference
      - otherwise:
          * creates tile dirs inside ./data/tiles/<year>/
          * tiles BIG image + BIG mask (with auto-alignment)
          * runs inference on each tile
          * merges tiles back into single prediction
    """

    
    year = config["year"]
    big_img = Path(config["input_path"])
    tile_size = config.get("tile_size", 1024)

    # --- Check size first ---
    with rasterio.open(big_img) as src:
        width, height = src.width, src.height

    if width <= tile_size and height <= tile_size:
        print(
            f"ℹ️ Image {big_img.name} is smaller than or equal to tile size "
            f"({width}x{height} <= {tile_size}). Running single-tile inference."
        )
        return run_inference_single(config)

    # --- If bigger than tile_size → do tiling workflow ---
    big_mask = Path(config["mask_path"]) if config.get("mask_path") else None

    # Auto tile directories
    base_dir = Path(f"./data/tiles/{year}")
    img_tiles_dir  = base_dir / "img"
    mask_tiles_dir = base_dir / "mask"
    pred_tiles_dir = base_dir / "pred"

    img_tiles_dir.mkdir(parents=True, exist_ok=True)
    mask_tiles_dir.mkdir(parents=True, exist_ok=True)
    pred_tiles_dir.mkdir(parents=True, exist_ok=True)

    # --- Tile image + mask ---
    print(f"🧩 Tiling image + mask for year {year}:")
    print(f"   image: {big_img}")
    if big_mask:
        print(f"   mask : {big_mask}")

        tile_pairs = tile_raster_with_mask(
            img_tif=big_img,
            mask_tif=big_mask,
            out_dir_img=img_tiles_dir,
            out_dir_mask=mask_tiles_dir,
            tile_size=tile_size,
            prefix=big_img.stem,
            skip_empty_mask=True,
        )
    else:
        # No mask - tile image only
        print("   (no mask)")
        tile_pairs = tile_image_only(
            img_tif=big_img,
            out_dir_img=img_tiles_dir,
            tile_size=tile_size,
            prefix=big_img.stem,
        )

    print(f"   → {len(tile_pairs)} tile pairs created\n")

    # --- Load model once ---
    want_cuda = (config.get("device", "cpu") == "cuda")
    device = torch.device("cuda" if (want_cuda and torch.cuda.is_available()) else "cpu")
    
    clf = PrestoClassifier.load(config["model_path"], device=device)
    print(f"✅ Loaded model from: {config['model_path']}")
    print(f"🖥️  Using device: {clf.device}")

    # --- Inference on each tile ---
    for idx, tile_info in enumerate(tile_pairs):
        if isinstance(tile_info, tuple):
            img_tile, mask_tile = tile_info
        else:
            img_tile = tile_info
            mask_tile = None
        
        img_tile = Path(img_tile)
        
        print(f"🚀 Inference on tile {idx+1}/{len(tile_pairs)}: {img_tile.name}")
        
        try:
            loader, meta = inference_loader(
                str(img_tile),
                mask_path=str(mask_tile) if mask_tile else None,
                batch_size=config.get("batch_size", 2048),
                device=device,
            )
        
        except RuntimeError as e:
            msg = str(e)
        
            # Case: mask has no cropland pixels (value = 4)
            if "No pixels with value 4" in msg:
                print(
                    f"⚠️  Tile {idx+1}/{len(tile_pairs)} → No cropland (value 4). "
                    f"Saving a full-zero tile: {img_tile.name}"
                )
        
                # Read tile shape from image tile
                with rasterio.open(img_tile) as src:
                    H, W = src.height, src.width
                    profile = src.profile
        
                # Full-zero prediction tile
                full_pred = np.zeros((H, W), dtype=np.uint8)
        
                # Output path
                tile_pred_path = pred_tiles_dir / f"{img_tile.stem}_pred.tif"
        
                # Save fully zero TIFF
                save_pred_to_tiff(
                    out_arr=full_pred,
                    out_tiff=str(tile_pred_path),
                    ref_tiff=str(img_tile),
                    dtype="uint8",
                    nodata=0,
                    compress="lzw",
                )
        
                continue  # move to next tile
        
            # Any other error → real problem
            raise

        
        
        # Predict
        pred = clf.predict(loader)
        
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        else:
            pred = np.asarray(pred)
        
        # Label shift
        pred = pred + 1
        
        # Rebuild tile prediction
        H, W = meta["shape_hw"]
        idx_flat = meta["flat_indices"]
        
        full_pred = np.full((H * W,), fill_value=0, dtype=np.uint8)
        full_pred[idx_flat] = pred
        full_pred = full_pred.reshape(H, W)
        
        # Save tile prediction
        tile_pred_path = pred_tiles_dir / f"{img_tile.stem}_pred.tif"
        save_pred_to_tiff(
            out_arr=full_pred,
            out_tiff=str(tile_pred_path),
            ref_tiff=str(img_tile),
            dtype="uint8",
            nodata=0,
            compress="lzw",
        )

    print("✅ All tile predictions complete.")
    
    # --- Merge tiles back together ---
    final_output = Path(config["output_path"])
    merge_prediction_tiles(
        tile_dir=pred_tiles_dir,
        output_path=final_output,
        reference_tif=big_img,
        tile_size=tile_size
    )
    
    print(f"✅ Final prediction saved to {final_output}")
    
    # Return merged prediction
    with rasterio.open(final_output) as src:
        return src.read(1)
