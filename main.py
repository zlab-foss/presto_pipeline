from pathlib import Path

from get_data import S2S1PrestoDownloader
from get_landcover import esri_landuse
from inference import run_inference_big_tif
from utils import pair_by_idx
from geo_utils import _list_tifs


def run_presto_pipeline(configs: dict):

    required = ["asset_path", "year", "landuse_method", "device", "model_path"]
    for k in required:
        if k not in configs:
            raise ValueError(f"Missing required config key: {k}")

    year = configs["year"]
    landuse_method = configs["landuse_method"]
    skip_download = configs.get("skip_download", False)

    asset_path = Path(configs["asset_path"])
    roi_name = asset_path.stem    # e.g. "roi_0"

    # ----------------------------------------
    # Base directory per ROI
    # ----------------------------------------
    ROI_BASE = Path("./data") / roi_name
    INPUT_DIR  = ROI_BASE / "inputs"
    OUTPUT_DIR = ROI_BASE / "outputs"
    LULC_DIR   = ROI_BASE / "LULC"

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LULC_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------
    # 1) Download step
    # ----------------------------------------
    if not skip_download:
        downloader = S2S1PrestoDownloader(
            asset_path=str(asset_path),
            output_dir=str(INPUT_DIR),
            start_year=year,
        )
        downloader.run()
    else:
        print("⏭️  Skipping S2/S1 download step. Using existing TIFFs.\n")

    # ----------------------------------------
    # 2) Landcover or no landcover
    # ----------------------------------------
    if landuse_method == "ESRI":

        if "ESRI_mask_path" not in configs:
            raise RuntimeError("ESRI_mask_path must be provided for ESRI mode.")

        if not (2017 <= year <= 2024):
            raise RuntimeError(f"ESRI does not support year={year}.")

        # ESRI output directory (per ROI)
        esri_cfg = configs.copy()
        esri_cfg["LULC_output_dir"] = str(LULC_DIR)
        esri_landuse(esri_cfg)

        # Pair TIFFs by index
        pairs = pair_by_idx(str(INPUT_DIR), str(LULC_DIR))

        for pair in pairs:
            idx = pair["idx"]

            cfg = configs.copy()
            cfg["input_path"]  = pair["input"]
            cfg["mask_path"]   = pair["mask"]
            cfg["output_path"] = str(OUTPUT_DIR / f"irrigation_{year}_{idx}.tif")

            run_inference_big_tif(cfg)

    else:
        # Run inference without landcover masks
        tiffs = _list_tifs(str(INPUT_DIR))

        for idx, input_path in enumerate(tiffs):
            cfg = configs.copy()
            cfg["input_path"]  = input_path
            cfg["mask_path"]   = None
            cfg["output_path"] = str(OUTPUT_DIR / f"irrigation_{year}_{idx}.tif")

            run_inference_big_tif(cfg)

if __name__ == "__main__":
    
    

    for year in [2024, 2017, 2019]:
        shp_dir = Path("./data/ROI/sample_wetlands")

        for shp_path in sorted(shp_dir.glob("*.shp")):
            configs = {
                "asset_path": shp_path,
                "year": year,
                "landuse_method": "ESRI",
                "ESRI_mask_path": f"../LULC/landcover-{year}",
                "device": "cuda",
                "model_path": "./weights/tune_model.pth",
                "skip_download": False,
                "tile_size": 2048,
            }
        
            run_presto_pipeline(configs)
        

