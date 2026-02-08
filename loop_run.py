from main import run_pipeline
from pathlib import Path
import shutil


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def remove_temp(out_dir: str):
    """Remove TEMP directory inside output folder if it exists."""
    temp_dir = Path(out_dir) / "TEMP"
    if temp_dir.exists() and temp_dir.is_dir():
        try:
            shutil.rmtree(temp_dir)
            print(f"🧹 TEMP removed: {temp_dir}")
        except Exception as e:
            print(f"⚠️ Failed to remove TEMP: {e}")
    else:
        print("ℹ️ No TEMP directory found — skipping cleanup.")


def pick_landuse(year: int) -> str:
    """ESRI LULC valid only for 2017–2024."""
    if 2017 <= year <= 2024:
        return "ESRI"
    return "skip"


# ---------------------------------------------------------------------
# Standardized run plan (ALL entries have the same keys)
# ---------------------------------------------------------------------
to_run = {
    "oroumieh": {
        "region_name": "oroumieh",
        "asset_path": "./ROI/wetlands/oroumieh.shp",
        "sensor_type": ["landsat"],
        "year": [2024],
        "tile_idx_resume": 390,
        "skip_download": False,
    },

    "anzali": {
        "region_name": "anzali",
        "asset_path": "./ROI/wetlands/anzali.shp",
        "sensor_type": ["sentinel"],
        "year": [2024],
        "tile_idx_resume": -1,
        "skip_download": True,
    },
    
    "karkheh": {
        "region_name": "karkheh",
        "asset_path": "./ROI/wetlands/karkheh.shp",
        "sensor_type": ["sentinel"],
        "year": [2024],
        "tile_idx_resume": -1,
        "skip_download": True,
    },

}


# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------
def main():
    for run_key, cfg in to_run.items():
        region_name = cfg["region_name"]
        asset_path = cfg["asset_path"]

        for year in cfg["year"]:
            landuse_method = pick_landuse(year)

            for sensor_type in cfg["sensor_type"]:
                out_dir = f"./data/{region_name}/{year}/{sensor_type}/"

                configs = {
                    "asset_path": asset_path,
                    "credentials_path": "./credentials/earthengine_credentials (1).json",
                    "service_account": "get-data@ardent-window-473507-c5.iam.gserviceaccount.com",
                    "year": year,
                    "sensor_type": sensor_type,
                    "sentinel_bands": ["red", "green", "blue", "nir"],
                    "out_dir": out_dir,
                    "tile_size": 2048,
                    "landuse_method": 'skip',
                    "device": "cuda",
                    "skip_download": cfg["skip_download"],
                    "tile_idx_resume": cfg["tile_idx_resume"],
                }

                print(
                    f"\n=== Running {region_name} | year={year} | sensor={sensor_type} "
                    f"| landuse={landuse_method} | skip_download={cfg['skip_download']} "
                    f"| resume={cfg['tile_idx_resume']} ==="
                )

                try:
                    run_pipeline(configs)

                except KeyboardInterrupt:
                    print("\n⚠️ Pipeline interrupted by user")
                    return

                except Exception as e:
                    print(
                        f"\n❌ Fatal error for {region_name} "
                        f"({year}, {sensor_type}): {e}"
                    )
                    remove_temp(out_dir)
                    continue

                remove_temp(out_dir)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
