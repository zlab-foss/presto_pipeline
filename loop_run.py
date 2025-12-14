from main import run_pipeline
from pathlib import Path
import shutil


def remove_temp(out_dir: str):
    """
    Remove TEMP directory inside output folder if it exists.
    """
    temp_dir = Path(out_dir) / "TEMP"

    if temp_dir.exists() and temp_dir.is_dir():
        try:
            shutil.rmtree(temp_dir)
            print(f"🧹 TEMP removed: {temp_dir}")
        except Exception as e:
            print(f"⚠️ Failed to remove TEMP: {e}")
    else:
        print("ℹ️ No TEMP directory found — skipping cleanup.")


def main():
    for year in [2024, 2018, 2004]:
        for region in ["oroumieh", "karkheh", "gavkhoni", "anzali"]:

            # Smart sensor + landuse selection
            if year <= 2017:
                sensor_type = "landsat"
                # This block executes if year <= 2017
                if year == 2017:
                    landuse_method = "ESRI"
                else: # This covers all years <= 2016
                    landuse_method = "skip"
            else: 
                sensor_type = "sentinel"
                
                if year >= 2025:
                    landuse_method = "skip"
                else: # This covers all years <= 2016
                    landuse_method = "ESRI"

            out_dir = f"./data/{region}_{year}"

            configs = {
                "asset_path": f"./ROI/wetlands/{region}.shp",
                "credentials_path": "./credentials/earthengine_credentials.json",
                "service_account": "fanapanomaly@fanapanomaly.iam.gserviceaccount.com",
                "year": year,
                "sensor_type": sensor_type,
                "sentinel_bands": ["red", "green", "blue", "nir"],
                "out_dir": out_dir,
                "tile_size": 1024,
                "landuse_method": landuse_method,
                "device": "cuda",
                "skip_download": False,
            }
            
            if year == 2024 and region=='oroumieh':
                configs['tile_idx_resume'] =182
            else:
                configs['tile_idx_resume'] =-1

            print(f"\n=== Running pipeline for {region} ({year}) "
                  f"[sensor={sensor_type}, landuse={landuse_method}] ===")

            try:
                run_pipeline(configs)
            except KeyboardInterrupt:
                print("\n\n⚠️ Pipeline interrupted by user")
                return
            except Exception as e:
                print(f"\n\n❌ Fatal error for {region} ({year}): {e}")
                # Even if pipeline fails, still try to clean TEMP
                remove_temp(out_dir)
                continue

            # After successful pipeline run → remove TEMP dir
            remove_temp(out_dir)


if __name__ == "__main__":
    main()
