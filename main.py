from pathlib import Path

from shap_tiler import ShapefileTiler
from data_sources.pysentinel import S1GEEDownloader, S2GEEDownloader
from preprocess import PrestoTensorBuilder

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
configs = {
    "asset_path" : './ROI/karkheh.shp',
    "credentials_path": "./credentials/earthengine_credentials.json",
    "service_account": "fanapanomaly@fanapanomaly.iam.gserviceaccount.com",
    "year": 2024,
}

OUT_ROOT = Path("./data/karkheh")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Downloaders
# ---------------------------------------------------------
dl2 = S2GEEDownloader(
    credentials_path=configs["credentials_path"],
    service_account=configs["service_account"],
    output_dir=OUT_ROOT / "s2",
    bands=['red', 'green', 'blue', 'nir'],
)

dl1 = S1GEEDownloader(
    credentials_path=configs["credentials_path"],
    service_account=configs["service_account"],
    output_dir=OUT_ROOT / "s1",
)

# ---------------------------------------------------------
# Shapefile tiler
# ---------------------------------------------------------
tiler = ShapefileTiler(
    shp_path=configs['asset_path'],
    max_pixels=1024,           # 2500 * 10 m = 25 km tile
    temp_dir="./TEMP/",
)

# ---------------------------------------------------------
# Presto builder
# ---------------------------------------------------------
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

# If you don't have these yet, either:
#  - comment them out and set ERA5/SRTM to False in group_flags, or
#  - point them to your real stacks.

# ---------------------------------------------------------
# Loop over tiles
# ---------------------------------------------------------
for item in tiler:
    poly_idx = item["poly_idx"]
    tile_idx = item["tile_idx"]  # None or (iy, ix)

    # nice tile tag for filenames
    if tile_idx is None:
        tile_tag = "full"
    else:
        ty, tx = tile_idx
        tile_tag = f"t{ty:03d}_{tx:03d}"

    # output filenames (relative to output_dir in downloader)
    s2_name = f"s2_{configs['year']}_p{poly_idx:03d}_{tile_tag}.tif"
    s1_name = f"s1_{configs['year']}_p{poly_idx:03d}_{tile_tag}.tif"

    # -----------------------------------------------------
    # 1) Download S2 + S1 for this tile
    # -----------------------------------------------------
    dl2.download_from_shapefile(
        shp_path=item["shp_path"],
        out_tif=s2_name,
        season_year=configs["year"],
    )

    dl1.download_from_shapefile(
        shp_path=item["shp_path"],
        out_tif=s1_name,
        season_year=configs["year"],
    )

    # Full paths to the just-downloaded TIFs
    s2_path = (OUT_ROOT / "s2" / s2_name).resolve()
    s1_path = (OUT_ROOT / "s1" / s1_name).resolve()

    # -----------------------------------------------------
    # 2) Build Presto tensor for this tile
    # -----------------------------------------------------
    paths_for_builder = [
        s1_path,   # S1 
        s2_path,   # S2 

    ]

    x, mask, dw = builder.build_from_tifs(
        paths=paths_for_builder,
        ref_index=0,   # use S2 as reference grid
    )

    print(f"=== Tile poly={poly_idx}, tile={tile_idx} ===")
    print("  builder.shape (H,W):", builder.shape)
    print("  x.shape   =", tuple(x.shape))      # (12,17,N)
    print("  mask.shape=", tuple(mask.shape))   # (12,17,N)
    print("  dw.shape  =", tuple(dw.shape))     # (12,N)
