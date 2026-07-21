import os
import math
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import box
from tqdm import tqdm


# =========================
# Configuration
# =========================

GEOJSON_PATH = "annotation-sina.geojson"
OUTPUT_DIR = "raster_tiles"

LABEL_COLUMN = "label_value"

PIXEL_SIZE = 10          # Sentinel-2 resolution (meters)
TILE_SIZE = 256          # pixels

TILE_METERS = PIXEL_SIZE * TILE_SIZE


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =========================
# Load vector data
# =========================

gdf = gpd.read_file(GEOJSON_PATH)


if LABEL_COLUMN not in gdf.columns:
    raise ValueError(
        f"{LABEL_COLUMN} not found in GeoJSON"
    )


print(
    "CRS:",
    gdf.crs
)


# =========================
# Ensure projected CRS
# =========================

if gdf.crs.is_geographic:

    raise ValueError(
        """
GeoJSON is in latitude/longitude.
Convert it to a projected CRS (UTM)
before rasterization.
"""
    )


# =========================
# Compute bounds
# =========================

xmin, ymin, xmax, ymax = gdf.total_bounds


print(
    "Bounds:",
    xmin,
    ymin,
    xmax,
    ymax
)


# =========================
# Snap bounds to grid
# =========================

xmin = math.floor(xmin / TILE_METERS) * TILE_METERS
ymin = math.floor(ymin / TILE_METERS) * TILE_METERS

xmax = math.ceil(xmax / TILE_METERS) * TILE_METERS
ymax = math.ceil(ymax / TILE_METERS) * TILE_METERS



# Number of tiles

cols = int(
    (xmax - xmin) / TILE_METERS
)

rows = int(
    (ymax - ymin) / TILE_METERS
)


print(
    f"Grid: {cols} x {rows} tiles"
)



# =========================
# Rasterize tiles
# =========================

tile_id = 0


for row in tqdm(range(rows)):

    for col in range(cols):

        tile_xmin = xmin + col * TILE_METERS
        tile_ymax = ymax - row * TILE_METERS

        tile_xmax = tile_xmin + TILE_METERS
        tile_ymin = tile_ymax - TILE_METERS


        tile_geom = box(
            tile_xmin,
            tile_ymin,
            tile_xmax,
            tile_ymax
        )


        # Select polygons intersecting tile

        subset = gdf[
            gdf.intersects(tile_geom)
        ]


        if len(subset) == 0:
            continue



        shapes = [
            (
                geom,
                value
            )
            for geom, value in zip(
                subset.geometry,
                subset[LABEL_COLUMN]
            )
        ]


        transform = from_origin(
            tile_xmin,
            tile_ymax,
            PIXEL_SIZE,
            PIXEL_SIZE
        )


        raster = rasterize(
            shapes,
            out_shape=(
                TILE_SIZE,
                TILE_SIZE
            ),
            transform=transform,
            fill=0,
            dtype="uint16"
        )


        output_path = os.path.join(
            OUTPUT_DIR,
            f"tile_{row}_{col}.tif"
        )


        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=TILE_SIZE,
            width=TILE_SIZE,
            count=1,
            dtype=raster.dtype,
            crs=gdf.crs,
            transform=transform,
            compress="lzw"
        ) as dst:

            dst.write(
                raster,
                1
            )


        tile_id += 1



print(
    f"Finished. Generated {tile_id} raster tiles"
)