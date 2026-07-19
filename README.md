# Presto Pipeline — GEE Data Downloader

A CLI tool for downloading multi-source satellite and geospatial data from Google Earth Engine (GEE) over arbitrary ROIs defined by shapefiles. Large ROIs are automatically tiled so downloads stay within GEE limits.

## Data sources

| `--source` | Dataset | Resolution | Output bands |
|---|---|---|---|
| `s2` | Sentinel-2 SR monthly medians | 10 m | 12 × N bands (12 months × selected bands) |
| `s1` | Sentinel-1 SAR monthly medians | 10 m | 12 × 2 bands (VV, VH) |
| `landsat` | Landsat 5/7/8/9 C2 monthly medians | 30 m | 12 × 6 bands (blue→swir2) |
| `era5` | ERA5-Land hourly → monthly aggregates | 30 m | 12 × 2 bands (t2m °C, tp m) |
| `esri_lulc` | ESRI Global LULC 10m annual | 10 m | 1 band (`lulc`, uint8) |
| `worldcover` | ESA WorldCover (2020 or 2021 only) | 10 m | 1 band (`worldcover`, uint8) |
| `embedding` | Google Satellite Embedding V1 Annual (2017–2024) | 10 m | multi-band float32 |
| `srtm` | USGS SRTM DEM | 30 m | 1 band (`elevation`, float32) |

## Setup

```bash
conda create -n presto_pipeline python=3.11
conda activate presto_pipeline
pip install -r requirements.txt
```

You also need a GEE service account key. Place it at `./ee-rsai-service-account.json` (or update `DEFAULT_CONFIGS` in `get_data.py`).

## Usage

```bash
python get_data.py --shp <shapefile> --year <year> --source <source> [options]
```

**Required arguments:**

| Argument | Description |
|---|---|
| `--shp` | Path to input shapefile (`.shp`) — can contain multiple polygons |
| `--source` | One of: `s2`, `s1`, `landsat`, `era5`, `esri_lulc`, `worldcover`, `embedding`, `srtm` |

**Optional arguments:**

| Argument | Default | Description |
|---|---|---|
| `--year` | _(none)_ | Required for `esri_lulc`, `worldcover`, `embedding`. Not used for `s2`/`s1`/`landsat`/`era5` — those read their download window from each polygon's own `plant`/`harvest` shapefile columns (see [Crop calendar](#crop-calendar) below) |
| `--out` | `./data/dataset` | Output root directory |
| `--temp` | `./tmp_tiles` | Directory for temporary per-tile shapefiles |
| `--max_pixels` | `1024` | Max tile size in pixels (used to split large ROIs) |
| `--s2-bands` | `all` | Sentinel-2 bands to download, e.g. `red green blue nir` |
| `--limit` | _(all)_ | Max number of polygons to download (useful for testing) |
| `--start-after-poly-idx` | `-1` | Resume: only process polygons with `poly_idx` greater than this |

### Examples

```bash
# Sentinel-2, crop-calendar polygons (plant/harvest columns drive the window)
python get_data.py --shp ./ROI/wheat/shapefile-province-cc-refined.shp --source s2

# Sentinel-2 RGB + NIR only
python get_data.py --shp ./ROI/wheat/shapefile-province-cc-refined.shp --source s2 --s2-bands red green blue nir

# Landsat, custom output directory
python get_data.py --shp ./ROI/urmia.shp --source landsat --out ./data/urmia

# ERA5 climate data
python get_data.py --shp ./ROI/jask.shp --source era5

# SRTM elevation
python get_data.py --shp ./ROI/tehran.shp --source srtm

# WorldCover / LULC / Embedding require --year
python get_data.py --shp ./ROI/tehran.shp --year 2021 --source worldcover

# Download only the first 3 polygons (useful for quick testing)
python get_data.py --shp ./ROI/wheat/shapefile-province-cc-refined.shp --source s2 --limit 3
```

## Crop calendar

For `s2`/`s1`/`landsat`/`era5`, each polygon supplies its own `plant`/`harvest` dates (`YYYY-MM-DD`) as shapefile columns instead of a global `--year`; `utils/crop_calendar.py` reads them and bins the `[plant, harvest]` range into 12 windows for monthly compositing.

`ROI/<crop>/shapefile-province-cc-refined.shp` (barley, corn, kolza, rice, wheat) are produced from raw per-parcel crop-calendar shapefiles by two one-off scripts:

- `scripts/add_crop_calendar_to_province.py` — aggregates parcel-level dates to a per-province mode and writes them onto `ROI/province/province-crop-calendar.shp`.
- `scripts/refine_crop_shapefiles.py` — reduces each crop's parcel shapefile down to just `plant`/`harvest` columns, using that same per-province mode.

## Running multiple crops/sources in parallel

`run_parallel.py` wraps `get_data.py`, splitting one shapefile's polygons across `--workers` parallel subprocesses (each downloading a distinct polygon range). It also supports a `--config crops.yaml` mode that fans out over several crops and sources at once — one worker group per (crop, source) pair, all running concurrently. See `crops.yaml` for the config format; each crop entry needs `shp`/`out`/`temp`, and `workers`/`sources` can be set globally or overridden per crop.

```bash
# Single shapefile, 4 parallel workers
python run_parallel.py --workers 4 --shp ./ROI/wheat/shapefile-province-cc-refined.shp \
    --source s2 --out ./data/wheat --temp ./tmp_tiles_wheat

# All crops/sources defined in crops.yaml
python run_parallel.py --config crops.yaml
```

## Output structure

Downloads are saved as GeoTIFFs under `<out>/<source>/`:

```
data/dataset/
  s2/
    s2_y2024_p0000_full.tif        # small polygon, no tiling needed
    s2_y2024_p0001_t000_000.tif    # tiled polygon, row 0 col 0
    s2_y2024_p0001_t000_001.tif    # tiled polygon, row 0 col 1
  landsat/
    landsat_y2016_p0000_full.tif
```

## Project structure

```
get_data.py            # CLI entry point
run_parallel.py        # Multi-worker / multi-crop wrapper around get_data.py
crops.yaml              # Multi-crop config for run_parallel.py --config
utils/
  pysentinel.py       # S2GEEDownloader, S1GEEDownloader
  pylandsat.py        # LandsatGEEDownloader
  pysatellite.py      # ERA5, ESRI LULC, WorldCover, Embedding, SRTM downloaders
  shape_tiler.py      # ShapefileTiler — splits large ROIs into tiles
  crop_calendar.py    # Reads per-polygon plant/harvest dates, builds monthly bin windows
scripts/
  add_crop_calendar_to_province.py  # Aggregates parcel crop-calendar dates onto province shapefile
  refine_crop_shapefiles.py         # Reduces parcel shapefiles to plant/harvest columns
ROI/                  # Region shapefiles (sample ROIs + per-crop crop-calendar shapefiles)
```

## Notes

- The tiler splits large polygons into rectangular tiles in EPSG:3857, intersects them with the original geometry, and writes one temporary shapefile per tile. Temporary files remain in `--temp` after the run.
- Complex polygon geometries are progressively simplified (up to 20 km tolerance) before being sent to GEE. If simplification fails, the bounding box is used as a fallback.
- WorldCover is only available for years `2020` and `2021`.
- Alpha Embedding is available for years `2017`–`2024`.
