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
| `--year` | Season/target year (e.g. `2024`) |
| `--source` | One of: `s2`, `s1`, `landsat`, `era5`, `esri_lulc`, `worldcover`, `embedding`, `srtm` |

**Optional arguments:**

| Argument | Default | Description |
|---|---|---|
| `--out` | `./data/dataset` | Output root directory |
| `--temp` | `./tmp_tiles` | Directory for temporary per-tile shapefiles |
| `--max_pixels` | `1024` | Max tile size in pixels (used to split large ROIs) |
| `--s2-bands` | `all` | Sentinel-2 bands to download, e.g. `red green blue nir` |
| `--limit` | _(all)_ | Max number of polygons to download (useful for testing) |

### Examples

```bash
# Sentinel-2 all bands, 2024
python get_data.py --shp ./ROI/tehran.shp --year 2024 --source s2

# Sentinel-2 RGB + NIR only
python get_data.py --shp ./ROI/tehran.shp --year 2024 --source s2 --s2-bands red green blue nir

# Landsat, custom output directory
python get_data.py --shp ./ROI/urmia.shp --year 2016 --source landsat --out ./data/urmia

# ERA5 climate data
python get_data.py --shp ./ROI/jask.shp --year 2023 --source era5

# SRTM elevation (year is required by CLI but ignored for SRTM)
python get_data.py --shp ./ROI/tehran.shp --year 2024 --source srtm

# Download only the first 3 polygons (useful for quick testing)
python get_data.py --shp ./ROI/tehran.shp --year 2024 --source s2 --limit 3
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
get_data.py           # CLI entry point
utils/
  pysentinel.py       # S2GEEDownloader, S1GEEDownloader
  pylandsat.py        # LandsatGEEDownloader
  pysatellite.py      # ERA5, ESRI LULC, WorldCover, Embedding, SRTM downloaders
  shape_tiler.py      # ShapefileTiler — splits large ROIs into tiles
ROI/                  # Sample region shapefiles (Tehran, Urmia, Jask, Tonekabon)
```

## Notes

- The tiler splits large polygons into rectangular tiles in EPSG:3857, intersects them with the original geometry, and writes one temporary shapefile per tile. Temporary files remain in `--temp` after the run.
- Complex polygon geometries are progressively simplified (up to 20 km tolerance) before being sent to GEE. If simplification fails, the bounding box is used as a fallback.
- WorldCover is only available for years `2020` and `2021`.
- Alpha Embedding is available for years `2017`–`2024`.
