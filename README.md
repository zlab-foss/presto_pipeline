# Presto Pipeline

Tile a shapefile of ROIs and download Earth observation data per tile from Google Earth Engine. Supports Sentinel-2, Sentinel-1, Landsat, ERA5, ESA WorldCover, Esri LULC, Alpha Earth embeddings, and SRTM.

## Setup

```bash
pip install -r requirements.txt
```

Place a GEE service-account key at `./ee-rsai-service-account.json`.

## Configuration

All settings live in `config.yaml` at the project root:

```yaml
shp: ./ROI/shahrestan/shahrestan.shp
year: 2022

# Sources to download (processed sequentially to avoid GEE quota exhaustion)
sources:
  - s2
  - s1

out: ./data/dataset
temp: ./tmp_tiles

credentials_path: ./ee-rsai-service-account.json
service_account: your-service-account@project.iam.gserviceaccount.com

max_pixels: 1024
s2_bands: all       # all | [red, green, blue, nir, ...]
s1_bands:           # subset of [VV, VH]
  - VV
  - VH
worldcover_scale: 10

# Tiles per source downloaded in parallel
workers: 4
```

## Usage

Run with everything from `config.yaml`:

```bash
python get_data.py
```

Override any setting on the CLI (takes precedence over config):

```bash
python get_data.py --source s2                       # single source only
python get_data.py --year 2024 --workers 8
python get_data.py --config ./other_config.yaml
python get_data.py --source s2 --limit 10            # process first 10 polygons
python get_data.py --start-after-poly-idx 42         # resume from polygon 43
```

### How it works

1. The shapefile is tiled into pixel-bounded sub-shapefiles (one per polygon tile).
2. All tiles for a source are downloaded in parallel (`workers` concurrent threads).
3. Sources listed in `sources` run sequentially to stay within GEE quota limits.
4. Already-downloaded valid `.tif` files are skipped automatically (safe to resume).

### All CLI flags

| Flag | Default | Description |
|---|---|---|
| `--config` | `./config.yaml` | Config file path |
| `--shp` | from config | Path to input shapefile (`.shp`) |
| `--year` | from config | Season year |
| `--source` | from config `sources` | Run a single source instead of all config sources |
| `--out` | `./data/dataset` | Output root directory |
| `--temp` | `./tmp_tiles` | Temp directory for tiled shapefiles |
| `--max_pixels` | `1024` | Max pixels per tile |
| `--workers` | `4` | Parallel download threads per source |
| `--s2-bands` | `all` | Sentinel-2 bands, e.g. `red green blue nir` |
| `--s1-bands` | `all` | Sentinel-1 bands: `VV`, `VH`, or both |
| `--worldcover-scale` | `10` | ESA WorldCover export resolution (m/pixel) |
| `--limit` | — | Max polygons to process |
| `--start-after-poly-idx` | `-1` | Resume: skip polygons up to and including this index |

Output files are named `{source}_y{year}_p{poly_idx}_{tile_tag}.tif`.

## Utilities

Helper scripts in [utils/](utils/):

- [pysentinel.py](utils/pysentinel.py) — Sentinel-1 / Sentinel-2 GEE downloaders
- [pylandsat.py](utils/pylandsat.py) — Landsat GEE downloader
- [pysatellite.py](utils/pysatellite.py) — ERA5, Esri LULC, ESA WorldCover, Alpha Earth embeddings, SRTM
- [shape_tiler.py](utils/shape_tiler.py) — splits an ROI shapefile into pixel-bounded tiles
- [shape_chunker.py](utils/shape_chunker.py) — splits a shapefile into N sub-shapefiles
- [convertTodB.py](utils/convertTodB.py) — convert Sentinel-1 linear output to dB
- [mergeRastersByYear.py](utils/mergeRastersByYear.py) — merge rasters sharing a common substring in their filename
- [wcDistribution.js](utils/wcDistribution.js) — GEE Code Editor script to compute WorldCover class percentages per tile
- [wcFiltering.py](utils/wcFiltering.py) — filter a GeoJSON grid by WorldCover class thresholds
- [groupRegion.py](utils/groupRegion.py), [move.py](utils/move.py) — file/region organization helpers
