# Presto Pipeline

Tile a shapefile of ROIs and download Earth observation data per tile from Google Earth Engine. Supports Sentinel-2, Sentinel-1, Landsat, ERA5, ESA WorldCover, Esri LULC, Alpha Earth embeddings, and SRTM.

## Setup

```bash
pip install -r requirements.txt
```

Place a GEE service-account key at `./ee-rsai-service-account.json` (or update `DEFAULT_CONFIGS` in [get_data.py](get_data.py)).

## Usage

For simple ROIs (does not work on complex polygons — use the `amir-get-data-crop` branch for those):

```bash
python get_data.py --shp ./ROI/test-presto-crop.shp --year 2019 --source s1 --out ./output_path
python get_data.py --shp ./ROI/test-presto-crop.shp --year 2019 --source s2
python get_data.py --shp ./ROI/test-lulc.shp        --year 2023 --source s2 --s2-bands red green blue nir
python get_data.py --shp ./ROI/test-presto-crop.shp --year 2019 --source landsat
python get_data.py --shp ./ROI/test-presto-crop.shp --year 2021 --source worldcover --worldcover-scale 10
```

### Arguments

| Flag | Required | Default | Description |
|---|---|---|---|
| `--shp` | yes | — | Path to input shapefile (`.shp`) |
| `--year` | yes | — | Season year (or year for masks/embeddings) |
| `--source` | yes | — | One of: `s2`, `s1`, `landsat`, `era5`, `esri_lulc`, `worldcover`, `embedding`, `srtm` |
| `--out` | no | `./data/dataset` | Output root directory (a per-source subfolder is created) |
| `--temp` | no | `./tmp_tiles` | Temp directory for tiled shapefiles |
| `--max_pixels` | no | `1024` | Max pixels per tile for the tiler |
| `--s2-bands` | no | `all` | Sentinel-2 bands, e.g. `red green blue nir`, or `all` |
| `--worldcover-scale` | no | `10` | ESA WorldCover export resolution (m/pixel) |

Output files are named `{source}_y{year}_p{poly_idx}_{tile_tag}.tif`.

## Utilities

Helper scripts in [utils/](utils/):

- [pysentinel.py](utils/pysentinel.py) — Sentinel-1 / Sentinel-2 GEE downloaders
- [pylandsat.py](utils/pylandsat.py) — Landsat GEE downloader
- [pysatellite.py](utils/pysatellite.py) — ERA5, Esri LULC, ESA WorldCover, Alpha Earth embeddings, SRTM
- [shape_tiler.py](utils/shape_tiler.py) — splits an ROI shapefile into pixel-bounded tiles
- [convertTodB.py](utils/convertTodB.py) — convert Sentinel-1 linear output to dB
- [mergeRastersByYear.py](utils/mergeRastersByYear.py) — merge rasters sharing a common substring (e.g. a year) in their filename
- [wcDistribution.js](utils/wcDistribution.js) — GEE Code Editor script to compute WorldCover class percentages per tile (upload a grid shapefile and run)
- [wcFiltering.py](utils/wcFiltering.py) — filter a GeoJSON grid (with WorldCover percentages) by class thresholds
- [groupRegion.py](utils/groupRegion.py), [move.py](utils/move.py) — file/region organization helpers
