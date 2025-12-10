# Presto-Based Irrigation Mapping Pipeline

This repository provides an **end-to-end pipeline** for generating **pixel-level irrigation maps** (irrigated vs. rainfed) and optional land-use classification from satellite imagery using a trained **Presto** model.

## Pipeline Overview
The pipeline:
- Tiles large Areas of Interest (AOIs) from a shapefile
- Downloads **Sentinel-2 + Sentinel-1** or **Landsat-8 + ERA5** data via Google Earth Engine
- Optionally applies the **ESRI 2020 Land Use/Land Cover (10m)** map as a cropland mask
- Builds **Presto-ready spatiotemporal tensors** (12-month sequences)
- Runs a pre-trained **Presto irrigation classifier**
- Exports predictions as **GeoTIFFs** perfectly aligned with input imagery

---

## ✨ Key Features

- **Multi-sensor support**
  - `sensor_type="sentinel"` → Sentinel-2 (optical) + Sentinel-1 (SAR)
  - `sensor_type="landsat"` → Landsat-8 (optical) + ERA5 (reanalysis climate data)

- **Smart tiling for large AOIs**
  - Automatically splits large or complex polygons into manageable tiles (metric CRS)
  - Each tile processed independently and saved with consistent naming

- **Efficient cropland masking with ESRI LULC**
  - Only runs inference on cropland pixels (class 4)
  - Skips model entirely for tiles with **zero cropland** → saves time and GPU

- **Presto-ready tensor builder**
  - Flexible band grouping via `group_flags`
  - Outputs monthly sequences (N pixels × 12 months × channels)

- **Seamless GeoTIFF output**
  - Predictions saved as `uint8` GeoTIFFs with `nodata=0`
  - Perfectly aligned to original optical imagery

---

## 🔧 Requirements

Tested with **Python 3.12.2** and **PyTorch 2.3.1 + CUDA 12.1**

```bash
# In a fresh environment
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Then install remaining dependencies
pip install -r requirements.txt
```

---

## 🔐 Google Earth Engine Authentication

The pipeline supports **service account authentication** (recommended for headless/cloud use):

```json
{
  "credentials_path": "./credentials/earthengine_credentials.json",
  "service_account": "your-service-account@project.iam.gserviceaccount.com"
}
```

Or manually initialize in code (comment out service account lines in data_source):

```python
ee.Initialize()  # Uses default credentials (e.g., gcloud auth)
```

> Make sure your service account has access to Earth Engine and the JSON key is readable.

---

## ⚙️ Configuration

Edit the `configs` dictionary in `main.py` or pass your own:

```python
configs = {
    "asset_path": "./ROI/karkheh.shp",                    # Input AOI shapefile
    "year": 2024,                                         # Target year
    "sensor_type": "sentinel",                            # "sentinel" or "landsat"
    "sentinel_bands": ["red", "green", "blue", "nir"],     # Only for Sentinel
    "out_dir": "./data/karkheh",                          # Output root
    "tile_size": 1024,            # Approx 1km × 1km at 10m
    "landuse_method": "ESRI",     # "ESRI", "presto", or "skip"
    "device": "cuda",             # "cuda" or "cpu"
    "credentials_path": "./credentials/earthengine_credentials.json",
    "service_account": "your-service-account@project.iam.gserviceaccount.com",
}
```

### Important Config Options

| Field              | Description |
|-------------------|-----------|
| `sensor_type`      | `"sentinel"` or `"landsat"` |
| `tile_size`        | Target width/height in pixels (e.g. 1024 ≈ 1km at 10m resolution) |
| `landuse_method`   | `"ESRI"` = use ESRI cropland mask<br>`"presto"` = future Presto LULC model<br>`"skip"` = no mask (run on all pixels) |

---

## 🌱 ESRI LULC Cropland Masking (Recommended)

When `landuse_method="ESRI"` and `2017 ≤ year ≤ 2024`:

1. Downloads ESRI 10m LULC for each tile
2. Aligns it to the optical raster (reprojects + resamples)
3. Creates binary mask: `1 = cropland (class 4)`, `0 = everything else`
4. Only cropland pixels are fed to the model
5. Tiles with **no cropland** → skip inference, write all-zero GeoTIFF

→ Huge speedup and avoids false irrigation signals in forests, water, urban areas.

---

## 🧠 Presto Model & Tensor Format

### Tensor Builder
Customizable via `group_flags` in `create_presto_builder()`:

```python
builder = PrestoTensorBuilder(group_flags={
    "S1": True,
    "S2_RGB": True,
    "S2_NIR_10m": True,
    "S2_SWIR": False,
    "ERA5": False,
    "NDVI": True,
    # ... etc
})
```

Returns:
- `x`: `(N, 12, C)` – input features
- `dw`: `(N, 12)` 
- `latlons`: `(N, 2)`
- `mask`: `(N, 12, C)`
- `labels`: `(N,)` 
- Plus spatial shape `(H, W)` to reshape predictions

### Pre-trained Models (included)
- `Presto_S2RGBNIR_S1.pth` → Sentinel-2 (RGB+NIR) + Sentinel-1
- `Presto_S2Full_S1.pth`     → Full Sentinel-2 bands + S1
- `Presto_L8_ERA5.pth`      → Landsat-8 + ERA5

### Prediction Post-processing
```python
pred = clf.predict(loader)           # → (N_valid,)
pred = pred + 1                      # Shift: {0,1} → {1,2}
# 0 remains 0 for nodata / non-cropland
```

Final raster values:
- `0` → nodata / non-cropland
- `1` → rainfed
- `2` → irrigated

---

## ▶️ Running the Pipeline

```bash
python main.py
```

The script will:
1. Tiles the input shapefile
2. Processes each tile in parallel (optional)
3. Downloads imagery + optional LULC
4. Builds tensors → runs inference → saves aligned GeoTIFF

---

## 📤 Outputs (per tile)

```
data/karkheh/
├── s2/                     # Sentinel-2 stacks (or landsat/)
├── s1/                     # Sentinel-1 stacks (or era5/)
├── esri_lulc/              # Optional aligned ESRI mask
└── predictions/
    └── 2024_pXXX_tYYY_ZZZ.tif   → Final irrigation map (uint8, nodata=0)
```

Use GDAL or rasterio to mosaic tiles afterward:



---

## 📌 TODO / Future Work

- [ ] Implement `"presto"` landuse method (Presto-based cropland classification)

---

## License

MIT License (or specify your own)



