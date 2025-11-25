# **Presto — Irrigation + LandCover Integration**

This repository contains **Presto**, a lightweight temporal deep learning model for pixel-wise classification of **Irrigated vs. Rainfed agriculture** using multi-temporal **Sentinel-1** and **Sentinel-2** satellite imagery.

Land-cover integration is currently under development; for now, the pipeline uses **ESRI LandCover** masks.

---

## 🌾 **Model Overview**

Presto operates on a **168-band GeoTIFF**, representing a full agricultural year:

* **Sentinel-2:** 12 spectral bands × 12 months
* **Sentinel-1:** VV/VH (or processed features) × 12 months
* **Temporal coverage:** **September → next-year September**

For each cropland pixel (identified via land-cover mask), the model predicts:

> **Irrigated** or **Rainfed**

---

## 🛠 **Installation**

### **Conda / Pip Setup**

```bash
# Install PyTorch (adjust CUDA version if needed)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu122

# Install remaining dependencies
pip install -r requirements.txt
```

⚠️ **Note:** A VPN connection is required for downloading Earth Engine (GEE) datasets directly to local storage.

---

## 🚀 **Inference Pipeline**

Below is a minimal example of the complete pipeline:

```python
from get_data import S2S1PrestoDownloader
from get_landcover import esri_landuse
from inference import run_inference
from utils import pair_by_idx

configs = {
    "asset_path": "./data/ROI/sample_wetlands/sample.shp",
    "year": 2017,
    "landuse_method": "ESRI",
    "ESRI_mask_path": "../../landcover/LULC/landcover-2017",
    "device": "cuda",
    "model_path": "./weights/tune_model.pth"
}

# --------------------------------------------------------
# 1) Download multi-temporal S2 + S1 inputs from GEE
# --------------------------------------------------------
downloader = S2S1PrestoDownloader(
    asset_path=configs["asset_path"],
    output_dir="./data/inputs",
    start_year=configs["year"],
)

downloader.run()

# --------------------------------------------------------
# 2) Extract ESRI LandCover masks for each polygon
# --------------------------------------------------------
esri_landuse(configs)

# --------------------------------------------------------
# 3) Match input TIFs & landcover masks by index
# --------------------------------------------------------
pairs = pair_by_idx("./data/inputs", f"./data/LULC/{configs['year']}")

# --------------------------------------------------------
# 4) Run inference tile-by-tile
# --------------------------------------------------------
for pair in pairs:
    idx = pair["idx"]

    configs["input_path"] = pair["input"]
    configs["mask_path"] = pair["mask"]
    configs["output_path"] = f"./data/outputs/irrigation_{configs['year']}_{idx}.tif"

    irrigation_map = run_inference(configs)
```

Finally, configure your parameters inside **main.py**, then simply run:

```bash
python main.py
```

---


