import time
from pathlib import Path

import sys
import os

# 1. Get the path of the parent directory (one level up)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 2. Add the parent directory to the system path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


from data_sources.pysentinel import S2GEEDownloader, S1GEEDownloader
from data_sources.pylandsat import LandsatGEEDownloader
from data_sources.pysatellite import (
    ERA5GEEDownloader,
    EsriLULCMaskDownloader,
    ESAWorldCoverMaskDownloader,
    AlphaEmbeddingDownloader,
    SRTMDownloader
)

# =========================
# GLOBAL CONFIG
# =========================
configs = {
    'credentials_path': "..//credentials/earthengine_credentials.json",
    'service_account': 'fanapanomaly@fanapanomaly.iam.gserviceaccount.com',

}

ROI_SHP = "../ROI/sample.shp"
year = 2024

OUT_ROOT = Path("../data/test_outputs")
OUT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# SMALL WRAPPER to time execution
# ============================================================
def run_test(name: str, func):
    print(f"\n============== {name} ==============")
    start = time.time()
    try:
        func()
        end = time.time()
        print(f"✅ SUCCESS — completed in {end - start:.2f} sec")
    except Exception as e:
        print(f"❌ FAILED — {e}\n")


# ============================================================
# TEST FUNCTIONS FOR EACH DOWNLOADER
# ============================================================

def test_s2():
    dl = S2GEEDownloader(
        credentials_path=configs['credentials_path'],
        service_account=configs['service_account'],
        output_dir=OUT_ROOT / "s2",
        bands='all',
    )
    dl.download_from_shapefile(
        shp_path=ROI_SHP,
        out_tif=f"test_{year}.tif",
        season_year=year,
    )


def test_s1():
    dl = S1GEEDownloader(
        credentials_path=configs['credentials_path'],
        service_account=configs['service_account'],
        output_dir=OUT_ROOT / "s1",
    )
    dl.download_from_shapefile(
        shp_path=ROI_SHP,
        out_tif=f"test_{year}.tif",
        season_year=year,
    )


def test_landsat():
    dl = LandsatGEEDownloader(
        credentials_path=configs['credentials_path'],
        service_account=configs['service_account'],
        output_dir=OUT_ROOT / "landsat",
    )
    dl.download_from_shapefile(
        shp_path=ROI_SHP,
        out_tif=f"test_{year}.tif",
        season_year=year,
    )


def test_era5():
    dl = ERA5GEEDownloader(
        credentials_path=configs['credentials_path'],
        service_account=configs['service_account'],
        output_dir=OUT_ROOT / "era5",
    )
    dl.download_from_shapefile(
        shp_path=ROI_SHP,
        out_tif=f"test_{year}.tif",
        season_year=year,
    )


def test_esri():
    dl = EsriLULCMaskDownloader(
        credentials_path=configs['credentials_path'],
        service_account=configs['service_account'],
        output_dir=OUT_ROOT / "esri_lulc",
    )
    dl.download_from_shapefile(
        shp_path=ROI_SHP,
        out_tif=f"test_{year}.tif",
        year=year,
    )


def test_worldcover():
    dl = ESAWorldCoverMaskDownloader(
        credentials_path=configs['credentials_path'],
        service_account=configs['service_account'],
        output_dir=OUT_ROOT / "esa_lulc",
    )
    dl.download_from_shapefile(
        shp_path=ROI_SHP,
        out_tif="test_2021.tif",
        year=2021,
    )


def test_embedding():
    dl = AlphaEmbeddingDownloader(
        credentials_path=configs['credentials_path'],
        service_account=configs['service_account'],
        output_dir=OUT_ROOT / "embedding",
    )
    dl.download_from_shapefile(
        shp_path=ROI_SHP,
        out_tif=f"test_{year}.tif",
        year=year,
    )


def test_srtm():
    dl = SRTMDownloader(
        credentials_path=configs['credentials_path'],
        service_account=configs['service_account'],
        output_dir=OUT_ROOT / "srtm",
    )
    dl.download_from_shapefile(
        shp_path=ROI_SHP,
        out_tif=f"test_{year}.tif",
    )


# ============================================================
# RUN ALL TESTS IN ORDER
# ============================================================
if __name__ == "__main__":
    print("\n###########################################################")
    print("### Running full test suite for all GEE data downloaders ###")
    print("###########################################################\n")

    run_test("Sentinel-2", test_s2)
    run_test("Sentinel-1", test_s1)
    run_test("Landsat", test_landsat)
    run_test("ERA5", test_era5)
    run_test("ESRI LULC", test_esri)
    run_test("ESA WorldCover", test_worldcover)
    run_test("SRTM", test_srtm)
    run_test("Alpha Earth Embedding", test_embedding)

    print("\n########### ALL TESTS COMPLETED ###########\n")
