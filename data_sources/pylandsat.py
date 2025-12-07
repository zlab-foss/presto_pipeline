import os
from pathlib import Path
# ----------------------------------------------------------------------
# Fix PROJ / GDAL environment (important for geopandas / raster stuff)
# ----------------------------------------------------------------------
os.environ.pop("PROJ_LIB", None)
os.environ.pop("PROJ_DATA", None)

from pyproj import datadir
_pyproj_dir = datadir.get_data_dir()
os.environ["PROJ_DATA"] = _pyproj_dir
os.environ["PROJ_LIB"] = _pyproj_dir

if "CONDA_PREFIX" in os.environ:
    os.environ.setdefault("GDAL_DATA", f"{os.environ['CONDA_PREFIX']}/share/gdal")

import warnings
warnings.filterwarnings("ignore")

import ee
import geemap
import geedim
import geopandas as gpd




class LandsatGEEDownloader:
    """
    Download Landsat (8/9, 7, 5) monthly medians for a single ROI as a stacked GeoTIFF.

    - Uses Collection 2, Level-2 Surface Reflectance:
        * LANDSAT/LC08/C02/T1_L2 (Landsat 8)
        * LANDSAT/LC09/C02/T1_L2 (Landsat 9)
        * LANDSAT/LE07/C02/T1_L2 (Landsat 7)
        * LANDSAT/LT05/C02/T1_L2 (Landsat 5)
    - Chooses sensor based on season_year.
    - Outputs 12 × 6 bands: M01_blue, ..., M12_swir2
    - Bands (friendly names):
        ["blue", "green", "red", "nir", "swir1", "swir2"]

    Usage:
        ls_down = LandsatGEEDownloader(
            credentials_path="path/to/key.json",
            service_account="xxx@yyy.iam.gserviceaccount.com",
            output_dir="./data/gee_inputs_ls",
            export_scale=30,
            start_month=9,
            start_day=1,
        )

        ls_down.download_from_shapefile(
            shp_path="./data/ROI/benchmark.shp",
            out_tif="season2017_benchmark_landsat.tif",
            season_year=2017,
        )
    """

    # Fixed friendly band names
    LS_NAMES = ["blue", "green", "red", "nir", "swir1", "swir2"]

    def __init__(
        self,
        credentials_path: str,
        service_account: str,
        output_dir: str,
        export_scale: int = 30,
        start_month: int = 9,
        start_day: int = 1,
    ):
        # --------------------------------------------------------------
        # AUTH
        # --------------------------------------------------------------
        ee.Reset()
        credentials = ee.ServiceAccountCredentials(service_account, credentials_path)
        ee.Initialize(credentials)

        # --------------------------------------------------------------
        # CONFIG
        # --------------------------------------------------------------
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.export_scale = export_scale
        self.start_month = start_month
        self.start_day = start_day

        print("✅ Landsat downloader ready (bands: blue, green, red, nir, swir1, swir2)")

    # ==============================================================
    # GEOMETRY HELPERS (same pattern as S2/S1)
    # ==============================================================
    def _load_single_polygon(self, shp_path: str) -> ee.Feature:
        """
        Load a shapefile, ensure it has exactly 1 polygon, and convert to ee.Feature.
        """
        shp_path = Path(shp_path)
        if not shp_path.exists() or not shp_path.is_file():
            raise FileNotFoundError(f"Shapefile not found: {shp_path}")

        gdf = gpd.read_file(shp_path)
        if len(gdf) != 1:
            raise ValueError(
                f"Shapefile must contain exactly 1 feature (polygon). "
                f"Found: {len(gdf)}"
            )

        geom_type = gdf.geometry.iloc[0].geom_type
        if geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError(
                f"Geometry must be Polygon or MultiPolygon, found: {geom_type}"
            )

        print(f"📁 Loaded ROI from {shp_path} (single polygon, {geom_type}).")

        ee_fc = geemap.gdf_to_ee(gdf)
        return ee.Feature(ee_fc.first())

    def _rename_with_month(self, base_names, m):
        mtag = ee.Number(m).format("%02d")
        return ee.List(base_names).map(
            lambda b: ee.String("M").cat(mtag).cat("_").cat(ee.String(b))
        )

    def _season_month_to_calendar(self, season_year, m_idx):
        """
        Convert 1–12 season month index to (year, month) in calendar,
        given season start_month (e.g. 9 → Sep).
        """
        season_year = ee.Number(season_year)
        m_idx = ee.Number(m_idx)
        start_m = ee.Number(self.start_month)

        cal_month = start_m.subtract(1).add(m_idx.subtract(1)).mod(12).add(1)
        year_offset = ee.Number(ee.Algorithms.If(cal_month.gte(start_m), 0, 1))
        cal_year = season_year.add(year_offset)
        return ee.Dictionary({"year": cal_year, "month": cal_month})

    def _bin_start(self, season_year, m_idx):
        d = self._season_month_to_calendar(season_year, m_idx)
        return ee.Date.fromYMD(
            ee.Number(d.get("year")),
            ee.Number(d.get("month")),
            self.start_day,
        )

    def _bin_end(self, season_year, m_idx):
        return self._bin_start(season_year, m_idx).advance(1, "month")

    # ==============================================================
    # LANDSAT SENSOR / BAND SELECTION
    # ==============================================================
    def _get_landsat_collection_and_bands(self, season_year: int):
        """
        Pick Landsat collection & band mapping based on season_year.

        Returns:
            (ee.ImageCollection, band_codes_list)
        """
        # L8/L9 (2013+)
        if season_year >= 2013:
            # Merge L8 + L9 (same band scheme)
            col = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").merge(
                ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
            )
            band_codes = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]

        # L7 (1999–2012)
        elif season_year >= 1999:
            # ETM+ SR
            col = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
            band_codes = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]

        # Older → L5 SR (you can adjust if you want something else)
        else:
            col = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
            band_codes = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]

        return col, band_codes

    # ==============================================================
    # LANDSAT PROCESSING
    # ==============================================================
    def _mask_and_scale(self, img, band_codes):
        """
        - Use QA_PIXEL to mask clouds & cloud shadows.
        - Scale SR bands to surface reflectance (0–1-ish).
          Scale = 0.0000275, offset = -0.2 (Collection 2 L2 convention).
        """
        qa = img.select("QA_PIXEL")

        cloud_bit = 1 << 3      # Cloud
        cloud_shadow_bit = 1 << 4  # Cloud shadow

        mask = qa.bitwiseAnd(cloud_bit).eq(0).And(
            qa.bitwiseAnd(cloud_shadow_bit).eq(0)
        )

        img_masked = img.updateMask(mask)

        # Select SR bands, apply scale & offset
        sr = img_masked.select(band_codes).multiply(0.0000275).add(-0.2)

        # Rename to friendly names
        sr = sr.rename(self.LS_NAMES)

        return sr

    def _ls_month(self, roi, season_year, m_idx) -> ee.Image:
        """
        Build monthly composite for Landsat (L8/9, 7, or 5) in surface reflectance.
        """
        start = self._bin_start(season_year, m_idx)
        end = self._bin_end(season_year, m_idx)

        base_col, band_codes = self._get_landsat_collection_and_bands(season_year)

        def _prep(img):
            return self._mask_and_scale(img, band_codes)

        col = (
            base_col
            .filterBounds(roi)
            .filterDate(start, end)
            .map(_prep)
        )

        # Fallback if nothing available
        fallback = (
            ee.Image.constant([0] * len(self.LS_NAMES))
            .rename(self.LS_NAMES)
            .selfMask()          # fully masked instead of zeros
            .toFloat()
            .clip(roi)
        )

        return ee.Image(
            ee.Algorithms.If(
                col.size().gt(0),
                col.median().toFloat().clip(roi),
                fallback,
            )
        )

    # ==============================================================
    # BUILD SEASON STACK (12 × 6 bands)
    # ==============================================================
    def create_stack(self, roi, season_year: int) -> ee.Image:
        """
        Create 12-month Landsat median stack (surface reflectance)
        for given ROI and season year.
        """

        def per_month(m):
            ls = self._ls_month(roi, season_year, m)
            return ls.rename(self._rename_with_month(self.LS_NAMES, m))

        images = ee.List.sequence(1, 12).map(per_month)
        stack = ee.ImageCollection.fromImages(images).toBands()

        # Clean band names (same trick as S2/S1)
        old = stack.bandNames()
        new = old.map(lambda n: ee.String(n).split("_").slice(0).join("_"))
        return stack.rename(new).toFloat()

    # ==============================================================
    # PUBLIC: DOWNLOAD FROM SHAPEFILE
    # ==============================================================
    def download_from_shapefile(
        self,
        shp_path: str,
        out_tif: str,
        season_year: int,
    ):
        """
        Load a single-polygon shapefile and download the seasonal Landsat stack.

        shp_path    : path to a shapefile with exactly 1 polygon.
        out_tif     : output filename (full path or relative to output_dir).
        season_year : integer season year (e.g. 2017).
        """
        feature = self._load_single_polygon(shp_path)
        roi = feature.geometry()

        out_tif = Path(out_tif)
        if not out_tif.is_absolute():
            out_tif = self.output_dir / out_tif

        print("============================================================")
        print(f"[Landsat] Processing growing season {season_year}")
        print(f"ROI shapefile: {shp_path}")
        print(f"Output: {out_tif}")
        print("============================================================")

        img = self.create_stack(roi, season_year)
        gd_img = geedim.MaskedImage(img.clip(roi))

        try:
            gd_img.download(
                str(out_tif),
                region=roi,
                scale=self.export_scale,
                crs="EPSG:4326",
                dtype="float32",
                overwrite=True,
                max_tile_size=32,
            )
            print(f"✅ [Landsat] Download complete → {out_tif}")
        except Exception as e:
            print(f"❌ [Landsat] Error during download: {e}")
            raise
