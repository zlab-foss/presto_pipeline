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


## 1️⃣ ERA5 monthly (t2m, tp)


class ERA5GEEDownloader:
    """
    ERA5-Land HOURLY -> monthly aggregates (t2m & total precip)

    Dataset: ECMWF/ERA5_LAND/HOURLY
      - temperature_2m [K]              -> t2m [°C]  (monthly mean)
      - total_precipitation_hourly [m]  -> tp  [m]   (monthly sum)

    Output bands (12 × 2):
      M01_t2m, M01_tp, ..., M12_t2m, M12_tp
    """

    ERA5_COLLECTION = "ECMWF/ERA5_LAND/HOURLY"
    ERA5_NAMES = ["t2m", "tp"]

    def __init__(
        self,
        credentials_path: str,
        service_account: str,
        output_dir: str,
        export_scale: float = 30.0,   # use same as Landsat/S2 if you want same grid
        start_month: int = 9,
        start_day: int = 1,
    ):
        # AUTH (same pattern as your other downloaders)
        ee.Reset()
        credentials = ee.ServiceAccountCredentials(service_account, credentials_path)
        ee.Initialize(credentials)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.export_scale = export_scale
        self.start_month = start_month
        self.start_day = start_day

        print("✅ ERA5-Land HOURLY downloader ready (t2m °C, tp m)")

    # ------------------------------------------------------------------
    #  GEOMETRY: single-polygon shapefile → ee.Feature
    # ------------------------------------------------------------------
    def _load_single_polygon(self, shp_path: str) -> ee.Feature:
        shp_path = Path(shp_path)
        if not shp_path.exists() or not shp_path.is_file():
            raise FileNotFoundError(f"Shapefile not found: {shp_path}")

        gdf = gpd.read_file(shp_path)
        if len(gdf) != 1:
            raise ValueError(
                f"Shapefile must contain exactly 1 feature (polygon). Found: {len(gdf)}"
            )

        geom_type = gdf.geometry.iloc[0].geom_type
        if geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError(
                f"Geometry must be Polygon or MultiPolygon, found: {geom_type}"
            )

        print(f"📁 Loaded ROI from {shp_path} (single polygon, {geom_type}).")
        ee_fc = geemap.gdf_to_ee(gdf)
        return ee.Feature(ee_fc.first())

    # ------------------------------------------------------------------
    #  TIME HELPERS (same as in your L8S1PrestoDownloader)
    # ------------------------------------------------------------------
    def _rename_with_month(self, base_names, m):
        mtag = ee.Number(m).format("%02d")
        return ee.List(base_names).map(
            lambda b: ee.String("M").cat(mtag).cat("_").cat(ee.String(b))
        )

    def _season_month_to_calendar(self, season_year, m_idx):
        """
        season_month 1..12 → calendar (year, month),
        using season start_month (e.g. 9 for Sep).
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

    # ------------------------------------------------------------------
    #  ERA5-Land monthly aggregates for one season-month
    # ------------------------------------------------------------------
    def _era5_month(self, roi, season_year, m_idx) -> ee.Image:
        """
        For a given season_month index m_idx:
          - t2m = mean(temperature_2m) - 273.15  [°C]
          - tp  = sum(total_precipitation_hourly) [m]
        """
        start = self._bin_start(season_year, m_idx)
        end = self._bin_end(season_year, m_idx)

        col = (
            ee.ImageCollection(self.ERA5_COLLECTION)
            .filterDate(start, end)
            .filterBounds(roi)
        )

        # Fallback → zeros if no data
        fallback = (
            ee.Image.constant([0, 0])
            .rename(self.ERA5_NAMES)
            .toFloat()
            .clip(roi)
        )

        def _build(imgcol):
            imgcol = ee.ImageCollection(imgcol)

            # monthly mean temperature (K → °C)
            t2mK = imgcol.select("temperature_2m").mean()
            t2mC = t2mK.subtract(273.15)

            # monthly sum of hourly total precip
            tp_month = imgcol.select("total_precipitation_hourly").sum()

            era = t2mC.addBands(tp_month)
            return era.rename(self.ERA5_NAMES).toFloat().clip(roi)

        return ee.Image(
            ee.Algorithms.If(
                col.size().gt(0),
                _build(col),
                fallback,
            )
        )

    # ------------------------------------------------------------------
    #  BUILD 12-MONTH ERA5 STACK (same pattern as your L8 stack)
    # ------------------------------------------------------------------
    def create_stack(self, roi, season_year: int) -> ee.Image:
        def per_month(m):
            era = self._era5_month(roi, season_year, m)
            return era.rename(self._rename_with_month(self.ERA5_NAMES, m))

        images = ee.List.sequence(1, 12).map(per_month)
        stack = ee.ImageCollection.fromImages(images).toBands()
        return stack.toFloat()

    # ------------------------------------------------------------------
    #  PUBLIC: DOWNLOAD FROM SHAPEFILE
    # ------------------------------------------------------------------
    def download_from_shapefile(
        self,
        shp_path: str,
        out_tif: str,
        season_year: int,
    ):
        feature = self._load_single_polygon(shp_path)
        roi = feature.geometry()

        out_tif = Path(out_tif)
        if not out_tif.is_absolute():
            out_tif = self.output_dir / out_tif
        out_tif.parent.mkdir(parents=True, exist_ok=True)

        print("============================================================")
        print(f"[ERA5-Land HOURLY] Processing season {season_year}")
        print(f"ROI shapefile: {shp_path}")
        print(f"Output: {out_tif}")
        print("============================================================")

        img = self.create_stack(roi, season_year).clip(roi)

        gd_img = geedim.MaskedImage(img)

        try:
            gd_img.download(
                str(out_tif),
                region=roi,                  # same as L8S1PrestoDownloader
                scale=self.export_scale,     # use same value as S2/L8 if you want same grid
                crs="EPSG:4326",
                dtype="float32",
                overwrite=True,
                max_tile_size=32,
            )
            print(f"✅ [ERA5-Land HOURLY] Download complete → {out_tif}")
        except Exception as e:
            print(f"❌ [ERA5-Land HOURLY] Error during download: {e}")
            raise

## 2️⃣ ESRI Global LULC mask 
class EsriLULCMaskDownloader:
    """
    Download ESRI Global LULC 10m annual mask for a single ROI and year.

    - Dataset: projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS
    - Remaps original ESRI classes (REMAP_FROM → REMAP_TO) into a single band 'lulc'.
    - Output: 1-band uint8 GeoTIFF: 'lulc'.
    """

    COLLECTION = "projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS"
    REMAP_FROM = [1, 2, 4, 5, 7, 8, 9, 10, 11]
    REMAP_TO = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def __init__(
        self,
        credentials_path: str,
        service_account: str,
        output_dir: str,
        export_scale: int = 10,
    ):
        ee.Reset()
        credentials = ee.ServiceAccountCredentials(service_account, credentials_path)
        ee.Initialize(credentials)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.export_scale = export_scale

        print("✅ ESRI LULC downloader ready")

    # ---------- helpers ----------
    def _load_single_polygon(self, shp_path: str) -> ee.Feature:
        shp_path = Path(shp_path)
        if not shp_path.exists() or not shp_path.is_file():
            raise FileNotFoundError(f"Shapefile not found: {shp_path}")

        gdf = gpd.read_file(shp_path)
        if len(gdf) != 1:
            raise ValueError(
                f"Shapefile must contain exactly 1 feature (polygon). Found: {len(gdf)}"
            )

        geom_type = gdf.geometry.iloc[0].geom_type
        if geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError(
                f"Geometry must be Polygon or MultiPolygon, found: {geom_type}"
            )

        print(f"📁 Loaded ROI from {shp_path} (single polygon, {geom_type}).")
        ee_fc = geemap.gdf_to_ee(gdf)
        return ee.Feature(ee_fc.first())

    def _remap_landcover(self, image: ee.Image) -> ee.Image:
        image = ee.Image(image)
        first_band = ee.String(image.bandNames().get(0))
        single = image.select([first_band]).rename(["remapped"])
        remapped = single.remap(self.REMAP_FROM, self.REMAP_TO)
        return ee.Image(remapped).rename("lulc")

    def _annual_esri(self, year: int, roi) -> ee.Image:
        start = ee.Date.fromYMD(year, 1, 1)
        end = ee.Date.fromYMD(year, 12, 31)
        col = (
            ee.ImageCollection(self.COLLECTION)
            .filterDate(start, end)
            .filterBounds(roi)
        )

        mosaic = ee.Image(col.mosaic())
        lulc = self._remap_landcover(mosaic).clip(roi)
        # make sure it's byte and no nodata → 0
        lulc = lulc.unmask(0).toByte()
        return lulc

    # ---------- public ----------
    def download_from_shapefile(
        self,
        shp_path: str,
        out_tif: str,
        year: int,
    ):
        feature = self._load_single_polygon(shp_path)
        roi = feature.geometry()

        out_tif = Path(out_tif)
        if not out_tif.is_absolute():
            out_tif = self.output_dir / out_tif

        print("============================================================")
        print(f"[ESRI LULC] Processing year {year}")
        print(f"ROI shapefile: {shp_path}")
        print(f"Output: {out_tif}")
        print("============================================================")

        img = self._annual_esri(year, roi)
        gd_img = geedim.MaskedImage(img.clip(roi))

        try:
            gd_img.download(
                str(out_tif),
                region=roi,
                scale=self.export_scale,
                crs="EPSG:4326",
                dtype="uint8",
                overwrite=True,
                max_tile_size=32,
            )
            print(f"✅ [ESRI LULC] Download complete → {out_tif}")
        except Exception as e:
            print(f"❌ [ESRI LULC] Error during download: {e}")
            raise

## 3️⃣ ESA WorldCover mask

class ESAWorldCoverMaskDownloader:
    """
    Download ESA WorldCover mask for a single ROI and year.

    - Year 2020 → ESA/WorldCover/v100
    - Year 2021 → ESA/WorldCover/v200
    - Band: 'Map' (kept as 'worldcover')
    """

    def __init__(
        self,
        credentials_path: str,
        service_account: str,
        output_dir: str,
        export_scale: int = 10,
    ):
        ee.Reset()
        credentials = ee.ServiceAccountCredentials(service_account, credentials_path)
        ee.Initialize(credentials)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.export_scale = export_scale

        print("✅ ESA WorldCover downloader ready")

    def _load_single_polygon(self, shp_path: str) -> ee.Feature:
        shp_path = Path(shp_path)
        if not shp_path.exists() or not shp_path.is_file():
            raise FileNotFoundError(f"Shapefile not found: {shp_path}")

        gdf = gpd.read_file(shp_path)
        if len(gdf) != 1:
            raise ValueError(
                f"Shapefile must contain exactly 1 feature (polygon). Found: {len(gdf)}"
            )

        geom_type = gdf.geometry.iloc[0].geom_type
        if geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError(
                f"Geometry must be Polygon or MultiPolygon, found: {geom_type}"
            )

        print(f"📁 Loaded ROI from {shp_path} (single polygon, {geom_type}).")
        ee_fc = geemap.gdf_to_ee(gdf)
        return ee.Feature(ee_fc.first())

    def _worldcover_image(self, year: int, roi) -> ee.Image:
        if year == 2020:
            col_id = "ESA/WorldCover/v100"
        elif year == 2021:
            col_id = "ESA/WorldCover/v200"
        else:
            raise ValueError(
                f"WorldCover only available for YEAR 2020 (v100) or 2021 (v200). Got {year}"
            )

        img = ee.ImageCollection(col_id).first().select("Map").rename("worldcover")
        return img.clip(roi).toByte()

    def download_from_shapefile(
        self,
        shp_path: str,
        out_tif: str,
        year: int,
    ):
        feature = self._load_single_polygon(shp_path)
        roi = feature.geometry()

        out_tif = Path(out_tif)
        if not out_tif.is_absolute():
            out_tif = self.output_dir / out_tif

        print("============================================================")
        print(f"[WorldCover] Processing year {year}")
        print(f"ROI shapefile: {shp_path}")
        print(f"Output: {out_tif}")
        print("============================================================")

        img = self._worldcover_image(year, roi)
        gd_img = geedim.MaskedImage(img.clip(roi))

        try:
            gd_img.download(
                str(out_tif),
                region=roi,
                scale=self.export_scale,
                crs="EPSG:4326",
                dtype="uint8",
                overwrite=True,
                max_tile_size=32,
            )
            print(f"✅ [WorldCover] Download complete → {out_tif}")
        except Exception as e:
            print(f"❌ [WorldCover] Error during download: {e}")
            raise


## 4️⃣ Alpha Earth Embedding (GOOGLE/SATELLITE_EMBEDDING)

class AlphaEmbeddingDownloader:
    """
    Download Alpha Earth / Satellite Embedding annual image for a single ROI.

    - Dataset: GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL
    - Filters to one year, takes median composite (usually only 1 image).
    - Output: multi-band embedding image (bands as provided by the collection).
    """

    EMBED_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"

    def __init__(
        self,
        credentials_path: str,
        service_account: str,
        output_dir: str,
        export_scale: int = 10,
        crs: str = "EPSG:3857",  # embedding native CRS
    ):
        ee.Reset()
        credentials = ee.ServiceAccountCredentials(service_account, credentials_path)
        ee.Initialize(credentials)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.export_scale = export_scale
        self.crs = crs

        print("✅ Alpha Embedding downloader ready")

    def _load_single_polygon(self, shp_path: str) -> ee.Feature:
        shp_path = Path(shp_path)
        if not shp_path.exists() or not shp_path.is_file():
            raise FileNotFoundError(f"Shapefile not found: {shp_path}")

        gdf = gpd.read_file(shp_path)
        if len(gdf) != 1:
            raise ValueError(
                f"Shapefile must contain exactly 1 feature (polygon). Found: {len(gdf)}"
            )

        geom_type = gdf.geometry.iloc[0].geom_type
        if geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError(
                f"Geometry must be Polygon or MultiPolygon, found: {geom_type}"
            )

        print(f"📁 Loaded ROI from {shp_path} (single polygon, {geom_type}).")
        ee_fc = geemap.gdf_to_ee(gdf)
        return ee.Feature(ee_fc.first())

    def _embedding_image(self, year: int, roi) -> ee.Image:
        col = (
            ee.ImageCollection(self.EMBED_COLLECTION)
            .filter(ee.Filter.calendarRange(year, year, "year"))
        )

        # Optional: raise if empty, to avoid silent all-zero
        size_ok = ee.Algorithms.If(col.size().gt(0), 1, 0)
        # Force evaluation to Python side:
        if ee.Number(size_ok).getInfo() == 0:
            raise RuntimeError(
                f"Embedding collection empty for year {year}. "
                "Check allowed range (typically ~2017–2023)."
            )

        emb = col.median().clip(roi)  # multi-band
        return emb.toFloat()

    def download_from_shapefile(
        self,
        shp_path: str,
        out_tif: str,
        year: int,
    ):
        
        if year > 2024 or year < 2017 :
            raise ValueError(
                f"AlphaEarth only available for years 2017 to 2024. Got {year}"
            )
        feature = self._load_single_polygon(shp_path)
        roi = feature.geometry()

        out_tif = Path(out_tif)
        if not out_tif.is_absolute():
            out_tif = self.output_dir / out_tif

        print("============================================================")
        print(f"[Embedding] Processing year {year}")
        print(f"ROI shapefile: {shp_path}")
        print(f"Output: {out_tif}")
        print("============================================================")

        img = self._embedding_image(year, roi)
        gd_img = geedim.MaskedImage(img.clip(roi))

        try:
            gd_img.download(
                str(out_tif),
                region=roi,
                scale=self.export_scale,
                crs=self.crs,
                dtype="float32",
                overwrite=True,
                max_tile_size=32,
            )
            print(f"✅ [Embedding] Download complete → {out_tif}")
        except Exception as e:
            print(f"❌ [Embedding] Error during download: {e}")
            raise


## 5️⃣ SRTM DEM

class SRTMDownloader:
    """
    Download SRTM DEM for a single ROI.

    - Dataset: USGS/SRTMGL1_003 (30m)
    - Band: 'elevation' [meters]
    """

    SRTM_ID = "USGS/SRTMGL1_003"

    def __init__(
        self,
        credentials_path: str,
        service_account: str,
        output_dir: str,
        export_scale: int = 30,
    ):
        ee.Reset()
        credentials = ee.ServiceAccountCredentials(service_account, credentials_path)
        ee.Initialize(credentials)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.export_scale = export_scale

        print("✅ SRTM downloader ready")

    def _load_single_polygon(self, shp_path: str) -> ee.Feature:
        shp_path = Path(shp_path)
        if not shp_path.exists() or not shp_path.is_file():
            raise FileNotFoundError(f"Shapefile not found: {shp_path}")

        gdf = gpd.read_file(shp_path)
        if len(gdf) != 1:
            raise ValueError(
                f"Shapefile must contain exactly 1 feature (polygon). Found: {len(gdf)}"
            )

        geom_type = gdf.geometry.iloc[0].geom_type
        if geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError(
                f"Geometry must be Polygon or MultiPolygon, found: {geom_type}"
            )

        print(f"📁 Loaded ROI from {shp_path} (single polygon, {geom_type}).")
        ee_fc = geemap.gdf_to_ee(gdf)
        return ee.Feature(ee_fc.first())

    def _srtm_image(self, roi) -> ee.Image:
        img = ee.Image(self.SRTM_ID).select("elevation")
        return img.clip(roi).toFloat()

    def download_from_shapefile(
        self,
        shp_path: str,
        out_tif: str,
    ):
        feature = self._load_single_polygon(shp_path)
        roi = feature.geometry()

        out_tif = Path(out_tif)
        if not out_tif.is_absolute():
            out_tif = self.output_dir / out_tif

        print("============================================================")
        print("[SRTM] Processing ROI")
        print(f"ROI shapefile: {shp_path}")
        print(f"Output: {out_tif}")
        print("============================================================")

        img = self._srtm_image(roi)
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
            print(f"✅ [SRTM] Download complete → {out_tif}")
        except Exception as e:
            print(f"❌ [SRTM] Error during download: {e}")
            raise
