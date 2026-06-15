import os
from pathlib import Path
from typing import List, Union

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
import json

from shapely.geometry import mapping
try:
    from shapely import force_2d as _force_2d
except Exception:  # pragma: no cover
    def _force_2d(geom):
        return geom


class S2GEEDownloader:
    """
    Download Sentinel-2 monthly medians for a single ROI as a stacked GeoTIFF.

    Usage:
        downloader = S2GEEDownloader(
            credentials_path="path/to/key.json",
            service_account="xxx@yyy.iam.gserviceaccount.com",
            output_dir="./data/gee_inputs",
            bands=["B2", "B3", "B4", "B8"],  # or "all"
            s2_cloudy_pct=10,
            export_scale=10,
            start_month=9,
            start_day=1,
        )

        downloader.download_from_shapefile(
            shp_path="./data/ROI/benchmark.shp",  # MUST contain exactly 1 polygon
            out_tif="season2024_benchmark.tif",
            season_year=2024,
        )
    """

    _FULL_S2_BANDS = [
        "B2", "B3", "B4", "B5", "B6",
        "B7", "B8", "B8A", "B11", "B12",
    ]

    def __init__(
        self,
        credentials_path: str,
        service_account: str,
        output_dir: str,
        bands: Union[str, List[str]] = ("B2", "B3", "B4", "B8"),
        s2_cloudy_pct: int = 10,
        export_scale: int = 10,
        start_month: int = 9,
        start_day: int = 1,
    ):
        # ------------------------------------------------------------------
        #  AUTH
        # ------------------------------------------------------------------
        ee.Reset()
        credentials = ee.ServiceAccountCredentials(service_account, credentials_path)
        ee.Initialize(credentials)

        # ------------------------------------------------------------------
        #  STORE CONFIG
        # ------------------------------------------------------------------
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.export_scale = export_scale
        self.s2_cloudy_pct = s2_cloudy_pct
        self.start_month = start_month
        self.start_day = start_day

        # ------------------------------------------------------------------
        #  BAND SELECTION
        # ------------------------------------------------------------------
        if isinstance(bands, str) and bands.lower() == "all":
            self.S2_BANDS = list(self._FULL_S2_BANDS)
        else:
            if isinstance(bands, str):
                raise ValueError(
                    "If 'bands' is a string, it must be 'all'. "
                    "Otherwise provide a list, e.g. ['B2', 'B8']."
                )
            invalid = [b for b in bands if b not in self._FULL_S2_BANDS]
            if invalid:
                raise ValueError(
                    f"Invalid band codes: {invalid}. "
                    f"Valid codes: {self._FULL_S2_BANDS}"
                )
            self.S2_BANDS = list(bands)

        print(f"✅ S2 bands selected: {self.S2_BANDS}")

    # ----------------------------------------------------------------------
    #  INTERNAL HELPERS
    # ----------------------------------------------------------------------
    def _load_single_polygon(self, shp_path: str) -> ee.Feature:
        """
        Load a shapefile, ensure it has exactly 1 polygon, and convert to ee.Feature.
        """
        shp_path = Path(shp_path)
        if not shp_path.exists() or not shp_path.is_file():
            raise FileNotFoundError(f"Shapefile not found: {shp_path}")

        gdf = gpd.read_file(shp_path)
        if gdf.crs is None:
            raise ValueError(
                f"Shapefile {shp_path} has no CRS defined (missing/invalid .prj)."
            )

        # Earth Engine expects lon/lat (EPSG:4326). Also normalize geometry validity.
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.make_valid()
        gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
        if len(gdf) != 1:
            raise ValueError(
                f"Shapefile must contain exactly 1 feature (polygon). "
                f"Found: {len(gdf)}"
            )

        gdf = gdf.to_crs(epsg=4326)

        # Optional: check geometry type
        geom_type = gdf.geometry.iloc[0].geom_type
        if geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError(
                f"Geometry must be Polygon or MultiPolygon, found: {geom_type}"
            )

        print(f"📁 Loaded ROI from {shp_path} (single polygon, {geom_type}).")

        last_err: Exception | None = None
        for tol_m in (0, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 20000):
            try:
                if tol_m > 0:
                    gdf_try = gdf.to_crs(epsg=3857)
                    gdf_try["geometry"] = (
                        gdf_try.geometry.simplify(tol_m, preserve_topology=True)
                        .make_valid()
                    )

                    geom_metric = gdf_try.geometry.iloc[0]
                    if getattr(geom_metric, "geom_type", None) == "MultiPolygon" and getattr(
                        geom_metric, "geoms", None
                    ) is not None:
                        # Prefer the largest part to reduce complexity.
                        geom_metric = max(geom_metric.geoms, key=lambda gg: gg.area)
                        gdf_try.at[gdf_try.index[0], "geometry"] = geom_metric

                    gdf_try = gdf_try.to_crs(epsg=4326)
                else:
                    gdf_try = gdf

                geom = _force_2d(gdf_try.geometry.iloc[0])
                geojson = json.loads(json.dumps(mapping(geom)))
                roi = ee.Geometry(geojson, None, False)
                return ee.Feature(roi)
            except Exception as e:
                last_err = e

        # Last resort: use bounding box so the pipeline can proceed.
        try:
            minx, miny, maxx, maxy = gdf.geometry.iloc[0].bounds
            print(
                f"⚠️  Warning: falling back to ROI bounding box for {shp_path} "
                "(polygon too complex for Earth Engine)."
            )
            roi = ee.Geometry.Rectangle([minx, miny, maxx, maxy], None, False)
            return ee.Feature(roi)
        except Exception:
            raise RuntimeError(
                "Failed converting ROI shapefile to an Earth Engine geometry. "
                "This often happens for extremely detailed admin boundaries. "
                "Try dissolving/simplifying the ROI, or reduce `max_pixels` so tiles are smaller."
            ) from last_err

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

    # ----------------------------------------------------------------------
    #  S2 MONTHLY MEDIAN
    # ----------------------------------------------------------------------
    def _s2_month(self, roi, season_year, m_idx):
        start = self._bin_start(season_year, m_idx)
        end = self._bin_end(season_year, m_idx)

        def _mask(img):
            qa = img.select("QA60")
            cloud = 1 << 10
            cirrus = 1 << 11
            mask = qa.bitwiseAnd(cloud).eq(0).And(qa.bitwiseAnd(cirrus).eq(0))
            return img.updateMask(mask)

        col = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(roi)
            .filterDate(start, end)
            .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", self.s2_cloudy_pct))
            .map(_mask)
            .select(self.S2_BANDS)
        )

        # Fallback image if no scenes
        fallback = (
            ee.Image.constant([0] * len(self.S2_BANDS))
            .rename(self.S2_BANDS)
            .selfMask()
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

    # ----------------------------------------------------------------------
    #  BUILD S2-ONLY STACK (12 months × selected bands)
    # ----------------------------------------------------------------------
    def create_stack(self, roi, season_year: int):
        """
        Create 12-month S2 median stack for given ROI and season year.
        """
        def per_month(m):
            s2 = self._s2_month(roi, season_year, m)
            return s2.rename(self._rename_with_month(self.S2_BANDS, m))

        images = ee.List.sequence(1, 12).map(per_month)
        stack = ee.ImageCollection.fromImages(images).toBands()

        # Remove "M01_" prefix duplication (so bands like "M01_red" stay as is)
        old = stack.bandNames()
        new = old.map(lambda n: ee.String(n).split("_").slice(0).join("_"))
        return stack.rename(new).toFloat()

    # ----------------------------------------------------------------------
    #  PUBLIC: DOWNLOAD FROM SHAPEFILE
    # ----------------------------------------------------------------------
    def download_from_shapefile(
        self,
        shp_path: str,
        out_tif: str,
        season_year: int,
    ):
        """
        Load a single-polygon shapefile and download the seasonal S2 stack.

        shp_path  : path to a shapefile with exactly 1 polygon.
        out_tif   : output filename (full path or relative to output_dir).
        season_year : integer season year (e.g. 2024).
        """
        feature = self._load_single_polygon(shp_path)
        roi = feature.geometry()

        # Ensure output path
        out_tif = Path(out_tif)
        if not out_tif.is_absolute():
            out_tif = self.output_dir / out_tif

        print("============================================================")
        print(f"Processing growing season {season_year}")
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
                overwrite = True,
                max_tile_size=32,  
            )
            print(f"✅ Download complete → {out_tif}")
        except Exception as e:
            print(f"❌ Error during download: {e}")
            raise


class S1GEEDownloader:
    """
    Download Sentinel-1 (VV, VH) monthly medians for a single ROI as a stacked GeoTIFF.

    - Uses COPERNICUS/S1_GRD
    - Filters IW mode, 10 m, dual-pol VV/VH
    - Converts dB → linear before taking monthly median
    - Outputs 24 bands: M01_VV, M01_VH, ..., M12_VV, M12_VH

    Usage:
        s1_down = S1GEEDownloader(
            credentials_path="path/to/key.json",
            service_account="xxx@yyy.iam.gserviceaccount.com",
            output_dir="./data/gee_inputs_s1",
            export_scale=10,
            start_month=9,
            start_day=1,
        )

        s1_down.download_from_shapefile(
            shp_path="./data/ROI/benchmark.shp",
            out_tif="season2024_benchmark_s1.tif",
            season_year=2024,
        )
    """

    # Output band base names (before month prefix)
    S1_NAMES = ["VV", "VH"]

    def __init__(
        self,
        credentials_path: str,
        service_account: str,
        output_dir: str,
        export_scale: int = 10,
        start_month: int = 9,
        start_day: int = 1,
        orbit_pass: str | None = None,  # e.g. "ASCENDING" / "DESCENDING" / None
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
        self.orbit_pass = orbit_pass

        print("✅ S1 downloader ready (VV, VH)")

    # ------------------------------------------------------------------
    #  GEOMETRY HELPERS (same logic as S2GEEDownloader)
    # ------------------------------------------------------------------
    def _load_single_polygon(self, shp_path: str) -> ee.Feature:
        """
        Load a shapefile, ensure it has exactly 1 polygon, and convert to ee.Feature.
        """
        shp_path = Path(shp_path)
        if not shp_path.exists() or not shp_path.is_file():
            raise FileNotFoundError(f"Shapefile not found: {shp_path}")

        gdf = gpd.read_file(shp_path)
        if gdf.crs is None:
            raise ValueError(
                f"Shapefile {shp_path} has no CRS defined (missing/invalid .prj)."
            )

        # Earth Engine expects lon/lat (EPSG:4326). Also normalize geometry validity.
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.make_valid()
        gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
        if len(gdf) != 1:
            raise ValueError(
                f"Shapefile must contain exactly 1 feature (polygon). "
                f"Found: {len(gdf)}"
            )

        gdf = gdf.to_crs(epsg=4326)

        geom_type = gdf.geometry.iloc[0].geom_type
        if geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError(
                f"Geometry must be Polygon or MultiPolygon, found: {geom_type}"
            )

        print(f"📁 Loaded ROI from {shp_path} (single polygon, {geom_type}).")

        last_err: Exception | None = None
        for tol_m in (0, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 20000):
            try:
                if tol_m > 0:
                    gdf_try = gdf.to_crs(epsg=3857)
                    gdf_try["geometry"] = (
                        gdf_try.geometry.simplify(tol_m, preserve_topology=True)
                        .make_valid()
                    )

                    geom_metric = gdf_try.geometry.iloc[0]
                    if getattr(geom_metric, "geom_type", None) == "MultiPolygon" and getattr(
                        geom_metric, "geoms", None
                    ) is not None:
                        geom_metric = max(geom_metric.geoms, key=lambda gg: gg.area)
                        gdf_try.at[gdf_try.index[0], "geometry"] = geom_metric

                    gdf_try = gdf_try.to_crs(epsg=4326)
                else:
                    gdf_try = gdf

                geom = _force_2d(gdf_try.geometry.iloc[0])
                geojson = json.loads(json.dumps(mapping(geom)))
                roi = ee.Geometry(geojson, None, False)
                return ee.Feature(roi)
            except Exception as e:
                last_err = e

        try:
            minx, miny, maxx, maxy = gdf.geometry.iloc[0].bounds
            print(
                f"⚠️  Warning: falling back to ROI bounding box for {shp_path} "
                "(polygon too complex for Earth Engine)."
            )
            roi = ee.Geometry.Rectangle([minx, miny, maxx, maxy], None, False)
            return ee.Feature(roi)
        except Exception:
            raise RuntimeError(
                "Failed converting ROI shapefile to an Earth Engine geometry. "
                "This often happens for extremely detailed admin boundaries. "
                "Try dissolving/simplifying the ROI, or reduce `max_pixels` so tiles are smaller."
            ) from last_err

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

    # ------------------------------------------------------------------
    #  S1 HELPERS
    # ------------------------------------------------------------------
    def _s1_month(self, roi, season_year, m_idx) -> ee.Image:
        """
        Build monthly composite for Sentinel-1 VV/VH in dB scale.
        """
        start = self._bin_start(season_year, m_idx)
        end = self._bin_end(season_year, m_idx)

        col_db = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(roi)
            .filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .filter(ee.Filter.eq("resolution_meters", 10))
            .select(["VV", "VH"])
        )

        # Optional filter by orbit pass
        if self.orbit_pass is not None:
            col_db = col_db.filter(ee.Filter.eq("orbitProperties_pass", self.orbit_pass))

        # Fallback image if no scenes
        fallback = (
            ee.Image.constant([0, 0])
            .rename(self.S1_NAMES)
            .selfMask()              # fully masked (no data) instead of zeros
            .toFloat()
            .clip(roi)
        )

        return ee.Image(
            ee.Algorithms.If(
                col_db.size().gt(0),
                col_db.median()
                .select(self.S1_NAMES, self.S1_NAMES)
                .toFloat()
                .clip(roi),
                fallback,
            )
        )

    # ------------------------------------------------------------------
    #  BUILD S1 STACK (12 months × 2 bands)
    # ------------------------------------------------------------------
    def create_stack(self, roi, season_year: int) -> ee.Image:
        """
        Create 12-month S1 VV/VH median stack (dB scale) for given ROI and season year.
        """

        def per_month(m):
            s1 = self._s1_month(roi, season_year, m)
            return s1.rename(self._rename_with_month(self.S1_NAMES, m))

        images = ee.List.sequence(1, 12).map(per_month)
        stack = ee.ImageCollection.fromImages(images).toBands()

        # Remove duplicated prefix artifacts (same trick as in S2 class)
        old = stack.bandNames()
        new = old.map(lambda n: ee.String(n).split("_").slice(0).join("_"))
        return stack.rename(new).toFloat()

    # ------------------------------------------------------------------
    #  PUBLIC: DOWNLOAD FROM SHAPEFILE
    # ------------------------------------------------------------------
    def download_from_shapefile(
        self,
        shp_path: str,
        out_tif: str,
        season_year: int,
    ):
        """
        Load a single-polygon shapefile and download the seasonal S1 stack.

        shp_path    : path to a shapefile with exactly 1 polygon.
        out_tif     : output filename (full path or relative to output_dir).
        season_year : integer season year (e.g. 2024).
        """
        feature = self._load_single_polygon(shp_path)
        roi = feature.geometry()

        out_tif = Path(out_tif)
        if not out_tif.is_absolute():
            out_tif = self.output_dir / out_tif

        print("============================================================")
        print(f"[S1] Processing growing season {season_year}")
        print(f"ROI shapefile: {shp_path}")
        print(f"Output: {out_tif}")
        if self.orbit_pass:
            print(f"Orbit pass filter: {self.orbit_pass}")
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
            print(f"✅ [S1] Download complete → {out_tif}")
        except Exception as e:
            print(f"❌ [S1] Error during download: {e}")
            raise
