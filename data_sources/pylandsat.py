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
import json

from shapely.geometry import mapping
try:
    from shapely import force_2d as _force_2d
except Exception:  # pragma: no cover
    def _force_2d(geom):
        return geom


class LandsatGEEDownloader:
    """
    Download Landsat (8/9, 7, 5) monthly medians for a single ROI as a stacked GeoTIFF.

    - Uses Collection 2, Level-2 Surface Reflectance:
        * LANDSAT/LC08/C02/T1_L2 (Landsat 8)
        * LANDSAT/LC09/C02/T1_L2 (Landsat 9)
        * LANDSAT/LE07/C02/T1_L2 (Landsat 7)
        * LANDSAT/LT05/C02/T1_L2 (Landsat 5)

    - Outputs 12 × 6 bands: M01_blue, ..., M12_swir2
    - Bands (friendly names):
        ["blue", "green", "red", "nir", "swir1", "swir2"]
    """

    LS_NAMES = ["blue", "green", "red", "nir", "swir1", "swir2"]

    def __init__(
        self,
        credentials_path: str,
        service_account: str,
        output_dir: str,
        export_scale: int = 30,
        start_month: int = 9,
        start_day: int = 1,
        # -----------------------------
        # Gap filling controls
        # -----------------------------
        gapfill_enabled: bool = True,
        gapfill_apply_to_l5: bool = True,   # you asked for L5 too
        gapfill_window_months: int = 2,     # use +/- N months as temporal fill
        gapfill_focal_radius_px: float = 2, # last-resort local interpolation radius (pixels)
        gapfill_focal_iterations: int = 1,
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

        self.gapfill_enabled = gapfill_enabled
        self.gapfill_apply_to_l5 = gapfill_apply_to_l5
        self.gapfill_window_months = int(gapfill_window_months)
        self.gapfill_focal_radius_px = float(gapfill_focal_radius_px)
        self.gapfill_focal_iterations = int(gapfill_focal_iterations)

        print("✅ Landsat downloader ready (bands: blue, green, red, nir, swir1, swir2)")
        if self.gapfill_enabled:
            print(
                f"🩹 Gap-fill enabled: window=±{self.gapfill_window_months} months, "
                f"focalMean radius={self.gapfill_focal_radius_px}px x{self.gapfill_focal_iterations} iters"
            )

    # ==============================================================
    # GEOMETRY HELPERS
    # ==============================================================
    def _load_single_polygon(self, shp_path: str) -> ee.Feature:
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
                f"Shapefile must contain exactly 1 feature (polygon). Found: {len(gdf)}"
            )

        gdf = gdf.to_crs(epsg=4326)

        geom_type = gdf.geometry.iloc[0].geom_type
        if geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError(f"Geometry must be Polygon or MultiPolygon, found: {geom_type}")

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
        if season_year >= 2013:
            col = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").merge(
                ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
            )
            band_codes = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
            sensor = "L8/9"
        elif season_year >= 1999:
            col = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
            band_codes = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]
            sensor = "L7"
        else:
            col = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
            band_codes = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]
            sensor = "L5"

        return col, band_codes, sensor

    # ==============================================================
    # MASK + SCALE
    # ==============================================================
    def _mask_and_scale(self, img, band_codes):
        """
        Mask clouds & cloud shadows using QA_PIXEL, then scale SR bands:
        scale=0.0000275, offset=-0.2 (Landsat C2 L2 convention).
        """
        qa = img.select("QA_PIXEL")
        cloud_bit = 1 << 3         # Cloud
        cloud_shadow_bit = 1 << 4  # Cloud shadow

        mask = qa.bitwiseAnd(cloud_bit).eq(0).And(
            qa.bitwiseAnd(cloud_shadow_bit).eq(0)
        )

        img_masked = img.updateMask(mask)
        sr = img_masked.select(band_codes).multiply(0.0000275).add(-0.2)
        return sr.rename(self.LS_NAMES)

    # ==============================================================
    # GAP FILLING (L7 SLC-OFF style + generic masked-pixel fill)
    # ==============================================================
    def _should_gapfill(self, sensor: str) -> bool:
        if not self.gapfill_enabled:
            return False
        if sensor == "L7":
            return True
        if sensor == "L5":
            return bool(self.gapfill_apply_to_l5)
        return False

    def _gap_fill_image(self, primary: ee.Image, fill: ee.Image) -> ee.Image:
        """
        Fill masked pixels in 'primary' using 'fill' (temporal median),
        then optionally fill any remaining holes using focalMean.
        """
        # 1) Temporal fill: replace masked pixels with fill values
        filled = primary.unmask(fill)

        # 2) Last resort: local interpolation for any remaining masked pixels
        # (focalMean uses neighborhood stats; can blur, so keep radius small)
        if self.gapfill_focal_radius_px > 0 and self.gapfill_focal_iterations > 0:
            interp = filled.focalMean(
                radius=self.gapfill_focal_radius_px,
                kernelType="circle",
                units="pixels",
                iterations=self.gapfill_focal_iterations,
            )
            filled = filled.unmask(interp)

        return filled

    def _ls_month(self, roi, season_year, m_idx) -> ee.Image:
        start = self._bin_start(season_year, m_idx)
        end = self._bin_end(season_year, m_idx)

        base_col, band_codes, sensor = self._get_landsat_collection_and_bands(season_year)

        def _prep(img):
            return self._mask_and_scale(img, band_codes)

        month_col = (
            base_col
            .filterBounds(roi)
            .filterDate(start, end)
            .map(_prep)
        )

        # If no images at all, return fully masked fallback
        fallback = (
            ee.Image.constant([0] * len(self.LS_NAMES))
            .rename(self.LS_NAMES)
            .selfMask()
            .toFloat()
            .clip(roi)
        )

        primary = ee.Image(
            ee.Algorithms.If(
                month_col.size().gt(0),
                month_col.median().toFloat(),
                fallback,
            )
        ).clip(roi)

        # Gap fill (L7 + optional L5)
        def _do_fill():
            # Wider temporal pool (±N months) to exploit non-aligned gaps across scenes
            w = ee.Number(self.gapfill_window_months)
            fill_start = start.advance(w.multiply(-1), "month")
            fill_end = end.advance(w, "month")

            fill_col = (
                base_col
                .filterBounds(roi)
                .filterDate(fill_start, fill_end)
                .map(_prep)
            )

            fill_img = ee.Image(
                ee.Algorithms.If(
                    fill_col.size().gt(0),
                    fill_col.median().toFloat(),
                    fallback,
                )
            ).clip(roi)

            return self._gap_fill_image(primary, fill_img).toFloat().clip(roi)

        return ee.Image(
            ee.Algorithms.If(
                ee.Algorithms.IsEqual(self._should_gapfill(sensor), True),
                _do_fill(),
                primary.toFloat().clip(roi),
            )
        )

    # ==============================================================
    # BUILD SEASON STACK (12 × 6 bands)
    # ==============================================================
    def create_stack(self, roi, season_year: int) -> ee.Image:
        def per_month(m):
            ls = self._ls_month(roi, season_year, m)
            return ls.rename(self._rename_with_month(self.LS_NAMES, m))

        images = ee.List.sequence(1, 12).map(per_month)
        stack = ee.ImageCollection.fromImages(images).toBands()

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


# ======================