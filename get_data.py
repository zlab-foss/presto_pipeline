import os

# Clean any stale settings
os.environ.pop("PROJ_LIB", None)
os.environ.pop("PROJ_DATA", None)

# Use pyproj's own (new) database (schema ≥3)
from pyproj import datadir
pyproj_proj = datadir.get_data_dir()          # e.g. .../site-packages/pyproj/proj_dir/share/proj
os.environ["PROJ_DATA"] = pyproj_proj
os.environ["PROJ_LIB"]  = pyproj_proj          # keep both for compatibility

# Keep GDAL paths pointed at your env (fine to leave as-is if already set)
if "CONDA_PREFIX" in os.environ:
    os.environ.setdefault("GDAL_DATA", f"{os.environ['CONDA_PREFIX']}/share/gdal")

from pathlib import Path
import ee
import geemap
import geedim

import warnings
warnings.filterwarnings("ignore")


class S2S1PrestoDownloader:
    """
    Download 168-band (12×S2 + 12×S1) for a growing season
    """

    def __init__(
        self,
        asset_path: str,
        output_dir: str = "season_stacks",
        export_scale: float = 10.0,
        s2_cloudy_pct: int = 10,
        start_year: int = 2024,
        end_year: int | None = None,
        max_features_per_batch: int = 5,
        season_start_month: int = 9,
        season_start_day: int = 1,
        gee_project: str | None = None,
    ):

        # ------------------------------------------------------------------
        #  AUTH + INIT (using credentials)
        # ----------------------------------------------------------------
        credentials_path =  "credentials/earthengine_credentials.json"
        service_account = 'fanapanomaly@fanapanomaly.iam.gserviceaccount.com'

        ee.Reset()
        credentials = ee.ServiceAccountCredentials(service_account, credentials_path)
        ee.Initialize(credentials)
        
        # ------------------------------------------------------------------
        #  STORE CONFIG
        # ------------------------------------------------------------------
        self.asset_path = asset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # NEW: base name per shapefile / asset for filenames
        path = Path(asset_path)
        if path.exists():
            self.asset_name = path.stem  # e.g. wetlands2.shp → "wetlands2"
        else:
            # for EE assets, take last part and sanitize a bit
            self.asset_name = asset_path.split("/")[-1].replace(":", "_")

        self.export_scale = export_scale          # meters per pixel
        self.s2_cloudy_pct = s2_cloudy_pct
        self.start_year = start_year
        self.end_year = end_year or start_year
        self.max_features_per_batch = max_features_per_batch
        self.season_start_month = season_start_month
        self.season_start_day = season_start_day

        self.S2_BANDS = [
            "B1", "B2", "B3", "B4", "B5", "B6",
            "B7", "B8", "B8A", "B9", "B11", "B12",
        ]
        self.S2_NAMES = [
            "coastal", "blue", "green", "red",
            "red_edge1", "red_edge2", "red_edge3",
            "nir", "red_edge4", "water_vapor",
            "swir1", "swir2",
        ]
        self.S1_NAMES = ["vv", "vh"]

        # Load feature collection (local file OR EE asset)
        self.fc = self._load_feature_collection(asset_path)

    # ----------------------------------------------------------------------
    #  LOCAL SHP / GEOJSON OR EE ASSET
    # ----------------------------------------------------------------------
    def _load_feature_collection(self, asset_path: str):
        """
        If path exists on disk → load local shapefile.
        Otherwise → treat as an Earth Engine asset ID.
        """
        path = Path(asset_path)

        if path.exists() and path.is_file():
            print(f"📁 Loading local vector file: {asset_path}")
            try:
                ee_fc = geemap.shp_to_ee(str(path))
                print("✔ Loaded local shapefile into EE FeatureCollection.\n")
                return ee_fc
            except Exception as e:
                raise RuntimeError(f"Error reading local shapefile: {e}")

        else:
            print(f"🌐 Loading Earth Engine asset: {asset_path}")
            try:
                return ee.FeatureCollection(asset_path)
            except Exception as e:
                raise RuntimeError(f"Invalid EE asset path or asset not found: {e}")

    # ----------------------------------------------------------------------
    #  HELPERS
    # ----------------------------------------------------------------------
    def _rename_with_month(self, base_names, m):
        mtag = ee.Number(m).format("%02d")
        return ee.List(base_names).map(
            lambda b: ee.String("M").cat(mtag).cat("_").cat(ee.String(b))
        )

    def _db_to_linear(self, img_db):
        vv_lin = ee.Image(10).pow(img_db.select("VV").divide(10))
        vh_lin = ee.Image(10).pow(img_db.select("VH").divide(10))
        return vv_lin.addBands(vh_lin).rename(["VV", "VH"])

    def _season_month_to_calendar(self, season_year, m_idx):
        season_year = ee.Number(season_year)
        m_idx = ee.Number(m_idx)
        start_m = ee.Number(self.season_start_month)

        cal_month = start_m.subtract(1).add(m_idx.subtract(1)).mod(12).add(1)
        year_offset = ee.Number(ee.Algorithms.If(cal_month.gte(start_m), 0, 1))
        cal_year = season_year.add(year_offset)

        return ee.Dictionary({"year": cal_year, "month": cal_month})

    def _bin_start(self, season_year, m_idx):
        d = self._season_month_to_calendar(season_year, m_idx)
        return ee.Date.fromYMD(
            ee.Number(d.get("year")),
            ee.Number(d.get("month")),
            self.season_start_day,
        )

    def _bin_end(self, season_year, m_idx):
        return self._bin_start(season_year, m_idx).advance(1, "month")

    # ----------------------------------------------------------------------
    #  S2 / S1 MONTHLY MEDIANS (FLOAT32)
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
            .select(self.S2_BANDS, self.S2_NAMES)
        )

        fallback = (
            ee.Image.constant([0] * len(self.S2_NAMES))
            .rename(self.S2_NAMES)
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

    def _s1_month(self, roi, season_year, m_idx):
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

        col_lin = col_db.map(self._db_to_linear)

        fallback = (
            ee.Image.constant([0, 0])
            .rename(self.S1_NAMES)
            .selfMask()
            .toFloat()
            .clip(roi)
        )

        return ee.Image(
            ee.Algorithms.If(
                col_lin.size().gt(0),
                col_lin.median()
                .select(["VV", "VH"], self.S1_NAMES)
                .toFloat()
                .clip(roi),
                fallback,
            )
        )

    # ----------------------------------------------------------------------
    #  BUILD 168-BAND STACK (FLOAT32)
    # ----------------------------------------------------------------------
    def create_stack(self, roi, season_year):
        def per_month(m):
            s2 = self._s2_month(roi, season_year, m)
            s1 = self._s1_month(roi, season_year, m)
            return s2.rename(self._rename_with_month(self.S2_NAMES, m)).addBands(
                s1.rename(self._rename_with_month(self.S1_NAMES, m))
            )

        images = ee.List.sequence(1, 12).map(per_month)
        stack = ee.ImageCollection.fromImages(images).toBands()

        old = stack.bandNames()
        new = old.map(lambda n: ee.String(n).split("_").slice(1).join("_"))
        return stack.rename(new).toFloat()

    # ----------------------------------------------------------------------
    #  DOWNLOAD FEATURE AS TILED CROPPED IMAGES
    # ----------------------------------------------------------------------
    def download_feature(self, feature, feat_idx, season_year):
        roi = feature.geometry()
        img = self.create_stack(roi, season_year)
    
        # NEW: include shapefile/asset base name in the filename
        # e.g. wetlands2_season2024_feat0.tif
        out_path = self.output_dir / f"{self.asset_name}_season{season_year}_feat{feat_idx}.tif"
        
        gd_img = geedim.MaskedImage(img.clip(roi))
    
        print(f"  ⬇️  Downloading Feature {feat_idx} to {out_path}...")
    
        try:
            gd_img.download(
                str(out_path), 
                region=roi, 
                scale=self.export_scale,
                crs="EPSG:4326",
                dtype="float32" 
            )
            print(f"  ✓ Saved Feature {feat_idx}")
        except Exception as e:
            print(f"  ❌ Error downloading feature {feat_idx}: {e}")

    # ----------------------------------------------------------------------
    #  MAIN DRIVER
    # ----------------------------------------------------------------------
    def run(self):
        print("Loading feature collection...")
        n = self.fc.size().getInfo()
        print(f"Found {n} features.\n")

        feats = self.fc.toList(n)
        count = 0

        for yr in range(self.start_year, self.end_year + 1):
            print("=" * 60)
            print(
                f"Processing season starting "
                f"{yr}-{self.season_start_month:02d}-{self.season_start_day:02d}"
            )
            print("=" * 60)

            for i in range(n):
                feat = ee.Feature(feats.get(i))
                self.download_feature(feat, i, yr)
                count += 1

        print(f"\n✓ All done → {count} features processed. Output: {self.output_dir}")


if __name__ == "__main__":
    # Example: loop over all .shp in a folder
    shp_dir = Path("./data/ROI/sample_wetlands")

    for shp_path in sorted(shp_dir.glob("*.shp")):
        print(f"\n\n==============================")
        print(f"Processing shapefile: {shp_path.name}")
        print(f"==============================")

        downloader = S2S1PrestoDownloader(
            asset_path=str(shp_path),
            output_dir="./data/inputs",
            start_year=2024,
        )
        downloader.run()
