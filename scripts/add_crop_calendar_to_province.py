"""
Add per-crop plant/harvest date columns to the province shapefile.

Spatially joins parcel-level crop calendar data (ROI/<crop>/shapefile-province-cc*.shp)
to province polygons (ROI/province/province-crop-calendar.shp, used here purely as a
source of province geometry + name): each parcel's centroid is located inside a
province, parcels are grouped by province, and the most frequent
(plant_s, plant_e, harvest_s, harvest_e) combination is taken as that province's calendar
for the crop. Writes ROI/province/province-crop-calendar.shp with the
original `province` field plus plant_<crop>/harvest_<crop> for each crop below.
Shapefile DBF field names are capped at 10 chars, so these are truncated
(plant_barley -> plant_barl, harvest_barley -> harvest_ba, etc; see PLANT_FIELD /
HARVEST_FIELD below for the exact truncated names used).

Run: python3 scripts/add_crop_calendar_to_province.py
"""

import collections
import os

from osgeo import ogr

ogr.UseExceptions()

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROVINCE_SHP = os.path.join(REPO_ROOT, "ROI/province/province-crop-calendar.shp")
OUTPUT_SHP = os.path.join(REPO_ROOT, "ROI/province/province-crop-calendar.shp")

CROP_SHAPEFILES = {
    "barley": "ROI/barley/shapefile-province-cc-shuffled.shp",
    "corn": "ROI/corn/shapefile-province-cc.shp",
    "kolza": "ROI/kolza/shapefile-province-cc-shuffled.shp",
    "rice": "ROI/rice/shapefile-province-cc-shuffled.shp",
    "wheat": "ROI/wheat/shapefile-province-cc-shuffled.shp",
}

# Shapefile DBF field names are capped at 10 chars, so plant_<crop>/harvest_<crop>
# get truncated by GDAL (e.g. plant_barley -> plant_barl). Define the truncated
# names explicitly so they stay stable and unambiguous.
PLANT_FIELD = {
    "barley": "plant_barl",
    "corn": "plant_corn",
    "kolza": "plant_kolz",
    "rice": "plant_rice",
    "wheat": "plant_whea",
}
HARVEST_FIELD = {
    "barley": "harvest_ba",
    "corn": "harvest_co",
    "kolza": "harvest_ko",
    "rice": "harvest_ri",
    "wheat": "harvest_wh",
}

DATE_FIELDS = ("plant_s", "plant_e", "harvest_s", "harvest_e")


def load_provinces(path):
    ds = ogr.Open(path, 0)
    layer = ds.GetLayer()
    provinces = []
    for feature in layer:
        geom = feature.GetGeometryRef().Clone()
        provinces.append(
            {
                "fid": feature.GetFID(),
                "name": feature.GetField("province"),
                "geom": geom,
                "envelope": geom.GetEnvelope(),  # (minx, maxx, miny, maxy)
            }
        )
    ds = None
    return provinces


def find_province(provinces, point):
    x, y = point.GetX(), point.GetY()
    for prov in provinces:
        minx, maxx, miny, maxy = prov["envelope"]
        if not (minx <= x <= maxx and miny <= y <= maxy):
            continue
        if prov["geom"].Contains(point):
            return prov
    # fall back to intersects for centroids that fall exactly on a boundary
    for prov in provinces:
        if prov["geom"].Intersects(point):
            return prov
    return None


def aggregate_crop_calendar(crop_shp_path, provinces):
    """Returns {province_fid: (plant_s, harvest_s)} mode date combo per province."""
    ds = ogr.Open(crop_shp_path, 0)
    layer = ds.GetLayer()

    combos_by_province = collections.defaultdict(collections.Counter)

    for feature in layer:
        values = [feature.GetField(f) for f in DATE_FIELDS]
        if any(v is None or v == "" for v in values):
            continue
        geom = feature.GetGeometryRef()
        if geom is None:
            continue
        centroid = geom.Centroid()
        prov = find_province(provinces, centroid)
        if prov is None:
            continue
        combos_by_province[prov["fid"]][tuple(values)] += 1

    ds = None

    result = {}
    for fid, counter in combos_by_province.items():
        plant_s, plant_e, harvest_s, harvest_e = counter.most_common(1)[0][0]
        result[fid] = (plant_s, harvest_s)
    return result


def write_output(provinces, calendars):
    driver = ogr.GetDriverByName("ESRI Shapefile")
    tmp_output = OUTPUT_SHP + ".tmp"
    if os.path.exists(tmp_output):
        driver.DeleteDataSource(tmp_output)

    src_ds = ogr.Open(PROVINCE_SHP, 0)
    src_layer = src_ds.GetLayer()
    srs = src_layer.GetSpatialRef()

    out_ds = driver.CreateDataSource(tmp_output)
    out_layer = out_ds.CreateLayer(
        "province-crop-calendar", srs, ogr.wkbPolygon, options=["ENCODING=UTF-8"]
    )

    out_layer.CreateField(ogr.FieldDefn("province", ogr.OFTString))
    for crop in CROP_SHAPEFILES:
        out_layer.CreateField(ogr.FieldDefn(PLANT_FIELD[crop], ogr.OFTString))
        out_layer.CreateField(ogr.FieldDefn(HARVEST_FIELD[crop], ogr.OFTString))

    layer_defn = out_layer.GetLayerDefn()
    for prov in provinces:
        feature = ogr.Feature(layer_defn)
        feature.SetGeometry(prov["geom"])
        feature.SetField("province", prov["name"])
        for crop in CROP_SHAPEFILES:
            plant_s, harvest_s = calendars[crop].get(prov["fid"], ("", ""))
            feature.SetField(PLANT_FIELD[crop], plant_s)
            feature.SetField(HARVEST_FIELD[crop], harvest_s)
        out_layer.CreateFeature(feature)
        feature = None

    out_ds = None
    src_ds = None

    # swap tmp files into place (PROVINCE_SHP == OUTPUT_SHP: rebuilding from itself)
    for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        tmp_path = OUTPUT_SHP[: -len(".shp")] + ".tmp" + ext
        final_path = OUTPUT_SHP[: -len(".shp")] + ext
        if os.path.exists(tmp_path):
            os.replace(tmp_path, final_path)


def main():
    provinces = load_provinces(PROVINCE_SHP)
    print(f"Loaded {len(provinces)} province features")

    calendars = {}
    for crop, rel_path in CROP_SHAPEFILES.items():
        crop_path = os.path.join(REPO_ROOT, rel_path)
        print(f"Processing {crop} ({crop_path}) ...")
        calendars[crop] = aggregate_crop_calendar(crop_path, provinces)
        populated = len(calendars[crop])
        print(f"  {populated}/{len(provinces)} provinces got a {crop} calendar")

    write_output(provinces, calendars)
    print(f"Wrote {OUTPUT_SHP}")


if __name__ == "__main__":
    main()
