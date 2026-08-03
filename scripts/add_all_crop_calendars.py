"""
Build, for each crop, a parcel shapefile carrying all five candidate crop
calendars (not just its own): ROI/<crop>/shapefile-all-cc.shp.

For every parcel in ROI/<crop>/shapefile-province-cc-refined.shp, the
parcel's centroid is located inside a province (using
ROI/province/province-crop-calendar.shp), and that province's per-crop
plant/harvest dates are copied onto the parcel as candidate columns
(pl_<crop>/hv_<crop> for each of barley/corn/kolza/rice/wheat), alongside
`province` and `actual` (the crop this shapefile's parcels actually are).
A province with no calendar for a given crop (e.g. rice in Zanjan) leaves
that pair as empty strings; downstream code skips that (parcel, candidate)
combination rather than dropping the parcel.

The raw shapefile-province-cc-shuffled.shp sources used by
add_crop_calendar_to_province.py are no longer on disk, so this script
builds from the refined per-crop shapefiles + the province layer instead.

Run: python3 scripts/add_all_crop_calendars.py
"""

import collections
import os

from osgeo import ogr

from add_crop_calendar_to_province import (
    load_provinces,
    find_province,
    PLANT_FIELD,
    HARVEST_FIELD,
)

ogr.UseExceptions()

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROVINCE_SHP = os.path.join(REPO_ROOT, "ROI/province/province-crop-calendar.shp")

CROPS = ["barley", "corn", "kolza", "rice", "wheat"]

REFINED_SHP = {
    crop: os.path.join(REPO_ROOT, f"ROI/{crop}/shapefile-province-cc-refined.shp")
    for crop in CROPS
}
OUTPUT_SHP = {
    crop: os.path.join(REPO_ROOT, f"ROI/{crop}/shapefile-all-cc.shp")
    for crop in CROPS
}

# Truncated (<=10 char) DBF field names for each candidate's plant/harvest
# columns. Must match utils/crop_calendar.CANDIDATE_PLANT_FIELD /
# CANDIDATE_HARVEST_FIELD.
CAND_PLANT_FIELD = {c: f"pl_{c}" for c in CROPS}
CAND_HARVEST_FIELD = {c: f"hv_{c}" for c in CROPS}


def load_province_calendars(province_path):
    """Returns {province_fid: {crop: (plant_s, harvest_s) | None}}."""
    ds = ogr.Open(province_path, 0)
    layer = ds.GetLayer()
    calendars = {}
    for feature in layer:
        fid = feature.GetFID()
        per_crop = {}
        for crop in CROPS:
            plant = feature.GetField(PLANT_FIELD[crop])
            harvest = feature.GetField(HARVEST_FIELD[crop])
            per_crop[crop] = (plant, harvest) if plant and harvest else None
        calendars[fid] = per_crop
    ds = None
    return calendars


def build_all_cc_for_crop(crop, provinces, province_calendars):
    src_path = REFINED_SHP[crop]
    out_path = OUTPUT_SHP[crop]

    src_ds = ogr.Open(src_path, 0)
    src_layer = src_ds.GetLayer()
    srs = src_layer.GetSpatialRef()
    geom_type = src_layer.GetGeomType()

    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(out_path):
        driver.DeleteDataSource(out_path)

    out_ds = driver.CreateDataSource(out_path)
    layer_name = os.path.splitext(os.path.basename(out_path))[0]
    out_layer = out_ds.CreateLayer(layer_name, srs, geom_type, options=["ENCODING=UTF-8"])

    out_layer.CreateField(ogr.FieldDefn("province", ogr.OFTString))
    out_layer.CreateField(ogr.FieldDefn("actual", ogr.OFTString))
    for c in CROPS:
        out_layer.CreateField(ogr.FieldDefn(CAND_PLANT_FIELD[c], ogr.OFTString))
        out_layer.CreateField(ogr.FieldDefn(CAND_HARVEST_FIELD[c], ogr.OFTString))

    layer_defn = out_layer.GetLayerDefn()

    coverage = collections.Counter()
    total = 0
    no_province = 0

    for feature in src_layer:
        geom = feature.GetGeometryRef()
        if geom is None:
            continue
        total += 1
        centroid = geom.Centroid()
        prov = find_province(provinces, centroid)

        out_feature = ogr.Feature(layer_defn)
        out_feature.SetGeometry(geom.Clone())
        out_feature.SetField("actual", crop)

        per_crop = province_calendars.get(prov["fid"], {}) if prov is not None else {}
        out_feature.SetField("province", prov["name"] if prov is not None else "")
        if prov is None:
            no_province += 1

        for c in CROPS:
            cal = per_crop.get(c)
            if cal is None:
                out_feature.SetField(CAND_PLANT_FIELD[c], "")
                out_feature.SetField(CAND_HARVEST_FIELD[c], "")
            else:
                plant_s, harvest_s = cal
                out_feature.SetField(CAND_PLANT_FIELD[c], plant_s)
                out_feature.SetField(CAND_HARVEST_FIELD[c], harvest_s)
                coverage[c] += 1

        out_layer.CreateFeature(out_feature)
        out_feature = None

    out_ds = None
    src_ds = None

    if no_province:
        print(f"  WARNING: {no_province}/{total} parcels had no matching province")

    return total, coverage


def main():
    provinces = load_provinces(PROVINCE_SHP)
    print(f"Loaded {len(provinces)} province features")
    province_calendars = load_province_calendars(PROVINCE_SHP)

    coverage_table = {}
    for crop in CROPS:
        print(f"\nProcessing {crop} ({REFINED_SHP[crop]}) ...")
        total, coverage = build_all_cc_for_crop(crop, provinces, province_calendars)
        coverage_table[crop] = (total, coverage)
        print(f"  Wrote {OUTPUT_SHP[crop]}: {total} parcels")

    print("\n=== Coverage: actual crop x candidate calendar availability ===")
    header = "actual".ljust(10) + "total".rjust(10) + "".join(c.rjust(12) for c in CROPS)
    print(header)
    for crop in CROPS:
        total, coverage = coverage_table[crop]
        row = crop.ljust(10) + str(total).rjust(10)
        row += "".join(str(coverage.get(c, 0)).rjust(12) for c in CROPS)
        print(row)


if __name__ == "__main__":
    main()
