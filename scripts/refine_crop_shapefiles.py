"""
Refine each crop's parcel-level shapefile down to two attributes: plant, harvest.

Each source shapefile (ROI/<crop>/shapefile-province-cc*.shp) has parcel-level
fields `type` (abi/deym/zorat_olofee/zorat_daneie/etc), `plant_s`, `plant_e`,
`harvest_s`, `harvest_e` (and `index`/`class` for wheat/corn). This script drops all
of that down to a single representative `plant` and `harvest` date per parcel, using
the same per-province mode logic as add_crop_calendar_to_province.py: every parcel is
assigned to a province (by centroid containment), and all parcels within a province
get that province's most frequent (plant_s, harvest_s) combo for the crop. Original
parcel geometries are kept as-is; only the attribute table is replaced.

Writes ROI/<crop>/shapefile-province-cc-refined.shp for each crop, leaving the
original source shapefiles untouched.

Run: python3 scripts/refine_crop_shapefiles.py
"""

import os

from osgeo import ogr

from add_crop_calendar_to_province import (
    CROP_SHAPEFILES,
    PROVINCE_SHP,
    aggregate_crop_calendar,
    find_province,
    load_provinces,
)

ogr.UseExceptions()

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def refine_crop_shapefile(crop, rel_path, provinces, calendar):
    src_path = os.path.join(REPO_ROOT, rel_path)
    out_path = os.path.join(
        os.path.dirname(src_path), "shapefile-province-cc-refined.shp"
    )

    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(out_path):
        driver.DeleteDataSource(out_path)

    src_ds = ogr.Open(src_path, 0)
    src_layer = src_ds.GetLayer()
    srs = src_layer.GetSpatialRef()
    geom_type = src_layer.GetGeomType()

    out_ds = driver.CreateDataSource(out_path)
    out_layer = out_ds.CreateLayer(
        "shapefile-province-cc-refined", srs, geom_type, options=["ENCODING=UTF-8"]
    )
    out_layer.CreateField(ogr.FieldDefn("plant", ogr.OFTString))
    out_layer.CreateField(ogr.FieldDefn("harvest", ogr.OFTString))
    layer_defn = out_layer.GetLayerDefn()

    written = 0
    matched = 0
    for feature in src_layer:
        geom = feature.GetGeometryRef()
        if geom is None:
            continue
        centroid = geom.Centroid()
        prov = find_province(provinces, centroid)
        plant_s, harvest_s = calendar.get(prov["fid"], ("", "")) if prov else ("", "")
        if plant_s or harvest_s:
            matched += 1

        out_feature = ogr.Feature(layer_defn)
        out_feature.SetGeometry(geom.Clone())
        out_feature.SetField("plant", plant_s)
        out_feature.SetField("harvest", harvest_s)
        out_layer.CreateFeature(out_feature)
        out_feature = None
        written += 1

    src_ds = None
    out_ds = None

    print(f"  wrote {written} parcels ({matched} with a plant/harvest value) -> {out_path}")


def main():
    provinces = load_provinces(PROVINCE_SHP)
    print(f"Loaded {len(provinces)} province features")

    for crop, rel_path in CROP_SHAPEFILES.items():
        crop_path = os.path.join(REPO_ROOT, rel_path)
        print(f"Processing {crop} ({crop_path}) ...")
        calendar = aggregate_crop_calendar(crop_path, provinces)
        print(f"  {len(calendar)}/{len(provinces)} provinces got a {crop} calendar")
        refine_crop_shapefile(crop, rel_path, provinces, calendar)


if __name__ == "__main__":
    main()
