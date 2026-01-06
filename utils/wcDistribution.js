// Google Earth Engine (JavaScript)
// Goal: For each ROI (feature) in a shapefile/FeatureCollection, compute ESA WorldCover 2021
// class area percentages and write them as columns, then export as a Shapefile.

// -------------------------
// 1) INPUTS (edit these)
// -------------------------

// Option A: If you uploaded your shapefile as an Asset FeatureCollection:
var rois = ee.FeatureCollection("projects/ee-chrcheel/assets/iran-grid");

// Option B: If you're using an imported layer in the Code Editor, comment the line above and use:
// var rois = YOUR_IMPORTED_LAYER_NAME;

// Export folder/name
var EXPORT_DESC = "rois_worldcover2021_class_percent";
var EXPORT_FOLDER = "GEE_exports"; // optional, can be null/empty

// -------------------------
// 2) ESA WorldCover 2021
// -------------------------
// WorldCover v200 corresponds to 2021 in GEE.
var wc2021 = ee.Image("ESA/WorldCover/v200/2021").select("Map");

// WorldCover class codes present in WorldCover:
var CLASS_CODES = ee.List([10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]);

// For column naming (Shapefile field names should be short; keep them <= ~10 chars)
var CLASS_FIELDS = ee.Dictionary({
  10:  "p_tc",   // Tree cover
  20:  "p_shr",  // Shrubland
  30:  "p_grs",  // Grassland
  40:  "p_crp",  // Cropland
  50:  "p_bui",  // Built-up
  60:  "p_bar",  // Bare/sparse vegetation
  70:  "p_sni",  // Snow/ice
  80:  "p_wat",  // Permanent water bodies
  90:  "p_hwt",  // Herbaceous wetland
  95:  "p_man",  // Mangroves
  100: "p_mli"   // Moss/lichen
});

// -------------------------
// 3) Helper: compute % area per class for one feature
// -------------------------
function addWorldCoverPercents(feat) {
  var geom = feat.geometry();

  // Compute area (m²) per class inside this feature using grouped reducer.
  var areaImg = ee.Image.pixelArea().addBands(wc2021);

  var grouped = areaImg.reduceRegion({
    reducer: ee.Reducer.sum().group({
      groupField: 1,     // band index of class band (wc2021) in (pixelArea, class)
      groupName: "class"
    }),
    geometry: geom,
    scale: 10,          // WorldCover is 10m
    maxPixels: 1e13,
    tileScale: 4
  });

  // grouped.get("groups") is a list of dicts: [{class: <code>, sum: <area_m2>}, ...]
  var groups = ee.List(grouped.get("groups"));

  // Total mapped area (m²) inside ROI (sum over all classes found)
  var totalArea = ee.Number(
    groups.iterate(function (it, acc) {
      it = ee.Dictionary(it);
      return ee.Number(acc).add(ee.Number(it.get("sum")));
    }, 0)
  );

  // Build a dictionary of percent fields, ensuring missing classes become 0.
  // First, convert groups list into a {classCode: areaM2} dictionary.
  var areaByClass = ee.Dictionary(
    groups.iterate(function (it, acc) {
      it = ee.Dictionary(it);
      var cls = ee.Number(it.get("class")).format(); // string key
      var area = ee.Number(it.get("sum"));
      return ee.Dictionary(acc).set(cls, area);
    }, ee.Dictionary({}))
  );

  // For every known class code, compute percent = 100 * area / totalArea
  var percentDict = ee.Dictionary(
    CLASS_CODES.iterate(function (code, acc) {
      code = ee.Number(code);
      var key = code.format(); // matches areaByClass string keys
      var area = ee.Number(areaByClass.get(key, 0));
      var pct = ee.Algorithms.If(
        totalArea.gt(0),
        area.divide(totalArea).multiply(100),
        0
      );

      var fieldName = ee.String(CLASS_FIELDS.get(code));
      return ee.Dictionary(acc).set(fieldName, ee.Number(pct));
    }, ee.Dictionary({}))
  );

  // Optionally add total area as well (m²)
  percentDict = percentDict.set("a_m2", totalArea);

  return feat.set(percentDict);
}

// -------------------------
// 4) Run + export
// -------------------------
var roisOut = rois.map(addWorldCoverPercents);

// Inspect one feature
print("Example feature with % fields:", roisOut.first());

Export.table.toDrive({
  collection: roisOut,
  description: EXPORT_DESC,
  fileFormat: "GeoJSON"   // or "CSV"
});

// Optional: visualize
Map.centerObject(rois, 7);
Map.addLayer(wc2021, {}, "WorldCover 2021");
Map.addLayer(rois, {}, "ROIs");
