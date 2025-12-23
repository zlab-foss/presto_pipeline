
For simple ROIs (Does not work on complex polygons, use amir-get-data-crop branch for them)
```
python get_data.py   --shp ./ROI/test-presto-crop.shp   --year 2019   --source s1 --out ./output_path
python get_data.py   --shp ./ROI/test-presto-crop.shp   --year 2019   --source s2
python get_data.py   --shp ./ROI/test-lulc.shp   --year 2023   --source s2   --s2-bands red green blue nir
python get_data.py   --shp ./ROI/test-presto-crop.shp   --year 2019   --source landsat
python get_data.py   --shp ./ROI/test-presto-crop.shp   --year 2021   --source worldcover   --worldcover-scale 10

```