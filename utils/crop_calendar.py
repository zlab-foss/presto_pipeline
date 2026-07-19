"""
Shared crop-calendar helpers: reading per-polygon plant/harvest dates from a
shapefile, computing centered 14-day bin windows across [plant, harvest],
and gap-filling empty bins from their immediate neighbor bins.
"""

from pathlib import Path
from typing import Callable, List, Tuple

import geopandas as gpd
import pandas as pd
import ee


def read_plant_harvest(shp_path) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Read 'plant'/'harvest' (YYYY-MM-DD strings) from row 0 of a per-polygon
    shapefile (e.g. wheat/barley/rice/kolza crop-calendar shapefiles).
    """
    shp_path = Path(shp_path)
    if not shp_path.exists() or not shp_path.is_file():
        raise FileNotFoundError(f"Shapefile not found: {shp_path}")

    gdf = gpd.read_file(shp_path)
    if len(gdf) < 1:
        raise ValueError(f"Shapefile {shp_path} has no features.")

    for col in ("plant", "harvest"):
        if col not in gdf.columns:
            raise ValueError(
                f"Shapefile {shp_path} missing required column '{col}'. "
                f"Available columns: {list(gdf.columns)}"
            )

    row = gdf.iloc[0]
    plant = pd.to_datetime(row["plant"])
    harvest = pd.to_datetime(row["harvest"])

    if pd.isna(plant) or pd.isna(harvest):
        raise ValueError(f"Shapefile {shp_path}: null plant/harvest value.")
    if harvest <= plant:
        raise ValueError(
            f"Shapefile {shp_path}: harvest ({harvest.date()}) must be after "
            f"plant ({plant.date()})."
        )

    return plant, harvest


def to_ee_dates(plant: pd.Timestamp, harvest: pd.Timestamp) -> Tuple[ee.Date, ee.Date]:
    """Convert pandas Timestamps to ee.Date objects."""
    return ee.Date(plant.strftime("%Y-%m-%d")), ee.Date(harvest.strftime("%Y-%m-%d"))


def bin_window(
    plant: ee.Date,
    harvest: ee.Date,
    bin_idx0: int,
    n_bins: int = 12,
    window_days: float = 14.0,
) -> Tuple[ee.Date, ee.Date]:
    """
    Return a (start, end) ee.Date pair: a `window_days`-wide window centered
    on the midpoint of equal segment `bin_idx0` (0-indexed) of [plant, harvest]
    split into `n_bins` equal-length segments.
    """
    seg_len = harvest.difference(plant, "day").divide(n_bins)
    midpoint = plant.advance(seg_len.multiply(bin_idx0 + 0.5), "day")
    half = window_days / 2.0
    return midpoint.advance(-half, "day"), midpoint.advance(half, "day")


def apply_bin_gapfill(
    bin_dicts: List[ee.Dictionary],
    rename_fn: Callable[[int], "ee.List"],
    n_bins: int = 12,
) -> List[ee.Image]:
    """
    bin_dicts[i] = ee.Dictionary({'img': ee.Image, 'empty': ee.Number(0/1)}).

    Empty bins are replaced by the mean of their immediate already-computed
    neighbor bin(s) (edge bins fall back to their single available
    neighbor). Returns `n_bins` renamed ee.Image objects, ready for
    ee.ImageCollection.fromImages(...).toBands().
    """

    def _img(i):
        return ee.Image(ee.Dictionary(bin_dicts[i]).get("img"))

    out = []
    for i in range(n_bins):
        img_i = _img(i)
        empty_i = ee.Number(ee.Dictionary(bin_dicts[i]).get("empty"))

        if i == 0:
            neighbor = _img(1)
        elif i == n_bins - 1:
            neighbor = _img(n_bins - 2)
        else:
            neighbor = ee.ImageCollection([_img(i - 1), _img(i + 1)]).mean()

        filled = ee.Image(ee.Algorithms.If(empty_i.eq(1), neighbor, img_i))
        out.append(filled.rename(rename_fn(i + 1)).toFloat())

    return out
