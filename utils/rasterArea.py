#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform


def iter_rasters(root: Path, exts: Tuple[str, ...]) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def local_laea_equal_area_crs(src_crs: CRS, bounds: Tuple[float, float, float, float]) -> CRS:
    wgs84 = CRS.from_epsg(4326)
    tr = Transformer.from_crs(src_crs, wgs84, always_xy=True)

    minx, miny, maxx, maxy = bounds
    lon_min, lat_min = tr.transform(minx, miny)
    lon_max, lat_max = tr.transform(maxx, maxy)

    lon0 = 0.5 * (lon_min + lon_max)
    lat0 = 0.5 * (lat_min + lat_max)

    proj4 = f"+proj=laea +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    return CRS.from_proj4(proj4)


def is_meter_projected(crs: CRS) -> bool:
    if crs is None or not crs.is_projected:
        return False
    axis = crs.axis_info
    if not axis or len(axis) < 2:
        return False
    u0 = (axis[0].unit_name or "").lower()
    u1 = (axis[1].unit_name or "").lower()
    return ("metre" in u0 or "meter" in u0) and ("metre" in u1 or "meter" in u1)


def pixel_area_from_transform(transform) -> float:
    return abs(transform.a * transform.e - transform.b * transform.d)


def window_grid(width: int, height: int, tile: int) -> Iterable[Tuple[int, int, int, int]]:
    for r0 in range(0, height, tile):
        h = min(tile, height - r0)
        for c0 in range(0, width, tile):
            w = min(tile, width - c0)
            yield c0, r0, w, h


def count_class_pixels(data: np.ma.MaskedArray, target_class: int) -> int:
    # Avoid data.compressed() to prevent extra allocations.
    if np.ma.isMaskedArray(data):
        m = data.mask
        if m is np.False_:
            return int(np.count_nonzero(data.data == target_class))
        return int(np.count_nonzero((data.data == target_class) & (~m)))
    return int(np.count_nonzero(data == target_class))


def compute_class_area_for_raster(path: Path, target_class: int, tile: int, force_equal_area: bool) -> Tuple[float, int]:
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"{path}: missing CRS; cannot compute correct areas")

        need_equal_area = force_equal_area or src.crs.is_geographic or not is_meter_projected(src.crs)

        if not need_equal_area:
            px_area = pixel_area_from_transform(src.transform)
            npx = 0
            for c0, r0, w, h in window_grid(src.width, src.height, tile):
                win = rasterio.windows.Window(c0, r0, w, h)
                arr = src.read(1, window=win, masked=True)
                npx += count_class_pixels(arr, target_class)
            return px_area, npx

        # Warp to local equal-area CRS (LAEA) and count there.
        b = src.bounds
        dst_crs = local_laea_equal_area_crs(src.crs, (b.left, b.bottom, b.right, b.top))

        # Estimate destination resolution at raster center to keep similar sampling.
        tf = src.transform
        cx, cy = src.xy(src.height // 2, src.width // 2, offset="center")
        col_step = (tf.a, tf.d)
        row_step = (tf.b, tf.e)

        tr = Transformer.from_crs(src.crs, dst_crs, always_xy=True)
        x0, y0 = tr.transform(cx, cy)
        x1, y1 = tr.transform(cx + col_step[0], cy + col_step[1])
        x2, y2 = tr.transform(cx + row_step[0], cy + row_step[1])

        resx = float(np.hypot(x1 - x0, y1 - y0))
        resy = float(np.hypot(x2 - x0, y2 - y0))

        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            b.left, b.bottom, b.right, b.top,
            resolution=(resx, resy),
        )

        with WarpedVRT(
            src,
            crs=dst_crs,
            transform=dst_transform,
            width=dst_width,
            height=dst_height,
            resampling=Resampling.nearest,
        ) as vrt:
            px_area = pixel_area_from_transform(vrt.transform)
            npx = 0
            for c0, r0, w, h in window_grid(vrt.width, vrt.height, tile):
                win = rasterio.windows.Window(c0, r0, w, h)
                arr = vrt.read(1, window=win, masked=True)
                npx += count_class_pixels(arr, target_class)
            return px_area, npx


def write_csv(out_csv: Path, rows: Iterable[Tuple[str, int, float]]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["raster", "class", "area_m2", "area_ha", "area_km2"])
        for raster_name, cls, area_m2 in rows:
            w.writerow([raster_name, cls, area_m2, area_m2 / 10_000.0, area_m2 / 1_000_000.0])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute area (m²) for a single categorical class for rasters in a directory; uses equal-area warping when needed."
    )
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("class_area.csv"))
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--class", dest="target_class", type=int, default=4, help="Class value to compute area for (default: 4)")
    ap.add_argument("--ext", action="append", default=[".tif", ".tiff"], help="File extension(s) to include (repeatable)")
    ap.add_argument("--force-equal-area", action="store_true", help="Always warp to local equal-area CRS")
    args = ap.parse_args()

    exts = tuple(e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.ext)

    out_rows = []
    for p in iter_rasters(args.input_dir, exts):
        px_area, npx = compute_class_area_for_raster(
            p, target_class=args.target_class, tile=args.tile, force_equal_area=args.force_equal_area
        )
        area_m2 = float(npx) * px_area
        out_rows.append((p.name, args.target_class, area_m2))

    write_csv(args.out, out_rows)


if __name__ == "__main__":
    main()
