"""
Split a shapefile into N roughly equal sub-shapefiles (chunks).

Each chunk contains a contiguous slice of features so spatially
adjacent polygons stay together. Chunk files are written to
out_dir/chunk_0000.shp, chunk_0001.shp, ...
"""

from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd


def chunk_shapefile(shp_path: Path, n_chunks: int, out_dir: Path) -> list[Path]:
    """
    Split shp_path into n_chunks sub-shapefiles written under out_dir.

    Returns the list of written chunk paths (may be fewer than n_chunks
    if the shapefile has fewer features than requested chunks).
    """
    shp_path = Path(shp_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(shp_path)
    total = len(gdf)
    if total == 0:
        raise ValueError(f"Shapefile {shp_path} has no features.")

    n_chunks = min(n_chunks, total)
    chunk_size = math.ceil(total / n_chunks)

    paths: list[Path] = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, total)
        if start >= total:
            break

        chunk_gdf = gdf.iloc[start:end].copy().reset_index(drop=True)
        out_path = out_dir / f"chunk_{i:04d}.shp"
        chunk_gdf.to_file(out_path, driver="ESRI Shapefile")
        paths.append(out_path)
        print(f"  chunk {i:04d}: features {start}–{end - 1} → {out_path.name}")

    return paths
