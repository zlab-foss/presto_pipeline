import math
import tempfile
from pathlib import Path
from typing import Dict, Iterator, Tuple

import geopandas as gpd
from shapely.geometry import box


class ShapefileTiler:
    """
    Utility class to iterate over sub-ROIs derived from a shapefile.

    Features:
    ----------
    1. If the input shapefile contains multiple polygons (features),
       it will treat each feature as a separate ROI.

    2. For each polygon, if its extent (in meters) exceeds the
       configured max tile size (e.g. 2500 x 2500 pixels at 10 m),
       the polygon is tiled into smaller rectangular tiles. Each tile
       is intersected with the polygon, and only non-empty intersections
       are kept.

    3. For each ROI or tile, a temporary shapefile is written containing
       **exactly one polygon**, preserving the original attributes.
       The iterator yields a small dict with metadata, including the
       temp shapefile path.

    Typical usage:
    --------------
        tiler = ShapefileTiler(
            shp_path="data/ROI/big_multi.shp",
            pixel_size=10.0,
            max_pixels=2500,
            temp_dir="tmp_tiles"
        )

        for item in tiler:
            print(item["shp_path"], item["poly_idx"], item["tile_idx"])

    Yields:
    -------
    Each iteration yields a dict with:
        {
          "shp_path": Path,         # path to temp shapefile (one polygon)
          "poly_idx": int,          # index of original polygon (0-based)
          "tile_idx": Optional[Tuple[int,int]],  # (row, col) tile index or None
          "geometry": shapely geometry (original CRS),
          "crs": original CRS
        }

    Notes:
    ------
    - The tiling is done in a metric CRS (EPSG:3857) for approximate
      meter-based sizes. The resulting geometries are reprojected back
      to the original CRS before writing.
    - Temp directory is not automatically deleted; you can remove it
      when done.
    """

    def __init__(
        self,
        shp_path: str | Path,
        max_pixels: int = 2500,
        temp_dir: str | Path | None = None,
        start_after_poly_idx: int = -1,
        limit: int | None = None,
    ):
        """
        Parameters
        ----------
        shp_path : str or Path
            Path to the input shapefile containing one or more polygons.
        max_pixels : int
            Maximum number of pixels along one dimension. If a polygon's
            width or height exceeds max_pixels * pixel_size (in meters),
            it will be tiled.
        temp_dir : str or Path or None
            Directory where temporary shapefiles will be written.
            If None, a new temporary directory will be created.
        start_after_poly_idx : int
            Only iterate (and write temp shapefiles for) polygons whose
            poly_idx is strictly greater than this value. Polygons at or
            below it are skipped *before* any file is written. This keeps
            parallel workers, which share one temp_dir, from writing the
            same temp shapefile path concurrently.
        limit : int or None
            Maximum number of polygons to iterate over. None means no limit.
        """
        self.shp_path = Path(shp_path)
        if not self.shp_path.exists():
            raise FileNotFoundError(f"Shapefile not found: {self.shp_path}")

        self.pixel_size = float(10)
        self.max_pixels = int(max_pixels)
        self.max_size_m = self.pixel_size * self.max_pixels  # e.g. 25000 m
        self.start_after_poly_idx = int(start_after_poly_idx)
        self.limit = limit

        if temp_dir is None:
            self.temp_root = Path(tempfile.mkdtemp(prefix="roi_tiles_"))
        else:
            self.temp_root = Path(temp_dir)
            self.temp_root.mkdir(parents=True, exist_ok=True)

        # Load shapefile
        self.gdf_orig = gpd.read_file(self.shp_path)
        if self.gdf_orig.empty:
            raise ValueError(f"Shapefile {self.shp_path} has no features.")

        if self.gdf_orig.crs is None:
            raise ValueError(f"Shapefile {self.shp_path} has no CRS defined.")

        self.orig_crs = self.gdf_orig.crs

        # For tiling, work in a metric CRS (Web Mercator or UTM)
        # Try to use appropriate UTM zone based on centroid
        try:
            self.gdf_metric = self._to_metric_crs(self.gdf_orig)
        except:
            # Fallback to Web Mercator
            self.gdf_metric = self.gdf_orig.to_crs(epsg=3857)

    def _to_metric_crs(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Convert GeoDataFrame to an appropriate metric CRS.
        Tries to use UTM zone based on centroid, falls back to Web Mercator.
        """
        # Use the midpoint of the overall bounding box as a representative
        # point. This is O(1) on the existing index and avoids a full
        # unary_union cascade over every polygon (which, for a nationwide
        # shapefile with tens of thousands of features, can take a very
        # long time just to pick a UTM zone).
        if gdf.crs.is_geographic:
            minx, miny, maxx, maxy = gdf.total_bounds
            lon, lat = (minx + maxx) / 2, (miny + maxy) / 2
        else:
            # Convert to lat/lon first
            gdf_latlon = gdf.to_crs(epsg=4326)
            minx, miny, maxx, maxy = gdf_latlon.total_bounds
            lon, lat = (minx + maxx) / 2, (miny + maxy) / 2
        
        # Calculate UTM zone
        utm_zone = int((lon + 180) / 6) + 1
        
        # Determine hemisphere (north/south)
        if lat >= 0:
            epsg_code = 32600 + utm_zone  # Northern hemisphere
        else:
            epsg_code = 32700 + utm_zone  # Southern hemisphere
        
        return gdf.to_crs(epsg=epsg_code)

    # ------------------------------------------------------------------
    #  Iterator interface
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[Dict]:
        """
        Iterate over all sub-ROIs (original polygons and/or tiles).

        For each ROI or tile, writes a temporary shapefile containing a
        single polygon and yields metadata as a dict.
        """
        # Loop over original polygons (use enumerate for proper indexing)
        seen_polys: set[int] = set()
        for poly_idx, (idx, row) in enumerate(self.gdf_metric.iterrows()):
            # Restrict to this worker's assigned range *before* writing any
            # temp shapefile, so parallel workers sharing one temp_dir never
            # write to the same path concurrently.
            if poly_idx <= self.start_after_poly_idx:
                continue
            if (
                self.limit is not None
                and len(seen_polys) >= self.limit
                and poly_idx not in seen_polys
            ):
                break
            seen_polys.add(poly_idx)

            geom_metric = row.geometry

            # Skip empty / invalid geometries
            if geom_metric is None or geom_metric.is_empty:
                continue
            
            # Ensure valid geometry
            if not geom_metric.is_valid:
                geom_metric = geom_metric.buffer(0)
                if not geom_metric.is_valid:
                    print(f"Warning: Skipping invalid geometry at index {poly_idx}")
                    continue

            bounds = geom_metric.bounds  # (minx, miny, maxx, maxy)
            minx, miny, maxx, maxy = bounds
            width_m = maxx - minx
            height_m = maxy - miny

            # Decide whether to tile or not
            if width_m <= self.max_size_m and height_m <= self.max_size_m:
                # No tiling needed, just write a single ROI shapefile
                yield from self._write_single_roi(row, poly_idx, idx)
            else:
                # Tiling required
                yield from self._yield_tiles_for_polygon(row, poly_idx, idx, bounds)

    # ------------------------------------------------------------------
    #  Internal writers
    # ------------------------------------------------------------------
    def _write_single_roi(self, row, poly_idx: int, original_idx) -> Iterator[Dict]:
        """
        Write a single-polygon shapefile for the given row (no tiling).
        """
        # Create a copy of the row to avoid modifying original
        row_copy = row.copy()
        
        # Convert single row back to original CRS
        single_gdf_metric = gpd.GeoDataFrame([row_copy], crs=self.gdf_metric.crs)
        single_gdf = single_gdf_metric.to_crs(self.orig_crs)

        # Build output path
        out_dir = self.temp_root / f"poly_{poly_idx:04d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"roi_poly_{poly_idx:04d}.shp"

        single_gdf.to_file(out_path, driver='ESRI Shapefile')

        yield {
            "shp_path": out_path,
            "poly_idx": poly_idx,
            "original_idx": original_idx,
            "tile_idx": None,
            "geometry": single_gdf.geometry.iloc[0],
            "crs": self.orig_crs,
        }

    def _yield_tiles_for_polygon(
        self,
        row,
        poly_idx: int,
        original_idx,
        bounds: Tuple[float, float, float, float],
    ) -> Iterator[Dict]:
        """
        Tile a single polygon (in metric CRS) if its bounding box exceeds
        the max tile size. Each tile is intersected with the polygon,
        and non-empty intersections are saved as separate shapefiles.
        """
        geom_metric = row.geometry
        minx, miny, maxx, maxy = bounds
        width_m = maxx - minx
        height_m = maxy - miny

        # Number of tiles in each direction
        nx = max(1, math.ceil(width_m / self.max_size_m))
        ny = max(1, math.ceil(height_m / self.max_size_m))

        print(f"Polygon {poly_idx}: Tiling into {ny} x {nx} = {ny*nx} tiles")
        print(f"  Bounds: width={width_m:.0f}m, height={height_m:.0f}m")

        tile_count = 0

        for iy in range(ny):
            tile_miny = miny + iy * self.max_size_m
            tile_maxy = min(tile_miny + self.max_size_m, maxy)

            for ix in range(nx):
                tile_minx = minx + ix * self.max_size_m
                tile_maxx = min(tile_minx + self.max_size_m, maxx)

                tile_box = box(tile_minx, tile_miny, tile_maxx, tile_maxy)

                # Intersect tile with polygon
                try:
                    tile_geom_metric = geom_metric.intersection(tile_box)
                except Exception as e:
                    print(f"  Warning: Intersection failed for tile ({iy},{ix}): {e}")
                    continue

                if tile_geom_metric.is_empty:
                    continue

                # Handle MultiPolygon results
                if tile_geom_metric.geom_type == 'GeometryCollection':
                    # Extract only Polygon/MultiPolygon from collection
                    from shapely.geometry import Polygon, MultiPolygon
                    polygons = [g for g in tile_geom_metric.geoms 
                               if isinstance(g, (Polygon, MultiPolygon))]
                    if not polygons:
                        continue
                    if len(polygons) == 1:
                        tile_geom_metric = polygons[0]
                    else:
                        tile_geom_metric = MultiPolygon(polygons)

                # Create a copy of attributes
                row_copy = row.copy()
                
                # Build GeoDataFrame for this tile in metric CRS
                tile_gdf_metric = gpd.GeoDataFrame([row_copy], crs=self.gdf_metric.crs)
                tile_gdf_metric.loc[tile_gdf_metric.index[0], 'geometry'] = tile_geom_metric

                # Reproject to original CRS
                try:
                    tile_gdf = tile_gdf_metric.to_crs(self.orig_crs)
                except Exception as e:
                    print(f"  Warning: Reprojection failed for tile ({iy},{ix}): {e}")
                    continue

                # Skip if empty after reprojection
                if tile_gdf.geometry.iloc[0].is_empty:
                    continue

                # Build output path
                out_dir = self.temp_root / f"poly_{poly_idx:04d}" / f"tile_{iy:03d}_{ix:03d}"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"roi_poly_{poly_idx:04d}_tile_{iy:03d}_{ix:03d}.shp"

                tile_gdf.to_file(out_path, driver='ESRI Shapefile')

                tile_count += 1

                yield {
                    "shp_path": out_path,
                    "poly_idx": poly_idx,
                    "original_idx": original_idx,
                    "tile_idx": (iy, ix),
                    "geometry": tile_gdf.geometry.iloc[0],
                    "crs": self.orig_crs,
                }
        
        print(f"  Generated {tile_count} valid tiles")

    def cleanup(self):
        """Remove temporary directory and all files"""
        import shutil
        if self.temp_root.exists():
            shutil.rmtree(self.temp_root)
            print(f"Cleaned up temporary directory: {self.temp_root}")


# Example usage
if __name__ == "__main__":
    # Example 1: Basic usage
    print("Example 1: Basic Usage")
    print("=" * 80)
    
    tiler = ShapefileTiler(
        shp_path="ROI/karkheh.shp",
        max_pixels=4096,
        temp_dir="tmp_tiles"
    )
    
    for item in tiler:
        print("\nProcessing:")
        print(f"  Shapefile: {item['shp_path']}")
        print(f"  Polygon index: {item['poly_idx']}")
        print(f"  Tile index: {item['tile_idx']}")
        print(f"  CRS: {item['crs']}")
        

    
    # Cleanup when done
    # tiler.cleanup()
    print("\n" + "=" * 80)