"""
Tile a user shapefile and download a selected datasource for each tile.

Usage examples:
  python get_data.py --shp ./ROI/sample.shp --year 2024 --source s2 --out ./data/test_outputs
  python get_data.py --shp ./ROI/sample.shp --year 2016 --source landsat
  python get_data.py --shp ./ROI/sample.shp --year 2024 --source era5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List, Union

# ---------------------------------------------------------------------
# Your project imports (adjust module paths if needed)
# ---------------------------------------------------------------------
from data_sources.pysentinel import S2GEEDownloader, S1GEEDownloader
from data_sources.pylandsat import LandsatGEEDownloader
from data_sources.pysatellite import (
    ERA5GEEDownloader,
    EsriLULCMaskDownloader,
    ESAWorldCoverMaskDownloader,
    AlphaEmbeddingDownloader,
    SRTMDownloader,
)

from shape_tiler import ShapefileTiler


# =========================
# DEFAULT CONFIG
# =========================
DEFAULT_CONFIGS = {
    "credentials_path": "./ee-rsai-service-account.json",
    "service_account": "fanapanomaly@fanapanomaly.iam.gserviceaccount.com",
}


# =========================
# Helpers
# =========================
def _ensure_shp_exists(shp_path: Path) -> None:
    if not shp_path.exists():
        raise FileNotFoundError(f"Shapefile not found: {shp_path}")
    for ext in [".dbf", ".shx", ".prj"]:
        p = shp_path.with_suffix(ext)
        if not p.exists():
            print(f"⚠️ Warning: missing sidecar {p.name} next to {shp_path.name}")


def _safe_int(x: Any, default: int) -> int:
    if x is None:
        return default
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def _normalize_tile_id(item: Dict[str, Any], fallback_i: int) -> Tuple[int, str]:
    """
    Normalize tiler outputs across different ShapefileTiler versions.

    Returns:
      poly_idx (int)
      tile_tag (str): either "full" or "t{ty}_{tx}" / "t####" depending on available info
    """
    poly_idx = _safe_int(item.get("poly_idx"), default=fallback_i)

    raw_tile = item.get("tile_idx")

    # Case 1: no tiling happened (whole polygon)
    if raw_tile is None:
        return poly_idx, "full"

    # Case 2: tile_idx is a tuple like (ty, tx)
    if isinstance(raw_tile, (tuple, list)) and len(raw_tile) == 2:
        ty = _safe_int(raw_tile[0], default=0)
        tx = _safe_int(raw_tile[1], default=0)
        return poly_idx, f"t{ty:03d}_{tx:03d}"

    # Case 3: tile_idx is a single number
    tile_i = _safe_int(raw_tile, default=fallback_i)
    return poly_idx, f"t{tile_i:04d}"


def _mk_out_name(
    source: str,
    year: Optional[int],
    poly_idx: int,
    tile_tag: str,
    suffix: str = ".tif",
) -> str:
    y = f"{year}" if year is not None else "na"
    return f"{source}_y{y}_p{poly_idx:04d}_{tile_tag}{suffix}"


def _build_downloader(
    source: str,
    out_dir: Path,
    configs: Dict[str, str],
    s2_bands: Union[str, List[str]] = "all",
):
    source = source.lower().strip()

    common = dict(
        credentials_path=configs["credentials_path"],
        service_account=configs["service_account"],
        output_dir=out_dir,
    )

    if source == "s2":
        return S2GEEDownloader(**common, bands=s2_bands)
    if source == "s1":
        return S1GEEDownloader(**common)
    if source == "landsat":
        return LandsatGEEDownloader(**common)
    if source == "era5":
        return ERA5GEEDownloader(**common)
    if source == "esri_lulc":
        return EsriLULCMaskDownloader(**common)
    if source == "worldcover":
        return ESAWorldCoverMaskDownloader(**common)
    if source == "embedding":
        return AlphaEmbeddingDownloader(**common)
    if source == "srtm":
        return SRTMDownloader(**common)

    raise ValueError(
        f"Unknown source '{source}'. Choose from: "
        f"s2, s1, landsat, era5, esri_lulc, worldcover, embedding, srtm"
    )


def _download_one_tile(
    dl,
    source: str,
    tile_shp_path: str,
    out_tif: str,
    year: int,
) -> None:
    source = source.lower().strip()

    if source in {"s2", "s1", "landsat", "era5"}:
        dl.download_from_shapefile(
            shp_path=tile_shp_path,
            out_tif=out_tif,
            season_year=year,
        )
        return

    if source == "esri_lulc":
        dl.download_from_shapefile(
            shp_path=tile_shp_path,
            out_tif=out_tif,
            year=year,
        )
        return

    if source == "worldcover":
        dl.download_from_shapefile(
            shp_path=tile_shp_path,
            out_tif=out_tif,
            year=2021,
        )
        return

    if source == "embedding":
        dl.download_from_shapefile(
            shp_path=tile_shp_path,
            out_tif=out_tif,
            year=year,
        )
        return

    if source == "srtm":
        dl.download_from_shapefile(
            shp_path=tile_shp_path,
            out_tif=out_tif,
        )
        return

    raise ValueError(f"Unsupported source '{source}' in _download_one_tile()")


def run_tiled_download(
    shp_path: Path,
    year: int,
    source: str,
    out_root: Path,
    temp_dir: Path,
    max_pixels: int = 1024,
    s2_bands: Union[str, List[str]] = "all",
) -> None:
    _ensure_shp_exists(shp_path)
    
    
    if s2_bands == ["all"]:
        s2_bands = "all"
    else:
        s2_bands = s2_bands

    source = source.lower().strip()
    out_dir = out_root / source
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    dl = _build_downloader(source=source, out_dir=out_dir, configs=DEFAULT_CONFIGS, s2_bands=s2_bands)

    tiler = ShapefileTiler(
        shp_path=str(shp_path),
        max_pixels=max_pixels,
        temp_dir=str(temp_dir),
    )

    total = ok = fail = 0
    failed_rois: set[int] = set()
    failed_tiles_by_roi: Dict[int, List[str]] = {}

    print(f"\n=== TILED DOWNLOAD START ===")
    print(f"Source      : {source}")
    print(f"Year        : {year}")
    print(f"Input SHP   : {shp_path}")
    print(f"Out dir     : {out_dir}")
    print(f"Temp tiles  : {temp_dir}")
    print(f"Max pixels  : {max_pixels}")
    if source == "s2":
        print(f"S2 bands    : {s2_bands}")

    for i, item in enumerate(tiler, 1):
        total += 1
        tile_shp = item["shp_path"]

        poly_idx, tile_tag = _normalize_tile_id(item, fallback_i=i)


        print("\nProcessing tile:")
        print(f"  Tile shp     : {tile_shp}")
        print(f"  Polygon idx  : {item.get('poly_idx')} -> {poly_idx}")
        print(f"  Tile idx     : {item.get('tile_idx')} -> {tile_tag}")
        print(f"  CRS          : {item.get('crs')}")

        out_name = _mk_out_name(source=source, year=year, poly_idx=poly_idx, tile_tag=tile_tag)

        t0 = time.time()
        try:
            _download_one_tile(
                dl=dl,
                source=source,
                tile_shp_path=tile_shp,
                out_tif=out_name,
                year=year,
            )
            dt = time.time() - t0
            ok += 1
            print(f"  ✅ Saved: {out_dir / out_name}  ({dt:.2f}s)")
        except Exception as e:
            dt = time.time() - t0
            fail += 1
            failed_rois.add(poly_idx)
            failed_tiles_by_roi.setdefault(poly_idx, []).append(tile_tag)
            print(f"  ❌ Failed tile (after {dt:.2f}s): {e}")

    print("\n=== SUMMARY ===")
    print(f"Total tiles : {total}")
    print(f"Success     : {ok}")
    print(f"Failed      : {fail}")
    if failed_rois:
        failed_sorted = sorted(failed_rois)
        print(f"Failed ROI poly_idx values ({len(failed_sorted)}): {failed_sorted}")
        print("Failed tiles by ROI:")
        for roi_idx in failed_sorted:
            tiles = failed_tiles_by_roi.get(roi_idx, [])
            tiles_str = ", ".join(tiles) if tiles else "(unknown tiles)"
            print(f"  - ROI {roi_idx}: {tiles_str}")
    print("=== DONE ===\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--shp", required=True, type=str, help="Path to input shapefile (.shp)")
    p.add_argument("--year", required=True, type=int, help="Season year (or year for masks/embedding)")
    p.add_argument(
        "--source",
        required=True,
        type=str,
        choices=["s2", "s1", "landsat", "era5", "esri_lulc", "worldcover", "embedding", "srtm"],
        help="Datasource to download",
    )
    p.add_argument("--out", default="./data/dataset", type=str, help="Output root directory")
    p.add_argument("--temp", default="./tmp_tiles", type=str, help="Temp directory for tiled shapefiles")
    p.add_argument("--max_pixels", default=1024, type=int, help="Max pixels for tiler")
    p.add_argument(
    "--s2-bands",
    nargs="+",
    default=["all"],
    help="Sentinel-2 bands (e.g. red green blue nir) or 'all'",
)

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    


    run_tiled_download(
        shp_path=Path(args.shp),
        year=args.year,
        source=args.source,
        out_root=Path(args.out),
        temp_dir=Path(args.temp),
        max_pixels=args.max_pixels,
        s2_bands=args.s2_bands,
    )
