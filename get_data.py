"""
Tile a user shapefile and download a selected datasource for each tile.

Usage examples:
  python get_data.py --shp ./ROI/sample.shp --year 2024 --source s2 --out ./data/test_outputs
  python get_data.py --shp ./ROI/sample.shp --year 2016 --source landsat
  python get_data.py --shp ./ROI/sample.shp --year 2024 --source era5
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List, Union

import rasterio

# ---------------------------------------------------------------------
# Your project imports (adjust module paths if needed)
# ---------------------------------------------------------------------
from utils.pysentinel import S2GEEDownloader, S1GEEDownloader
from utils.pylandsat import LandsatGEEDownloader
from utils.pysatellite import (
    ERA5GEEDownloader,
    EsriLULCMaskDownloader,
    ESAWorldCoverMaskDownloader,
    AlphaEmbeddingDownloader,
    SRTMDownloader,
)

from utils.shape_tiler import ShapefileTiler
from utils.crop_calendar import read_plant_harvest, read_all_calendars, read_row0_field


# =========================
# DEFAULT CONFIG
# =========================
DEFAULT_CONFIGS = {
    "credentials_path": "./ee-rsai-service-account.json",
    "service_account": "fanapanomaly@fanapanomaly.iam.gserviceaccount.com",
}

# Sources whose download window comes from each polygon's own 'plant'/
# 'harvest' shapefile columns rather than a CLI-wide --year.
CROP_CAL_SOURCES = {"s2", "s1", "landsat", "era5"}

# The five candidate crops evaluated for every training parcel of a
# crop-calendar source: one tile per (parcel, candidate), cropped to that
# candidate's own phenological window, labeled 1 iff candidate == actual.
CANDIDATE_CROPS = ["barley", "corn", "kolza", "rice", "wheat"]

MANIFEST_FIELDS = [
    "path",
    "source",
    "actual_crop",
    "candidate_crop",
    "target",
    "parcel_id",
    "poly_idx",
    "tile_tag",
    "province",
    "plant",
    "harvest",
    "season_days",
    "start_month",
]


# =========================
# Helpers
# =========================
def _is_valid_tif(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with rasterio.open(path) as ds:
            _ = ds.profile
        return True
    except Exception:
        return False


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
    actual: Optional[str] = None,
    candidate: Optional[str] = None,
    suffix: str = ".tif",
) -> str:
    y = f"{year}" if year is not None else "na"
    if actual is not None and candidate is not None:
        # e.g. s2_y2018_rice_p000000_full_cwheat.tif -- 6-digit poly_idx
        # since candidate-crop shapefiles run up to ~90k parcels (wheat).
        return f"{source}_y{y}_{actual}_p{poly_idx:06d}_{tile_tag}_c{candidate}{suffix}"
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
    year: Optional[int],
    plant=None,
    harvest=None,
) -> None:
    source = source.lower().strip()

    if source in CROP_CAL_SOURCES:
        dl.download_from_shapefile(
            shp_path=tile_shp_path,
            out_tif=out_tif,
            plant=plant,
            harvest=harvest,
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
    year: Optional[int],
    source: str,
    out_root: Path,
    temp_dir: Path,
    max_pixels: int = 1024,
    s2_bands: Union[str, List[str]] = "all",
    limit: Optional[int] = None,
    start_after_poly_idx: int = -1,
    actual_crop: Optional[str] = None,
    candidates: Optional[List[str]] = None,
    manifest_dir: Optional[Path] = None,
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

    is_candidate_run = source in CROP_CAL_SOURCES
    if is_candidate_run:
        if not actual_crop:
            raise ValueError(f"--actual-crop is required for crop-calendar source '{source}'")
        if not manifest_dir:
            raise ValueError(f"--manifest-dir is required for crop-calendar source '{source}'")
        candidates = candidates or CANDIDATE_CROPS
        manifest_dir = Path(manifest_dir)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"{source}_{actual_crop}_w{start_after_poly_idx}.csv"
        manifest_file = open(manifest_path, "w", newline="")
        manifest_writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_FIELDS)
        manifest_writer.writeheader()
    else:
        manifest_path = None
        manifest_file = None
        manifest_writer = None

    dl = _build_downloader(source=source, out_dir=out_dir, configs=DEFAULT_CONFIGS, s2_bands=s2_bands)

    tiler = ShapefileTiler(
        shp_path=str(shp_path),
        max_pixels=max_pixels,
        temp_dir=str(temp_dir),
        start_after_poly_idx=start_after_poly_idx,
        limit=limit,
    )

    total = ok = fail = skipped = 0
    started_processing = start_after_poly_idx < 0
    failed_rois: set[int] = set()
    failed_tiles_by_roi: Dict[int, List[str]] = {}

    print(f"\n=== TILED DOWNLOAD START ===")
    print(f"Source      : {source}")
    print(f"Year        : {year}")
    print(f"Input SHP   : {shp_path}")
    print(f"Out dir     : {out_dir}")
    print(f"Temp tiles  : {temp_dir}")
    print(f"Max pixels  : {max_pixels}")
    if limit is not None:
        print(f"Poly limit  : {limit}")
    if start_after_poly_idx >= 0:
        print(f"Start after poly_idx : {start_after_poly_idx}")
    if source == "s2":
        print(f"S2 bands    : {s2_bands}")
    if is_candidate_run:
        print(f"Actual crop : {actual_crop}")
        print(f"Candidates  : {candidates}")
        print(f"Manifest    : {manifest_path}")

    seen_polys: set[int] = set()

    try:
        for i, item in enumerate(tiler, 1):
            poly_idx_raw = _safe_int(item.get("poly_idx"), default=i)
            if limit is not None and len(seen_polys) >= limit and poly_idx_raw not in seen_polys:
                break
            seen_polys.add(poly_idx_raw)
            tile_shp = item["shp_path"]

            poly_idx, tile_tag = _normalize_tile_id(item, fallback_i=i)

            if not started_processing:
                if poly_idx > start_after_poly_idx:
                    started_processing = True
                    print(f"\nResume point reached: poly_idx {poly_idx} > {start_after_poly_idx}. Starting downloads.")
                else:
                    continue

            print("\nProcessing tile:")
            print(f"  Tile shp     : {tile_shp}")
            print(f"  Polygon idx  : {item.get('poly_idx')} -> {poly_idx}")
            print(f"  Tile idx     : {item.get('tile_idx')} -> {tile_tag}")
            print(f"  CRS          : {item.get('crs')}")

            if not is_candidate_run:
                total += 1
                t0 = time.time()
                try:
                    out_name = _mk_out_name(source=source, year=year, poly_idx=poly_idx, tile_tag=tile_tag)
                    out_path = out_dir / out_name

                    if _is_valid_tif(out_path):
                        ok += 1
                        skipped += 1
                        print(f"  Already downloaded and valid, skipping: {out_name}")
                        continue

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
                continue

            # --- crop-calendar candidate loop: one tile per (parcel, candidate) ---
            try:
                cals = read_all_calendars(tile_shp, CANDIDATE_CROPS)
                province = read_row0_field(tile_shp, "province")
            except Exception as e:
                total += 1
                fail += 1
                failed_rois.add(poly_idx)
                failed_tiles_by_roi.setdefault(poly_idx, []).append(f"{tile_tag}:<calendar-read>")
                print(f"  ❌ Failed to read calendars for tile: {e}")
                continue

            parcel_id = f"{actual_crop}_{poly_idx:06d}"

            for candidate in candidates:
                cal = cals.get(candidate)
                if cal is None:
                    # No calendar for this (parcel, candidate) pair: skip only
                    # this pair, never drop the parcel or other candidates.
                    continue

                total += 1
                t0 = time.time()
                plant_dt, harvest_dt = cal
                tile_year = plant_dt.year

                out_name = _mk_out_name(
                    source=source, year=tile_year, poly_idx=poly_idx, tile_tag=tile_tag,
                    actual=actual_crop, candidate=candidate,
                )
                out_path = out_dir / out_name

                manifest_row = {
                    "path": f"{source}/{out_name}",
                    "source": source,
                    "actual_crop": actual_crop,
                    "candidate_crop": candidate,
                    "target": int(actual_crop == candidate),
                    "parcel_id": parcel_id,
                    "poly_idx": poly_idx,
                    "tile_tag": tile_tag,
                    "province": province,
                    "plant": plant_dt.date().isoformat(),
                    "harvest": harvest_dt.date().isoformat(),
                    "season_days": (harvest_dt - plant_dt).days,
                    "start_month": plant_dt.month - 1,
                }

                if _is_valid_tif(out_path):
                    ok += 1
                    skipped += 1
                    print(f"  Already downloaded and valid, skipping: {out_name}")
                    manifest_writer.writerow(manifest_row)
                    manifest_file.flush()
                    continue

                try:
                    _download_one_tile(
                        dl=dl,
                        source=source,
                        tile_shp_path=tile_shp,
                        out_tif=out_name,
                        year=tile_year,
                        plant=plant_dt,
                        harvest=harvest_dt,
                    )
                    dt = time.time() - t0
                    ok += 1
                    print(f"  ✅ Saved: {out_dir / out_name}  ({dt:.2f}s)")
                    manifest_writer.writerow(manifest_row)
                    manifest_file.flush()
                except Exception as e:
                    dt = time.time() - t0
                    fail += 1
                    failed_rois.add(poly_idx)
                    failed_tiles_by_roi.setdefault(poly_idx, []).append(f"{tile_tag}:{candidate}")
                    print(f"  ❌ Failed candidate '{candidate}' (after {dt:.2f}s): {e}")
    finally:
        if manifest_file is not None:
            manifest_file.close()

    print("\n=== SUMMARY ===")
    print(f"Total tiles : {total}")
    print(f"Success     : {ok}")
    print(f"Failed      : {fail}")
    print(f"Skipped     : {skipped}")
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
    p.add_argument(
        "--year",
        required=False,
        default=None,
        type=int,
        help=(
            "Year for masks/embedding sources (esri_lulc, worldcover, embedding). "
            "Not used for s2/s1/landsat/era5 - those read their download window "
            "from each polygon's own 'plant'/'harvest' shapefile columns."
        ),
    )
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
    p.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Max number of polygons to download (default: all)",
    )
    p.add_argument(
        "--start-after-poly-idx",
        type=int,
        default=-1,
        help="Only process tiles with poly_idx greater than this value (for resuming).",
    )
    p.add_argument(
        "--actual-crop",
        default=None,
        type=str,
        choices=CANDIDATE_CROPS,
        help=(
            "Actual crop of the input shapefile's parcels. Required for "
            "crop-calendar sources (s2, s1, landsat, era5): each parcel is "
            "downloaded once per --candidates crop, cropped to that "
            "candidate's own calendar window, labeled 1 iff candidate == actual-crop."
        ),
    )
    p.add_argument(
        "--candidates",
        nargs="+",
        default=None,
        choices=CANDIDATE_CROPS,
        help=f"Candidate crops to evaluate per parcel (default: all of {CANDIDATE_CROPS}).",
    )
    p.add_argument(
        "--manifest-dir",
        default=None,
        type=str,
        help="Directory for this worker's manifest CSV. Required for crop-calendar sources.",
    )

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.source in {"esri_lulc", "worldcover", "embedding"} and args.year is None:
        raise SystemExit(f"--year is required for source '{args.source}'")

    if args.source in CROP_CAL_SOURCES:
        if not args.actual_crop:
            raise SystemExit(f"--actual-crop is required for source '{args.source}'")
        if not args.manifest_dir:
            raise SystemExit(f"--manifest-dir is required for source '{args.source}'")

    run_tiled_download(
        shp_path=Path(args.shp),
        year=args.year,
        source=args.source,
        out_root=Path(args.out),
        temp_dir=Path(args.temp),
        max_pixels=args.max_pixels,
        s2_bands=args.s2_bands,
        limit=args.limit,
        start_after_poly_idx=args.start_after_poly_idx,
        actual_crop=args.actual_crop,
        candidates=args.candidates,
        manifest_dir=Path(args.manifest_dir) if args.manifest_dir else None,
    )
