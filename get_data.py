"""
Tile a shapefile of ROIs and download EO data per tile from GEE.

Simplest usage — reads everything from config.yaml:
  python get_data.py

Override individual settings on the CLI (takes precedence over config):
  python get_data.py --source s2 --year 2024
  python get_data.py --config ./my_config.yaml --workers 8
"""

from __future__ import annotations
import argparse
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List, Union

import rasterio
import yaml


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


# =========================
# DEFAULT CONFIG
# =========================
_CONFIG_FILE = Path(__file__).parent / "config.yaml"

def _load_configs() -> Dict[str, str]:
    if _CONFIG_FILE.exists():
        with open(_CONFIG_FILE) as f:
            cfg = yaml.safe_load(f) or {}
        return {
            "credentials_path": cfg.get("credentials_path", "./ee-rsai-service-account.json"),
            "service_account": cfg.get("service_account", "fanapanomaly@fanapanomaly.iam.gserviceaccount.com"),
        }
    return {
        "credentials_path": "./ee-rsai-service-account.json",
        "service_account": "fanapanomaly@fanapanomaly.iam.gserviceaccount.com",
    }

DEFAULT_CONFIGS = _load_configs()


# =========================
# Helpers
# =========================
def _is_valid_tif(path: Path) -> bool:
    """Return True if path exists, is non-empty, and rasterio can open it."""
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
    suffix: str = ".tif",
) -> str:
    y = f"{year}" if year is not None else "na"
    return f"{source}_y{y}_p{poly_idx:04d}_{tile_tag}{suffix}"


def _build_downloader(
    source: str,
    out_dir: Path,
    configs: Dict[str, str],
    s2_bands: Union[str, List[str]] = "all",
    s1_bands: Union[str, List[str]] = "all",
    worldcover_scale: int = 10,
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
        return S1GEEDownloader(**common, bands=s1_bands)
    if source == "landsat":
        return LandsatGEEDownloader(**common)
    if source == "era5":
        return ERA5GEEDownloader(**common)
    if source == "esri_lulc":
        return EsriLULCMaskDownloader(**common)
    if source == "worldcover":
        return ESAWorldCoverMaskDownloader(**common, export_scale=worldcover_scale)
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


# =========================
# Tile worker (shared by sequential + parallel paths)
# =========================
# A task is a fully picklable tuple describing one tile to download.
#   (tile_shp, poly_idx_raw, tile_idx_raw, crs, poly_idx, tile_tag, year)
Task = Tuple[str, Any, Any, Any, int, str, int]


def _run_tile_task(dl, source: str, out_dir: Path, task: Task) -> Tuple[str, int, str]:
    tile_shp, poly_idx_raw, tile_idx_raw, crs, poly_idx, tile_tag, year = task
    out_name = _mk_out_name(source=source, year=year, poly_idx=poly_idx, tile_tag=tile_tag)
    out_path = out_dir / out_name

    print(f"\nProcessing tile:")
    print(f"  Tile shp     : {tile_shp}")
    print(f"  Polygon idx  : {poly_idx_raw} -> {poly_idx}")
    print(f"  Tile idx     : {tile_idx_raw} -> {tile_tag}")
    print(f"  CRS          : {crs}")

    if _is_valid_tif(out_path):
        print(f"  ⏭ Already downloaded and valid, skipping: {out_name}")
        return ("skip", poly_idx, tile_tag)

    t0 = time.time()
    try:
        _download_one_tile(dl=dl, source=source, tile_shp_path=tile_shp, out_tif=out_name, year=year)
        dt = time.time() - t0
        print(f"  ✅ Saved: {out_dir / out_name}  ({dt:.2f}s)")
        return ("ok", poly_idx, tile_tag)
    except Exception as e:
        dt = time.time() - t0
        print(f"  ❌ Failed tile (after {dt:.2f}s): {e}")
        return ("fail", poly_idx, tile_tag)


# Per-process state for the parallel path. Each worker process builds its own
# downloader (its own ee.Initialize / geedim asyncio loop / executor) exactly
# once, in the initializer — never shared across processes.
_WORKER: Dict[str, Any] = {}


def _init_worker(
    source: str,
    out_dir: str,
    configs: Dict[str, str],
    s2_bands: Union[str, List[str]],
    s1_bands: Union[str, List[str]],
    worldcover_scale: int,
) -> None:
    global _WORKER
    dl = _build_downloader(
        source=source,
        out_dir=Path(out_dir),
        configs=configs,
        s2_bands=s2_bands,
        s1_bands=s1_bands,
        worldcover_scale=worldcover_scale,
    )
    _WORKER = {"dl": dl, "source": source, "out_dir": Path(out_dir)}


def _worker_run(task: Task) -> Tuple[str, int, str]:
    return _run_tile_task(_WORKER["dl"], _WORKER["source"], _WORKER["out_dir"], task)


def collect_tile_tasks(
    shp_path: Path,
    year: int,
    temp_dir: Path,
    max_pixels: int = 1024,
    limit: Optional[int] = None,
    start_after_poly_idx: int = -1,
) -> Tuple[List[Task], int]:
    """
    Tile the ROI shapefile ONCE and return (tasks, skipped).

    Tiling depends only on the ROI + max_pixels, NOT on the data source, so the
    resulting tiles are shared by every source (s2, s1, ...). The temp tile
    shapefiles are written to ``temp_dir`` and read (read-only) by all source
    download processes.
    """
    _ensure_shp_exists(shp_path)
    temp_dir.mkdir(parents=True, exist_ok=True)

    tiler = ShapefileTiler(
        shp_path=str(shp_path),
        max_pixels=max_pixels,
        temp_dir=str(temp_dir),
    )

    print(f"\n=== TILING (shared across all sources) ===")
    print(f"Input SHP   : {shp_path}")
    print(f"Temp tiles  : {temp_dir}")
    print(f"Max pixels  : {max_pixels}")
    if limit is not None:
        print(f"Poly limit  : {limit}")
    print(f"Start after poly_idx : {start_after_poly_idx}")

    pending: List[Tuple[int, Dict[str, Any], int, str]] = []
    seen_polys: set[int] = set()
    started_processing = start_after_poly_idx < 0
    skipped = 0

    for i, item in enumerate(tiler, 1):
        poly_idx_raw = _safe_int(item.get("poly_idx"), default=i)
        if limit is not None and len(seen_polys) >= limit and poly_idx_raw not in seen_polys:
            break
        seen_polys.add(poly_idx_raw)

        poly_idx, tile_tag = _normalize_tile_id(item, fallback_i=i)

        if not started_processing:
            if poly_idx > start_after_poly_idx:
                started_processing = True
                print(
                    f"\nResume point reached: poly_idx {poly_idx} > {start_after_poly_idx}. Starting downloads."
                )
            else:
                skipped += 1
                print(
                    f"\nSkipping tile: poly_idx {poly_idx} is not greater than {start_after_poly_idx}"
                )
                continue

        pending.append((i, item, poly_idx, tile_tag))

    # Build picklable tasks (geedim sessions/ee objects are NOT picklable, so we
    # only pass plain data across the process boundary).
    tasks: List[Task] = [
        (
            item["shp_path"],
            item.get("poly_idx"),
            item.get("tile_idx"),
            item.get("crs"),
            poly_idx,
            tile_tag,
            year,
        )
        for (i, item, poly_idx, tile_tag) in pending
    ]

    print(f"Collected {len(tasks)} tiles ({skipped} skipped by resume).\n")
    return tasks, skipped


def run_tiled_download(
    source: str,
    out_root: Path,
    tasks: List[Task],
    skipped: int = 0,
    year: Optional[int] = None,
    s2_bands: Union[str, List[str]] = "all",
    s1_bands: Union[str, List[str]] = "all",
    worldcover_scale: int = 10,
    workers: int = 4,
) -> None:
    """Download all pre-computed ``tasks`` for a single ``source``."""
    if s2_bands == ["all"]:
        s2_bands = "all"
    if s1_bands == ["all"]:
        s1_bands = "all"

    source = source.lower().strip()
    out_dir = out_root / source
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== DOWNLOAD START ===")
    print(f"Source      : {source}")
    print(f"Year        : {year}")
    print(f"Out dir     : {out_dir}")
    print(f"Tiles       : {len(tasks)}")
    print(f"Workers     : {workers}")
    if source == "s2":
        print(f"S2 bands    : {s2_bands}")
    if source == "s1":
        print(f"S1 bands    : {s1_bands}")
    if source == "worldcover":
        print(f"WorldCover scale : {worldcover_scale}")

    total = skipped + len(tasks)
    ok = fail = 0
    failed_rois: set[int] = set()
    failed_tiles_by_roi: Dict[int, List[str]] = {}

    def _record(result: Tuple[str, int, str]) -> None:
        nonlocal ok, fail, skipped
        status, p_idx, t_tag = result
        if status == "skip":
            ok += 1
            skipped += 1
        elif status == "ok":
            ok += 1
        else:
            fail += 1
            failed_rois.add(p_idx)
            failed_tiles_by_roi.setdefault(p_idx, []).append(t_tag)

    if workers <= 1:
        # Sequential: build one downloader in-process and reuse it.
        dl = _build_downloader(
            source=source,
            out_dir=out_dir,
            configs=DEFAULT_CONFIGS,
            s2_bands=s2_bands,
            s1_bands=s1_bands,
            worldcover_scale=worldcover_scale,
        )
        for task in tasks:
            _record(_run_tile_task(dl, source, out_dir, task))
    else:
        # Parallel: each worker process gets its OWN ee.Initialize / geedim
        # asyncio loop / executor. "spawn" guarantees clean, isolated state and
        # avoids inheriting the parent's ee/asyncio globals (which is what broke
        # the old thread-based approach).
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(source, str(out_dir), DEFAULT_CONFIGS, s2_bands, s1_bands, worldcover_scale),
        ) as pool:
            futures = {pool.submit(_worker_run, task): task for task in tasks}
            for fut in as_completed(futures):
                _record(fut.result())

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


_VALID_SOURCES = ["s2", "s1", "landsat", "era5", "esri_lulc", "worldcover", "embedding", "srtm"]


def _load_full_config(config_path: Path) -> Dict[str, Any]:
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default="./config.yaml", metavar="PATH",
                   help="Config file (default: ./config.yaml). CLI flags override it.")
    p.add_argument("--shp", type=str, help="Path to input shapefile (.shp)")
    p.add_argument("--year", type=int, help="Season year")
    p.add_argument("--source", type=str, choices=_VALID_SOURCES,
                   help="Single datasource (overrides config 'sources'; runs only this one)")
    p.add_argument("--out", type=str, help="Output root directory")
    p.add_argument("--temp", type=str, help="Temp directory for tiled shapefiles")
    p.add_argument("--max_pixels", type=int, help="Max pixels per tile")
    p.add_argument("--s2-bands", nargs="+", metavar="BAND",
                   help="Sentinel-2 bands (e.g. red green blue nir) or 'all'")
    p.add_argument("--s1-bands", nargs="+", metavar="BAND",
                   help="Sentinel-1 bands: VV VH or both")
    p.add_argument("--worldcover-scale", type=int,
                   help="ESA WorldCover export resolution (m/pixel)")
    p.add_argument("--limit", type=int, help="Max polygons to process")
    p.add_argument("--start-after-poly-idx", type=int,
                   help="Resume: skip polygons up to and including this index")
    p.add_argument("--workers", type=int,
                   help="Parallel download workers per source")

    args = p.parse_args()
    cfg = _load_full_config(Path(args.config))

    # Fill unset CLI args from config, then fall back to hard defaults
    def _cfg(key: str, default: Any = None) -> Any:
        return cfg.get(key, default)

    if args.shp is None:
        args.shp = _cfg("shp")
    if args.year is None:
        args.year = _cfg("year")
    if args.out is None:
        args.out = _cfg("out", "./data/dataset")
    if args.temp is None:
        args.temp = _cfg("temp", "./tmp_tiles")
    if args.max_pixels is None:
        args.max_pixels = int(_cfg("max_pixels", 1024))
    if args.s2_bands is None:
        v = _cfg("s2_bands", "all")
        args.s2_bands = v if isinstance(v, list) else [str(v)]
    if args.s1_bands is None:
        v = _cfg("s1_bands", "all")
        args.s1_bands = v if isinstance(v, list) else [str(v)]
    if args.worldcover_scale is None:
        args.worldcover_scale = int(_cfg("worldcover_scale", 10))
    if args.limit is None:
        args.limit = _cfg("limit")
    if args.start_after_poly_idx is None:
        args.start_after_poly_idx = int(_cfg("start_after_poly_idx", -1))
    if args.workers is None:
        args.workers = int(_cfg("workers", 4))

    # Resolve source list: CLI --source wins; else use config 'sources'/'source'
    if args.source:
        args.sources = [args.source]
    else:
        raw = _cfg("sources") or _cfg("source")
        if raw is None:
            p.error("--source is required (or set 'sources' in config.yaml)")
        args.sources = [raw] if isinstance(raw, str) else list(raw)

    if args.shp is None:
        p.error("--shp is required (or set 'shp' in config.yaml)")
    if args.year is None:
        p.error("--year is required (or set 'year' in config.yaml)")

    return args


if __name__ == "__main__":
    args = parse_args()

    # Tile the ROI ONCE — the tiles depend only on the ROI + max_pixels, so all
    # sources (s2, s1, ...) share the exact same temp tiles. No redundant tiling.
    tasks, skipped = collect_tile_tasks(
        shp_path=Path(args.shp),
        year=args.year,
        temp_dir=Path(args.temp),
        max_pixels=args.max_pixels,
        limit=args.limit,
        start_after_poly_idx=args.start_after_poly_idx,
    )

    def _source_kwargs(source: str) -> Dict[str, Any]:
        return dict(
            source=source,
            out_root=Path(args.out),
            tasks=tasks,
            skipped=skipped,
            year=args.year,
            s2_bands=args.s2_bands,
            s1_bands=args.s1_bands,
            worldcover_scale=args.worldcover_scale,
            workers=args.workers,
        )

    if len(args.sources) <= 1:
        for source in args.sources:
            run_tiled_download(**_source_kwargs(source))
    else:
        # Run every source CONCURRENTLY, each in its own process. Each source
        # process in turn spawns its own pool of tile workers, so S1 and S2
        # download at the same time instead of one after the other.
        ctx = mp.get_context("spawn")
        procs: List[Tuple[str, mp.process.BaseProcess]] = []
        for source in args.sources:
            pr = ctx.Process(
                target=run_tiled_download,
                kwargs=_source_kwargs(source),
                name=f"source-{source}",
            )
            pr.start()
            procs.append((source, pr))
            print(f"🚀 Launched source '{source}' (pid {pr.pid})")

        failures = []
        for source, pr in procs:
            pr.join()
            if pr.exitcode != 0:
                failures.append((source, pr.exitcode))
                print(f"⚠️ Source '{source}' exited with code {pr.exitcode}")

        if failures:
            raise SystemExit(
                "Some sources failed: "
                + ", ".join(f"{s} (exit {c})" for s, c in failures)
            )
