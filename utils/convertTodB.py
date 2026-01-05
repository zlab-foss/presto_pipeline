from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import rasterio


@dataclass(frozen=True)
class Job:
    src_path: Path
    dst_path: Path


def _build_jobs(root_dir: Path, out_dir: Optional[Path], suffix: str) -> list[Job]:
    jobs: list[Job] = []
    for tif_path in root_dir.rglob("*.tif"):
        if "s1" not in tif_path.name.lower():
            continue

        if out_dir is None:
            dst_path = tif_path.with_name(f"{tif_path.stem}{suffix}{tif_path.suffix}")
        else:
            rel = tif_path.relative_to(root_dir)
            dst_path = out_dir / rel
            dst_path.parent.mkdir(parents=True, exist_ok=True)

        jobs.append(Job(tif_path, dst_path))
    return jobs


def _convert_one(
    job: Job,
    out_nodata: float,
    mode: str,
    epsilon: float,
    gdal_num_threads: str = "ALL_CPUS",
) -> int:
    # Per-process GDAL threading for IO/DEFLATE, etc.
    env_opts = {"GDAL_NUM_THREADS": gdal_num_threads} if gdal_num_threads else {}

    with rasterio.Env(**env_opts):
        with rasterio.open(job.src_path) as src:
            profile = src.profile.copy()
            profile.update(dtype=rasterio.float32, nodata=out_nodata)

            src_nodata = src.nodata  # can be None
            src_desc = src.descriptions  # keep band names as-is

            with rasterio.open(job.dst_path, "w", **profile) as dst:
                # Keep original band names (descriptions)
                for b in range(1, src.count + 1):
                    desc = src_desc[b - 1]
                    if desc is not None:
                        dst.set_band_description(b, desc)

                out_nodata32 = np.float32(out_nodata)
                eps32 = np.float32(epsilon)

                # Block-wise processing (fast + low RAM)
                for _, window in src.block_windows(1):
                    block = src.read(window=window, out_dtype="float32")  # (B,H,W)

                    # invalid: non-finite OR <=0 OR equals nodata (if defined)
                    invalid = ~np.isfinite(block) | (block <= 0)
                    if src_nodata is not None and np.isfinite(src_nodata):
                        invalid |= (block == np.float32(src_nodata))

                    if mode == "clip_epsilon":
                        # ensure finite positive values everywhere (clipped)
                        safe = np.where(invalid, eps32, block)
                        safe = np.maximum(safe, eps32)
                        block_db = np.empty_like(safe, dtype=np.float32)
                        np.log10(safe, out=block_db)
                        block_db *= np.float32(10.0)
                    else:
                        # compute only on valid pixels, set invalid directly to nodata (avoids NaNs/log warnings)
                        block_db = np.empty_like(block, dtype=np.float32)
                        block_db[invalid] = out_nodata32

                        valid = ~invalid
                        if np.any(valid):
                            tmp = block[valid]
                            tmp_db = np.empty_like(tmp, dtype=np.float32)
                            np.log10(tmp, out=tmp_db)
                            tmp_db *= np.float32(10.0)
                            block_db[valid] = tmp_db

                    dst.write(block_db, window=window)

    return 1


def convert_s1_tifs_linear_to_db(
    root_dir: str | Path,
    out_dir: Optional[str | Path] = None,
    suffix: str = "_db",
    out_nodata: float = -9999.0,
    mode: str = "set_nodata",  # "set_nodata" or "clip_epsilon"
    epsilon: float = 1e-10,
    workers: Optional[int] = None,
    gdal_num_threads: str = "ALL_CPUS",
) -> int:
    """Parallel conversion: one process per file (fastest/cleanest for raster IO + numpy)."""
    if mode not in {"set_nodata", "clip_epsilon"}:
        raise ValueError("mode must be 'set_nodata' or 'clip_epsilon'")

    root_dir_p = Path(root_dir)
    out_dir_p = Path(out_dir) if out_dir is not None else None

    jobs = _build_jobs(root_dir_p, out_dir_p, suffix)
    if not jobs:
        return 0

    # Default: use a sensible number of workers (avoid oversubscribing)
    if workers is None:
        cpu = os.cpu_count() or 1
        workers = min(8, cpu)  # good default; raise if your storage is fast

    written = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(_convert_one, job, out_nodata, mode, epsilon, gdal_num_threads)
            for job in jobs
        ]
        for f in as_completed(futs):
            written += f.result()

    return written


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert S1 GeoTIFFs from linear power to dB (parallel)")
    parser.add_argument("root_dir", type=str, help="Directory to search recursively")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (preserves relative paths). If omitted, writes *_db.tif next to inputs.",
    )
    parser.add_argument("--suffix", type=str, default="_db", help="Output suffix")
    parser.add_argument("--out-nodata", type=float, default=-9999.0, help="Nodata value for output dB GeoTIFF")
    parser.add_argument(
        "--mode",
        type=str,
        default="set_nodata",
        choices=["set_nodata", "clip_epsilon"],
        help="How to handle invalid pixels (<=0/NaN/inf/nodata).",
    )
    parser.add_argument("--epsilon", type=float, default=1e-10, help="Used only in clip_epsilon mode")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel processes (default: min(8, CPU))")
    parser.add_argument(
        "--gdal-num-threads",
        type=str,
        default="ALL_CPUS",
        help="GDAL_NUM_THREADS for internal IO/decompression (e.g. ALL_CPUS, 1, 2, ...).",
    )
    args = parser.parse_args()

    n = convert_s1_tifs_linear_to_db(
        args.root_dir,
        out_dir=args.out_dir,
        suffix=args.suffix,
        out_nodata=args.out_nodata,
        mode=args.mode,
        epsilon=args.epsilon,
        workers=args.workers,
        gdal_num_threads=args.gdal_num_threads,
    )
    print(f"Wrote {n} file(s)")
