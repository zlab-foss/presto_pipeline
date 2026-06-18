"""
Parallel wrapper for get_data.py.

Single crop (same as before):
  python run_parallel.py --workers 4 \\
      --shp ./ROI/barley/shapefile-province-cc-shuffled.shp \\
      --year 2019 --source s2 \\
      --out ./data/barley --temp ./tmp_tiles_barley

Multiple crops via config file:
  python run_parallel.py --config crops.yaml

crops.yaml format:
  workers: 4          # workers per crop
  year: 2019
  source: s2
  crops:
    - shp:  ./ROI/barley/shapefile-province-cc-shuffled.shp
      out:  ./data/barley
      temp: ./tmp_tiles_barley
    - shp:  ./ROI/wheat/shapefile-province-cc-shuffled.shp
      out:  ./data/wheat
      temp: ./tmp_tiles_wheat
"""

from __future__ import annotations

import argparse
import math
import struct
import subprocess
import sys
import threading
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_polygons(shp_path: Path) -> int:
    dbf = shp_path.with_suffix(".dbf")
    with open(dbf, "rb") as f:
        f.read(4)
        (count,) = struct.unpack("<I", f.read(4))
    return count


def _stream(src, sinks: list) -> None:
    for line in iter(src.readline, b""):
        text = line.decode(errors="replace")
        for s in sinks:
            s.write(text)
            s.flush()
    src.close()


def _run_worker(
    label: str,
    start_after: int,
    limit: int,
    base_args: list[str],
    log_path: Path,
) -> int:
    cmd = [
        sys.executable, "get_data.py",
        *base_args,
        "--start-after-poly-idx", str(start_after),
        "--limit", str(limit),
    ]
    prefix = f"[{label}] "
    print(f"{prefix}start_after={start_after}  limit={limit}  log={log_path}")
    with open(log_path, "w") as log_file:
        log_file.write(" ".join(cmd) + "\n\n")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        class PrefixWriter:
            def __init__(self, inner):
                self._inner = inner
                self._at_line_start = True

            def write(self, text: str) -> None:
                if not text:
                    return
                parts = text.split("\n")
                out = []
                for i, chunk in enumerate(parts):
                    if i < len(parts) - 1:
                        out.append((prefix if self._at_line_start else "") + chunk + "\n")
                        self._at_line_start = True
                    else:
                        if chunk:
                            out.append((prefix if self._at_line_start else "") + chunk)
                            self._at_line_start = False
                self._inner.write("".join(out))

            def flush(self) -> None:
                self._inner.flush()

        _stream(proc.stdout, [PrefixWriter(sys.stdout), log_file])
        proc.wait()
    return proc.returncode


# ---------------------------------------------------------------------------
# Per-crop launcher
# ---------------------------------------------------------------------------

def _launch_crop(
    crop_name: str,
    base_args: list[str],
    workers: int,
    log_dir: Path,
    total_polys: int | None,
) -> tuple[list[threading.Thread], list[int]]:
    """Return started threads for all workers of one crop."""
    shp_path: Path | None = None
    for i, tok in enumerate(base_args):
        if tok == "--shp" and i + 1 < len(base_args):
            shp_path = Path(base_args[i + 1])
            break

    total = total_polys
    if total is None:
        if shp_path is None:
            sys.exit(f"ERROR [{crop_name}]: --shp is required.")
        if not shp_path.with_suffix(".dbf").exists():
            sys.exit(f"ERROR [{crop_name}]: Cannot find {shp_path.with_suffix('.dbf')}")
        total = _count_polygons(shp_path)
        print(f"[{crop_name}] Total polygons: {total}")

    chunk = math.ceil(total / workers)
    ranges = []
    for i in range(workers):
        start_after = i * chunk - 1
        limit = min(chunk, total - i * chunk)
        if limit <= 0:
            break
        ranges.append((start_after, limit))

    print(f"[{crop_name}] Launching {len(ranges)} workers, ~{chunk} polygons each.")

    results: list[int] = [0] * len(ranges)
    threads: list[threading.Thread] = []

    def run(idx: int, start_after: int, limit: int) -> None:
        label = f"{crop_name} w{idx:02d}"
        log_path = log_dir / f"{crop_name}_worker_{idx:02d}.log"
        results[idx] = _run_worker(label, start_after, limit, base_args, log_path)

    for idx, (start_after, limit) in enumerate(ranges):
        t = threading.Thread(target=run, args=(idx, start_after, limit), daemon=True)
        t.start()
        threads.append(t)

    return threads, results


# ---------------------------------------------------------------------------
# Config-file mode
# ---------------------------------------------------------------------------

def _run_from_config(config_path: Path) -> None:
    try:
        import yaml
    except ImportError:
        sys.exit("ERROR: PyYAML is required for --config mode. Install with: pip install pyyaml")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    workers: int = cfg.get("workers", 4)
    log_dir = Path(cfg.get("log_dir", "./logs_parallel"))
    log_dir.mkdir(parents=True, exist_ok=True)

    # Shared passthrough args (year, max_pixels, s2_bands)
    shared_passthrough: list[str] = []
    for key in ("year", "max_pixels"):
        if key in cfg:
            shared_passthrough += [f"--{key.replace('_', '-')}", str(cfg[key])]
    if "s2_bands" in cfg:
        shared_passthrough += ["--s2-bands"] + cfg["s2_bands"]

    # Sources: accept either `source: s2` or `sources: [s1, s2]`
    if "sources" in cfg:
        global_sources = list(cfg["sources"])
    elif "source" in cfg:
        global_sources = [cfg["source"]]
    else:
        global_sources = []

    all_threads: list[threading.Thread] = []
    all_results: list[list[int]] = []
    group_names: list[str] = []

    for crop_cfg in cfg["crops"]:
        crop_name = crop_cfg.get("name") or Path(crop_cfg["shp"]).parts[-2]
        crop_sources = crop_cfg.get("sources") or (
            [crop_cfg["source"]] if "source" in crop_cfg else global_sources
        )
        if not crop_sources:
            sys.exit(f"ERROR [{crop_name}]: no source(s) configured.")

        for source in crop_sources:
            group_name = f"{crop_name}-{source}"
            passthrough = shared_passthrough + [
                "--source", source,
                "--shp",  crop_cfg["shp"],
                "--out",  crop_cfg["out"],
                "--temp", crop_cfg["temp"],
            ]
            threads, results = _launch_crop(
                crop_name=group_name,
                base_args=passthrough,
                workers=crop_cfg.get("workers", workers),
                log_dir=log_dir,
                total_polys=crop_cfg.get("total_polys"),
            )
            all_threads.extend(threads)
            all_results.append(results)
            group_names.append(group_name)

    print(f"\nAll {len(all_threads)} workers launched across {len(group_names)} crop/source groups.\n")
    for t in all_threads:
        t.join()

    failed_groups = [
        name for name, results in zip(group_names, all_results)
        if any(rc != 0 for rc in results)
    ]
    if failed_groups:
        print(f"\nGroups with errors: {failed_groups}")
        sys.exit(1)
    print("\nAll crop/source groups completed successfully.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description="Run get_data.py across multiple parallel workers (single or multi-crop).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None, help="YAML config for multi-crop runs")
    p.add_argument("--workers", type=int, default=4, help="Workers per crop (single-crop mode)")
    p.add_argument("--log-dir", default="./logs_parallel", help="Directory for per-worker log files")
    p.add_argument("--total-polys", type=int, default=None, help="Skip shapefile read if count is known")
    return p.parse_known_args()


def main() -> None:
    args, passthrough = parse_args()

    if args.config:
        _run_from_config(Path(args.config))
        return

    # Single-crop mode (original behaviour)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    shp_path: Path | None = None
    for i, tok in enumerate(passthrough):
        if tok == "--shp" and i + 1 < len(passthrough):
            shp_path = Path(passthrough[i + 1])
            break

    crop_name = shp_path.parts[-2] if shp_path else "crop"
    threads, results = _launch_crop(
        crop_name=crop_name,
        base_args=passthrough,
        workers=args.workers,
        log_dir=log_dir,
        total_polys=args.total_polys,
    )

    for t in threads:
        t.join()

    failed = [i for i, rc in enumerate(results) if rc != 0]
    if failed:
        print(f"\nWorkers that exited with errors: {failed}")
        sys.exit(1)
    print("\nAll workers completed successfully.")


if __name__ == "__main__":
    main()
