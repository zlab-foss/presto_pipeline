#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


YEAR_RE = re.compile(r"(?:^|_)y(19\d{2}|20\d{2})(?:_|$)", re.IGNORECASE)


def extract_year(path: Path) -> int | None:
    m = YEAR_RE.search(path.name)
    return int(m.group(1)) if m else None


def group_by_year(input_dir: Path, exts: tuple[str, ...] = (".tif", ".tiff")) -> Dict[int, List[Path]]:
    groups: Dict[int, List[Path]] = defaultdict(list)
    for p in sorted(input_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        y = extract_year(p)
        if y is None:
            continue
        groups[y].append(p)
    return dict(groups)


def run_gdalbuildvrt(vrt_path: Path, tif_paths: List[Path]) -> None:
    # Use an argument file to avoid overly-long command lines.
    list_path = vrt_path.with_suffix(".txt")
    list_path.write_text("\n".join(str(p) for p in tif_paths) + "\n", encoding="utf-8")

    cmd = ["gdalbuildvrt", "-input_file_list", str(list_path), str(vrt_path)]
    subprocess.run(cmd, check=True)


def run_gdalwarp_to_tif(
    vrt_path: Path,
    out_tif: Path,
    *,
    dst_nodata: float | int | None,
    resampling: str,
    compress: str,
    bigtiff: str,
    overwrite: bool,
) -> None:
    cmd = ["gdalwarp", "-of", "GTiff"]

    if overwrite:
        cmd.append("-overwrite")

    # Resampling suitable for categorical rasters (default: near)
    cmd += ["-r", resampling]

    if dst_nodata is not None:
        cmd += ["-dstnodata", str(dst_nodata)]

    # Creation options
    cmd += ["-co", f"COMPRESS={compress}", "-co", "TILED=YES", "-co", f"BIGTIFF={bigtiff}"]

    cmd += [str(vrt_path), str(out_tif)]
    subprocess.run(cmd, check=True)


def merge_year_group(
    year: int,
    tif_paths: List[Path],
    out_dir: Path,
    *,
    prefix: str,
    dst_nodata: float | int | None,
    resampling: str,
    compress: str,
    bigtiff: str,
    overwrite: bool,
    keep_vrt: bool,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tif = out_dir / f"{prefix}_y{year}.tif"

    if out_tif.exists() and not overwrite:
        return out_tif

    vrt_path = out_dir / f"{prefix}_y{year}.vrt"
    run_gdalbuildvrt(vrt_path, tif_paths)
    run_gdalwarp_to_tif(
        vrt_path,
        out_tif,
        dst_nodata=dst_nodata,
        resampling=resampling,
        compress=compress,
        bigtiff=bigtiff,
        overwrite=True,
    )

    if not keep_vrt:
        try:
            os.remove(vrt_path)
            txt = vrt_path.with_suffix(".txt")
            if txt.exists():
                os.remove(txt)
        except OSError:
            pass

    return out_tif


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Group rasters by year (from filename like *_y2017_*) and mosaic each year into one GeoTIFF."
    )
    ap.add_argument("--input", required=True, type=Path, help="Input directory containing tiles")
    ap.add_argument("--output", required=True, type=Path, help="Output directory for yearly mosaics")
    ap.add_argument("--prefix", default="esri_lulc", help="Output filename prefix (default: esri_lulc)")
    ap.add_argument(
        "--dst-nodata",
        default=None,
        help="Set output nodata (e.g., 0). If omitted, gdalwarp chooses from inputs.",
    )
    ap.add_argument(
        "--resampling",
        default="near",
        choices=["near", "mode", "bilinear", "cubic"],
        help="Resampling method (categorical => near or mode). Default: near",
    )
    ap.add_argument("--compress", default="DEFLATE", choices=["DEFLATE", "LZW", "ZSTD", "NONE"])
    ap.add_argument(
        "--bigtiff",
        default="IF_SAFER",
        choices=["YES", "NO", "IF_SAFER", "IF_NEEDED"],
        help="BIGTIFF setting. Default: IF_SAFER",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    ap.add_argument("--keep-vrt", action="store_true", help="Keep intermediate .vrt and .txt files")

    args = ap.parse_args()

    dst_nodata = None
    if args.dst_nodata is not None:
        # accept int/float from string
        try:
            dst_nodata = int(args.dst_nodata)
        except ValueError:
            dst_nodata = float(args.dst_nodata)

    groups = group_by_year(args.input)
    if not groups:
        raise SystemExit("No rasters with a *_yYYYY_* year token found in filenames.")

    for year in sorted(groups):
        paths = groups[year]
        out = merge_year_group(
            year,
            paths,
            args.output,
            prefix=args.prefix,
            dst_nodata=dst_nodata,
            resampling=args.resampling,
            compress=args.compress,
            bigtiff=args.bigtiff,
            overwrite=args.overwrite,
            keep_vrt=args.keep_vrt,
        )
        print(f"[y{year}] {len(paths)} tiles -> {out}")


if __name__ == "__main__":
    main()
