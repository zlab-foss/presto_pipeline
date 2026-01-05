#!/usr/bin/env python3
"""
Group-move Landsat + WorldCover GeoTIFFs by region id (p####).

Example:
  landsat_y2020_p0049_t002_002.tif  -> <dest_root>/p0049/landsat_y2020_p0049_t002_002.tif
  worldcover_y2021_p0049_t...       -> <dest_root>/p0049/worldcover_y2021_p0049_t...tif
"""

import argparse
import os
import re
import shutil


# Matches: landsat_y2020_p0049_t002_002.tif / worldcover_y2021_p0019_t003_002.tif
_RX = re.compile(r"^(landsat|worldcover)_y\d{4}_p(\d{4})_.*\.tif$", re.IGNORECASE)


def move_by_region(src_dir: str, dst_root: str, recursive: bool, dry_run: bool) -> int:
    match = _RX.match
    join = os.path.join
    makedirs = os.makedirs
    move = shutil.move
    exists = os.path.exists

    moved = 0
    skipped = 0
    conflicts = 0

    if recursive:
        walker = os.walk(src_dir)
        for root, _, files in walker:
            for name in files:
                m = match(name)
                if not m:
                    skipped += 1
                    continue

                p_digits = m.group(2)           # "0049"
                region = "p" + p_digits         # "p0049"
                dst_dir = join(dst_root, region)
                src_path = join(root, name)
                dst_path = join(dst_dir, name)

                if exists(dst_path):
                    conflicts += 1
                    continue

                if not dry_run:
                    makedirs(dst_dir, exist_ok=True)
                    move(src_path, dst_path)
                moved += 1
    else:
        with os.scandir(src_dir) as it:
            for entry in it:
                if not entry.is_file():
                    continue

                name = entry.name
                m = match(name)
                if not m:
                    skipped += 1
                    continue

                p_digits = m.group(2)
                region = "p" + p_digits
                dst_dir = join(dst_root, region)
                src_path = entry.path
                dst_path = join(dst_dir, name)

                if exists(dst_path):
                    conflicts += 1
                    continue

                if not dry_run:
                    makedirs(dst_dir, exist_ok=True)
                    move(src_path, dst_path)
                moved += 1

    print(f"moved={moved} skipped_nonmatch={skipped} conflicts_existing={conflicts}")
    return 0 if conflicts == 0 else 2


def main() -> int:
    ap = argparse.ArgumentParser(description="Move landsat/worldcover files into per-region (p####) folders.")
    ap.add_argument("src_dir", help="Source directory to scan")
    ap.add_argument("dst_root", help="Destination root directory where p#### folders will be created")
    ap.add_argument("--recursive", action="store_true", help="Scan src_dir recursively")
    ap.add_argument("--dry-run", action="store_true", help="Print counts only; do not move files")
    args = ap.parse_args()

    return move_by_region(args.src_dir, args.dst_root, args.recursive, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
