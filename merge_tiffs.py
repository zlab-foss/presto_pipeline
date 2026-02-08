from __future__ import annotations

from pathlib import Path
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling


def _find_tifs(in_dir: Path) -> list[Path]:
    return sorted([*in_dir.glob("*.tif"), *in_dir.glob("*.tiff")])


def _get_resampling(method: str) -> Resampling:
    return {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
    }.get(str(method).lower(), Resampling.nearest)


def _band_names(src: rasterio.io.DatasetReader) -> list[str]:
    desc = list(src.descriptions or [])
    while len(desc) < src.count:
        desc.append("")
    return [("" if d is None else str(d)).strip().lower() for d in desc]


def _select_lulc_band(src: rasterio.io.DatasetReader, lulc_band_name: str, fill_mask_name: str) -> list[int]:
    """
    Return exactly one band index (1-based) to use for LULC.
    - Prefer band with description containing lulc_band_name
    - Else fall back to band 1
    - If chosen band is actually fill_mask, fall back to band 1
    """
    names = _band_names(src)

    # prefer band with "lulc" in name
    for i, name in enumerate(names, start=1):
        if lulc_band_name in name and fill_mask_name not in name:
            return [i]

    # fallback to band 1 (unless it's fill_mask, then pick first non-fill_mask band)
    if names and fill_mask_name in names[0]:
        for i, name in enumerate(names, start=1):
            if fill_mask_name not in name:
                return [i]
    return [1]


def merge_tiffs(config: dict) -> Path:
    in_dir = Path(config["in_dir"])
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / config.get("out_name", "merged.tif")

    dtype = str(config.get("dtype", "int16")).lower()
    resampling = _get_resampling(config.get("resampling", "nearest"))

    # nodata rules
    input_nodata_values = list(config.get("input_nodata_values", [0, 255]))
    output_nodata = int(config.get("output_nodata", 0))

    # LULC band rules
    keep_lulc_only = bool(config.get("keep_lulc_only", False))
    lulc_band_name = str(config.get("lulc_band_name", "lulc")).lower()
    drop_fill_mask = bool(config.get("drop_fill_mask", False))
    fill_mask_name = str(config.get("fill_mask_name", "fill_mask")).lower()

    tifs = _find_tifs(in_dir)
    if not tifs:
        raise ValueError(f"No GeoTIFFs found in {in_dir}")

    # Use first raster as reference for CRS + resolution (not extent!)
    with rasterio.open(tifs[0]) as ref:
        ref_crs = ref.crs
        ref_res = ref.res  # (xres, yres)

    vrts: list[WarpedVRT] = []
    srcs: list[rasterio.io.DatasetReader] = []

    try:
        for p in tifs:
            src = rasterio.open(p)
            srcs.append(src)

            # Decide which bands to use
            if keep_lulc_only:
                bands = _select_lulc_band(src, lulc_band_name, fill_mask_name)
            else:
                bands = list(range(1, src.count + 1))
                if drop_fill_mask:
                    names = _band_names(src)
                    bands = [b for b in bands if fill_mask_name not in (names[b - 1] if b - 1 < len(names) else "")]

            if not bands:
                raise ValueError(f"No bands left after filtering for {p.name}")

            # Build VRT: common CRS + resolution, preserve each raster's extent
            vrt = WarpedVRT(
                src,
                crs=ref_crs,
                resolution=ref_res,
                resampling=resampling,
                src_nodata=src.nodata,     # keep original nodata if set
                nodata=output_nodata,      # output nodata in the VRT
                add_alpha=False,
            )

            # Wrap VRT with a lightweight "band selector" by remembering which bands to read later
            # rasterio.merge.merge reads all bands, so we create a small proxy by using vrt.read in merge? Not possible.
            # Practical solution: if keep_lulc_only -> ensure vrt is 1-band by reading & writing to a MemoryFile.
            if len(bands) == vrt.count:
                vrts.append(vrt)
            else:
                # Create a 1-band/filtered-band in-memory dataset so merge() sees the right band count
                data = vrt.read(bands)  # (len(bands), h, w)

                # Normalize nodata: map any 255 -> 0 (and any other nodata values -> 0 if configured)
                for nd in input_nodata_values:
                    if nd != output_nodata:
                        data = np.where(data == nd, output_nodata, data)

                # Build clean profile without inheriting problematic tiling settings
                mem_profile = {
                    'driver': 'GTiff',
                    'dtype': data.dtype,
                    'width': vrt.width,
                    'height': vrt.height,
                    'count': data.shape[0],
                    'crs': vrt.crs,
                    'transform': vrt.transform,
                    'nodata': output_nodata,
                    'compress': 'LZW',
                }

                mem = rasterio.io.MemoryFile()
                ds = mem.open(**mem_profile)
                ds.write(data)

                # Keep references so they don't get GC'ed
                vrts.append(ds)
                srcs.append(mem)  # store mem in srcs list just to keep it alive

        # Merge using union extent
        mosaic, out_transform = merge(vrts, nodata=output_nodata)

        # Normalize nodata values globally again (handle 255 anywhere)
        for nd in input_nodata_values:
            if nd != output_nodata:
                mosaic = np.where(mosaic == nd, output_nodata, mosaic)

        # Cast dtype safely
        if dtype == "int16":
            info = np.iinfo(np.int16)
            mosaic = np.clip(mosaic, info.min, info.max).astype(np.int16)
        else:
            mosaic = mosaic.astype(dtype)

        # Build clean output profile without problematic tiling settings
        out_profile = {
            'driver': 'GTiff',
            'dtype': dtype,
            'width': mosaic.shape[2],
            'height': mosaic.shape[1],
            'count': mosaic.shape[0],
            'crs': vrts[0].crs,
            'transform': out_transform,
            'nodata': output_nodata,
            'compress': 'LZW',
        }

        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(mosaic)

        print(f"✅ Merged {len(tifs)} file(s) → {out_path} | bands={mosaic.shape[0]} | nodata={output_nodata}")
        return out_path

    finally:
        # Close VRTs / datasets
        for v in vrts:
            try:
                v.close()
            except Exception:
                pass
        for s in srcs:
            try:
                s.close()
            except Exception:
                pass


if __name__ == "__main__":

    # for region in ['anzali', 'karkheh', 'gavkhoni', 'urmia']:
    for region in ['anzali', 'karkheh', 'urmia']:
        for year in [2024]:
            # for cls_type in ['irrigation', 'lulc']:
            for cls_type in ['irrigation']:

                config = {
                    "in_dir": f"./data/export/{cls_type}/{region}_{year}",
                    "out_dir": f"./data/{region}/{cls_type}",
                    "out_name": f"{region}_{year}_sentinel.tif",
                    "dtype": "int16",

                    # nodata normalization (your rule)
                    "input_nodata_values": [0, 255],
                    "output_nodata": 0,

                    "resampling": "nearest",
                }

                if cls_type == "lulc":
                    config.update({
                        "keep_lulc_only": True,     # keep only 1 band
                        "lulc_band_name": "lulc",   # if band desc contains this, prefer it
                        "drop_fill_mask": True,     # remove any band whose desc contains FILL_MASK
                        "fill_mask_name": "fill_mask",
                    })

                try:
                    merge_tiffs(config)
                except Exception as e:
                    print(f"⚠️ Failed: {region=} {year=} {cls_type=} -> {type(e).__name__}: {e}")
                    continue
