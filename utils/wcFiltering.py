from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional
import json


@dataclass(frozen=True)
class Range:
    min: Optional[float] = None
    max: Optional[float] = None


def iter_geojson_features(path: str):
    """
    Streaming iterator over FeatureCollection.features without loading the full file.
    Works with standard GeoJSON (single big JSON object).
    """
    dec = json.JSONDecoder()
    buf = ""
    in_features = False

    with open(path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(1 << 20)  # 1 MiB
            if not chunk:
                break
            buf += chunk

            if not in_features:
                k = buf.find('"features"')
                if k == -1:
                    # keep tail to handle split tokens across chunks
                    buf = buf[-65536:]
                    continue
                a = buf.find("[", k)
                if a == -1:
                    buf = buf[k:]
                    continue
                buf = buf[a + 1 :]
                in_features = True

            while True:
                # skip whitespace and commas between elements
                i = 0
                n = len(buf)
                while i < n and buf[i] in " \r\n\t,":
                    i += 1
                buf = buf[i:]
                if not buf:
                    break

                if buf[0] == "]":
                    return  # end of features array

                try:
                    obj, end = dec.raw_decode(buf)
                except json.JSONDecodeError:
                    break  # need more data

                yield obj
                buf = buf[end:]


def _to_float(v, *, missing_as_zero: bool) -> float:
    """
    Convert GEE-exported property to float.
    Arrays/dicts => treated as missing (0 if missing_as_zero else NaN behavior by returning None).
    """
    if v is None:
        return 0.0 if missing_as_zero else float("nan")
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return 0.0 if missing_as_zero else float("nan")
    # lists/dicts/etc (common GEE mistake) -> treat as missing
    return 0.0 if missing_as_zero else float("nan")


def feature_passes_thresholds(
    feature: dict,
    thresholds: Mapping[str, Range],
    *,
    missing_as_zero: bool = True,
) -> bool:
    props = feature.get("properties") or {}

    for col, r in thresholds.items():
        val = _to_float(props.get(col), missing_as_zero=missing_as_zero)

        if r.min is not None and not (val >= r.min):
            return False
        if r.max is not None and not (val <= r.max):
            return False

    return True


def filter_geojson_by_class_percent(
    in_path: str,
    out_path: str,
    thresholds: Mapping[str, Range],
    *,
    missing_as_zero: bool = True,
) -> tuple[int, int]:
    """
    Stream-filter FeatureCollection by thresholds and write a new GeoJSON.
    Returns (kept, total).
    """
    kept = 0
    total = 0
    first = True

    with open(out_path, "w", encoding="utf-8") as out:
        out.write('{"type":"FeatureCollection","features":[')

        for feat in iter_geojson_features(in_path):
            total += 1
            if feature_passes_thresholds(feat, thresholds, missing_as_zero=missing_as_zero):
                if not first:
                    out.write(",")
                else:
                    first = False
                out.write(json.dumps(feat, ensure_ascii=False, separators=(",", ":")))
                kept += 1

        out.write("]}")

    return kept, total


if __name__ == "__main__":
    in_path = "/home/amir/Downloads/iran-grid-wc-map.geojson" # this is the output of gee export which contains class percentages for each ROI
    out_path = "filtered.geojson"

    thresholds = {
        # "p_crp": Range(min=20.0), # Cropland
        # "p_bui": Range(min=10.0), # Built-up
        # "p_wat": Range(min=1.0), # Water
        # "p_bar": Range(max=10.0), # Barren
        # "p_shr": Range(max=20.0), # Shrubland
        # "p_grs": Range(min=70.0), # Grassland
        # "p_sni": Range(min=1.0), # Snow Ice
        "p_tc": Range(min=5.0), # Herbaceous wetland
        # "p_hwt": Range(min=1.0), # Woody wetland
        # "p_man": Range(min=1.0), # Mangroves
        # "p_mli": Range(min=1.0), # Moss Lichen
    }


    kept, total = filter_geojson_by_class_percent(in_path, out_path, thresholds)
    print(f"Kept {kept} / {total} ROIs -> {out_path}")
