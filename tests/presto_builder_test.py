import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt

import sys
import os

# 1. Get the path of the parent directory (one level up)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 2. Add the parent directory to the system path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


from preprocess import PrestoTensorBuilder


# =========================
# 1) S2 + S1 + ERA5 + SRTM
# =========================
builder = PrestoTensorBuilder(
    group_flags={
        "S1": True,
        "S2_RGB": True,
        "S2_Red_Edge": True,
        "S2_NIR_10m": True,
        "S2_NIR_20m": True,
        "S2_SWIR": True,
        "ERA5": True,
        "SRTM": True,
        "NDVI": True,
    },
    q_low=10.0,
    q_high=90.0,
)

paths_s2 = [
    Path("../data/test_outputs/s2/test_2024.tif"),
    Path("../data/test_outputs/s1/test_2024.tif"),
    Path("../data/test_outputs/era5/test_2024.tif"),
    Path("../data/test_outputs/srtm/test_2024.tif"),
]

x, mask, dw = builder.build_from_tifs(
    paths=paths_s2,
    ref_index=0,
)
print("Builder shape:", builder.shape)

print("\nS2 stack:")
print("  x.shape   =", tuple(x.shape))
print("  mask.shape=", tuple(mask.shape))
print("  dw.shape  =", tuple(dw.shape))


# =========================
# 2) Landsat + S1 only
# =========================
builder2 = PrestoTensorBuilder(
    group_flags={
        "S1": True,
        "S2_RGB": True,      # using L8 mapped to S2 names
        "S2_Red_Edge": False,
        "S2_NIR_10m": True,
        "S2_NIR_20m": False,
        "S2_SWIR": True,
        "ERA5": True,
        "SRTM": True,
        "NDVI": True,
    },
    q_low=10.0,
    q_high=90.0,
)

paths_l8 = [
    Path("../data/test_outputs/landsat/test_2024.tif"),
    Path("../data/test_outputs/s1/test_2024.tif"),
    Path("../data/test_outputs/era5/test_2024.tif"),
    Path("../data/test_outputs/srtm/test_2024.tif"),
]

x2, mask2, dw2 = builder2.build_from_tifs(
    paths=paths_l8,
    ref_index=0,
)

print("\nLandsat stack:")
print("  x2.shape   =", tuple(x2.shape))
print("  mask2.shape=", tuple(mask2.shape))
print("  dw2.shape  =", tuple(dw2.shape))


# =========================
# 3) Visualization helper
# =========================

PRESTO_CHANNEL_NAMES = [
    "S1_VV", "S1_VH",             # 0–1
    "S2_RED", "S2_GREEN", "S2_BLUE",  # 2–4
    "S2_RE1", "S2_RE2", "S2_RE3",     # 5–7
    "S2_NIR10",                      # 8
    "S2_NIR20",                      # 9
    "S2_SWIR1", "S2_SWIR2",          # 10–11
    "ERA5_T2M", "ERA5_TP",           # 12–13
    "SRTM_ELEV", "SRTM_SLOPE",       # 14–15
    "NDVI",                          # 16
]

CHANNEL_COLORS = [
    "#0047AB", "#2E86C1",       # S1
    "#E63946", "#52B788", "#0096C7",  # RGB
    "#9D4EDD", "#7B2CBF", "#5A189A",  # Red-edge
    "#1B4332", "#40916C",            # NIR10, NIR20
    "#8338EC", "#3A0CA3",            # SWIR1-2
    "#F1C40F", "#F39C12",            # ERA5
    "#8D99AE", "#2B2D42",            # SRTM
    "#2ECC71",                       # NDVI
]


def plot_mean_timeseries(x: torch.Tensor, title_suffix: str = ""):
    """
    Visualize mean time-series (per PRESTO channel) from tensor x.

    Accepts:
      - x: (12, 17, N)  [time, channel, pixel]
      - x: (N, 12, 17)  [pixel, time, channel]
    """

    if x.dim() != 3:
        raise ValueError(f"x must be 3-D, got shape {tuple(x.shape)}")

    s0, s1, s2 = x.shape
    dims = [s0, s1, s2]

    # must have one 12 and one 17 somewhere
    if 12 not in dims or 17 not in dims:
        raise ValueError(
            f"Expected one dimension=12 (months) and one=17 (channels), got {tuple(x.shape)}"
        )

    # bring to (N, 12, 17)
    if s0 == 12 and s1 == 17:
        # (12,17,N) -> (N,12,17)
        x = x.permute(2, 0, 1)
    elif s0 == 12 and s2 == 17:
        # (12,N,17) -> (N,12,17)
        x = x.permute(1, 0, 2)
    elif s1 == 12 and s2 == 17:
        # (N,12,17) already OK
        pass
    elif s1 == 12 and s0 == 17:
        # (17,12,N) -> (N,12,17)
        x = x.permute(2, 1, 0)
    elif s2 == 12 and s0 == 17:
        # (17,N,12) -> (N,12,17)
        x = x.permute(1, 2, 0)
    elif s2 == 12 and s1 == 17:
        # (N,17,12) -> (N,12,17)
        x = x.permute(0, 2, 1)
    else:
        raise ValueError(f"Ambiguous shape for (N,12,17) or (12,17,N): {tuple(x.shape)}")

    x = x.cpu().float()  # (N,12,17)
    N, T, C = x.shape
    assert T == 12 and C == 17

    # mean over pixels
    mean_ts = torch.nanmean(x, dim=0).numpy()  # (12,17)
    months = np.arange(1, 13)

    plt.figure(figsize=(14, 8))

    for ch in range(C):
        plt.plot(
            months,
            mean_ts[:, ch],
            label=PRESTO_CHANNEL_NAMES[ch],
            color=CHANNEL_COLORS[ch],
            linewidth=2,
        )

    plt.xticks(months)
    plt.xlabel("Month")
    plt.ylabel("Mean value across all pixels")
    plt.title(f"PRESTO Input — Mean Time Series (All 17 Channels){title_suffix}")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Plot for Landsat+S1 case
    plot_mean_timeseries(x2, title_suffix=" — Landsat + S1 + ERA5 + SRTM")
    # If you also want to see the S2+S1+ERA5+SRTM version, uncomment:
    plot_mean_timeseries(x, title_suffix=" — S2 + S1 + ERA5 + SRTM")