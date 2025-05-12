#!/usr/bin/env python3
"""
lamb_weather_types_slp.py  -- **synoptic-scale version**

Changes compared with the previous revision
-------------------------------------------
* **Synoptic remapping step** – ERA5 SLP is first aggregated to a user-set coarse
  resolution (default **2.5°** ≃ 1000–1500 km synoptic scale) before the Lamb/Jenkinson-
  Collison weather-type calculation.
* New CLI flag **--coarse_deg** allows quick experimentation (set to 0 to bypass).

Why this matters
----------------
Smoothing (or coarse re-gridding) removes mesoscale pressure anomalies produced by
Alpine orography, yielding more stable flow/shear gradients and cleaner weather types.

Usage examples
--------------
    # Default 2.5° synoptic grid (recommended)
    python lamb_weather_types_slp.py --region southern_alps

    # Try a 1.5° grid (finer) on 8 CPU cores
    python lamb_weather_types_slp.py --region southern_alps --coarse_deg 1.5 --ncores 8

    # Disable remapping (backward-compatible behaviour)
    python lamb_weather_types_slp.py --region southern_alps --coarse_deg 0
"""

import os
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from datetime import timedelta
from multiprocessing import Pool
import logging
import warnings

# --------------------------------------
# Logging & misc. setup
# --------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(os.path.basename(__file__))
warnings.filterwarnings("ignore", ".*non-contiguous coordinate*")

# --------------------------------------
# Sub-region definitions
# --------------------------------------
SUBREGIONS = {
    "western_alps":  {"lon_min":   3, "lon_max":  8, "lat_min": 43, "lat_max": 49},
    "southern_alps": {"lon_min": 7.5, "lon_max": 13, "lat_min": 43, "lat_max": 46},
    "dinaric_alps":  {"lon_min":12.5, "lon_max": 20, "lat_min": 42, "lat_max": 46},
    "eastern_alps":  {"lon_min":   8, "lon_max": 17, "lat_min": 46, "lat_max": 49},
    "britain": {"lon_min": -20, "lon_max": 10, "lat_min": 45, "lat_max": 65},
}

# --------------------------------------
# Helper functions
# --------------------------------------

def reorder_lat(ds: xr.Dataset) -> xr.Dataset:
    """Ensure latitude is in ascending order (south → north)."""
    if ds.latitude.values[0] > ds.latitude.values[-1]:
        ds = ds.reindex(latitude=np.sort(ds.latitude.values))
    return ds


def get_target_grid(region_bounds):
    """Return the 5×7 Lamb grid (lat asc., lon asc.) centred on region."""
    lat_min, lat_max = region_bounds["lat_min"], region_bounds["lat_max"]
    lon_min, lon_max = region_bounds["lon_min"], region_bounds["lon_max"]
    c_lat = 0.5 * (lat_min + lat_max)
    c_lon = 0.5 * (lon_min + lon_max)

    target_lats = np.sort([c_lat + 10, c_lat + 5, c_lat, c_lat - 5, c_lat - 10])
    target_lons = np.array([
        c_lon - 15, c_lon - 10, c_lon - 5, c_lon,
        c_lon + 5,  c_lon + 10, c_lon + 15,
    ])
    return target_lats, target_lons, c_lat, c_lon


# ==============================================================
#  Synoptic-scale remapping utilities
# ==============================================================

def synoptic_regrid(da: xr.DataArray, coarse_deg: float,
                    target_lats: np.ndarray, target_lons: np.ndarray) -> xr.DataArray:
    """Aggregate *da* to *coarse_deg* grid and interpolate to Lamb target grid.

    Parameters
    ----------
    da : xr.DataArray
        ERA5 sea-level pressure (time × lat × lon) @ 0.25° native grid.
    coarse_deg : float
        Desired coarse resolution in degrees. If 0 (or < native res.), remapping is skipped.
    target_lats, target_lons : array-like
        Coordinates of the 5×7 Lamb grid.
    """
    # Skip if user requests 0 or finer than native
    if coarse_deg <= 0:
        logger.debug("Skipping synoptic remapping (coarse_deg <= 0)")
        return da.interp(latitude=target_lats, longitude=target_lons, method="linear")

    # Compute integer aggregation factor (rounded to nearest)
    dlat = float(abs(da.latitude.values[1] - da.latitude.values[0]))
    dlon = float(abs(da.longitude.values[1] - da.longitude.values[0]))
    factor_lat = max(int(round(coarse_deg / dlat)), 1)
    factor_lon = max(int(round(coarse_deg / dlon)), 1)

    logger.debug("Coarsening ERA5: %dx%d (%.3f°, %.3f° → %.2f°)" % (
        factor_lat, factor_lon, dlat, dlon, coarse_deg))

    # Coarsen (mean aggregation)
    da_coarse = da.coarsen(latitude=factor_lat, longitude=factor_lon, boundary="trim").mean()

    # Interpolate the coarse field to the fine target grid
    return da_coarse.interp(latitude=target_lats, longitude=target_lons, method="linear")


# ==============================================================
#  Lamb-type core (unchanged mathematical routines)
# ==============================================================


def calc_const():
    """Return four numeric constants used in the JC-WT equations."""
    const1 = 1 / np.cos(np.radians(45))
    const2 = np.sin(np.radians(45)) / np.sin(np.radians(40))
    const3 = np.sin(np.radians(45)) / np.sin(np.radians(50))
    const4 = 1 / (2 * (np.cos(np.radians(45)) ** 2))
    return const1, const2, const3, const4


def calc_westerly_flow(cube):
    return 0.5 * (cube[:, 1, 2] + cube[:, 1, 4]) - 0.5 * (cube[:, 3, 2] + cube[:, 3, 4])


def calc_southerly_flow(cube, const1):
    return const1 * (
        0.25 * (cube[:, 3, 4] + 2 * cube[:, 2, 4] + cube[:, 1, 4])
        - 0.25 * (cube[:, 3, 2] + 2 * cube[:, 2, 2] + cube[:, 1, 2])
    )


def calc_resultant_flow(w_flow, s_flow):
    return np.sqrt(w_flow ** 2 + s_flow ** 2)


def calc_westerly_shear_velocity(cube, const2, const3):
    return (
        const2 * (0.5 * (cube[:, 0, 2] + cube[:, 0, 4]) - 0.5 * (cube[:, 2, 2] + cube[:, 2, 4]))
        - const3 * (0.5 * (cube[:, 2, 2] + cube[:, 2, 4]) - 0.5 * (cube[:, 4, 2] + cube[:, 4, 4]))
    )


def calc_southerly_shear_velocity(cube, const4):
    return const4 * (
        0.25 * (cube[:, 3, 6] + 2 * cube[:, 2, 6] + cube[:, 1, 6])
        - 0.25 * (cube[:, 3, 4] + 2 * cube[:, 2, 4] + cube[:, 1, 4])
        - 0.25 * (cube[:, 3, 2] + 2 * cube[:, 2, 2] + cube[:, 1, 2])
        + 0.25 * (cube[:, 3, 0] + 2 * cube[:, 2, 0] + cube[:, 1, 0])
    )


def calc_total_shear_velocity(w_shear, s_shear):
    return w_shear + s_shear


# == Main Lamb/Jenkinson-Collison logic (unchanged) ==


def wt_algorithm(cube):
    """Return array of 27 Lamb types for each time-step."""
    c1, c2, c3, c4 = calc_const()

    w_flow = calc_westerly_flow(cube)
    s_flow = calc_southerly_flow(cube, c1)
    total_flow = calc_resultant_flow(w_flow, s_flow)

    w_shear = calc_westerly_shear_velocity(cube, c2, c3)
    s_shear = calc_southerly_shear_velocity(cube, c4)
    total_shear = calc_total_shear_velocity(w_shear, s_shear)

    n = cube.shape[0]
    lwt = np.zeros(n, dtype=int)

    for i in range(n):
        # Flow direction (meteorological convention)
        direction = (np.degrees(np.arctan2(-w_flow[i], -s_flow[i])) + 360) % 360

        # ---- Pure directional types ----
        if abs(total_shear[i]) < total_flow[i]:
            if (direction >= 337.5 or direction < 22.5):
                lwt[i] = 1
            elif direction < 67.5:
                lwt[i] = 2
            elif direction < 112.5:
                lwt[i] = 3
            elif direction < 157.5:
                lwt[i] = 4
            elif direction < 202.5:
                lwt[i] = 5
            elif direction < 247.5:
                lwt[i] = 6
            elif direction < 292.5:
                lwt[i] = 7
            else:  # < 337.5
                lwt[i] = 8

        # ---- Pure cyclonic / anticyclonic types ----
        elif (2 * total_flow[i]) < abs(total_shear[i]):
            lwt[i] = 9 if total_shear[i] > 0 else 10

        # ---- Hybrid (synoptic-directional) types ----
        elif total_flow[i] < abs(total_shear[i]) < (2 * total_flow[i]):
            if total_shear[i] > 0:
                offset = 11
            else:
                offset = 19
            # Directional bucket 0-7
            dir_idx = int(((direction + 22.5) % 360) // 45)
            lwt[i] = offset + dir_idx

        # ---- Weak / indeterminate circulation ----
        elif abs(total_shear[i]) < 6 and total_flow[i] < 6:
            lwt[i] = 27

    return lwt


def map_lwt_to_slwt(lwt):
    """Convert 27 Lamb types to 9 simplified groups (paper mapping)."""
    mapping = {1: 1, 2: 1, 19: 1, 27: 1,
               3: 2, 4: 2, 20: 2, 21: 2, 22: 2,
               5: 3, 6: 3, 15: 3, 16: 3,
               7: 4, 8: 4, 17: 4, 18: 4,
               9: 5,
               10: 6,
               11: 7, 12: 7, 13: 7, 14: 7,
               23: 8, 24: 8,
               25: 9, 26: 9}
    return np.array([mapping.get(t, 0) for t in lwt], dtype=int)


# ==============================================================
#  Per-year processing function
# ==============================================================

def process_year(year: int, region_bounds: dict, target_lats: np.ndarray, target_lons: np.ndarray,
                 data_path: str, coarse_deg: float) -> pd.DataFrame:
    nc_file = os.path.join(data_path, f"slp_{year}_NA.nc")
    logger.info("Processing %s", nc_file)

    try:
        ds = xr.open_dataset(nc_file)
    except Exception as exc:
        logger.error("Cannot open %s → %s", nc_file, exc)
        return pd.DataFrame()

    ds = reorder_lat(ds)
    da = ds["msl"].resample(time="1D").mean()

    # --- NEW: Synoptic-scale remapping ---
    da_synoptic = synoptic_regrid(da, coarse_deg, target_lats, target_lons)

    # Lamb cube shape: (time, 5, 7)
    cube = da_synoptic.data
    lwt = wt_algorithm(cube)
    slwt = map_lwt_to_slwt(lwt)


    base_times = pd.to_datetime(da_synoptic.time.values) + timedelta(minutes=30)
    datetimes = base_times.repeat(24) + pd.to_timedelta(np.tile(np.arange(24), base_times.size), unit="h")

    df = pd.DataFrame({
        "datetime": datetimes,
        "wt": np.repeat(lwt, 24),
        "slwt": np.repeat(slwt, 24),
    })
    return df


# ==============================================================
#  Main CLI driver
# ==============================================================

def main():
    parser = argparse.ArgumentParser(description="Hourly Lamb weather types (synoptic version)")
    parser.add_argument("--region", required=True, choices=SUBREGIONS.keys())
    parser.add_argument("--ncores", type=int, default=32, help="Parallel workers (default 32)")
    parser.add_argument("--serial", action="store_true", help="Run serially (debug)")
    parser.add_argument("--data_path", default="/data/reloclim/normal/INTERACT/ERA5/surface",
                        help="Directory with ERA5 SLP NetCDFs")
    parser.add_argument("--coarse_deg", type=float, default=0,
                        help="Coarse-grid resolution in degrees for synoptic remap (0 = off)")
    args = parser.parse_args()

    # --- Target Lamb grid ---
    tgt_lats, tgt_lons, c_lat, c_lon = get_target_grid(SUBREGIONS[args.region])
    logger.info("%s centre: %.2f°N, %.2f°E – Lamb grid built", args.region, c_lat, c_lon)

    years = range(2001, 2021)
    job_args = [(yr, SUBREGIONS[args.region], tgt_lats, tgt_lons, args.data_path, args.coarse_deg)
                for yr in years]

    if args.serial or args.ncores == 1:
        results = [process_year(*a) for a in job_args]
    else:
        with Pool(args.ncores) as pool:
            results = pool.starmap(process_year, job_args)

    if results:
        df_all = pd.concat(results, ignore_index=True).sort_values("datetime")
        # out_dir = "./csv/weather_types"; os.makedirs(out_dir, exist_ok=True)
        out_dir = "./preprocess/weather_types"
        out_file = os.path.join(out_dir, f"weather_types_{args.region}_daily.csv")
        df_all.to_csv(out_file, index=False)
        logger.info("Saved → %s", out_file)
    else:
        logger.error("No data were processed – check logs above.")


if __name__ == "__main__":
    main()
