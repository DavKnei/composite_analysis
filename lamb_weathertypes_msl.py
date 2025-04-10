#!/usr/bin/env python3
"""
lamb_weather_types_slp.py

This script calculates hourly Lamb weather types from ERA5 SLP data (variable "msl")
for the years 2000 to 2020. The weather‐typing grid is centered on the region specified
by the --region flag (using its center point computed from preset boundaries). The full ERA5
dataset is used for interpolation onto this grid. In the end the 27 Lamb weather types are
mapped to 9 simplified weather types (slwt) following the paper.
Results (including lwt and slwt) are saved to a CSV file named "weather_types_{region}.csv".

Usage examples:
    python lamb_weather_types_slp.py --region southern_alps
    python lamb_weather_types_slp.py --region southern_alps --ncores 16
    python lamb_weather_types_slp.py --region southern_alps --serial
    python lamb_weather_types_slp.py --region southern_alps --data_path /my/data/path
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(os.path.basename(__file__))

# Define subregions and their boundaries
SUBREGIONS = {
    'western_alps': {'lon_min': 3,   'lon_max': 8,   'lat_min': 43, 'lat_max': 49},
    'southern_alps': {'lon_min': 7.5, 'lon_max': 13,  'lat_min': 43, 'lat_max': 46},
    'dinaric_alps':  {'lon_min': 12.5,'lon_max': 20,  'lat_min': 42, 'lat_max': 46},
    'eastern_alps':  {'lon_min': 8,   'lon_max': 17,  'lat_min': 46, 'lat_max': 49}
}

warnings.filterwarnings("ignore", ".*non-contiguous coordinate*")

def reorder_lat(ds):
    """
    Ensure that the latitude coordinate is in ascending order.
    """
    if ds.latitude.values[0] > ds.latitude.values[-1]:
        ds = ds.reindex(latitude=np.sort(ds.latitude.values))
    return ds

def get_target_grid(region_bounds):
    """
    Given region boundaries, compute the center and define a 5x7 target grid.
    The grid is defined relative to the center as:
      latitudes: [center+10, center+5, center, center-5, center-10] (sorted ascending)
      longitudes: [center-15, center-10, center-5, center, center+5, center+10, center+15]
    """
    lat_min = region_bounds['lat_min']
    lat_max = region_bounds['lat_max']
    lon_min = region_bounds['lon_min']
    lon_max = region_bounds['lon_max']
    center_lat = (lat_min + lat_max) / 2.
    center_lon = (lon_min + lon_max) / 2.
    target_lats = np.array([center_lat + 10, center_lat + 5, center_lat,
                             center_lat - 5, center_lat - 10])
    target_lons = np.array([center_lon - 15, center_lon - 10, center_lon - 5,
                             center_lon, center_lon + 5, center_lon + 10, center_lon + 15])
    target_lats = np.sort(target_lats)
    return target_lats, target_lons, center_lat, center_lon

#
# Functions following ESMValTool naming and logic:
#

def calc_const():
    """Calculate the four constants for the Lamb weathertyping algorithm."""
    const1 = 1 / np.cos(np.radians(45))
    const2 = np.sin(np.radians(45)) / np.sin(np.radians(40))
    const3 = np.sin(np.radians(45)) / np.sin(np.radians(50))
    const4 = 1 / (2 * (np.cos(np.radians(45))**2))
    return const1, const2, const3, const4

def calc_westerly_flow(cube):
    """Calculate the westerly flow over the target grid."""
    return 0.5 * (cube[:, 1, 2] + cube[:, 1, 4]) - 0.5 * (cube[:, 3, 2] + cube[:, 3, 4])

def calc_southerly_flow(cube, const1):
    """Calculate the southerly flow over the target grid."""
    return const1 * (0.25 * (cube[:, 3, 4] + 2 * cube[:, 2, 4] + cube[:, 1, 4]) -
                     0.25 * (cube[:, 3, 2] + 2 * cube[:, 2, 2] + cube[:, 1, 2]))

def calc_resultant_flow(westerly_flow, southerly_flow):
    """Calculate the resultant flow."""
    return np.sqrt(westerly_flow**2 + southerly_flow**2)

def calc_westerly_shear_velocity(cube, const2, const3):
    """Calculate the westerly shear velocity."""
    return (const2 * (0.5 * (cube[:, 0, 2] + cube[:, 0, 4]) -
                      0.5 * (cube[:, 2, 2] + cube[:, 2, 4])) -
            const3 * (0.5 * (cube[:, 2, 2] + cube[:, 2, 4]) -
                      0.5 * (cube[:, 4, 2] + cube[:, 4, 4])))

def calc_southerly_shear_velocity(cube, const4):
    """Calculate the southerly shear velocity."""
    return const4 * (0.25 * (cube[:, 3, 6] + 2 * cube[:, 2, 6] + cube[:, 1, 6]) -
                     0.25 * (cube[:, 3, 4] + 2 * cube[:, 2, 4] + cube[:, 1, 4]) -
                     0.25 * (cube[:, 3, 2] + 2 * cube[:, 2, 2] + cube[:, 1, 2]) +
                     0.25 * (cube[:, 3, 0] + 2 * cube[:, 2, 0] + cube[:, 1, 0]))

def calc_total_shear_velocity(westerly_shear_velocity, southerly_shear_velocity):
    """Calculate the total shear velocity."""
    return westerly_shear_velocity + southerly_shear_velocity

def wt_algorithm(cube, dataset):
    """
    Calculate Lamb weathertypes (lwt) for each time step using the 5x7 grid.
    Follows the ESMValTool logic.
    """
    logger.info("Calculating Lamb Weathertypes for %s", dataset)
    const1, const2, const3, const4 = calc_const()
    w_flow = calc_westerly_flow(cube)
    s_flow = calc_southerly_flow(cube, const1)
    total_flow = calc_resultant_flow(w_flow, s_flow)
    w_shear = calc_westerly_shear_velocity(cube, const2, const3)
    s_shear = calc_southerly_shear_velocity(cube, const4)
    total_shear = calc_total_shear_velocity(w_shear, s_shear)
    
    n = cube.shape[0]
    lwt = np.zeros(n, dtype=int)
    for i in range(n):
        direction = np.degrees(np.arctan2(-w_flow[i], -s_flow[i]))
        direction = (direction + 360) % 360 
        # Lamb pure directional types:
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
            elif direction < 337.5:
                lwt[i] = 8
        # Lamb’s pure cyclonic/anticyclonic types:
        elif (2 * total_flow[i]) < abs(total_shear[i]):
            if total_shear[i] > 0:
                lwt[i] = 9
            elif total_shear[i] < 0:
                lwt[i] = 10
        # Lamb’s synoptic/directional hybrid types:
        elif total_flow[i] < abs(total_shear[i]) < (2 * total_flow[i]):
            if total_shear[i] > 0:
                if (direction >= 337.5 or direction < 22.5):
                    lwt[i] = 11
                elif direction < 67.5:
                    lwt[i] = 12
                elif direction < 112.5:
                    lwt[i] = 13
                elif direction < 157.5:
                    lwt[i] = 14
                elif direction < 202.5:
                    lwt[i] = 15
                elif direction < 247.5:
                    lwt[i] = 16
                elif direction < 292.5:
                    lwt[i] = 17
                elif direction < 337.5:
                    lwt[i] = 18
            elif total_shear[i] < 0:
                if (direction >= 337.5 or direction < 22.5):
                    lwt[i] = 19
                elif direction < 67.5:
                    lwt[i] = 20
                elif direction < 112.5:
                    lwt[i] = 21
                elif direction < 157.5:
                    lwt[i] = 22
                elif direction < 202.5:
                    lwt[i] = 23
                elif direction < 247.5:
                    lwt[i] = 24
                elif direction < 292.5:
                    lwt[i] = 25
                elif direction < 337.5:
                    lwt[i] = 26
        elif abs(total_shear[i]) < 6 and total_flow[i] < 6:
            lwt[i] = 27
    return lwt

def map_lwt_to_slwt(lwt):
    """
    Map the 27 Lamb weathertypes (lwt) to 9 simplified weathertypes (slwt) exactly as in the paper.
    The mapping is defined as:
      - WT1: {1, 2, 19, 27}         -> N, NE, AN (group 1)
      - WT2: {3, 4, 20, 21, 22}      -> E, SE, AE, ASE (group 2)
      - WT3: {5, 6, 15, 16}          -> S, SW, CS, CSW (group 3)
      - WT4: {7, 8, 17, 18}          -> W, NW, CN, CNW (group 4)
      - WT5: {9}                   -> C, CW (group 5)
      - WT6: {10}                  -> A, ANE (group 6)
      - WT7: {11, 12, 13, 14}       -> CNE, CE, CSE (group 7)
      - WT8: {23, 24}              -> AS, ASW (group 8)
      - WT9: {25, 26}              -> AW, ANW (group 9)
    """
    mapping = {}
    for i in range(1, 28):
        if i in [1, 2, 19, 27]:
            mapping[i] = 1
        elif i in [3, 4, 20, 21, 22]:
            mapping[i] = 2
        elif i in [5, 6, 15, 16]:
            mapping[i] = 3
        elif i in [7, 8, 17, 18]:
            mapping[i] = 4
        elif i == 9:
            mapping[i] = 5
        elif i == 10:
            mapping[i] = 6
        elif i in [11, 12, 13, 14]:
            mapping[i] = 7
        elif i in [23, 24]:
            mapping[i] = 8
        elif i in [25, 26]:
            mapping[i] = 9
        else:
            mapping[i] = 0
    slwt = np.array([mapping.get(val, 0) for val in lwt], dtype=int)
    return slwt

#
# Processing routine per year
#
def process_year(year, region_bounds, target_lats, target_lons, data_path):
    file_path = os.path.join(data_path, f"slp_{year}_NA.nc")
    logger.info("Processing %s ...", file_path)
    try:
        ds = xr.open_dataset(file_path)
    except Exception as e:
        logger.error("Error opening %s: %s", file_path, e)
        return pd.DataFrame()
    ds = reorder_lat(ds)
    da = ds["msl"]
    # Interpolate the full ERA5 dataset to the target grid:
    da_interp = da.interp(latitude=target_lats, longitude=target_lons, method="linear")

    # Use the interpolated data (shape: time x 5 x 7) as our cube
    cube = da_interp.data
    lwt = wt_algorithm(cube, f"slp_{year}")
    slwt = map_lwt_to_slwt(lwt)
    time_vals = da_interp["time"].data
    df = pd.DataFrame({
        "datetime": pd.to_datetime(time_vals) + timedelta(minutes=30),
        "lwt": lwt,
        "slwt": slwt
    })
    return df

#
# Main routine with argparse and parallel processing
#
def main():
    parser = argparse.ArgumentParser(
        description="Calculate hourly Lamb weather types from ERA5 SLP data.")
    parser.add_argument("--region", type=str, required=True,
                        choices=list(SUBREGIONS.keys()),
                        help="Region name (e.g. southern_alps)")
    parser.add_argument("--ncores", type=int, default=32,
                        help="Number of cores for parallel processing (default: 32)")
    parser.add_argument("--serial", action="store_true",
                        help="Run processing in serial mode for debugging")
    parser.add_argument("--data_path", type=str,
                        default="/data/reloclim/normal/INTERACT/ERA5/surface",
                        help="Path to ERA5 SLP data")
    args = parser.parse_args()

    region = args.region
    region_bounds = SUBREGIONS[region]
    target_lats, target_lons, center_lat, center_lon = get_target_grid(region_bounds)
    logger.info("Region %s: using target grid centered at (%.2f, %.2f)", region, center_lat, center_lon)
    logger.info("Target latitudes (ascending): %s", target_lats)
    logger.info("Target longitudes: %s", target_lons)

    years = list(range(2000, 2021))
    results = []
    if args.serial:
        for year in years:
            df_year = process_year(year, region_bounds, target_lats, target_lons, args.data_path)
            results.append(df_year)
    else:
        pool = Pool(processes=args.ncores)
        tasks = [(year, region_bounds, target_lats, target_lons, args.data_path)
                 for year in years]
        results = pool.starmap(process_year, tasks)
        pool.close()
        pool.join()

    if results:
        df_all = pd.concat(results, ignore_index=True)
        df_all.sort_values("datetime", inplace=True)
        output_file = f"./csv/weather_types_{region}_sorted.csv"
        df_all.to_csv(output_file, index=False)
        logger.info("Output saved to %s", output_file)
    else:
        logger.error("No data processed.")

if __name__ == "__main__":
    main()
