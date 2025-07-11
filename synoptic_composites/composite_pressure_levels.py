#!/usr/bin/env python3
"""
Compute MJJAS ERA5 composites for MCS composite times at pressure levels
for specific climatological periods and save as a multidimensional netCDF file
with an extra weather type dimension. Includes WT0 (all events).

Theta_e for event composites is calculated per timestep using an external module (calc_atmospheric_variables.py).
Climatological reference for theta_e is loaded from a pre-calculated climatology file.
Climatology files are expected to contain May-September data for specific periods,
but composites are calculated ONLY for MJJAS (May, June, July, August, Sep).

Reads a composite CSV file containing MCS event times and weather types ('wt').
Events are filtered for MJJAS AND for the selected period (e.g., 2001-2020),
then grouped by weather type, month, and time offset. Corresponding ERA5 data
and pre-calculated climatology data are read.

Output Composite File Contains:
- <var>_mean: Mean of the raw variable for the events in the group.
- <var>_clim_mean: Mean of the corresponding monthly-hourly climatology fields
                   for the specific event times in the group.
- event_count: Number of events contributing at each grid cell.

Dimensions: (weather_type, month, time_diff, level, latitude, longitude)
where 'month' will be [5, 6, 7, 8, 9].

Usage:
    python composite_pressure_levels.py --era5_dir /path/to/era5/pressure_levels/ \\
         --clim_base_dir ./climatology_output_custom/ \\
         --period evaluation \\
         --comp_csv_base ./csv/composite_ --region southern_alps --levels 250,500,850 \\
         --output_dir ./composite_output_plev_custom/ [--ncores 32] [--serial] [--debug]

Author: David Kneidinger
Date: 2025-05-07
Last Modified: 2025-06-18
"""

import os
import sys
import argparse
from datetime import timedelta
import pandas as pd
import numpy as np
import xarray as xr
from multiprocessing import Pool
import metpy.calc as mpcalc
from metpy.units import units
import warnings
import logging
from typing import List, Dict, Tuple, Any, Optional, Set
from pathlib import Path

from calc_atmospheric_variables import calculate_theta_e_on_single_level

# --- Configuration ---
DOMAIN_LAT_MIN, DOMAIN_LAT_MAX = 25, 65
DOMAIN_LON_MIN, DOMAIN_LON_MAX = -20, 43
VAR_LIST = ['z', 't', 'q', 'u', 'v', 'w']
CALCULATED_VARS_EVENT = ['theta_e']
ALL_VARS_TO_SAVE = VAR_LIST + CALCULATED_VARS_EVENT
CLIM_VARS_TO_LOAD = VAR_LIST + ['theta_e']
PERIODS = {
    "historical": {"start": 1991, "end": 2020, "name_in_file": "historical"}
}
TARGET_MONTHS = [5, 6, 7, 8, 9]
CLIMATOLOGY_MONTHS = list(range(5, 10))


# --- Helper Functions ---
def get_era5_file(era5_dir: Path, year: int, month: int) -> Path:
    """Construct the ERA5 monthly filename (e.g., "2005-08_NA.nc")."""
    fname = f"{year}-{month:02d}_NA.nc"
    return era5_dir / fname

def reorder_lat(ds: xr.Dataset) -> xr.Dataset:
    """Ensure latitude is in ascending order."""
    lat_coord_name = None
    if 'latitude' in ds.coords: lat_coord_name = 'latitude'
    elif 'lat' in ds.coords: lat_coord_name = 'lat'

    if lat_coord_name and ds[lat_coord_name].values.size > 1 and ds[lat_coord_name].values[0] > ds[lat_coord_name].values[-1]:
        logging.debug(f"Reordering {lat_coord_name} to ascending.")
        sorted_lat_values = np.sort(ds[lat_coord_name].values)
        ds = ds.reindex({lat_coord_name: sorted_lat_values})
    return ds

def fix_lat_lon_names(ds: xr.Dataset) -> xr.Dataset:
    """Ensure standard 'latitude' and 'longitude' coordinate names."""
    rename_dict = {}
    if 'lat' in ds.coords and 'latitude' not in ds.coords:
        rename_dict['lat'] = 'latitude'
    if 'lon' in ds.coords and 'longitude' not in ds.coords:
        rename_dict['lon'] = 'longitude'
    if rename_dict:
        ds = ds.rename(rename_dict)
    return ds

def create_offset_cols(df: pd.DataFrame) -> Dict[int, str]:
    """Automatically extract offset column names from the DataFrame."""
    offset_cols = {}
    found_base = False
    time_cols_in_csv = [c for c in df.columns if 'time' in c.lower()]
    for col in time_cols_in_csv:
        offset_val_str = col.replace("time_minus", "").replace("time_plus", "").replace("h", "").replace("time_", "")
        if col.startswith("time_minus") or col.startswith("time_plus") or col == "time_0h":
            offset_val = int(offset_val_str)
            if col.startswith("time_minus"):
                offset_cols[-offset_val] = col
            elif col.startswith("time_plus"):
                offset_cols[offset_val] = col
            elif col == "time_0h":
                offset_cols[0] = col
                found_base = True
    if not found_base and 0 not in offset_cols:
        logging.warning("Base time column 'time_0h' not found.")
        if time_cols_in_csv:
            potential_base_candidates = [c for c in time_cols_in_csv if "minus" not in c and "plus" not in c]
            if potential_base_candidates:
                offset_cols[0] = potential_base_candidates[0]
                logging.info(f"Using '{offset_cols[0]}' as base time (offset 0) due to missing 'time_0h'.")
            else:
                 raise ValueError("No suitable base time column (e.g., 'time_0h' or similar) found in CSV.")
        else:
            raise ValueError("No suitable time column found in CSV.")
    return offset_cols


# --- Core Processing Function (Optimized) ---
def process_month_level_events(task: Tuple) -> Dict[str, Any]:
    """
    Process one (year, month, times_pd, era5_dir, levels, clim_ds) task.
    This version is optimized to reduce memory pressure within the worker.
    """
    year, month, times_pd, era5_dir, levels, clim_ds_input = task
    task_label = f"Y{year}-M{month:02d}"
    logging.debug(f"--- Starting Task: {task_label} ---")
    file_path = get_era5_file(era5_dir, year, month)

    comp_data = {}
    for lev_val in levels:
        level_key = str(lev_val)
        comp_data[level_key] = {f"{var}_sum": None for var in ALL_VARS_TO_SAVE}
        comp_data[level_key].update({f"{var}_clim_sum": None for var in ALL_VARS_TO_SAVE})
        comp_data[level_key].update({'count': None, 'lat': None, 'lon': None})

    if not file_path.exists():
        logging.warning(f"Task {task_label}: ERA5 file {file_path} not found. Skipping.")
        return {"composite": comp_data, "year": year, "month": month}

    # OPTIMIZATION: Keep climatology lazy. Do NOT .load() the whole month.
    clim_month_ds = None
    if clim_ds_input is not None and month in TARGET_MONTHS:
        clim_month_ds = clim_ds_input.sel(month=month)

    logging.debug(f"Task {task_label}: Loading raw data from {file_path}")
    with xr.open_dataset(file_path, chunks={'time': 'auto'}) as ds:
        ds = fix_lat_lon_names(ds)
        ds = reorder_lat(ds)
        ds = ds.sel(latitude=slice(DOMAIN_LAT_MIN, DOMAIN_LAT_MAX),
                    longitude=slice(DOMAIN_LON_MIN, DOMAIN_LON_MAX))

        ds_sel_raw = ds.reindex(time=times_pd, method='nearest', tolerance=pd.Timedelta('1H')).dropna(dim="time", how="all")

        if ds_sel_raw.time.size == 0:
             logging.warning(f"Task {task_label}: No valid MCS data found for specified times in {file_path}.")
             return {"composite": comp_data, "year": year, "month": month}

        actual_event_times = pd.to_datetime(ds_sel_raw['time'].values)
        event_hours = actual_event_times.hour.to_numpy()

        for lev_val in levels:
            level_key = str(lev_val)
            ds_level_raw_data = ds_sel_raw.sel(level=lev_val, method="nearest")

            if ds_level_raw_data.time.size == 0:
                continue

            # --- Calculate event sums ---
            count_arr = ds_level_raw_data[VAR_LIST[0]].notnull().sum(dim='time').compute()
            comp_data[level_key]['count'] = count_arr
            comp_data[level_key]['lat'] = ds_level_raw_data['latitude'].values
            comp_data[level_key]['lon'] = ds_level_raw_data['longitude'].values

            for var in VAR_LIST:
                if var in ds_level_raw_data:
                    comp_data[level_key][f"{var}_sum"] = ds_level_raw_data[var].sum(dim='time', skipna=True).compute()

            # --- PRESERVED .load() for external function compatibility ---
            if 't' in ds_level_raw_data and 'q' in ds_level_raw_data:
                input_ds_for_theta_e = ds_level_raw_data[['t', 'q']].load()
                theta_e_instantaneous_raw = calculate_theta_e_on_single_level(input_ds_for_theta_e)
                comp_data[level_key]["theta_e_sum"] = theta_e_instantaneous_raw.sum(dim='time', skipna=True).values
                del input_ds_for_theta_e, theta_e_instantaneous_raw

            # --- Calculate climatology sums (Optimized) ---
            if clim_month_ds is not None:
                clim_level_month_ds = clim_month_ds.sel(level=lev_val, method="nearest")
                # OPTIMIZATION: Select hours BEFORE loading data
                clim_data_for_events = clim_level_month_ds.sel(hour=xr.DataArray(event_hours, dims="event_time"))

                # Now load only the small, selected slice and sum it
                clim_sums_loaded = clim_data_for_events[CLIM_VARS_TO_LOAD].load()
                for var in CLIM_VARS_TO_LOAD:
                    if var in clim_sums_loaded:
                        comp_data[level_key][f"{var}_clim_sum"] = np.nansum(clim_sums_loaded[var].values, axis=0)
                del clim_sums_loaded, clim_data_for_events

    logging.debug(f"--- Finishing Task: {task_label} ---")
    return {"composite": comp_data, "year": year, "month": month}


# --- Combination Functions ---
def combine_tasks_results(task_results: List[Dict], levels: List[int]) -> Dict[str, Any]:
    """Combine task results for composites."""
    logging.debug(f"--- Combining results for {len(task_results)} tasks ---")
    overall = {}
    for lev_val in levels:
        key = str(lev_val)
        overall[key] = {f"{var}_sum": None for var in ALL_VARS_TO_SAVE}
        overall[key].update({f"{var}_clim_sum": None for var in ALL_VARS_TO_SAVE})
        overall[key].update({'count': None, 'lat': None, 'lon': None})

    valid_task_count = 0
    for i, result_dict in enumerate(task_results):
        result = result_dict.get("composite", {})
        if not result:
            logging.debug(f"  Combine task {i+1}: Skipping (no 'composite' key)")
            continue

        task_had_valid_level = False
        for lev_val in levels:
            key = str(lev_val)
            res_level = result.get(key)
            if res_level is None or res_level.get('count') is None:
                continue

            count_val = res_level.get('count')
            if isinstance(count_val, (int, float)): count_val = np.array([[count_val]])
            if count_val.size == 0 or np.all(count_val == 0):
                continue

            task_had_valid_level = True

            if overall[key]['lat'] is None and res_level.get('lat') is not None:
                overall[key]['lat'] = res_level.get('lat')
                overall[key]['lon'] = res_level.get('lon')
                overall[key]['count'] = count_val.astype(np.int32)
                for var in ALL_VARS_TO_SAVE:
                    overall[key][f"{var}_sum"] = res_level.get(f"{var}_sum") if res_level.get(f"{var}_sum") is not None else np.zeros_like(count_val, dtype=np.float64)
                    overall[key][f"{var}_clim_sum"] = res_level.get(f"{var}_clim_sum") if res_level.get(f"{var}_clim_sum") is not None else np.zeros_like(count_val, dtype=np.float64)
            elif overall[key]['lat'] is not None:
                if overall[key]['count'] is not None and overall[key]['count'].shape == count_val.shape:
                    overall[key]['count'] += count_val.astype(np.int32)
                    for var in ALL_VARS_TO_SAVE:
                        for sum_type_key_suffix in ["_sum", "_clim_sum"]:
                            sum_type = f"{var}{sum_type_key_suffix}"
                            current_sum = overall[key].get(sum_type)
                            new_sum = res_level.get(sum_type)
                            if new_sum is not None:
                                if current_sum is not None and current_sum.shape == new_sum.shape:
                                    overall[key][sum_type] += new_sum
                                elif current_sum is None:
                                    overall[key][sum_type] = new_sum.copy()
                                else:
                                     logging.warning(f"    Combine task {i+1} L{lev_val} Type={sum_type}: Shape mismatch. Overall: {current_sum.shape}, New: {new_sum.shape}. Skipping.")
                else:
                    logging.warning(f"  Combine task {i+1} L{lev_val}: Shape mismatch for count. Overall: {overall[key]['count'].shape if overall[key]['count'] is not None else 'None'}, New: {count_val.shape}. Skipping.")
        if task_had_valid_level:
            valid_task_count += 1

    logging.debug(f"--- Finished Combining Results. Processed {valid_task_count}/{len(task_results)} valid tasks. ---")
    for lev_val in levels:
        key = str(lev_val)
        if overall[key].get('lat') is not None and overall[key].get('lon') is not None:
            final_shape = (len(overall[key]['lat']), len(overall[key]['lon']))
            if overall[key].get('count') is None:
                 logging.debug(f"  Final combined count L{lev_val} is None. Filling with zeros.")
                 overall[key]['count'] = np.zeros(final_shape, dtype=np.int32)

            for var in ALL_VARS_TO_SAVE:
                if overall[key].get(f"{var}_sum") is None:
                    logging.debug(f"  Final combined {var}_sum L{lev_val} is None. Filling with NaN.")
                    overall[key][f"{var}_sum"] = np.full(final_shape, np.nan, dtype=np.float64)
                if overall[key].get(f"{var}_clim_sum") is None:
                    logging.debug(f"  Final combined {var}_clim_sum L{lev_val} is None. Filling with NaN.")
                    overall[key][f"{var}_clim_sum"] = np.full(final_shape, np.nan, dtype=np.float64)
        else:
             logging.warning(f"  No valid lat/lon found after combining tasks for L{lev_val}.")
    return overall


# --- NetCDF Saving Functions ---
def save_composites_to_netcdf(results_wt: Dict, weather_types: List, months: List[int],
                              time_offsets: List[int], levels: List[int],
                              lat: np.ndarray, lon: np.ndarray,
                              output_file: Path, period_details: Dict,
                              processed_years: Set[int], missing_years: List[int]):
    """
    Save composite means (raw and climatology) and counts to a NetCDF file.
    """
    logging.info(f"--- Saving final composite NetCDF to: {output_file} ---")
    n_wt = len(weather_types); n_months = len(months); n_offsets = len(time_offsets)
    n_levels = len(levels); nlat = len(lat); nlon = len(lon)

    comp_arrays_mean = {var: np.full((n_wt,n_months,n_offsets,n_levels,nlat,nlon), np.nan, dtype=np.float32) for var in ALL_VARS_TO_SAVE}
    comp_arrays_clim_mean = {var: np.full((n_wt,n_months,n_offsets,n_levels,nlat,nlon), np.nan, dtype=np.float32) for var in ALL_VARS_TO_SAVE}
    comp_arrays_count_grid = np.full((n_wt,n_months,n_offsets,n_levels,nlat,nlon), 0, dtype=np.int32)

    logging.debug("Populating final arrays for NetCDF...")
    for wi, wt in enumerate(weather_types):
        for mi, m_val in enumerate(months):
            for oi, off in enumerate(time_offsets):
                comp_month_data = results_wt.get(wt, {}).get(off, {}).get(m_val, None)
                if comp_month_data is None:
                    continue
                for li, lev_val in enumerate(levels):
                    key = str(lev_val)
                    comp_level_data = comp_month_data.get(key)
                    if comp_level_data is None:
                        continue

                    count_arr = comp_level_data.get('count')
                    if count_arr is None or count_arr.size == 0 or np.all(count_arr == 0):
                        continue

                    comp_arrays_count_grid[wi,mi,oi,li,:,:] = count_arr

                    for var in ALL_VARS_TO_SAVE:
                        for sum_type_suffix, mean_array_target in [("_sum", comp_arrays_mean[var]),
                                                                    ("_clim_sum", comp_arrays_clim_mean[var])]:
                            sum_type_key = f"{var}{sum_type_suffix}"
                            var_sum_val = comp_level_data.get(sum_type_key)
                            if var_sum_val is not None:
                                with np.errstate(divide='ignore', invalid='ignore'):
                                    mean_val = var_sum_val / count_arr
                                mean_array_target[wi,mi,oi,li,:,:] = np.where(count_arr > 0, mean_val, np.nan)

    logging.debug("Creating xarray Dataset...")
    ds_vars = {}
    for var in ALL_VARS_TO_SAVE:
        ds_vars[f"{var}_mean"] = xr.DataArray(comp_arrays_mean[var], dims=("weather_type","month","time_diff","level","latitude","longitude"), coords={"weather_type":weather_types, "month":months, "time_diff":time_offsets, "level":levels, "latitude":lat, "longitude":lon}, name=f"{var}_mean", attrs={"long_name":f"Mean raw {var}"})
        ds_vars[f"{var}_clim_mean"] = xr.DataArray(comp_arrays_clim_mean[var], dims=("weather_type","month","time_diff","level","latitude","longitude"), coords={"weather_type":weather_types, "month":months, "time_diff":time_offsets, "level":levels, "latitude":lat, "longitude":lon}, name=f"{var}_clim_mean", attrs={"long_name":f"Mean MJJAS climatology of {var}"})
        if var == 'theta_e':
             ds_vars[f"{var}_mean"].attrs['units'] = 'K'
             ds_vars[f"{var}_clim_mean"].attrs['units'] = 'K'

    ds_vars["event_count"] = xr.DataArray(comp_arrays_count_grid, dims=("weather_type","month","time_diff","level","latitude","longitude"), coords={"weather_type":weather_types, "month":months, "time_diff":time_offsets, "level":levels, "latitude":lat, "longitude":lon}, name="event_count", attrs={"long_name":"Number of events at each grid cell"})

    ds = xr.Dataset(ds_vars)
    ds.attrs["description"] = (f"MJJAS ERA5 composites for pressure-level variables for MCS environments, "
                               f"stratified by weather type. Theta_e for events calculated per timestep using calc_atmospheric_variables.py. "
                               f"Climatology for theta_e loaded from pre-calculated file. "
                               f"Climatology based on MJJAS for period '{period_details['name_in_file']}' ({period_details['start']}-{period_details['end']}).")
    ds.attrs["history"] = f"Created by composite_pressure_levels.py on {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S %Z')}"
    ds.attrs["source"] = "ERA5 pressure level data"
    ds.attrs["climatology_source_period_name"] = period_details['name_in_file']
    ds.attrs["climatology_source_period_years"] = f"{period_details['start']}-{period_details['end']}"
    ds.attrs["climatology_months_included"] = "May-September (used for clim), composites for June-August"
    ds.attrs["data_period_years_processed"] = ", ".join(map(str, sorted(list(processed_years))))
    ds.attrs["data_period_years_missing_in_csv"] = ", ".join(map(str, sorted(missing_years))) if missing_years else "None"

    encoding = {vname: {'zlib': True, 'complevel': 4, '_FillValue': np.float32(1e20)} for vname in ds.data_vars}
    output_file.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_file, encoding=encoding, mode='w')
    logging.info(f"Saved composite data to {output_file}")

# --- Main Script ---
def main():
    parser = argparse.ArgumentParser(
        description="Compute MJJAS monthly ERA5 pressure level composites. Theta_e for events calculated per timestep via external module. Theta_e climatology loaded from file.")
    # ... (Argument parsing is unchanged) ...
    parser.add_argument("--era5_dir", type=Path, default='/data/reloclim/normal/INTERACT/ERA5/pressure_levels/',
                        help="Directory containing ERA5 monthly files (e.g., 2005-08_NA.nc)")
    parser.add_argument("--clim_base_dir", type=Path, default="/home/dkn/climatology/ERA5/",
                        help="Base directory where period-specific MJJAS climatology files are stored.")
    parser.add_argument("--period", type=str, default="historical", choices=PERIODS.keys(),
                        help=f"Climatology period to use: {list(PERIODS.keys())}. Default: historical.")
    parser.add_argument("--comp_csv_base", type=str, default='/nas/home/dkn/Desktop/MoCCA/composites/scripts/synoptic_composites/csv/composite_',
                        help="Base path for composite CSV files (e.g., './csv/composite_')")
    parser.add_argument("--levels", type=str, default="250,500,850",
                        help="Comma-separated pressure levels in hPa (e.g., 250,500,850)")
    parser.add_argument("--region", type=str, required=True,
                        help="Subregion name, used to find CSV file (e.g., Alps)")
    parser.add_argument("--output_dir", type=Path, default='/home/dkn/composites/ERA5/',
                        help="Directory to save output composite netCDF files")
    parser.add_argument("--ncores", type=int, default=32, help="Number of cores for parallel processing")
    parser.add_argument("--serial", action='store_true', help="Run in serial mode for debugging")
    parser.add_argument("--time_offsets", type=str, default="-12,0,12",
                        help="Comma-separated list of time offsets in hours (e.g., -12,-6,0,6,12)")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG level logging.")
    parser.add_argument("--noMCS", action="store_true", help="False: composite of initMCS; True: composite of noMCS; Default=False")
    args = parser.parse_args()


    # ... (Logging setup and initial file/period setup is unchanged) ...
    log_level = logging.DEBUG if args.debug else logging.INFO
    log_filename = "composite_pressure_level.log"
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='w'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    logging.info("--- Starting Pressure Level Composite Script (Theta_e event calc via module, Theta_e clim loaded) ---")
    logging.info(f"Run arguments: {args}")
    logging.info(f"Logging output to: {log_filename}")

    levels_parsed = [int(l.strip()) for l in args.levels.split(',')]

    comp_csv_file_suffix = "_nomcs.csv" if args.noMCS else "_mcs.csv"
    comp_csv_file = Path(f"{args.comp_csv_base}{args.region}{comp_csv_file_suffix}")
    if not comp_csv_file.exists():
        logging.error(f"Composite CSV file not found: {comp_csv_file}")
        sys.exit(1)

    args_time_offsets_str = "0" if args.noMCS else args.time_offsets

    selected_period_details = PERIODS[args.period]
    clim_period_name_in_file = selected_period_details["name_in_file"]
    clim_start_year_period = selected_period_details["start"]
    clim_end_year_period = selected_period_details["end"]

    # ... (Climatology file loading logic is unchanged) ...
    clim_files_to_load = []
    clim_base_var_dir = args.clim_base_dir
    logging.info(f"Attempting to load climatology files for variables: {CLIM_VARS_TO_LOAD}")
    for var in CLIM_VARS_TO_LOAD:
        fname = f"era5_plev_{var}_clim_may_sep_{clim_period_name_in_file}_{clim_start_year_period}-{clim_end_year_period}.nc"
        fpath = clim_base_var_dir / fname
        if not fpath.exists():
            logging.error(f"Required climatology file not found: {fpath}")
            logging.error(f"Ensure climatology for '{var}' (and all others in CLIM_VARS_TO_LOAD) exists for May-Sep.")
            sys.exit(1)
        clim_files_to_load.append(fpath)

    clim_ds = None
    datasets_to_merge = []
    logging.info(f"Loading and merging {len(clim_files_to_load)} climatology files for period '{args.period}'...")
    for f in clim_files_to_load:
        logging.debug(f"  Opening clim file: {f}")
        datasets_to_merge.append(xr.open_dataset(f))

    clim_ds = xr.merge(datasets_to_merge, compat='override')
    logging.info("Climatology files merged.")

    clim_ds = fix_lat_lon_names(clim_ds)
    clim_ds = clim_ds.sel(latitude=slice(DOMAIN_LAT_MIN, DOMAIN_LAT_MAX),
                          longitude=slice(DOMAIN_LON_MIN, DOMAIN_LON_MAX))
    clim_ds = reorder_lat(clim_ds)

    if not {'month', 'hour', 'level'}.issubset(clim_ds.dims):
        raise ValueError("Merged climatology must have 'month', 'hour', 'level' dimensions.")

    missing_vars_in_clim = [v for v in CLIM_VARS_TO_LOAD if v not in clim_ds]
    if missing_vars_in_clim:
        raise ValueError(f"Variables {missing_vars_in_clim} not found in merged climatology dataset. Ensure all files (including for theta_e) were loaded correctly.")

    clim_months_present = np.sort(clim_ds.month.values)
    if not np.array_equal(clim_months_present, CLIMATOLOGY_MONTHS):
         logging.warning(f"Merged climatology months {clim_months_present} do not match expected May-Sep {CLIMATOLOGY_MONTHS}.")
    logging.info(f"Climatology for PLEV vars (Period: {args.period}, Months: May-Sep) loaded and merged.")

    for ds_item in datasets_to_merge: ds_item.close()

    # ... (DataFrame loading and filtering is unchanged) ...
    base_col = "datetime" if args.noMCS else 'time_0h'
    df_all = pd.read_csv(comp_csv_file, parse_dates=[base_col])
    df_all[base_col] = df_all[base_col].dt.round("H")

    df_all['year'] = df_all[base_col].dt.year
    df_all['event_month_for_filter'] = df_all[base_col].dt.month

    logging.info(f"Filtering events for period: {args.period} ({clim_start_year_period}-{clim_end_year_period})")
    df_period = df_all[(df_all['year'] >= clim_start_year_period) & (df_all['year'] <= clim_end_year_period)].copy()

    target_years_set = set(range(clim_start_year_period, clim_end_year_period + 1))
    processed_years_set = set(df_period['year'].unique())
    missing_years_list = sorted(list(target_years_set - processed_years_set))

    if not processed_years_set:
         logging.warning(f"No events found in the CSV for the selected period {args.period}. Exiting.")
         if clim_ds is not None: clim_ds.close()
         sys.exit(0)
    logging.info(f"Years processed from CSV within period: {sorted(list(processed_years_set))}")
    if missing_years_list:
        logging.warning(f"Years missing in CSV within period {args.period}: {missing_years_list}")
    else:
        logging.info(f"All years within period {args.period} present in CSV.")

    logging.info(f"Filtering events for target months (MJJAS): {TARGET_MONTHS}")
    df_filtered = df_period[df_period['event_month_for_filter'].isin(TARGET_MONTHS)].copy()

    if df_filtered.empty:
        logging.info(f"No events found in MJJAS for region {args.region} within period {args.period}. Exiting.")
        if clim_ds is not None: clim_ds.close()
        sys.exit(0)

    offset_col_names = create_offset_cols(df_filtered)
    time_offsets_parsed = sorted([int(o) for o in args_time_offsets_str.split(',')])

    missing_offsets = [off for off in time_offsets_parsed if off not in offset_col_names]
    if missing_offsets:
         raise ValueError(f"Requested time offsets {missing_offsets} not found in CSV columns: {list(offset_col_names.values())}")

    months_to_process_for_output = sorted(df_filtered['event_month_for_filter'].unique())
    logging.info(f"Processing events for months: {months_to_process_for_output} for region {args.region}")

    if 'wt' not in df_filtered.columns:
        raise KeyError(f"Weather type column 'wt' not found in {comp_csv_file}.")
    weather_types_from_csv = sorted(list(df_filtered['wt'].unique()))
    weather_types_to_process = [0] + weather_types_from_csv if 0 not in weather_types_from_csv else weather_types_from_csv
    logging.info(f"Processing weather types: {weather_types_to_process}")


    # --- OPTIMIZED Main Processing Loops ---
    results_wt = {wt: {off: {} for off in time_offsets_parsed} for wt in weather_types_to_process}

    logging.info("--- Starting Processing Loops ---")
    for wt_val in weather_types_to_process:
        df_wt = df_filtered[df_filtered['wt'] == wt_val].copy() if wt_val != 0 else df_filtered.copy()
        logging.info(f"Processing WT {wt_val} ('{'All Events' if wt_val==0 else f'Type {wt_val}'}') - {len(df_wt)} events (MJJAS, Period: {args.period})")
        if df_wt.empty: continue

        for off_val in time_offsets_parsed:
            offset_col = offset_col_names[off_val]
            logging.info(f"  Processing Offset: {off_val}h (Column: {offset_col})")
            df_wt_off = df_wt.copy()
            if not pd.api.types.is_datetime64_any_dtype(df_wt_off[offset_col]):
                 df_wt_off[offset_col] = pd.to_datetime(df_wt_off[offset_col], errors='coerce')
            df_wt_off = df_wt_off.dropna(subset=[offset_col])

            df_wt_off['year_for_grouping'] = df_wt_off[offset_col].dt.year
            df_wt_off['month_for_grouping'] = df_wt_off['event_month_for_filter']
            groups = df_wt_off.groupby(['year_for_grouping', 'month_for_grouping'])

            tasks = []
            for (year_group, month_group), group_data in groups:
                t_list_pd = pd.DatetimeIndex(group_data[offset_col].tolist())
                if not t_list_pd.empty:
                    tasks.append((year_group, month_group, t_list_pd, args.era5_dir, levels_parsed, clim_ds))

            logging.info(f"    WT {wt_val}, Offset {off_val}h: Created {len(tasks)} tasks (year-month groups).")
            if not tasks: continue

            tasks_by_month_comp = {m: [] for m in months_to_process_for_output}

            pool_context = Pool(processes=args.ncores) if not args.serial else None
            task_iterator = pool_context.imap_unordered(process_month_level_events, tasks) if pool_context else map(process_month_level_events, tasks)

            logging.info(f"    Running {len(tasks)} tasks and collecting results iteratively...")
            for result in task_iterator:
                if result['month'] in tasks_by_month_comp:
                    tasks_by_month_comp[result['month']].append(result)

            if pool_context:
                pool_context.close()
                pool_context.join()

            month_composites = {}
            logging.info(f"    Combining results for WT {wt_val}, Offset {off_val}h...")
            for m_out in months_to_process_for_output:
                if m_out in tasks_by_month_comp and tasks_by_month_comp[m_out]:
                    month_composites[m_out] = combine_tasks_results(tasks_by_month_comp[m_out], levels_parsed)
                else:
                    month_composites[m_out] = {
                        str(lvl): {
                            **{f"{var}_sum": None for var in ALL_VARS_TO_SAVE},
                            **{f"{var}_clim_sum": None for var in ALL_VARS_TO_SAVE},
                            'count': None, 'lat': None, 'lon': None
                        } for lvl in levels_parsed
                    }
            results_wt[wt_val][off_val] = month_composites
            logging.info(f"    Finished combining for WT {wt_val}, Offset {off_val}h.")
    logging.info("--- Finished Processing Loops ---")


    # --- Post-processing and Saving Results (Unchanged) ---
    logging.info("--- Post-processing and Saving Results ---")
    lat_coords, lon_coords = None, None
    found_coords = False
    for wt_chk in weather_types_to_process:
        for off_chk in time_offsets_parsed:
             for m_chk in months_to_process_for_output:
                  for lev_chk in levels_parsed:
                      sample_comp = results_wt.get(wt_chk,{}).get(off_chk,{}).get(m_chk,{}).get(str(lev_chk),None)
                      if sample_comp and sample_comp.get('lat') is not None and sample_comp.get('lon') is not None:
                          lat_coords = sample_comp['lat']; lon_coords = sample_comp['lon']
                          found_coords = True; break
                  if found_coords: break
             if found_coords: break
        if found_coords: break

    if not found_coords:
        logging.warning("Could not find valid lat/lon in any processed composite. Trying clim_ds for coords.")
        if clim_ds is not None and 'latitude' in clim_ds and 'longitude' in clim_ds:
            lat_coords = clim_ds['latitude'].values; lon_coords = clim_ds['longitude'].values
            if lat_coords is not None and lon_coords is not None: found_coords = True
        if not found_coords:
            raise ValueError("Could not find valid lat/lon coordinates from results or climatology. Cannot save.")

    if lat_coords is None or lon_coords is None:
        raise ValueError("Lat/lon coordinate information is missing. Cannot save output file.")

    output_file_name_suffix = "_nomcs.nc" if args.noMCS else ".nc"
    output_file_comp = args.output_dir / f"composite_plev_{args.region}_wt_clim_{args.period}{output_file_name_suffix}"

    save_composites_to_netcdf(results_wt, weather_types_to_process, months_to_process_for_output,
                              time_offsets_parsed, levels_parsed, lat_coords, lon_coords, output_file_comp,
                              selected_period_details,
                              processed_years=processed_years_set,
                              missing_years=missing_years_list)

    if clim_ds is not None: clim_ds.close()
    logging.info("--- Processing complete. ---")


if __name__ == '__main__':
    warnings.filterwarnings("ignore", message="Relative humidity >120%, ensure proper units.")
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    warnings.filterwarnings("ignore", message=".*Slicing is producing a large chunk.*")
    warnings.filterwarnings("ignore", message=".*Converting non-nanosecond precision datetime values to nanosecond precision.*")
    main()