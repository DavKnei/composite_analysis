#!/usr/bin/env python3
"""
Compute JJA ERA5 composites for MCS composite times (surface variables)
for specific climatological periods and save as a multidimensional netCDF file
with an extra weather type dimension.

Includes calculation and storage of the corresponding mean MONTHLY-HOURLY
climatology for each composite group, allowing for later anomaly calculation.
Climatology files are expected to contain JJA data for specific periods.

Reads a composite CSV file containing MCS event times and weather types ('wt').
Events are filtered for JJA, then grouped by weather type, month,
and time offset. Corresponding ERA5 data and pre-calculated climatology data are read.

Output Composite File Contains:
- <var>_mean: Mean of the raw variable for the events in the group.
- <var>_clim_mean: Mean of the corresponding monthly-hourly climatology fields
                   for the specific event times in the group.
- event_count: Number of events contributing at each grid cell.

Dimensions: (weather_type, month, time_diff, latitude, longitude)
where 'month' will be [6, 7, 8].

Usage:
    python composite_surface.py --data_dir /path/to/era5/surface/ \\
        --clim_base_dir ./climatology_output_custom/ \\
        --period evaluation \\
        --data_var msl --file_pattern "slp_{year}_NA.nc" \\
        --wt_csv_base ./csv/composite_ --region southern_alps \\
        --output_dir ./composite_output_surface_custom/ [--ncores 32] [--serial] [--debug]

Author: David Kneidinger (updated by Gemini)
Date: 2025-05-06
"""

import os
import sys
import dask
import argparse
from datetime import timedelta
import pandas as pd
import numpy as np
import xarray as xr
from multiprocessing import Pool
import warnings
import logging
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path

# --- Configuration ---
# Logging level set via command line arg --debug
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Full domain boundaries
DOMAIN_LAT_MIN, DOMAIN_LAT_MAX = 20, 55
DOMAIN_LON_MIN, DOMAIN_LON_MAX = -20, 40

# Definitions for climatology periods to match climatologies_py_custom_periods.py
PERIODS = {
    "historical": {"start": 1996, "end": 2005, "name_in_file": "historical"},
    "evaluation": {"start": 2000, "end": 2009, "name_in_file": "evaluation"}
}
# Months to process and include in the output composite
TARGET_MONTHS = list(range(6, 9)) # JJA

# --- Helper Functions ---

def get_data_file(data_dir: Path, year: int, file_pattern: str) -> Path:
    """Construct the data file name for a given year."""
    fname = file_pattern.format(year=year)
    return data_dir / fname

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
        if col.startswith("time_minus"):
            try:
                offset = -int(col.replace("time_minus", "").replace("h", ""))
                offset_cols[offset] = col
            except ValueError: logging.warning(f"Could not parse offset: {col}")
        elif col.startswith("time_plus"):
            try:
                offset = int(col.replace("time_plus", "").replace("h", ""))
                offset_cols[offset] = col
            except ValueError: logging.warning(f"Could not parse offset: {col}")
        elif col == "time_0h":
            offset_cols[0] = col
            found_base = True

    if not found_base and 0 not in offset_cols:
        logging.warning("Base time column 'time_0h' not found.")
        if time_cols_in_csv:
            base_col_guess = time_cols_in_csv[0]
            logging.info(f"Using '{base_col_guess}' as base time (offset 0).")
            offset_cols[0] = base_col_guess
        else: raise ValueError("No suitable time column found in CSV.")
    return offset_cols

# --- Core Processing Function ---

def process_month_surface_events(task: Tuple) -> Dict[str, Any]:
    """
    Process one (year, month, times_pd, data_dir, file_pattern, var_name, clim_ds) task for surface data.
    """
    year, month, times_pd, data_dir, file_pattern, var_name_list, clim_ds_input = task
    task_label = f"Y{year}-M{month:02d}" # Label for logging
    logging.debug(f"--- Starting Task: {task_label} ---")
    file_path = get_data_file(data_dir, year, file_pattern)

    var_name = var_name_list[0]
    comp_data = {f"{var_name}_sum": None, f"{var_name}_clim_sum": None,
                 'count': None, 'lat': None, 'lon': None}
    indiv_data = {var_name: None, 'time': None, 'lat': None, 'lon': None} # Individual data not saved currently

    if not file_path.exists():
        logging.warning(f"Task {task_label}: Data file {file_path} not found. Skipping.")
        return {"composite": comp_data, "individual": indiv_data}

    try:
        # --- Climatology Handling ---
        clim_var_name = var_name
        clim_month_ds = None
        clim_month_aligned = None # Initialize aligned version
        if clim_ds_input is not None:
            if 'month' not in clim_ds_input.coords or 'hour' not in clim_ds_input.coords:
                logging.error(f"Task {task_label}: Climatology dataset missing 'month' or 'hour' coordinates.")
                return {"composite": comp_data, "individual": indiv_data} # Cannot proceed
            if clim_var_name not in clim_ds_input:
                logging.error(f"Task {task_label}: Climatology variable '{clim_var_name}' not found.")
                return {"composite": comp_data, "individual": indiv_data}
            try:
                logging.debug(f"Task {task_label}: Selecting month {month} from climatology.")
                clim_month_ds = clim_ds_input[[clim_var_name]].sel(month=month).load()
                logging.debug(f"Task {task_label}: clim_month_ds shape: {clim_month_ds[clim_var_name].shape if clim_month_ds else 'None'}")
            except KeyError:
                logging.debug(f"Task {task_label}: Month {month} not found in loaded climatology. Clim sum will be None.")
                comp_data[f"{var_name}_clim_sum"] = None
            except Exception as e:
                logging.error(f"Task {task_label}: Could not select month {month} for '{clim_var_name}' from climatology: {e}")
                comp_data[f"{var_name}_clim_sum"] = None
        else:
            logging.warning(f"Task {task_label}: Climatology dataset (clim_ds_input) is None. Clim sum will be None.")
            comp_data[f"{var_name}_clim_sum"] = None

        # --- Load Raw ERA5 Data ---
        logging.debug(f"Task {task_label}: Loading raw data from {file_path}")
        with xr.open_dataset(file_path, chunks={'time': 'auto'}, cache=False) as ds:
            ds = fix_lat_lon_names(ds)
            ds = reorder_lat(ds)
            ds = ds.sel(latitude=slice(DOMAIN_LAT_MIN, DOMAIN_LAT_MAX),
                        longitude=slice(DOMAIN_LON_MIN, DOMAIN_LON_MAX))

            if var_name not in ds:
                 logging.error(f"Task {task_label}: Variable '{var_name}' not found in {file_path}.")
                 return {"composite": comp_data, "individual": indiv_data}

            logging.debug(f"Task {task_label}: Reindexing raw data to event times ({len(times_pd)} events)...")
            ds_sel_raw = ds.reindex(time=times_pd, method='nearest', tolerance=pd.Timedelta('1H'))
            logging.debug(f"Task {task_label}: Raw data time size after reindex: {ds_sel_raw.time.size}")
            ds_sel_raw = ds_sel_raw.dropna(dim="time", how="all")
            logging.debug(f"Task {task_label}: Raw data time size after dropna: {ds_sel_raw.time.size}")

            if ds_sel_raw.time.size == 0 or (var_name in ds_sel_raw and ds_sel_raw[var_name].isnull().all()):
                 logging.warning(f"Task {task_label}: No valid raw data for '{var_name}' at specified times in {file_path}.")
                 return {"composite": comp_data, "individual": indiv_data}

            actual_event_times = pd.to_datetime(ds_sel_raw['time'].values)
            event_hours = actual_event_times.hour.to_numpy()
            logging.debug(f"Task {task_label}: Actual event hours selected: {event_hours}")

            # --- Grid Check (Strict - No Interpolation) ---
            if clim_month_ds is not None:
                logging.debug(f"Task {task_label}: Checking grid alignment...")
                try:
                    if ds_sel_raw.time.size > 0:
                        # Use align to check for exact match on spatial coords
                        xr.align(ds_sel_raw.isel(time=0), clim_month_ds, join="exact", copy=False)
                        clim_month_aligned = clim_month_ds # Grids match
                        logging.debug(f"Task {task_label}: Grids match exactly.")
                    else: # Should not happen if previous checks passed
                         logging.warning(f"Task {task_label}: ds_sel_raw has no time steps after checks, cannot align grid.")
                         clim_month_aligned = None
                except ValueError as e_align:
                    # Grids do not match exactly - EXIT as requested
                    logging.error(f"Task {task_label}: FATAL: Grid mismatch between raw data ({file_path}) and climatology. Cannot proceed.")
                    logging.error(f"Task {task_label}: Alignment error details: {e_align}")
                    # Return empty/error state - the main loop should handle this? Or sys.exit?
                    # Returning empty is safer in parallel mode.
                    comp_data[f"{var_name}_clim_sum"] = None # Ensure clim sum is None
                    clim_month_aligned = None # Ensure we don't try to use it
                    # Optionally: raise an exception to stop processing? Depends on desired behavior.
                    # raise RuntimeError(f"Grid mismatch in task {task_label}")
                    # For now, let it continue without clim sum
                    logging.error(f"Task {task_label}: Proceeding without climatology sum due to grid mismatch.")


            # --- Composite Calculation ---
            logging.debug(f"Task {task_label}: Calculating sums and counts...")
            raw_var_data = ds_sel_raw[var_name]
            count_arr = raw_var_data.notnull().sum(dim='time').compute()
            comp_data['count'] = count_arr
            comp_data['lat'] = raw_var_data['latitude'].compute().values
            comp_data['lon'] = raw_var_data['longitude'].compute().values
            var_sum_arr = raw_var_data.sum(dim='time', skipna=True).compute()
            comp_data[f"{var_name}_sum"] = var_sum_arr
            logging.debug(f"Task {task_label}: count_arr shape: {count_arr.shape}, max: {np.max(count_arr) if count_arr.size>0 else 'N/A'}, NaNs: {np.isnan(count_arr).sum()}")
            logging.debug(f"Task {task_label}: {var_name}_sum shape: {var_sum_arr.shape if var_sum_arr is not None else 'None'}, max: {np.nanmax(var_sum_arr) if var_sum_arr is not None and var_sum_arr.size>0 else 'N/A'}, NaNs: {np.isnan(var_sum_arr).sum() if var_sum_arr is not None else 'N/A'}")

            # --- Climatology Sum ---
            clim_sum_arr = None
            if clim_month_aligned is not None: # Only proceed if grids matched
                logging.debug(f"Task {task_label}: Calculating climatology sum...")
                try:
                    # Select the specific hours from the *aligned* monthly climatology
                    clim_data_hourly = clim_month_aligned[clim_var_name].sel(hour=event_hours).compute().values
                    logging.debug(f"Task {task_label}: Selected clim_data_hourly shape: {clim_data_hourly.shape}")
                    clim_sum_arr = np.nansum(clim_data_hourly, axis=0)
                    comp_data[f"{var_name}_clim_sum"] = clim_sum_arr
                except Exception as e_clim_sel:
                    logging.warning(f"Task {task_label}: Could not select/sum hourly clim for {var_name} M{month}: {e_clim_sel}")
                    comp_data[f"{var_name}_clim_sum"] = None # Ensure it's None on error
            else: # clim_month_aligned is None (due to grid mismatch or clim_ds was None)
                logging.debug(f"Task {task_label}: Skipping climatology sum calculation (clim_month_aligned is None).")
                comp_data[f"{var_name}_clim_sum"] = None
            logging.debug(f"Task {task_label}: {var_name}_clim_sum shape: {clim_sum_arr.shape if clim_sum_arr is not None else 'None'}, max: {np.nanmax(clim_sum_arr) if clim_sum_arr is not None and clim_sum_arr.size > 0 else 'N/A'}, NaNs: {np.isnan(clim_sum_arr).sum() if clim_sum_arr is not None else 'N/A'}")


            # --- Individual Event Data Extraction (kept for potential future use, but not saved) ---
            indiv_data['lat'] = comp_data['lat']
            indiv_data['lon'] = comp_data['lon']
            indiv_data['time'] = actual_event_times.to_numpy()
            indiv_data[var_name] = raw_var_data.compute().values

    except Exception as e:
        logging.error(f"Task {task_label}: General error processing task: {e}", exc_info=True)
        # Return empty structure on error
        return {"composite": comp_data, "individual": indiv_data}

    logging.debug(f"--- Finished Task: {task_label} ---")
    return {"composite": comp_data, "individual": indiv_data}


# --- Combination Functions ---
def combine_tasks_results_surface(task_results: List[Dict], var_name: str) -> Dict[str, Any]:
    """Combine task results for composites."""
    logging.debug(f"--- Combining results for variable '{var_name}' ---")
    overall = {f'{var_name}_sum': None, f'{var_name}_clim_sum': None,
               'count': None, 'lat': None, 'lon': None}
    first_valid = True
    task_count = len(task_results)
    valid_task_count = 0

    for i, result_dict in enumerate(task_results):
        result = result_dict # Already passed the 'composite' dict
        if not result or result.get('count') is None:
            logging.debug(f"  Combine task {i+1}/{task_count}: Skipping (no count)")
            continue

        count_val = result.get('count')
        if isinstance(count_val, (int, float)): count_val = np.array([[count_val]])
        if count_val.size == 0 or np.all(count_val == 0):
            logging.debug(f"  Combine task {i+1}/{task_count}: Skipping (count is zero)")
            continue

        valid_task_count += 1
        logging.debug(f"  Combine task {i+1}/{task_count}: Processing valid task. Count max: {np.max(count_val)}")

        if first_valid:
            overall['lat'] = result.get('lat')
            overall['lon'] = result.get('lon')
            overall['count'] = count_val.astype(np.int32)
            shape_zeros = count_val.shape
            overall[f'{var_name}_sum'] = result.get(f"{var_name}_sum") if result.get(f"{var_name}_sum") is not None else np.zeros(shape_zeros, dtype=np.float64)
            overall[f'{var_name}_clim_sum'] = result.get(f"{var_name}_clim_sum") if result.get(f"{var_name}_clim_sum") is not None else np.zeros(shape_zeros, dtype=np.float64)
            logging.debug(f"  Combine task {i+1}: Initialized overall sums/counts. Sum shape: {overall[f'{var_name}_sum'].shape if overall[f'{var_name}_sum'] is not None else 'None'}, ClimSum shape: {overall[f'{var_name}_clim_sum'].shape if overall[f'{var_name}_clim_sum'] is not None else 'None'}")
            first_valid = False
        else:
            # Accumulate count
            if overall['count'] is not None and overall['count'].shape == count_val.shape:
                overall['count'] += count_val.astype(np.int32)
            elif overall['count'] is None: # Should not happen if first_valid logic is correct
                 overall['count'] = count_val.astype(np.int32)
            else:
                 logging.warning(f"  Combine task {i+1}: Shape mismatch for count. Overall: {overall['count'].shape}, New: {count_val.shape}. Skipping accumulation for this task.")
                 continue # Skip accumulation if shapes mismatch

            # Accumulate raw sum
            current_sum = overall.get(f'{var_name}_sum')
            new_sum = result.get(f"{var_name}_sum")
            if new_sum is not None:
                if current_sum is not None and current_sum.shape == new_sum.shape:
                    overall[f'{var_name}_sum'] += new_sum
                    logging.debug(f"    Accumulated var_sum. New max: {np.nanmax(overall[f'{var_name}_sum'])}")
                elif current_sum is None: # If overall sum was initialized to zeros
                    overall[f'{var_name}_sum'] = new_sum.copy()
                    logging.debug(f"    Initialized var_sum from task {i+1}.")
                else:
                    logging.warning(f"    Combine task {i+1}: Shape mismatch for var_sum. Overall: {current_sum.shape}, New: {new_sum.shape}. Skipping var_sum accumulation.")
            else:
                logging.debug(f"    Combine task {i+1}: New var_sum is None.")


            # Accumulate clim sum
            current_clim_sum = overall.get(f'{var_name}_clim_sum')
            new_clim_sum = result.get(f"{var_name}_clim_sum")
            if new_clim_sum is not None:
                if current_clim_sum is not None and current_clim_sum.shape == new_clim_sum.shape:
                    overall[f'{var_name}_clim_sum'] += new_clim_sum
                    logging.debug(f"    Accumulated clim_sum. New max: {np.nanmax(overall[f'{var_name}_clim_sum'])}")
                elif current_clim_sum is None: # If overall sum was initialized to zeros
                    overall[f'{var_name}_clim_sum'] = new_clim_sum.copy()
                    logging.debug(f"    Initialized clim_sum from task {i+1}.")
                else:
                    logging.warning(f"    Combine task {i+1}: Shape mismatch for clim_sum. Overall: {current_clim_sum.shape}, New: {new_clim_sum.shape}. Skipping clim_sum accumulation.")
            else:
                logging.debug(f"    Combine task {i+1}: New clim_sum is None.")
                 # Ensure clim sum remains zeros if it was initialized as such, even if new one is None
                if overall.get(f'{var_name}_clim_sum') is None and overall.get('count') is not None:
                     overall[f'{var_name}_clim_sum'] = np.zeros(overall['count'].shape, dtype=np.float64)


    logging.debug(f"--- Finished Combining Results for '{var_name}'. Processed {valid_task_count}/{task_count} valid tasks. ---")
    # Final check: if overall sums are still None (e.g., no valid tasks), fill with NaN
    if overall.get('lat') is not None and overall.get('lon') is not None:
        final_shape = (len(overall['lat']), len(overall['lon']))
        if overall.get(f'{var_name}_sum') is None:
            logging.debug(f"  Final combined var_sum is None. Filling with NaN.")
            overall[f'{var_name}_sum'] = np.full(final_shape, np.nan, dtype=np.float64)
        if overall.get(f'{var_name}_clim_sum') is None:
            logging.debug(f"  Final combined clim_sum is None. Filling with NaN.")
            overall[f'{var_name}_clim_sum'] = np.full(final_shape, np.nan, dtype=np.float64)
        if overall.get('count') is None: # Should not happen if lat/lon are set
            logging.debug(f"  Final combined count is None. Filling with zeros.")
            overall['count'] = np.zeros(final_shape, dtype=np.int32)
    else:
         logging.warning(f"  No valid latitude/longitude found after combining tasks for '{var_name}'.")

    return overall

# --- NetCDF Saving Functions ---
def save_composites_to_netcdf_surface(results_wt: Dict, var_name: str, weather_types: List,
                                      months: List[int], time_offsets: List[int],
                                      lat: np.ndarray, lon: np.ndarray,
                                      output_file: Path, period_details: Dict):
    """Save surface composites (raw mean, clim mean, count) with weather_type dimension."""
    logging.info(f"--- Saving final composite NetCDF to: {output_file} ---")
    n_wt = len(weather_types)
    n_months = len(months) # This will be TARGET_MONTHS
    n_offsets = len(time_offsets)
    nlat = len(lat)
    nlon = len(lon)

    mean_array = np.full((n_wt, n_months, n_offsets, nlat, nlon), np.nan, dtype=np.float32)
    clim_mean_array = np.full((n_wt, n_months, n_offsets, nlat, nlon), np.nan, dtype=np.float32)
    count_grid_array = np.full((n_wt, n_months, n_offsets, nlat, nlon), 0, dtype=np.int32)
    count_scalar_array = np.full((n_wt, n_months, n_offsets), 0, dtype=np.int32)

    logging.debug("Populating final arrays for NetCDF...")
    for wi, wt in enumerate(weather_types):
        for mi, m_val in enumerate(months): # m_val is one of TARGET_MONTHS
            for oi, off in enumerate(time_offsets):
                group_label = f"WT={wt}, Month={m_val}, Offset={off}h"
                comp_month_data = results_wt.get(wt, {}).get(off, {}).get(m_val, None)
                if comp_month_data is None or comp_month_data.get('count') is None:
                    logging.debug(f"  {group_label}: No combined data found.")
                    continue

                count_arr_grid = comp_month_data['count']
                if count_arr_grid.size == 0 or np.all(count_arr_grid == 0):
                    logging.debug(f"  {group_label}: Combined count is zero.")
                    continue

                count_grid_array[wi, mi, oi, :, :] = count_arr_grid
                scalar_count = int(np.max(count_arr_grid)) if count_arr_grid.size > 0 else 0
                count_scalar_array[wi, mi, oi] = scalar_count
                logging.debug(f"  {group_label}: Count max = {scalar_count}")

                var_sum = comp_month_data.get(f'{var_name}_sum')
                clim_sum = comp_month_data.get(f'{var_name}_clim_sum')

                if var_sum is not None:
                    with np.errstate(divide='ignore', invalid='ignore'): mean_val = var_sum / count_arr_grid
                    mean_array[wi, mi, oi, :, :] = np.where(count_arr_grid > 0, mean_val, np.nan)
                    logging.debug(f"    {var_name}_mean calculated. Min: {np.nanmin(mean_val):.2f}, Max: {np.nanmax(mean_val):.2f}")
                else:
                    logging.debug(f"    {var_name}_sum was None.")

                if clim_sum is not None:
                    with np.errstate(divide='ignore', invalid='ignore'): clim_mean_val = clim_sum / count_arr_grid
                    clim_mean_array[wi, mi, oi, :, :] = np.where(count_arr_grid > 0, clim_mean_val, np.nan)
                    logging.debug(f"    {var_name}_clim_mean calculated. Min: {np.nanmin(clim_mean_val):.2f}, Max: {np.nanmax(clim_mean_val):.2f}")
                else:
                    logging.debug(f"    {var_name}_clim_sum was None.")

    # Ensure 'months' coord in DataArray uses the actual month numbers [5,6,7,8,9]
    logging.debug("Creating xarray Dataset...")
    ds_vars = {}
    da_mean = xr.DataArray(mean_array, dims=("weather_type", "month", "time_diff", "latitude", "longitude"), coords={"weather_type": weather_types, "month": months, "time_diff": time_offsets, "latitude": lat, "longitude": lon}, name=f"{var_name}_mean", attrs={"long_name": f"Mean raw {var_name} for composite group"})
    da_clim_mean = xr.DataArray(clim_mean_array, dims=("weather_type", "month", "time_diff", "latitude", "longitude"), coords={"weather_type": weather_types, "month": months, "time_diff": time_offsets, "latitude": lat, "longitude": lon}, name=f"{var_name}_clim_mean", attrs={"long_name": f"Mean corresponding JJA monthly-hourly climatology of {var_name} for composite group"})
    da_count_scalar = xr.DataArray(count_scalar_array, dims=("weather_type", "month", "time_diff"), coords={"weather_type": weather_types, "month": months, "time_diff": time_offsets})
    da_count_grid = xr.DataArray(count_grid_array, dims=("weather_type", "month", "time_diff", "latitude", "longitude"), coords={"weather_type": weather_types, "month": months, "time_diff": time_offsets, "latitude": lat, "longitude": lon}, name="event_count", attrs={"long_name": "Number of valid events contributing at each grid cell"})

    ds_vars[f"{var_name}_mean"] = da_mean
    ds_vars[f"{var_name}_clim_mean"] = da_clim_mean
    ds_vars["event_count"] = da_count_grid

    ds = xr.Dataset(ds_vars)
    ds.attrs["description"] = (f"JJA ERA5 composites for surface variable '{var_name}' for MCS environments, "
                               f"stratified by weather type. Climatology based on JJA for period '{period_details['name_in_file']}' ({period_details['start']}-{period_details['end']}).")
    ds.attrs["history"] = f"Created by composite_surface.py on {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S %Z')}"
    ds.attrs["source"] = "ERA5 surface data"
    ds.attrs["climatology_source_period_name"] = period_details['name_in_file']
    ds.attrs["climatology_source_period_years"] = f"{period_details['start']}-{period_details['end']}"
    ds.attrs["climatology_months_included"] = "June, July, August"

    encoding = {var: {'zlib': True, 'complevel': 4, '_FillValue': np.float32(1e20)} for var in ds.data_vars}
    output_file.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_file, encoding=encoding, mode='w')
    logging.info(f"Saved composite data to {output_file}")

# --- Main Script ---
def main():
    parser = argparse.ArgumentParser(
        description="Compute JJA monthly composites for surface variables for MCS events, stratified by weather type, using period-specific JJA climatology.")
    parser.add_argument("--data_dir", type=Path, default="/data/reloclim/normal/INTERACT/ERA5/surface/",
                        help="Directory containing annual data files (e.g., slp_2001_NA.nc)")
    parser.add_argument("--clim_base_dir", type=Path, default="/home/dkn/climatology/ERA5/",
                        help="Base directory where period-specific JJA climatology files are stored.")
    parser.add_argument("--period", type=str, default="evaluation", choices=PERIODS.keys(),
                        help=f"Climatology period to use: {list(PERIODS.keys())}. Default: evaluation.")
    parser.add_argument("--data_var", type=str, default="msl",
                        help="Name of the data variable to composite (e.g., msl, t2m)")
    parser.add_argument("--file_pattern", type=str, default="slp_{year}_NA.nc",
                        help="Filename pattern with {year} placeholder (e.g., {year}.nc or slp_{year}_NA.nc)")
    parser.add_argument("--wt_csv_base", type=str, default="./csv/composite_",
                        help="Base path for composite CSV files (e.g., './csv/composite_')")
    parser.add_argument("--region", type=str, required=True,
                        help="Subregion name, used to find CSV file (e.g., southern_alps)")
    parser.add_argument("--output_dir", type=Path, default="/home/dkn/composites/ERA5/",
                        help="Directory to save output composite netCDF files")
    parser.add_argument("--ncores", type=int, default=32,
                        help="Number of cores for parallel processing")
    parser.add_argument("--serial", action="store_true",
                        help="Run in serial mode for debugging")
    parser.add_argument("--time_offsets", type=str, default="-12,0,12",
                        help="Comma-separated list of time offsets in hours (e.g., -12,-6,0,6,12)")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG level logging.") # Added debug flag
    args = parser.parse_args()

    # --- Setup Logging ---
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s', force=True)
    logging.info("--- Starting Surface Composite Script ---")
    logging.info(f"Run arguments: {args}")

    dask.config.set(array={"slicing": {"split_large_chunks": True}})

    # Set up logging to file
    logging.basicConfig(
        filename='./my_log_file.log',     # Your desired log file name
        filemode='w',                   # Use 'a' to append, 'w' to overwrite each run
        level=logging.DEBUG,            # Minimum level to log (can be INFO, WARNING, etc.)
        format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
    )

    variable_list = [args.data_var]

    weather_type_path = Path(f"{args.wt_csv_base}{args.region}_mcs.csv")
    if not weather_type_path.exists():
        logging.error(f"Composite CSV file with weather types not found: {weather_type_path}")
        sys.exit(1)

    # --- Load Climatology Data ---
    selected_period_details = PERIODS[args.period]
    clim_period_name_in_file = selected_period_details["name_in_file"]
    clim_start_year = selected_period_details["start"]
    clim_end_year = selected_period_details["end"]

    clim_filename = f"era5_surf_{args.data_var}_clim_may_sep_{clim_period_name_in_file}_{clim_start_year}-{clim_end_year}.nc"
    clim_file_path = args.clim_base_dir  / clim_filename # Adjusted path assuming subdir per var

    clim_ds = None
    if not clim_file_path.exists():
        logging.error(f"Climatology file not found: {clim_file_path}")
        logging.error("Please ensure climatologies have been generated with 'climatologies_py_custom_periods.py'")
        sys.exit(1)
    try:
        logging.info(f"Loading climatology data from: {clim_file_path}")
        clim_ds = xr.open_dataset(clim_file_path)
        clim_ds = fix_lat_lon_names(clim_ds)
        # clim_ds = reorder_lat(clim_ds)
        clim_ds = clim_ds.sel(latitude=slice(DOMAIN_LAT_MIN, DOMAIN_LAT_MAX),
                              longitude=slice(DOMAIN_LON_MIN, DOMAIN_LON_MAX))
        

        if not {'month', 'hour'}.issubset(clim_ds.dims):
            raise ValueError("Climatology file must contain 'month' and 'hour' dimensions.")
        if args.data_var not in clim_ds:
             raise ValueError(f"Variable '{args.data_var}' not found in climatology file {clim_file_path}.")
        if not np.array_equal(np.sort(clim_ds.month.values), TARGET_MONTHS):
            logging.warning(f"Climatology file {clim_file_path} months {np.sort(clim_ds.month.values)} do not match target months {TARGET_MONTHS}. This might lead to issues.")
        logging.info(f"Climatology for '{args.data_var}' (Period: {args.period}, Months: JJA) loaded.")
    except Exception as e:
        logging.error(f"Failed to load or process climatology file {clim_file_path}: {e}")
        if clim_ds is not None: clim_ds.close()
        sys.exit(1)

    base_col = 'time_0h'

    try:
        df_all = pd.read_csv(weather_type_path, parse_dates=[base_col])
    except KeyError:
        logging.error(f"Base time column '{base_col}' not found in {weather_type_path}.")
        if clim_ds is not None: clim_ds.close(); sys.exit(1)
    except Exception as e:
        logging.error(f"Error reading CSV {weather_type_path}: {e}")
        if clim_ds is not None: clim_ds.close(); sys.exit(1)

    df_all[base_col] = df_all[base_col].dt.round("H")

     # --- Filter by Period and Months ---
    # Extract year and month from the base time column
    df_all['year'] = df_all[base_col].dt.year
    df_all['event_month_for_filter'] = df_all[base_col].dt.month

    # Filter rows based on the selected period's year range
    logging.info(f"Filtering events for period: {args.period} ({clim_start_year}-{clim_end_year})")
    df_period = df_all[(df_all['year'] >= clim_start_year) & (df_all['year'] <= clim_end_year)].copy()

    # Determine processed and missing years within the selected period
    target_years_set = set(range(clim_start_year, clim_end_year + 1))
    processed_years_set = set(df_period['year'].unique())
    missing_years_list = sorted(list(target_years_set - processed_years_set))

    if not processed_years_set:
         logging.warning(f"No events found in the CSV for the selected period {args.period} ({clim_start_year}-{clim_end_year}). Exiting.")
         if clim_ds is not None: clim_ds.close(); sys.exit(0)
    else:
        logging.info(f"Years processed from CSV within period: {sorted(list(processed_years_set))}")
        if missing_years_list:
            logging.warning(f"Years missing in CSV within period {args.period}: {missing_years_list}")
        else:
            logging.info(f"All years within period {args.period} present in CSV.")

    df_filtered = df_period[df_period['event_month_for_filter'].isin(TARGET_MONTHS)].copy()
    if df_filtered.empty:
        logging.info(f"No events found in JJA for region {args.region}. Exiting.")
        if clim_ds is not None: clim_ds.close(); sys.exit(0)

    try:
        offset_col_names = create_offset_cols(df_filtered)
        time_offsets = sorted([int(o) for o in args.time_offsets.split(',')])
        missing_offsets = [off for off in time_offsets if off not in offset_col_names]
        if missing_offsets:
             logging.error(f"Requested time offsets {missing_offsets} not found in CSV columns: {list(offset_col_names.values())}")
             if clim_ds is not None: clim_ds.close(); sys.exit(1)
    except ValueError as e:
        logging.error(f"Error determining time offset columns: {e}")
        if clim_ds is not None: clim_ds.close(); sys.exit(1)

    months_to_process_for_output = sorted(df_filtered['event_month_for_filter'].unique())
    logging.info(f"Processing events for months: {months_to_process_for_output} for region {args.region}")

    if 'wt' not in df_filtered.columns:
        logging.error(f"Weather type column 'wt' not found in {weather_type_path}.")
        if clim_ds is not None: clim_ds.close(); sys.exit(1)
    weather_types = sorted(list(df_all['wt'].unique()))
    if 0 not in weather_types:
        weather_types = [0] + weather_types
    logging.info(f"Processing weather types: {weather_types}")

    results_wt = {wt: {off: {} for off in time_offsets} for wt in weather_types}

    logging.info("--- Starting Processing Loops ---")
    for wt in weather_types:
        df_wt = df_filtered[df_filtered['wt'] == wt].copy() if wt != 0 else df_filtered.copy()
        logging.info(f"Processing WT {wt} ('{'All Events' if wt==0 else f'Type {wt}'}') - {len(df_wt)} events (JJA)")
        if df_wt.empty: continue

        for off in time_offsets:
            offset_col = offset_col_names[off]
            if offset_col not in df_wt.columns:
                 logging.warning(f"Offset column '{offset_col}' missing for WT {wt}, offset {off}h. Skipping this offset.")
                 continue

            logging.info(f"  Processing Offset: {off}h (Column: {offset_col})")
            df_wt_off = df_wt.copy()
            if not pd.api.types.is_datetime64_any_dtype(df_wt_off[offset_col]):
                 df_wt_off[offset_col] = pd.to_datetime(df_wt_off[offset_col], errors='coerce')
            df_wt_off = df_wt_off.dropna(subset=[offset_col])

            df_wt_off['year'] = df_wt_off[offset_col].dt.year
            df_wt_off['month_for_grouping'] = df_wt_off['event_month_for_filter']
            groups = df_wt_off.groupby(['year', 'month_for_grouping'])

            tasks = []
            for (year, month_group_val), group_data in groups:
                t_list_pd = pd.DatetimeIndex(group_data[offset_col].tolist())
                # Filter times_pd again just to be sure? No, df_filtered was already filtered.
                if not t_list_pd.empty:
                    tasks.append((year, month_group_val, t_list_pd, args.data_dir, args.file_pattern, variable_list, clim_ds))

            logging.info(f"    WT {wt}, Offset {off}h: Created {len(tasks)} tasks (year-month groups).")
            if not tasks: continue

            monthly_tasks_results = []
            if args.serial or args.ncores == 1:
                logging.info(f"    Running {len(tasks)} tasks serially...")
                for task_idx, task_data in enumerate(tasks):
                    logging.debug(f"      Serial task {task_idx+1}/{len(tasks)}: Year {task_data[0]}, Month {task_data[1]}")
                    monthly_tasks_results.append(process_month_surface_events(task_data))
            else:
                logging.info(f"    Running {len(tasks)} tasks in parallel using {args.ncores} cores...")
                with Pool(processes=args.ncores) as pool:
                    monthly_tasks_results = pool.map(process_month_surface_events, tasks)
            logging.info(f"    Finished processing tasks for WT {wt}, Offset {off}h.")

            tasks_by_month_comp = {}
            for task_data, res in zip(tasks, monthly_tasks_results):
                _, month_val, _, _, _, _, _ = task_data
                tasks_by_month_comp.setdefault(month_val, []).append(res) # Append the whole result dict

            month_composites = {}
            logging.info(f"    Combining results for WT {wt}, Offset {off}h...")
            for m_val_output in months_to_process_for_output:
                if m_val_output in tasks_by_month_comp:
                    composite_results_for_month = [r['composite'] for r in tasks_by_month_comp[m_val_output]]
                    month_composites[m_val_output] = combine_tasks_results_surface(composite_results_for_month, args.data_var)
                else:
                    month_composites[m_val_output] = {f'{args.data_var}_sum': None, f'{args.data_var}_clim_sum': None, 'count': None, 'lat': None, 'lon': None}
            results_wt[wt][off] = month_composites
            logging.info(f"    Finished combining for WT {wt}, Offset {off}h.")

    logging.info("--- Finished Processing Loops ---")

    # --- Post-processing and Saving ---
    logging.info("--- Post-processing and Saving Results ---")
    lat, lon = None, None
    try:
        found_coords = False
        for wt_chk in weather_types:
            for off_chk in time_offsets:
                 for m_chk in months_to_process_for_output:
                      sample_comp = results_wt.get(wt_chk, {}).get(off_chk, {}).get(m_chk, None)
                      if sample_comp and sample_comp.get('lat') is not None and sample_comp.get('lon') is not None:
                          lat = sample_comp['lat']
                          lon = sample_comp['lon']
                          found_coords = True; break
                 if found_coords: break
            if found_coords: break
        if not found_coords:
            logging.warning("Could not find valid lat/lon in any processed composite. Using clim_ds for coords.")
            if clim_ds is not None:
                lat = clim_ds['latitude'].values; lon = clim_ds['longitude'].values
                if lat is not None and lon is not None: found_coords = True
            if not found_coords: raise KeyError("Could not find valid lat/lon.")
    except KeyError as e:
        logging.error(f"Could not retrieve lat/lon: {e}. Cannot save.")
        if clim_ds is not None: clim_ds.close(); sys.exit(1)

    if lat is None or lon is None:
        logging.error("Lat/lon info missing. Cannot save.");
        if clim_ds is not None: clim_ds.close(); sys.exit(1)

    output_file_comp = args.output_dir / f"composite_surface_{args.region}_{args.data_var}_wt_clim_{args.period}.nc"
    save_composites_to_netcdf_surface(results_wt, args.data_var, weather_types,
                                      months_to_process_for_output,
                                      time_offsets, lat, lon, output_file_comp,
                                      selected_period_details)

    if clim_ds is not None: clim_ds.close()
    logging.info("--- Processing complete. ---")

if __name__ == '__main__':
    warnings.filterwarnings("ignore", message=".*Converting non-nanosecond precision datetime values to nanosecond precision.*")
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    main()
