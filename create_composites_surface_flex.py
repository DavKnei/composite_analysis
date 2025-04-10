#!/usr/bin/env python3
"""
Compute monthly ERA5 composites for MCS composite times and save as a multidimensional netCDF file.
Optionally, if a weather type CSV (with a column "lwt") is provided via --wt_csv,
composites are computed separately for each weather type and the output dimensions become:
    weather_type, month, time_diff, latitude, longitude

Usage (without weather type):
    python create_composites_surface.py --data_dir /data/reloclim/backup/MCS_database \
         --data_var precipitation --file_pattern "{year}.nc" \
         --comp_dir ./csv/ --region southern_alps --time_offsets -6,-3,0,3,6 \
         --output_dir output_composites [--ncores 32] [--serial]

Usage (with weather type):
    python create_composites_surface.py --data_dir /data/reloclim/backup/MCS_database \
         --data_var precipitation --file_pattern "{year}.nc" \
         --wt_csv ./csv/composite_ --region southern_alps --time_offsets -6,-3,0,3,6 \
         --output_dir output_composites [--ncores 32] [--serial]

Author: David Kneidinger (adapted)
Date: 2025-04-01
"""

import os
import argparse
from datetime import timedelta
import pandas as pd
import numpy as np
import xarray as xr
import netCDF4 as nc
from multiprocessing import Pool
import warnings

# Full domain boundaries
DOMAIN_LAT_MIN, DOMAIN_LAT_MAX = 20, 55
DOMAIN_LON_MIN, DOMAIN_LON_MAX = -20, 40

# Subregion definitions
SUBREGIONS = {
    'western_alps': {'lon_min': 3,   'lon_max': 8,   'lat_min': 43, 'lat_max': 49},
    'southern_alps': {'lon_min': 7.5, 'lon_max': 13,  'lat_min': 43, 'lat_max': 46},
    'dinaric_alps':  {'lon_min': 12.5,'lon_max': 20,  'lat_min': 42, 'lat_max': 46},
    'eastern_alps':  {'lon_min': 8,   'lon_max': 17,  'lat_min': 46, 'lat_max': 49}
}

def read_composite_csv(comp_csv_file, time_offset_col):
    """
    Read the composite CSV file for the region.
    The CSV must include the column specified by time_offset_col (e.g. "time_0h").
    The datetime values are rounded to the nearest hour.
    A column 'year' and 'month' are added from the time_offset_col.
    """
    df = pd.read_csv(comp_csv_file, parse_dates=[time_offset_col])
    df[time_offset_col] = df[time_offset_col].dt.round("H")
    df['year'] = df[time_offset_col].dt.year
    df['month'] = df[time_offset_col].dt.month
    return df

def get_data_file(data_dir, year, file_pattern):
    """
    Construct the data file name for a given year.
    """
    fname = file_pattern.format(year=year)
    return os.path.join(data_dir, fname)

def reorder_lat(ds):
    """
    Ensure that latitude is in ascending order.
    """
    if ds.latitude.values[0] > ds.latitude.values[-1]:
        ds = ds.reindex(latitude=list(np.sort(ds.latitude.values)))
    return ds

def fix_lat_lon(ds: xr.Dataset) -> xr.Dataset:
    """
    Check and fix the latitude and longitude coordinates and dimension names.
    """
    def is_2d(var):
        return var.ndim > 1

    # Process Latitude
    lat_candidate = None
    for dim in ds.dims:
        if dim.lower() in ["lat", "latitude"]:
            lat_candidate = dim
            break
    if lat_candidate is None:
        for key in ds.coords:
            if key.lower() in ["lat", "latitude"]:
                lat_candidate = key
                break
    if lat_candidate is None:
        raise ValueError("No candidate latitude coordinate found.")
    if lat_candidate != "latitude":
        if "latitude" in ds.data_vars:
            if is_2d(ds["latitude"]):
                ds = ds.rename({"latitude": "lat2d"})
        ds = ds.rename({lat_candidate: "latitude"})
    
    # Process Longitude
    lon_candidate = None
    for dim in ds.dims:
        if dim.lower() in ["lon", "longitude"]:
            lon_candidate = dim
            break
    if lon_candidate is None:
        for key in ds.coords:
            if key.lower() in ["lon", "longitude"]:
                lon_candidate = key
                break
    if lon_candidate is None:
        raise ValueError("No candidate longitude coordinate found.")
    if lon_candidate != "longitude":
        if "longitude" in ds.data_vars:
            if is_2d(ds["longitude"]):
                ds = ds.rename({"longitude": "lon2d"})
        ds = ds.rename({lon_candidate: "longitude"})
    
    if ds["latitude"].ndim == 1:
        if not np.all(np.diff(ds["latitude"].values) > 0):
            ds = ds.sortby("latitude")
    if ds["longitude"].ndim == 1:
        if not np.all(np.diff(ds["longitude"].values) > 0):
            ds = ds.sortby("longitude")
    
    return ds

def create_offset_cols(df):
    """
    Read available offset columns from the CSV.
    Returns a dictionary mapping offset (int) to column name.
    """
    offset_cols = {}
    for col in df.columns:
        if col.startswith("time_minus"):
            hours_str = col.replace("time_minus", "").replace("h", "")
            offset = -int(hours_str)
            offset_cols[offset] = col
        elif col.startswith("time_plus"):
            hours_str = col.replace("time_plus", "").replace("h", "")
            offset = int(hours_str)
            offset_cols[offset] = col
        elif col == "time_0h":
            offset_cols[0] = col
    return offset_cols

#########################################################################
# New functions for processing individual events along with composites
#########################################################################

def process_month_both_surface(task):
    """
    Process one (year, month, times, data_dir, file_pattern, var_list) task for surface data.
    Reads the annual file for the given year, selects data for the month,
    extracts the times using nearest neighbor and computes:
      - Composite: sum and count over events.
      - Individual: the data for each event as-is.
    Returns a dictionary with keys:
       "composite": { var: {'sum': composite_sum, 'count': composite_count, 'lat': lat, 'lon': lon} }
       "individual": { var: individual_event_array, "time": event_times, "lat": lat, "lon": lon }
    """
    year, month, times, data_dir, file_pattern, var_list = task
    file_path = get_data_file(data_dir, year, file_pattern)
    if not os.path.exists(file_path):
        print(f"Data file {file_path} not found for year {year}. Skipping.")
        return {"composite": {}, "individual": {}}
    try:
        ds = xr.open_dataset(file_path)
    except Exception as e:
        print(f"Error opening {file_path}: {e}")
        return {"composite": {}, "individual": {}}
    
    ds = fix_lat_lon(ds)
    ds = ds.sel(latitude=slice(DOMAIN_LAT_MIN, DOMAIN_LAT_MAX),
                longitude=slice(DOMAIN_LON_MIN, DOMAIN_LON_MAX))
    ds = ds.sel(time=ds.time.dt.month == month)
    
    time_vals = np.array([np.datetime64(t) for t in times])
    try:
        ds_sel = ds.sel(time=time_vals, method='nearest')
    except Exception as e:
        print(f"Error selecting times in {file_path}: {e}")
        ds.close()
        return {"composite": {}, "individual": {}}
    
    comp = {}
    indiv = {}
    for var in var_list:
        comp[var] = {
            'sum': None,
            'count': 0,
            'lat': None,
            'lon': None
        }
    for var in var_list:
        if var not in ds_sel:
            print(f"Variable '{var}' not found in {file_path}.")
            continue
        data = ds_sel[var]
        comp[var]['sum'] = data.sum(dim='time').values
        comp[var]['count'] = data.count(dim='time').values
        indiv[var] = data.values  # shape: (n_events, lat, lon)
    event_times = ds_sel['time'].values  # shape: (n_events,)
    if "latitude" in ds_sel.coords:
        lat_arr = np.sort(ds_sel["latitude"].values)
    else:
        lat_arr = None
    if "longitude" in ds_sel.coords:
        lon_arr = ds_sel["longitude"].values
    else:
        lon_arr = None
    for var in var_list:
        comp[var]['lat'] = lat_arr
        comp[var]['lon'] = lon_arr
    indiv_data = {}
    for var in var_list:
        indiv_data[var] = indiv[var]
    indiv_data["time"] = event_times
    indiv_data["lat"] = lat_arr
    indiv_data["lon"] = lon_arr
    
    ds.close()
    return {"composite": comp, "individual": indiv_data}

def combine_tasks_results(task_results, var_list):
    """
    Combine a list of task results into one composite dictionary.
    """
    overall = {}
    for var in var_list:
        overall[var] = {
            'sum': None,
            'count': 0,
            'lat': None,
            'lon': None
        }
    for result in task_results:
        if not result:
            continue
        for var in var_list:
            if var not in result:
                continue
            res = result[var]
            count_val = np.array(res.get("count", 0))
            if np.all(count_val == 0):
                continue
            if overall[var]['sum'] is None:
                overall[var]['sum'] = res.get("sum")
                overall[var]['count'] = count_val
                overall[var]['lat'] = np.array(res.get("lat"))
                overall[var]['lon'] = np.array(res.get("lon"))
            else:
                overall[var]['sum'] += res.get("sum")
                overall[var]['count'] += count_val
    return overall

def combine_tasks_individual_surface(task_results, var_list):
    """
    Combine a list of individual event task results (each is a dictionary keyed by variable)
    into one dictionary for a given month.
    For each variable, concatenate the event arrays along the event dimension.
    Also concatenate the time arrays.
    """
    overall = {}
    for var in var_list:
        overall[var] = []
    overall["time"] = []
    overall["lat"] = None
    overall["lon"] = None
    for result in task_results:
        if not result:
            continue
        for var in var_list:
            overall[var].append(result.get(var))
        overall["time"].append(result.get("time"))
        if overall["lat"] is None:
            overall["lat"] = result.get("lat")
            overall["lon"] = result.get("lon")
    for var in var_list:
        if overall[var]:
            overall[var] = np.concatenate(overall[var], axis=0)
        else:
            overall[var] = np.empty((0,))
    if overall["time"]:
        overall["time"] = np.concatenate(overall["time"], axis=0)
    else:
        overall["time"] = np.empty((0,))
    return overall

def save_to_netcdf_surface(composite_array, composite_counts_scalar, weather_types, months, time_offsets, lat, lon, output_file):
    """
    Save composites with weather_type dimension using xarray.
    Creates two variables per data field: mean and count.
    """
    ds_vars = {}
    for var in composite_array:
        da_mean = xr.DataArray(
            composite_array[var],
            dims=("weather_type", "month", "time_diff", "latitude", "longitude"),
            coords={
                "weather_type": weather_types,
                "month": months,
                "time_diff": time_offsets,
                "latitude": lat,
                "longitude": lon,
            },
            name=f"{var}_mean"
        )
        ds_vars[f"{var}_mean"] = da_mean
        
        da_count = xr.DataArray(
            composite_counts_scalar[var],
            dims=("weather_type", "month", "time_diff"),
            coords={
                "weather_type": weather_types,
                "month": months,
                "time_diff": time_offsets,
            },
            name=f"{var}_count"
        )
        ds_vars[f"{var}_count"] = da_count
        
    ds = xr.Dataset(ds_vars)
    ds.attrs["description"] = (
        "Monthly composites for variable(s) for MCS environments, computed separately for each weather type.\n"
        "Dimensions: weather_type, month, time_diff, latitude, longitude.\n"
        "Variables '<var>_count' give the number of events used per composite."
    )
    ds.attrs["history"] = "Created by create_composites_surface.py with weather_type dimension"
    ds.attrs["source"] = "Input data composites"
    ds.to_netcdf(output_file)
    print(f"Saved composite data to {output_file}")

def save_individual_events_to_netcdf_surface(results_wt_ind, weather_types, months, time_offsets, var_list, lat, lon, output_file):
    """
    Create an xarray Dataset from the individual events for surface composites with weather type.
    Dimensions:
       (weather_type, month, time_diff, time, latitude, longitude)
    Since groups may have different numbers of events, pad with NaNs up to the maximum number.
    """
    n_wt = len(weather_types)
    n_months = len(months)
    n_offsets = len(time_offsets)
    nlat= len(lat)
    nlon = len(lon)
    # Determine maximum number of events over all groups
    max_events = 0
    for wi, wt in enumerate(weather_types):
        for mi, m in enumerate(months):
            for oi, off in enumerate(time_offsets):
                group = results_wt_ind.get(wt, {}).get(off, {}).get(m, None)
                if group is None:
                    continue
                n_ev = group["time"].shape[0]
                if n_ev > max_events:
                    max_events = n_ev
    if max_events == 0:
        print("No individual event data found to save.")
        return
    
    indiv_arrays = {}
    for var in var_list:
        indiv_arrays[var] = np.full((n_wt, n_months, n_offsets, max_events, nlat, nlon),
                                     np.nan, dtype=np.float32)
    time_array = np.full((n_wt, n_months, n_offsets, max_events),
                         np.datetime64('NaT'), dtype='datetime64[ns]')
    
    for wi, wt in enumerate(weather_types):
        for mi, m in enumerate(months):
            for oi, off in enumerate(time_offsets):
                group = results_wt_ind.get(wt, {}).get(off, {}).get(m, None)
                if group is None:
                    continue
                n_ev = group["time"].shape[0]
                for var in var_list:
                    indiv_arrays[var][wi, mi, oi, :n_ev, :, :] = group[var]
                time_array[wi, mi, oi, :n_ev] = group["time"]

    ds_vars = {}
    for var in var_list:
        da = xr.DataArray(
            indiv_arrays[var],
            dims=("weather_type", "month", "time_diff", "time", "latitude", "longitude"),
            coords={
                "weather_type": weather_types,
                "month": months,
                "time_diff": time_offsets,
                "latitude": lat,  # placeholder for actual latitudes
                "longitude": lon,  # placeholder for actual longitudes
            },
            name=var
        )
        ds_vars[var] = da
    ds_vars["event_time"] = xr.DataArray(
        time_array,
        dims=("weather_type", "month", "time_diff", "time"),
        coords={
            "weather_type": weather_types,
            "month": months,
            "time_diff": time_offsets,
        },
        name="event_time"
    )
    
    ds = xr.Dataset(ds_vars)
    ds.attrs["description"] = (
        "Individual surface data for MCS events computed separately for each weather type. "
        "Dimensions: weather_type, month, time_diff, time, latitude, longitude. "
        "The 'time' dimension holds the event datetime."
    )
    ds.attrs["history"] = "Created by create_composites_surface.py (individual events added)"
    ds.attrs["source"] = "Surface data composites"
    ds.to_netcdf(output_file)
    print(f"Saved individual event data to {output_file}")


###########################################################################
# Main script
###########################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Compute monthly composites for MCS composite times and save as a multidimensional netCDF file. "
                    "Optionally, if a weather type CSV (--wt_csv) is provided, composites are computed per weather type."
    )
    parser.add_argument("--data_dir", type=str, default="/data/reloclim/backup/MCS_database/",
                        help="Directory containing annual data files (e.g., 2001.nc, 2002.nc, etc.)")
    parser.add_argument("--data_var", type=str, default="precipitation",
                        help="Name of the data variable to composite (e.g., msl)")
    parser.add_argument("--file_pattern", type=str, default="{year}.nc",
                        help="Filename pattern with {year} placeholder (e.g., {year}.nc)")
    parser.add_argument("--comp_dir", type=str, default="./csv/",
                        help="Directory containing composite CSV files (used if --wt_csv is not provided)")
    parser.add_argument("--wt_csv", type=str, default="./csv/composite_",
                        help="Base path for composite CSV files with weather type information (e.g., './csv/composite_'). "
                             "If provided, the file composite_<region>_mcs.csv is used and composites are computed per weather type.")
    parser.add_argument("--region", type=str, required=True,
                        help="Subregion to process (e.g., western_alps, southern_alps, dinaric_alps, eastern_alps)")
    parser.add_argument("--output_dir", type=str, default="/data/reloclim/normal/MoCCA/composites/composite_files/",
                        help="Base directory to save output composite netCDF files")
    parser.add_argument("--ncores", type=int, default=32,
                        help="Number of cores to use for parallel processing")
    parser.add_argument("--serial", action="store_true",
                        help="Run in serial mode for debugging")
    parser.add_argument("--time_offsets", type=str, default="-6,-3,0,3,6",
                        help="Comma-separated list of time offsets in hours (e.g., -6,-3,0,3,6)")
    args = parser.parse_args()
    
    variable_list = [args.data_var]
    
    # Check if weather-type mode is requested.
    if args.wt_csv is not None:
        weather_type_path = args.wt_csv + args.region + '_mcs.csv'
        if not os.path.exists(weather_type_path):
            print(f"Composite CSV file with weather types not found: {weather_type_path}")
            return
        
        base_col = 'time_0h'
        df_all = pd.read_csv(weather_type_path, parse_dates=[base_col])
        df_all[base_col] = df_all[base_col].dt.round("H")
        df_all['month'] = df_all[base_col].dt.month
        
        offset_col_names = create_offset_cols(df_all)
        time_offsets = sorted(list(offset_col_names.keys()))
        months = sorted(df_all['month'].unique())
        print(f"Processing months: {months} for region {args.region} with weather type composites")
        
        weather_types = list(range(0, 28))
        results_wt = { wt: { off: {} for off in time_offsets } for wt in weather_types }
        results_wt_ind = { wt: { off: {} for off in time_offsets } for wt in weather_types }
        
        for wt in weather_types:
            if wt == 0:
                df_wt = df_all.copy()
            else:
                df_wt = df_all[df_all['lwt'] == wt].copy()
            if df_wt.empty:
                print(f"No events found for weather type {wt} in region {args.region}.")
                continue
            for off in time_offsets:
                df_offset = pd.read_csv(weather_type_path, parse_dates=[offset_col_names[off]])
                df_offset[offset_col_names[off]] = df_offset[offset_col_names[off]].dt.round("H")
                df_offset['year'] = df_offset[offset_col_names[off]].dt.year
                df_offset['month'] = df_offset[offset_col_names[off]].dt.month
                if wt != 0:
                    df_offset = df_offset[df_offset['lwt'] == wt]
                groups = df_offset.groupby(['year', 'month'])
                tasks = []
                for (year, month), group in groups:
                    t_list = group[offset_col_names[off]].tolist()
                    if 2000 <= year <= 2021:
                        tasks.append((year, month, t_list, args.data_dir, args.file_pattern, variable_list))
                print(f"Weather type {wt}, time offset {off}h: Found {len(tasks)} tasks (year-month groups).")
                if args.serial:
                    monthly_tasks = [process_month_both_surface(task) for task in tasks]
                else:
                    with Pool(processes=args.ncores) as pool:
                        monthly_tasks = pool.map(process_month_both_surface, tasks)
                tasks_by_month = {}
                tasks_by_month_ind = {}
                for task, res in zip(tasks, monthly_tasks):
                    _, month, _, _, _, _ = task
                    tasks_by_month.setdefault(month, []).append(res["composite"])
                    tasks_by_month_ind.setdefault(month, []).append(res["individual"])
                month_composites = {}
                month_individual = {}
                for m in months:
                    if m not in tasks_by_month:
                        continue
                    month_composites[m] = combine_tasks_results(tasks_by_month[m], variable_list)
                    month_individual[m] = combine_tasks_individual_surface(tasks_by_month_ind[m], variable_list)
                results_wt[wt][off] = month_composites
                results_wt_ind[wt][off] = month_individual
        
        sample_dict = results_wt[0].get(time_offsets[0], {}).get(months[0], None)
        if sample_dict is None or sample_dict.get(variable_list[0], {}).get("lat", None) is None:
            print("Could not retrieve latitude/longitude information from composites.")
            return
        lat = sample_dict[variable_list[0]]['lat']
        lon = sample_dict[variable_list[0]]['lon']
        nlat = len(lat)
        nlon = len(lon)
        n_months = len(months)
        n_offsets = len(time_offsets)
        n_wt = len(weather_types)
        
        composite_arrays_wt = {}
        composite_counts_wt = {}
        for var in variable_list:
            composite_arrays_wt[var] = np.empty((n_wt, n_months, n_offsets, nlat, nlon))
            composite_counts_wt[var] = np.empty((n_wt, n_months, n_offsets, nlat, nlon))
        for wi, wt in enumerate(weather_types):
            for mi, m in enumerate(months):
                for oi, off in enumerate(time_offsets):
                    comp_month = results_wt[wt].get(off, {}).get(m, None)
                    for var in variable_list:
                        if comp_month is None or comp_month.get(var, None) is None or np.all(comp_month[var].get("count", 0) == 0):
                            composite_arrays_wt[var][wi, mi, oi, :, :] = np.nan
                            composite_counts_wt[var][wi, mi, oi, :, :] = np.nan
                        else:
                            cnt = comp_month[var]["count"]
                            composite_arrays_wt[var][wi, mi, oi, :, :] = comp_month[var]["sum"] / cnt
                            composite_counts_wt[var][wi, mi, oi, :, :] = cnt
        composite_counts_scalar = {}
        for var in variable_list:
            composite_counts_scalar[var] = np.empty((n_wt, n_months, n_offsets))
            for wi in range(n_wt):
                for mi in range(n_months):
                    for oi in range(n_offsets):
                        composite_counts_scalar[var][wi, mi, oi] = composite_counts_wt[var][wi, mi, oi, 0, 0]
        weather_types_arr = np.array(weather_types)
        months_arr = np.array(months)
        time_offsets_arr = np.array(time_offsets)
        output_file_comp = os.path.join(args.output_dir, f"composite_surface_{args.region}_{args.data_var}_wt.nc")
        save_to_netcdf_surface(composite_arrays_wt, composite_counts_scalar,
                                   weather_types_arr, months_arr, time_offsets_arr, lat, lon, output_file_comp)
        
        output_file_ind = os.path.join(args.output_dir, f"individual_events_surface_{args.region}_{args.data_var}_wt.nc")
        #save_individual_events_to_netcdf_surface(results_wt_ind, weather_types_arr, months_arr, time_offsets_arr,
        #                                           variable_list, lat, lon, output_file_ind)
    else:
        print("Weathertype csv file not found")

if __name__ == '__main__':
    warnings.filterwarnings("ignore", message="Relative humidity >120%, ensure proper units.")
    warnings.filterwarnings("ignore", message="invalid value encountered in log")
    main()
