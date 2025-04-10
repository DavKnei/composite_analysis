#!/usr/bin/env python3
"""
Compute monthly ERA5 composites for MCS composite times at pressure levels and save as a multidimensional netCDF file 
with an extra weather type dimension.

This script reads a composite CSV file (e.g. "./csv/composite_southern_alps_mcs.csv")
which contains composite time indices for MCS events that initiated within a specific subregion.
It expects that the CSV has columns for multiple time offsets (e.g. "time_minus6h", "time_minus3h",
"time_0h", "time_plus3h", "time_plus6h"). For each time offset, events are grouped by month (ignoring year)
and the corresponding ERA5 monthly data are extracted from files (e.g. "2005-08_NA.nc") over the full domain 
(lat 20–55, lon -20 to 40) for the specified pressure levels (via --levels). Composite means for each variable 
(z, t, q, u, v, w, and computed theta_e) are computed for each month, each time offset and each pressure level.
In this updated version, an outer weather type dimension is added – using the Lamb weather type from the CSV:
    weather_type = 0   means “all events” (no filtering)
    weather_type = 1..27 filter events by the respective Lamb type.
For each composite we also compute a scalar count (assumed uniform over latitude/longitude) representing the number 
of events averaged.
Additionally, this script now saves a second netCDF file with the individual event data. In that file an extra 
dimension “time” (holding the event datetime) is added and the variables are stored as t, z, etc. (no mean, no count).

Usage:
    python compute_composites_nc_multidim.py --era5_dir /data/reloclim/normal/INTERACT/ERA5/pressure_levels/ \
         --comp_dir ./csv/ --levels 250,500,850 --region southern_alps --time_offsets -6,-3,0,3,6 \
         --output_dir output_composites [--ncores 32] [--serial]

Author: David Kneidinger (updated)
Date: 2025-03-25
"""

import os
import argparse
from datetime import timedelta
import pandas as pd
import numpy as np
import xarray as xr
from multiprocessing import Pool
import metpy.calc as mpcalc
from metpy.units import units
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

# ERA5 variables to extract at pressure levels.
VAR_LIST = ['z', 't', 'q', 'u', 'v', 'w']


def read_composite_csv(comp_csv_file, time_offset_col):
    """
    Read the composite CSV file for the region.
    The CSV must include the column specified by time_offset_col (e.g. "time_0h").
    Datetime values are rounded to the nearest hour and a column 'month' (ignoring year) is added.
    """
    df = pd.read_csv(comp_csv_file, parse_dates=[time_offset_col])
    df[time_offset_col] = df[time_offset_col].dt.round("H")
    df['month'] = df[time_offset_col].dt.month
    return df


def get_era5_file(era5_dir, year, month):
    """
    Construct the ERA5 monthly filename (e.g. "2005-08_NA.nc").
    """
    fname = f"{year}-{month:02d}_NA.nc"
    return os.path.join(era5_dir, fname)


def reorder_lat(ds):
    """
    Ensure latitude is in ascending order.
    """
    if ds.latitude.values[0] > ds.latitude.values[-1]:
        ds = ds.reindex(latitude=list(np.sort(ds.latitude.values)))
    return ds


def create_offset_cols(df):
    """
    Automatically extract offset column names from the DataFrame.
    Returns a dictionary mapping offset hours (int) to column names.
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

def process_month_both(task):
    """
    Process one (year, month, times, era5_dir, levels) task.
    Reads the ERA5 monthly file for (year, month) and selects data for the specified times (using nearest method).
    Computes:
      - Composite: sums over events and count (as before).
      - Individual: retains the data for each event (no aggregation) and computes theta_e per event.
    Returns a dictionary with keys:
       "composite": { level: { 'z_sum', 't_sum', ... , 'theta_e_sum', 'count', 'lat', 'lon' } }
       "individual": { level: { var: array of shape (n_events, lat, lon) for each var in VAR_LIST + ['theta_e'],
                                  "time": array of event datetimes,
                                  "lat": lat, "lon": lon } }
    """
    year, month, times, era5_dir, levels = task
    file_path = get_era5_file(era5_dir, year, month)
    if not os.path.exists(file_path):
        print(f"ERA5 file {file_path} not found for {year}-{month:02d}. Skipping.")
        return {"composite": {}, "individual": {}}
    try:
        ds = xr.open_dataset(file_path)
    except Exception as e:
        print(f"Error opening {file_path}: {e}")
        return {"composite": {}, "individual": {}}

    ds = ds.sel(latitude=slice(DOMAIN_LAT_MAX, DOMAIN_LAT_MIN),
                longitude=slice(DOMAIN_LON_MIN, DOMAIN_LON_MAX))
    ds = reorder_lat(ds)

    # Select only the times corresponding to this composite (using nearest method)
    time_vals = np.array([np.datetime64(t) for t in times])
    ds = ds.sel(time=time_vals, method='nearest')

    comp = {}
    indiv = {}
    # Loop over levels
    for lev in levels:
        try:
            ds_level = ds.sel(level=lev, method="nearest")
        except Exception as e:
            print(f"Level {lev} not found in {file_path}: {e}")
            continue

        # ---------------------------
        # Composite processing (as before)
        # ---------------------------
        comp_dict = {}
        for var in VAR_LIST:
            if var not in ds_level:
                print(f"Variable {var} not found in {file_path} at level {lev}.")
                continue
            data = ds_level[var]
            comp_dict[f"{var}_sum"] = data.sum(dim='time').values
            comp_dict["count"] = data.count(dim='time').values
        # Retrieve spatial coordinates
        lat_arr = np.sort(ds_level['latitude'].values)
        lon_arr = ds_level['longitude'].values
        comp_dict["lat"] = lat_arr
        comp_dict["lon"] = lon_arr

        # Compute theta_e sum for composite (summing over events)
        if 't' in ds_level and 'q' in ds_level:
            t_data = ds_level['t'].values  # in Kelvin, shape (n_events, lat, lon)
            q_data = ds_level['q'].values  # in kg/kg
            theta_e_sum = np.zeros_like(t_data[0])
            count = t_data.shape[0]
            p = lev * units.hPa
            for i in range(count):
                t_i = t_data[i] * units.kelvin
                q_i = q_data[i] * units('kg/kg')
                sat_mr = mpcalc.saturation_mixing_ratio(p, t_i)
                q_i = np.minimum(q_i, 1.2 * sat_mr)
                try:
                    dewpoint = mpcalc.dewpoint_from_specific_humidity(p, t_i, q_i)
                    theta_e_i = mpcalc.equivalent_potential_temperature(p, t_i, dewpoint)
                except Exception as e:
                    print(f"Error computing theta_e at index {i} for level {lev}: {e}")
                    theta_e_i = np.full_like(t_data[0], np.nan)
                theta_e_sum += theta_e_i.magnitude
            comp_dict["theta_e_sum"] = theta_e_sum
            comp_dict["count"] = count
        else:
            comp_dict["theta_e_sum"] = None
        comp[str(lev)] = comp_dict

        # ---------------------------
        # Individual events processing
        # ---------------------------
        indiv_dict = {}
        # For each variable in VAR_LIST, store the full event data (no sum)
        for var in VAR_LIST:
            if var in ds_level:
                # Shape: (n_events, lat, lon)
                indiv_dict[var] = ds_level[var].values
            else:
                print(f"Variable {var} not found for individual events in {file_path} at level {lev}.")
                indiv_dict[var] = np.full((ds_level.sizes['time'], len(ds_level.latitude), len(ds_level.longitude)), np.nan)
        # Compute theta_e for each event individually
        if ('t' in ds_level) and ('q' in ds_level):
            t_data = ds_level['t'].values
            q_data = ds_level['q'].values
            n_events = t_data.shape[0]
            theta_e_arr = np.empty_like(t_data)
            p = lev * units.hPa
            for i in range(n_events):
                t_i = t_data[i] * units.kelvin
                q_i = q_data[i] * units('kg/kg')
                sat_mr = mpcalc.saturation_mixing_ratio(p, t_i)
                q_i = np.minimum(q_i, 1.2 * sat_mr)
                try:
                    dewpoint = mpcalc.dewpoint_from_specific_humidity(p, t_i, q_i)
                    theta_e_i = mpcalc.equivalent_potential_temperature(p, t_i, dewpoint)
                    theta_e_arr[i] = theta_e_i.magnitude
                except Exception as e:
                    print(f"Error computing theta_e for individual event at index {i} for level {lev}: {e}")
                    theta_e_arr[i] = np.full_like(t_data[0], np.nan)
            indiv_dict['theta_e'] = theta_e_arr
        else:
            indiv_dict['theta_e'] = np.full((ds_level.sizes['time'], len(ds_level.latitude), len(ds_level.longitude)), np.nan)

        # Save the event times (as selected by ds.sel, the coordinate "time" remains)
        indiv_dict["time"] = ds_level['time'].values  # shape (n_events,)
        indiv_dict["lat"] = lat_arr
        indiv_dict["lon"] = lon_arr

        indiv[str(lev)] = indiv_dict

    ds.close()
    return {"composite": comp, "individual": indiv}


def combine_tasks_results(task_results, levels):
    """
    Combine a list of task results (each is a dictionary keyed by level) into one composite dictionary.
    For each level, sum the fields and the counts.
    """
    overall = {}
    for lev in levels:
        overall[str(lev)] = {
            'z_sum': None, 't_sum': None, 'q_sum': None,
            'u_sum': None, 'v_sum': None, 'w_sum': None,
            'theta_e_sum': None, 'count': 0, 'lat': None, 'lon': None
        }
    for result in task_results:
        if not result:
            continue
        for lev in levels:
            key = str(lev)
            if key not in result:
                continue
            res = result[key]
            if res.get("count", 0) == 0:
                continue
            if overall[key]['z_sum'] is None:
                overall[key]['z_sum'] = res.get("z_sum")
                overall[key]['t_sum'] = res.get("t_sum")
                overall[key]['q_sum'] = res.get("q_sum")
                overall[key]['u_sum'] = res.get("u_sum")
                overall[key]['v_sum'] = res.get("v_sum")
                overall[key]['w_sum'] = res.get("w_sum")
                overall[key]['theta_e_sum'] = res.get("theta_e_sum")
                overall[key]['count'] = res.get("count", 0)
                overall[key]['lat'] = np.array(res.get("lat"))
                overall[key]['lon'] = np.array(res.get("lon"))
            else:
                overall[key]['z_sum'] += res.get("z_sum")
                overall[key]['t_sum'] += res.get("t_sum")
                overall[key]['q_sum'] += res.get("q_sum")
                overall[key]['u_sum'] += res.get("u_sum")
                overall[key]['v_sum'] += res.get("v_sum")
                overall[key]['w_sum'] += res.get("w_sum")
                overall[key]['theta_e_sum'] += res.get("theta_e_sum")
                overall[key]['count'] += res.get("count", 0)
    return overall


def combine_tasks_individual(task_results, levels):
    """
    Combine a list of individual event task results (each is a dictionary keyed by level)
    into one dictionary for a given month.
    For each level, concatenate the event arrays along the event dimension.
    """
    overall = {}
    for lev in levels:
        overall[str(lev)] = {var: [] for var in VAR_LIST + ['theta_e']}
        overall[str(lev)]["time"] = []
        overall[str(lev)]["lat"] = None
        overall[str(lev)]["lon"] = None
    for result in task_results:
        if not result:
            continue
        for lev in levels:
            key = str(lev)
            if key not in result:
                continue
            res = result[key]
            # Append data for each variable
            for var in VAR_LIST + ['theta_e']:
                overall[key][var].append(res.get(var))
            overall[key]["time"].append(res.get("time"))
            if overall[key]["lat"] is None:
                overall[key]["lat"] = res.get("lat")
                overall[key]["lon"] = res.get("lon")
    # Concatenate along event dimension
    for lev in levels:
        key = str(lev)
        for var in VAR_LIST + ['theta_e']:
            if overall[key][var]:
                overall[key][var] = np.concatenate(overall[key][var], axis=0)
            else:
                overall[key][var] = np.empty((0,))
        if overall[key]["time"]:
            overall[key]["time"] = np.concatenate(overall[key]["time"], axis=0)
        else:
            overall[key]["time"] = np.empty((0,))
    return overall


#########################################################################
# New function to create and save the individual events netCDF file.
#########################################################################

def save_individual_events_to_netcdf(results_wt_ind, weather_types, months, time_offsets, levels, lat, lon, output_file):
    """
    Create an xarray Dataset from the individual events.
    The structure is similar to the composite file but with an extra "time" dimension that holds the event datetime.
    Dimensions:
       (weather_type, month, time_diff, level, time, latitude, longitude)
    Since different groups may have different numbers of events, we pad with NaNs up to the maximum number.
    """
    n_wt = len(weather_types)
    n_months = len(months)
    n_offsets = len(time_offsets)
    n_levels = len(levels)
    nlat = len(lat)
    nlon = len(lon)

    # Determine maximum number of events over all groups
    max_events = 0
    for wi, wt in enumerate(weather_types):
        for mi, m in enumerate(months):
            for oi, off in enumerate(time_offsets):
                group = results_wt_ind.get(wt, {}).get(off, {}).get(m, None)
                if group is None:
                    continue
                for li, lev in enumerate(levels):
                    key = str(lev)
                    if key in group:
                        n_ev = group[key]["time"].shape[0]
                        if n_ev > max_events:
                            max_events = n_ev
    if max_events == 0:
        print("No individual event data found to save.")
        return

    # Initialize arrays for each variable and for event times.
    indiv_arrays = {}
    for var in VAR_LIST + ['theta_e']:
        indiv_arrays[var] = np.full((n_wt, n_months, n_offsets, n_levels, max_events, nlat, nlon),
                                     np.nan, dtype=np.float32)
    # For event times, use datetime64[ns] and fill with NaT.
    time_array = np.full((n_wt, n_months, n_offsets, n_levels, max_events),
                         np.datetime64('NaT'), dtype='datetime64[ns]')

    # Fill arrays from results_wt_ind structure
    for wi, wt in enumerate(weather_types):
        for mi, m in enumerate(months):
            for oi, off in enumerate(time_offsets):
                group = results_wt_ind.get(wt, {}).get(off, {}).get(m, None)
                if group is None:
                    continue
                for li, lev in enumerate(levels):
                    key = str(lev)
                    if key not in group:
                        continue
                    n_ev = group[key]["time"].shape[0]
                    # For each variable, fill the first n_ev entries; the rest remain NaN.
                    for var in VAR_LIST + ['theta_e']:
                        indiv_arrays[var][wi, mi, oi, li, :n_ev, :, :] = group[key][var]
                    time_array[wi, mi, oi, li, :n_ev] = group[key]["time"]

    # Create xarray Dataset.
    ds_vars = {}
    for var in VAR_LIST + ['theta_e']:
        da = xr.DataArray(
            indiv_arrays[var],
            dims=("weather_type", "month", "time_diff", "level", "time", "latitude", "longitude"),
            coords={
                "weather_type": weather_types,
                "month": months,
                "time_diff": time_offsets,
                "level": levels,
                "latitude": lat, 
                "longitude": lon, 
            },
            name=var
        )
        ds_vars[var] = da
    # Add the event time coordinate as a separate variable (since it varies per group)
    ds_vars["event_time"] = xr.DataArray(
        time_array,
        dims=("weather_type", "month", "time_diff", "level", "time"),
        coords={
            "weather_type": weather_types,
            "month": months,
            "time_diff": time_offsets,
            "level": levels,
        },
        name="event_time"
    )

    # Set attributes and save to netCDF.
    ds = xr.Dataset(ds_vars)
    ds.attrs["description"] = (
        "Individual ERA5 event data for pressure-level variables (z, t, q, u, v, w, theta_e) "
        "for MCS environments computed separately for each Lamb weather type "
        "(0=all events, 1-27 filtered). An extra dimension 'time' holds the event datetime. "
        "Data are padded with NaNs for groups with fewer events."
    )
    ds.attrs["history"] = "Created by compute_composites_nc_multidim.py (individual events added)"
    ds.attrs["source"] = "ERA5 pressure level data"
    ds.to_netcdf(output_file)
    print(f"Saved individual event data to {output_file}")


###########################################################################
# Main script
###########################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Compute monthly ERA5 composites for MCS composite times (pressure levels) and save as a multidimensional netCDF file with a weather type dimension.")
    parser.add_argument("--era5_dir", type=str, default='/data/reloclim/normal/INTERACT/ERA5/pressure_levels/',
                        help="Directory containing ERA5 monthly files (e.g., 2005-08_NA.nc)")
    parser.add_argument("--comp_dir", type=str, default='./csv/',
                        help="Directory containing composite CSV files")
    parser.add_argument("--levels", type=str, default="250,500,850",
                        help="Comma-separated pressure levels in hPa (e.g., 250,500,850)")
    parser.add_argument("--region", type=str, required=True,
                        help="Subregion to process (e.g., western_alps, southern_alps, dinaric_alps, eastern_alps)")
    parser.add_argument("--output_dir", type=str, default='/data/reloclim/normal/MoCCA/composites/composite_files/',
                        help="Directory to save output composite netCDF file")
    parser.add_argument("--ncores", type=int, default=32,
                        help="Number of cores for parallel processing")
    parser.add_argument("--serial", action='store_true',
                        help="Run in serial mode for debugging")
    args = parser.parse_args()

    levels = [int(l.strip()) for l in args.levels.split(',')]

    # Construct CSV file path: e.g., "./csv/composite_southern_alps_mcs.csv"
    comp_csv_file = os.path.join(args.comp_dir, f"composite_{args.region}_mcs.csv")
    if not os.path.exists(comp_csv_file):
        print(f"Composite CSV file not found: {comp_csv_file}")
        exit()

    # Read composite CSV using base column "time_0h" to get month info.
    base_col = 'time_0h'
    df_all = pd.read_csv(comp_csv_file, parse_dates=[base_col])
    df_all[base_col] = df_all[base_col].dt.round("H")
    df_all['month'] = df_all[base_col].dt.month

    # Extract available offset columns.
    offset_col_names = create_offset_cols(df_all)
    time_offsets = sorted(list(offset_col_names.keys()))
   
    # Unique months present.
    months = sorted(df_all['month'].unique())
    print(f"Processing months: {months} for region {args.region}")

    # Now, add the weather type dimension.
    # Weather types: 0 = all events, 1 to 27 Lamb weather types   TODO: simplified weather types similar to: DOI: 10.1029/2020JD032824
    # 
    weather_types = list(range(0, 28))

    # Build nested dictionaries for composite and individual results:
    # results_wt[wt][off][month] = composite dictionary (per level)
    # results_wt_ind[wt][off][month] = individual events dictionary (per level)
    results_wt = {wt: {off: {} for off in time_offsets} for wt in weather_types}
    results_wt_ind = {wt: {off: {} for off in time_offsets} for wt in weather_types}

    # Loop over weather types.
    for wt in weather_types:
        if wt == 0:
            df_wt = df_all.copy()
        else:
            df_wt = df_all[df_all['lwt'] == wt].copy()
        if df_wt.empty:
            print(f"No events found for weather type {wt} in region {args.region}.")
            continue
        for off in time_offsets:
            # For each offset, read the corresponding column from the CSV.
            df_offset = pd.read_csv(comp_csv_file, parse_dates=[offset_col_names[off]])
            df_offset[offset_col_names[off]] = df_offset[offset_col_names[off]].dt.round("H")
            df_offset['year'] = df_offset[offset_col_names[off]].dt.year
            df_offset['month'] = df_offset[offset_col_names[off]].dt.month
            if wt != 0:
                df_offset = df_offset[df_offset['lwt'] == wt]
            groups = df_offset.groupby(['year', 'month'])
            tasks = []
            for (year, month), group in groups:
                t_list = group[offset_col_names[off]].tolist()
                tasks.append((year, month, t_list, args.era5_dir, levels))
            print(f"Weather type {wt}, time offset {off}h: Found {len(tasks)} tasks (year-month groups).")
            if args.serial:
                monthly_tasks = [process_month_both(task) for task in tasks]
            else:
                with Pool(processes=args.ncores) as pool:
                    monthly_tasks = pool.map(process_month_both, tasks)
            # Combine results per month for composites
            tasks_by_month = {}
            # And for individual events
            tasks_by_month_ind = {}
            for task, res in zip(tasks, monthly_tasks):
                _, month, _, _, _ = task
                tasks_by_month.setdefault(month, []).append(res["composite"])
                tasks_by_month_ind.setdefault(month, []).append(res["individual"])
            month_composites = {}
            month_individual = {}
            for m in months:
                if m not in tasks_by_month:
                    continue
                month_composites[m] = combine_tasks_results(tasks_by_month[m], levels)
                month_individual[m] = combine_tasks_individual(tasks_by_month_ind[m], levels)
            results_wt[wt][off] = month_composites
            results_wt_ind[wt][off] = month_individual

    # Retrieve lat, lon from one composite (from weather type 0) to define grid.
    sample_off = time_offsets[0]
    sample_month = months[0]
    sample_comp = results_wt[0].get(sample_off, {}).get(sample_month, None)
    if sample_comp is None or sample_comp.get(str(levels[0]), {}).get("lat", None) is None:
        print("Could not retrieve latitude/longitude information from composites.")
        exit()
    lat = sample_comp[str(levels[0])]['lat']
    lon = sample_comp[str(levels[0])]['lon']
    nlat = len(lat)
    nlon = len(lon)
    n_levels = len(levels)
    n_months = len(months)
    n_offsets = len(time_offsets)
    n_wt = len(weather_types)

    # Initialize composite arrays for each variable with dimensions:
    # (weather_type, month, time_diff, level, latitude, longitude)
    comp_arrays = {}
    count_arrays = {}
    for var in VAR_LIST + ['theta_e']:
        comp_arrays[var] = np.empty((n_wt, n_months, n_offsets, n_levels, nlat, nlon))
        count_arrays[var] = np.empty((n_wt, n_months, n_offsets, n_levels))

    # Loop over weather type, month, time offset, and pressure level to compute composite mean and count.
    for wi, wt in enumerate(weather_types):
        for mi, m in enumerate(months):
            for oi, off in enumerate(time_offsets):
                comp_month = results_wt[wt].get(off, {}).get(m, None)
                for li, lev in enumerate(levels):
                    key = str(lev)
                    if comp_month is None or comp_month.get(key, None) is None or comp_month[key].get("count", 0) == 0:
                        for var in VAR_LIST + ['theta_e']:
                            comp_arrays[var][wi, mi, oi, li, :, :] = np.nan
                        count_arrays['theta_e'][wi, mi, oi, li] = np.nan
                    else:
                        cnt = comp_month[key]["count"]
                        count_scalar = cnt[0, 0] if isinstance(cnt, np.ndarray) else cnt
                        count_arrays['theta_e'][wi, mi, oi, li] = count_scalar
                        for var in VAR_LIST:
                            comp_arrays[var][wi, mi, oi, li, :, :] = comp_month[key][f"{var}_sum"] / cnt
                        if comp_month[key]["theta_e_sum"] is not None:
                            comp_arrays['theta_e'][wi, mi, oi, li, :, :] = comp_month[key]["theta_e_sum"] / cnt
                        else:
                            comp_arrays['theta_e'][wi, mi, oi, li, :, :] = np.nan

    # Create xarray Dataset for composites.
    weather_types_arr = np.array(weather_types)
    months_arr = np.array(months)
    time_offsets_arr = np.array(time_offsets)
    levels_arr = np.array(levels)

    ds_vars = {}
    for var in VAR_LIST + ['theta_e']:
        da_mean = xr.DataArray(
            comp_arrays[var],
            dims=("weather_type", "month", "time_diff", "level", "latitude", "longitude"),
            coords={
                "weather_type": weather_types_arr,
                "month": months_arr,
                "time_diff": time_offsets_arr,
                "level": levels_arr,
                "latitude": lat,
                "longitude": lon,
            },
            name=f"{var}_mean"
        )
        da_count = xr.DataArray(
            count_arrays[var],
            dims=("weather_type", "month", "time_diff", "level"),
            coords={
                "weather_type": weather_types_arr,
                "month": months_arr,
                "time_diff": time_offsets_arr,
                "level": levels_arr,
            },
            name=f"{var}_count"
        )
        ds_vars[f"{var}_mean"] = da_mean
        ds_vars[f"{var}_count"] = da_count

    ds = xr.Dataset(ds_vars)
    ds.attrs["description"] = (
        "Monthly ERA5 composites for pressure-level variables (z, t, q, u, v, w, theta_e) "
        "for MCS environments computed separately for each Lamb weather type "
        "(0=all events, 1-27 filtered). Dimensions: weather_type, month, time_diff, level, latitude, longitude. "
        "Variables with suffix '_count' give the number of events used per composite "
        "(dimensions: weather_type x month x time_diff x level)."
    )
    ds.attrs["history"] = "Created by compute_composites_nc_multidim.py"
    ds.attrs["source"] = "ERA5 pressure level data"

    output_file_comp = os.path.join(args.output_dir, f"composite_multidim_{args.region}_wt.nc")
    ds.to_netcdf(output_file_comp)
    print(f"Saved composite data to {output_file_comp}")

    # Now, create and save the individual events file.
    #output_file_ind = os.path.join(args.output_dir, f"individual_events_{args.region}_wt.nc")
    #save_individual_events_to_netcdf(results_wt_ind, weather_types, months, time_offsets, levels, lat, lon, output_file_ind)


###############################################################################
# Entry point
###############################################################################
if __name__ == '__main__':
    warnings.filterwarnings("ignore", message="Relative humidity >120%, ensure proper units.")
    warnings.filterwarnings("ignore", message="invalid value encountered in log")
    main()
