#!/usr/bin/env python3
"""
Compute JJA monthly composites of precipitation for MCS events,
stratified by weather type, without climatology.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import timedelta
import logging
import warnings

import pandas as pd
import numpy as np
import xarray as xr
import dask
from multiprocessing import Pool
from typing import List, Dict, Tuple, Any

# Full domain boundaries
DOMAIN_LAT_MIN, DOMAIN_LAT_MAX = 20, 55
DOMAIN_LON_MIN, DOMAIN_LON_MAX = -20, 40

# Definitions for event periods
PERIODS = {
    "historical": {"start": 1996, "end": 2005, "name_in_file": "historical"},
    "evaluation": {"start": 2000, "end": 2009, "name_in_file": "evaluation"}
}

# Months to process and include in the output composite (JJA)
TARGET_MONTHS = list(range(6, 9))

def get_data_file(data_dir: Path, year: int, file_pattern: str) -> Path:
    """Construct the data file name for a given year."""
    fname = file_pattern.format(year=year)
    return data_dir / fname

def reorder_lat(ds: xr.Dataset) -> xr.Dataset:
    """Ensure latitude is in ascending order."""
    lat_coord_name = 'latitude' if 'latitude' in ds.coords else ('lat' if 'lat' in ds.coords else None)
    if lat_coord_name and ds[lat_coord_name].values[0] > ds[lat_coord_name].values[-1]:
        ds = ds.reindex({lat_coord_name: np.sort(ds[lat_coord_name].values)})
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
    time_cols = [c for c in df.columns if 'time' in c.lower()]
    for col in time_cols:
        if col.startswith("time_minus"):
            try:
                offset = -int(col.replace("time_minus", "").replace("h", ""))
                offset_cols[offset] = col
            except ValueError:
                logging.warning(f"Could not parse offset: {col}")
        elif col.startswith("time_plus"):
            try:
                offset = int(col.replace("time_plus", "").replace("h", ""))
                offset_cols[offset] = col
            except ValueError:
                logging.warning(f"Could not parse offset: {col}")
        elif col == "time_0h":
            offset_cols[0] = col
            found_base = True
        elif col == 'datetime':
            offset_cols[0] = col
            found_base = True
    if not found_base and 0 not in offset_cols:
        logging.warning("Base time column 'time_0h' not found.")
        if time_cols:
            offset_cols[0] = time_cols[0]
            logging.info(f"Using '{time_cols[0]}' as base time (offset 0).")
        else:
            raise ValueError("No suitable time column found in CSV.")
    return offset_cols

def process_month_precip_events(task: Tuple) -> Dict[str, Any]:
    """Process one (year, month, times, data_dir, file_pattern, var_name_list) task for precipitation."""
    year, month, times_pd, data_dir, file_pattern, var_name_list = task
    var_name = var_name_list[0]
    comp_data = {f"{var_name}_sum": None, 'count': None, 'lat': None, 'lon': None}
    file_path = get_data_file(data_dir, year, file_pattern)
    if not file_path.exists():
        logging.warning(f"Task Y{year}-M{month:02d}: Data file {file_path} not found. Skipping.")
        return comp_data
    try:
        with xr.open_dataset(file_path, chunks={'time': 'auto'}, cache=False) as ds:
            ds = ds.rename_vars({'latitude': 'lat2d', 'longitude': 'lon2d'})
            ds = fix_lat_lon_names(ds)
            ds = reorder_lat(ds)
            ds = ds.sel(latitude=slice(DOMAIN_LAT_MIN, DOMAIN_LAT_MAX),
                        longitude=slice(DOMAIN_LON_MIN, DOMAIN_LON_MAX))

            if var_name not in ds:
                logging.error(f"Task Y{year}-M{month:02d}: Variable '{var_name}' not found. Skipping.")
                return comp_data

            ds_sel = ds.reindex(time=times_pd, method='nearest', tolerance=pd.Timedelta('1H'))
            ds_sel = ds_sel.dropna(dim="time", how="all")
            if ds_sel.time.size == 0 or ds_sel[var_name].isnull().all():
                logging.warning(f"Task Y{year}-M{month:02d}: No valid data at event times. Skipping.")
                return comp_data

            raw = ds_sel[var_name]
            count_arr = raw.notnull().sum(dim='time').compute()
            sum_arr = raw.sum(dim='time', skipna=True).compute()
            lat_vals = raw['latitude'].compute().values
            lon_vals = raw['longitude'].compute().values

            comp_data['count'] = count_arr
            comp_data[f"{var_name}_sum"] = sum_arr
            comp_data['lat'] = lat_vals
            comp_data['lon'] = lon_vals
    except Exception as e:
        logging.error(f"Task Y{year}-M{month:02d}: Error processing data: {e}", exc_info=True)
    return comp_data

def combine_tasks_results_surface(task_results: List[Dict], var_name: str) -> Dict[str, Any]:
    """Combine task results for composites (sums and counts only)."""
    overall = {f"{var_name}_sum": None, 'count': None, 'lat': None, 'lon': None}
    first = True
    for result in task_results:
        count_val = result.get('count')
        if count_val is None: continue
        count_arr = count_val if isinstance(count_val, np.ndarray) else np.array(count_val)
        if count_arr.size == 0 or np.all(count_arr == 0): continue

        sum_val = result.get(f"{var_name}_sum")
        if first:
            overall['lat'] = result.get('lat')
            overall['lon'] = result.get('lon')
            overall['count'] = count_arr.astype(np.int32)
            overall[f"{var_name}_sum"] = sum_val if sum_val is not None else np.zeros_like(overall['count'], dtype=np.float64)
            first = False
        else:
            if overall['count'].shape == count_arr.shape:
                overall['count'] += count_arr.astype(np.int32)
            else:
                logging.warning(f"Shape mismatch in count accumulation: {overall['count'].shape} vs {count_arr.shape}")
            if sum_val is not None and overall[f"{var_name}_sum"].shape == sum_val.shape:
                overall[f"{var_name}_sum"] += sum_val
            elif sum_val is not None:
                logging.warning(f"Shape mismatch in sum accumulation for '{var_name}': {overall[f'{var_name}_sum'].shape} vs {sum_val.shape}")
    return overall

def save_composites_to_netcdf_precipitation(results_wt: Dict, var_name: str,
                                             weather_types: List[int],
                                             months: List[int], time_offsets: List[int],
                                             lat: np.ndarray, lon: np.ndarray,
                                             output_file: Path, period_details: Dict[str, Any]):
    """Save precipitation composites (mean and event_count) to a NetCDF file."""
    n_wt = len(weather_types)
    n_months = len(months)
    n_offsets = len(time_offsets)
    nlat = len(lat)
    nlon = len(lon)

    mean_arr = np.full((n_wt, n_months, n_offsets, nlat, nlon), np.nan, dtype=np.float32)
    count_arr = np.zeros((n_wt, n_months, n_offsets, nlat, nlon), dtype=np.int32)

    for wi, wt in enumerate(weather_types):
        for mi, m in enumerate(months):
            for oi, off in enumerate(time_offsets):
                comp = results_wt.get(wt, {}).get(off, {}).get(m)
                if not comp or comp.get('count') is None: continue
                cnt = comp['count']
                s = comp.get(f"{var_name}_sum")
                count_arr[wi, mi, oi, :, :] = cnt
                with np.errstate(divide='ignore', invalid='ignore'):
                    mean_val = s / cnt
                mean_arr[wi, mi, oi, :, :] = np.where(cnt > 0, mean_val, np.nan)

    da_mean = xr.DataArray(
        mean_arr,
        dims=("weather_type", "month", "time_diff", "latitude", "longitude"),
        coords={"weather_type": weather_types,
                "month": months,
                "time_diff": time_offsets,
                "latitude": lat,
                "longitude": lon},
        name=f"{var_name}_mean",
        attrs={"long_name": f"Mean {var_name} for composite group"}
    )

    da_count = xr.DataArray(
        count_arr,
        dims=("weather_type", "month", "time_diff", "latitude", "longitude"),
        coords={"weather_type": weather_types,
                "month": months,
                "time_diff": time_offsets,
                "latitude": lat,
                "longitude": lon},
        name="event_count",
        attrs={"long_name": "Number of valid events contributing at each grid cell"}
    )

    ds_out = xr.Dataset({f"{var_name}_mean": da_mean, "event_count": da_count})
    ds_out.attrs["description"] = (
        f"JJA composites for precipitation for MCS events, stratified by weather type. "
        f"Period filter: {period_details['name_in_file']} ({period_details['start']}-{period_details['end']})."
    )
    ds_out.attrs["history"] = (
        f"Created by composite_surface_precipitation.py on "
        f"{pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S %Z')}"
    )
    ds_out.attrs["source"] = "ERA5 precipitation data"

    encoding = {v: {'zlib': True, 'complevel': 4, '_FillValue': np.float32(1e20)}
                for v in ds_out.data_vars}
    output_file.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(output_file, encoding=encoding, mode='w')
    logging.info(f"Saved composite data to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Compute JJA monthly composites for surface variables for MCS events, stratified by weather type.")
    parser.add_argument("--data_dir", type=Path, default="/data/reloclim/backup/MCS_database",
                        help="Directory containing annual data files (e.g., era5_precipitation_2001.nc)")
    parser.add_argument("--period", type=str, default="evaluation", choices=PERIODS.keys(),
                        help=f"Period to filter MCS events: {list(PERIODS.keys())}. Default: evaluation.")
    parser.add_argument("--data_var", type=str, default="precipitation",
                        help="Name of the data variable to composite (e.g., precipitation, tp)")
    parser.add_argument("--file_pattern", type=str, default="{year}.nc",
                        help="Filename pattern with {year} placeholder (e.g., {year}.nc or era5_precip_{year}.nc)")
    parser.add_argument("--wt_csv_base", type=str, default="/nas/home/dkn/Desktop/MoCCA/composites/scripts/synoptic_composites/csv/composite_",
                        help="Base path for composite CSV files (e.g., './csv/composite_')")
    parser.add_argument("--region", type=str, required=True,
                        help="Subregion name, used to find CSV file (e.g., southern_alps)")
    parser.add_argument("--output_dir", type=Path, default="/home/dkn/composites/ERA5/",
                        help="Directory to save output composite netCDF files")
    parser.add_argument("--ncores", type=int, default=1,
                        help="Number of cores for parallel processing")
    parser.add_argument("--serial", action="store_true",
                        help="Run in serial mode for debugging (overrides ncores > 1)")
    parser.add_argument("--time_offsets", type=str, default="-12,0,12",
                        help="Comma-separated list of time offsets in hours (e.g., -12,-6,0,6,12)")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG level logging.")
    parser.add_argument("--noMCS", action="store_true",
                        help="False: composite of initMCS (default); True: composite of noMCS events")
    args = parser.parse_args()

    # Logging setup
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level,
                        format="%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s",
                        force=True)

    warnings.filterwarnings("ignore", message=".*Converting non-nanosecond precision datetime values to nanosecond precision.*")
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    warnings.filterwarnings("ignore", message="Mean of empty slice")

    logging.info("Starting precipitation composite script")
    dask.config.set(array={"slicing": {"split_large_chunks": True}})

    # Load event CSV
    if args.noMCS:
        csv_path = Path(f"{args.wt_csv_base}{args.region}_nomcs.csv")
        base_col = 'datetime'
        args_time_offsets = "0"
    else:
        csv_path = Path(f"{args.wt_csv_base}{args.region}_mcs.csv")
        base_col = 'time_0h'
        args_time_offsets = args.time_offsets

    if not csv_path.exists():
        logging.error(f"Composite CSV file not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path, parse_dates=[base_col])
    df[base_col] = df[base_col].dt.round("H")

    # Add year and month for filtering
    df['year'] = df[base_col].dt.year
    df['event_month_for_filter'] = df[base_col].dt.month

    # Filter by period
    period_details = PERIODS[args.period]
    start_year, end_year = period_details['start'], period_details['end']
    df_period = df[(df['year'] >= start_year) & (df['year'] <= end_year)].copy()
    processed_years_set = set(df_period['year'].unique())
    if not processed_years_set:
        logging.warning(f"No events found for period {args.period} ({start_year}-{end_year}). Exiting.")
        sys.exit(0)

    df_filtered = df_period[df_period['event_month_for_filter'].isin(TARGET_MONTHS)].copy()
    if df_filtered.empty:
        logging.info(f"No JJA events for region {args.region}. Exiting.")
        sys.exit(0)

    # Determine offset columns
    offset_cols = create_offset_cols(df_filtered)
    time_offsets = sorted(int(o) for o in args_time_offsets.split(','))
    missing = [off for off in time_offsets if off not in offset_cols]
    if missing:
        logging.error(f"Requested time offsets {missing} not in CSV columns")
        sys.exit(1)

    months_to_process = sorted(df_filtered['event_month_for_filter'].unique())
    weather_types = sorted(df['wt'].unique())
    if 0 not in weather_types:
        weather_types.insert(0, 0)

    results_wt = {wt: {off: {} for off in time_offsets} for wt in weather_types}

    # Processing loops
    for wt in weather_types:
        df_wt = df_filtered if wt == 0 else df_filtered[df_filtered['wt'] == wt]
        logging.info(f"Processing WT {wt} ('{'All Events' if wt==0 else f'Type {wt}'}') - {len(df_wt)} events (JJA, period-filtered)")

        if df_wt.empty:
            continue
        for off in time_offsets:
            col = offset_cols[off]
            df_off = df_wt.dropna(subset=[col]).copy()
            df_off['year'] = df_off[col].dt.year
            df_off['month_for_group'] = df_off['event_month_for_filter']
            groups = df_off.groupby(['year', 'month_for_group'])

            tasks = []
            for (yr, m), grp in groups:
                times = pd.DatetimeIndex(grp[col].tolist())
                if times.empty: continue
                tasks.append((yr, m, times, args.data_dir, args.file_pattern, [args.data_var]))

            # Run tasks
            if args.serial or args.ncores == 1:
                task_results = [process_month_precip_events(t) for t in tasks]
            else:
                with Pool(processes=args.ncores) as pool:
                    task_results = pool.map(process_month_precip_events, tasks)

            # Organize by month and combine
            by_month = {}
            for t, res in zip(tasks, task_results):
                _, m, _, _, _, _ = t
                by_month.setdefault(m, []).append(res)
            month_comps = {}
            for m in months_to_process:
                res_list = by_month.get(m, [])
                month_comps[m] = combine_tasks_results_surface(res_list, args.data_var)
            results_wt[wt][off] = month_comps

    # Retrieve lat/lon or fallback to first data file
    lat = lon = None
    found = False
    for wt in weather_types:
        for off in time_offsets:
            for m in months_to_process:
                comp = results_wt[wt][off].get(m)
                if comp and comp.get('lat') is not None:
                    lat = comp['lat']; lon = comp['lon']; found = True
                    break
            if found: break
        if found: break

    if not found:
        logging.warning("Could not find lat/lon in composites; using first data file")
        sample_year = sorted(processed_years_set)[0]
        sample_file = get_data_file(args.data_dir, sample_year, args.file_pattern)
        with xr.open_dataset(sample_file) as ds0:
            ds0 = ds0.rename_vars({'latitude': 'lat2d', 'longitude': 'lon2d'})
            ds0 = fix_lat_lon_names(ds0)
            ds0 = reorder_lat(ds0)
            ds0 = ds0.sel(latitude=slice(DOMAIN_LAT_MIN, DOMAIN_LAT_MAX),
                          longitude=slice(DOMAIN_LON_MIN, DOMAIN_LON_MAX))
            lat = ds0['latitude'].values
            lon = ds0['longitude'].values

    # Determine output filename
    suffix = "_nomcs" if args.noMCS else ""
    out_file = args.output_dir / f"composite_surface_{args.region}_{args.data_var}_wt_{args.period}{suffix}.nc"

    save_composites_to_netcdf_precipitation(results_wt, args.data_var,
                                             weather_types, months_to_process,
                                             time_offsets, lat, lon,
                                             out_file, period_details)

if __name__ == "__main__":
    main()
