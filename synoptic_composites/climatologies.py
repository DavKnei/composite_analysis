#!/usr/bin/env python3
"""
Calculate monthly-hourly mean climatologies from ERA5 data for specific
periods (historical: 1996-2005, evaluation: 2000-2009) and months (May-September).

Uses a robust year-by-year incremental approach.
Part 1: Calculates monthly-hourly means for May-September for each individual year
        in the range 1996-2009 and saves them.
Part 2: Averages these yearly means to produce the long-term climatologies for the
        'historical' and 'evaluation' periods.

This approach is more robust to interruptions and manages memory better.

Usage:
    python climatologies_custom_periods.py \\
        --plev_dir /path/to/era5/pressure_levels/ \\
        --surf_dir /path/to/era5/surface/ \\
        --output_dir ./climatology_output_custom/ \\
        [--ncores 4]

Note: The script internally defines the overall year range for intermediate files (1996-2009)
and the specific months (May-September).
"""

import argparse
import logging
import sys
import os
from pathlib import Path
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
import dask
import pandas as pd # For timestamps

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Variables to process
PLEV_VARS = ['z', 't', 'q', 'u', 'v', 'w']
SURF_VARS = ['tp']

# Filename patterns
PLEV_PATTERN = "{year}-{month:02d}_NA.nc"
SURF_PATTERN = "{year}_NA.nc"

# Domain for consistency
DOMAIN_LAT_MIN, DOMAIN_LAT_MAX = 20, 55
DOMAIN_LON_MIN, DOMAIN_LON_MAX = -20, 40

# Months to process (May to September)
MONTHS_TO_PROCESS = list(range(5, 10)) # 5 (May) to 9 (Sep)

# Overall year range for generating intermediate files
INTERMEDIATE_START_YEAR = 1996
INTERMEDIATE_END_YEAR = 2009

# Definitions for final climatology periods
PERIODS = {
    "historical": {"start": 1996, "end": 2005},
    "evaluation": {"start": 2000, "end": 2009}
}

# --- Helper Functions (from original script) ---
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

def preprocess_era5(ds: xr.Dataset) -> xr.Dataset:
    """Apply standard preprocessing: fix coords, reorder lat, select domain."""
    ds = fix_lat_lon_names(ds)
    ds = reorder_lat(ds)
    lat_coord_name = 'latitude' if 'latitude' in ds.coords else 'lat'
    lon_coord_name = 'longitude' if 'longitude' in ds.coords else 'lon'

    if lat_coord_name in ds.coords and lon_coord_name in ds.coords:
        if ds[lat_coord_name].ndim == 1 and ds[lon_coord_name].ndim == 1:
             ds = ds.sel({lat_coord_name: slice(DOMAIN_LAT_MIN, DOMAIN_LAT_MAX),
                          lon_coord_name: slice(DOMAIN_LON_MIN, DOMAIN_LON_MAX)})
        else:
             logging.warning("Latitude or Longitude coordinates are not 1D for .sel. Using .where for domain selection.")
             ds = ds.where((ds[lat_coord_name] >= DOMAIN_LAT_MIN) & (ds[lat_coord_name] <= DOMAIN_LAT_MAX) &
                           (ds[lon_coord_name] >= DOMAIN_LON_MIN) & (ds[lon_coord_name] <= DOMAIN_LON_MAX), drop=True)
    else:
        logging.warning("Latitude or Longitude coordinates not found for domain selection.")

    if 'time' in ds.coords and ds['time'].dtype != 'datetime64[ns]':
        try:
            logging.debug("Attempting to convert time coordinate to datetime64[ns]")
            if hasattr(ds['time'].values[0], 'strftime'): # Handle cftime objects
                 times_pd = ds.indexes['time'].to_datetimeindex(unsafe=True)
                 ds['time'] = ("time", times_pd.values, ds.time.attrs)
            else:
                 ds['time'] = ds['time'].astype('datetime64[ns]')
        except Exception as e:
            logging.warning(f"Could not convert time coordinate to datetime64[ns]: {e}")
    return ds

# --- Part 1: Function to calculate climatology for a single year (May-Sep) ---
def calculate_single_year_may_sep_climatology(
    data_dir: Path,
    file_pattern_template: str,
    variable_name: str,
    year: int,
    output_dir_yearly: Path,
    prefix: str
):
    logging.info(f"Processing {prefix} (May-Sep) for year {year}...")
    # Filename includes month range for clarity, though data itself will reflect it
    output_yearly_file = output_dir_yearly / f"{prefix}_{year}_may_sep_monthly_hourly.nc"

    if output_yearly_file.exists():
        logging.info(f"Yearly climatology {output_yearly_file.name} already exists. Skipping.")
        return

    paths_for_year_months = []
    if "{month:02d}" in file_pattern_template:
        for month in MONTHS_TO_PROCESS: # Only May-September
            fpath_str = file_pattern_template.format(year=year, month=month)
            fpath = data_dir / fpath_str
            if fpath.exists():
                paths_for_year_months.append(str(fpath))
            else:
                logging.debug(f"File not found: {fpath}")
    elif "{year}" in file_pattern_template: # Annual file
        # If input is annual, we'll select months after loading
        fpath_str = file_pattern_template.format(year=year)
        fpath = data_dir / fpath_str
        if fpath.exists():
            paths_for_year_months.append(str(fpath))
        else:
            logging.debug(f"File not found: {fpath}")
    else:
        logging.error(f"Invalid file_pattern_template: {file_pattern_template}")
        return

    if not paths_for_year_months:
        logging.warning(f"No input files found for {prefix}, year {year}, months {MONTHS_TO_PROCESS}. Skipping.")
        return

    logging.info(f"  Found {len(paths_for_year_months)} files for {prefix}, year {year} (May-Sep).")

    datasets_for_year = []
    for p_str in paths_for_year_months:
        try:
            with xr.open_dataset(p_str, chunks={'time': -1}, decode_times=False) as ds_single_raw:
                ds_single = xr.decode_cf(ds_single_raw, decode_times=True)
                
                # If the input file is annual, select only May-Sep here
                if "{month:02d}" not in file_pattern_template: # i.e., it's an annual file
                    ds_single = ds_single.sel(time=ds_single.time.dt.month.isin(MONTHS_TO_PROCESS))
                    if ds_single.time.size == 0:
                        logging.debug(f"No May-Sep data in annual file {p_str} for year {year}. Skipping this file.")
                        continue
                
                ds_processed = preprocess_era5(ds_single[[variable_name]])
                if variable_name in ds_processed.data_vars:
                    if 'time' in ds_processed.coords and ds_processed.time.size > 0:
                        datasets_for_year.append(ds_processed.copy(deep=False))
                    else:
                        logging.warning(f"Time missing or empty after preprocessing {p_str}. Skipping.")
                else:
                     logging.warning(f"Var '{variable_name}' not in {p_str} after preprocessing. Skipping.")
        except Exception as e:
            logging.error(f"Error opening/preprocessing {p_str} for year {year}: {e}", exc_info=True)

    if not datasets_for_year:
        logging.warning(f"No valid datasets for {prefix}, year {year} after loading. Skipping.")
        return

    try:
        with xr.concat(datasets_for_year, dim='time', coords='minimal', data_vars='minimal', compat='override', join='override') as ds_year:
            if 'time' not in ds_year.coords or ds_year.time.size == 0:
                logging.warning(f"No time data in combined dataset for {prefix}, year {year}. Skipping.")
                return

            chunks_spec = {'time': 'auto'}
            if 'latitude' in ds_year.dims: chunks_spec['latitude'] = 'auto'
            if 'longitude' in ds_year.dims: chunks_spec['longitude'] = 'auto'
            if 'level' in ds_year.dims: chunks_spec['level'] = 'auto'
            ds_year = ds_year.chunk(chunks_spec)

            logging.info(f"  Calculating May-Sep monthly-hourly means for {prefix}, year {year}...")
            with ProgressBar():
                climatology_year = ds_year.astype(np.float32).groupby('time.month').apply(
                    lambda x: x.groupby('time.hour').mean(dim='time', skipna=True)
                ).compute()
            
            # Ensure 'month' coordinate contains the actual month numbers (5-9)
            if not climatology_year.month.equals(xr.DataArray(MONTHS_TO_PROCESS, dims='month')):
                 logging.warning(f"Month coordinate in yearly climatology for {year} does not match expected {MONTHS_TO_PROCESS}. It is {climatology_year.month.values}. This might affect averaging if not all months are present.")


            climatology_year = climatology_year.assign_coords(year=year)
            climatology_year = climatology_year.expand_dims('year')

            output_dir_yearly.mkdir(parents=True, exist_ok=True)
            encoding = {var: {'zlib': True, 'complevel': 4, '_FillValue': np.float32(1e20)} for var in climatology_year.data_vars}
            climatology_year.to_netcdf(output_yearly_file, encoding=encoding, mode='w')
            logging.info(f"  Saved yearly May-Sep climatology: {output_yearly_file.name}")

    except Exception as e:
        logging.error(f"Error during calculation/saving for {prefix}, year {year}: {e}", exc_info=True)
    finally:
        for ds in datasets_for_year: ds.close()
        datasets_for_year.clear()

# --- Part 2: Function to average yearly climatologies for a specific period ---
def average_yearly_climatologies_for_period(
    input_dir_yearly: Path,
    glob_pattern_yearly: str, # e.g., "plev_z_*_may_sep_monthly_hourly.nc"
    output_final_file: Path,
    period_name: str,
    start_year_period: int,
    end_year_period: int
):
    logging.info(f"Averaging yearly May-Sep climatologies for PERIOD: {period_name} ({start_year_period}-{end_year_period})")
    logging.info(f"  Source dir: {input_dir_yearly}/{glob_pattern_yearly}")

    all_yearly_files = sorted(list(input_dir_yearly.glob(glob_pattern_yearly)))
    files_for_period = []
    for fpath in all_yearly_files:
        try:
            # Extract year from filename, e.g., plev_z_1996_may_sep_monthly_hourly.nc
            year_str = fpath.name.split('_')[-5] # Adjust index if filename structure changes
            file_year = int(year_str)
            if start_year_period <= file_year <= end_year_period:
                files_for_period.append(fpath)
        except (IndexError, ValueError) as e:
            logging.warning(f"Could not parse year from filename {fpath.name}: {e}. Skipping this file.")

    if not files_for_period:
        logging.error(f"No yearly climatology files found for period {period_name} ({start_year_period}-{end_year_period})")
        return

    logging.info(f"  Found {len(files_for_period)} yearly files for period {period_name}.")

    try:
        with xr.open_mfdataset(files_for_period, concat_dim="year", combine="nested",
                               chunks={'month': 'auto', 'hour': 'auto', 'level': 'auto'},
                               decode_times=False) as ds_period_years:

            logging.info(f"  Calculating final mean across years for period {period_name}...")
            with ProgressBar():
                final_climatology = ds_period_years.mean(dim='year', skipna=True).compute()

            final_climatology.attrs['title'] = f"Mean May-Sep Monthly-Hourly Climatology ({period_name.capitalize()} Period: {start_year_period}-{end_year_period})"
            final_climatology.attrs['months_included'] = "May, June, July, August, September"
            final_climatology.attrs['source_processing_method'] = "Yearly incremental averaging (May-Sep)"
            final_climatology.attrs['history'] = f"Created by climatologies_custom_periods.py on {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S %Z')}"
            try:
                with xr.open_dataset(files_for_period[0]) as sample_ds:
                    for var_name_attr in final_climatology.data_vars:
                        if var_name_attr in sample_ds.data_vars:
                            final_climatology[var_name_attr].attrs = sample_ds[var_name_attr].attrs.copy()
                            orig_long_name = sample_ds[var_name_attr].attrs.get('long_name', var_name_attr)
                            final_climatology[var_name_attr].attrs['long_name'] = f"Climatological mean (May-Sep, {period_name}) of {orig_long_name}"
            except Exception as e_attr:
                logging.warning(f"Could not copy variable attributes: {e_attr}")

            output_final_file.parent.mkdir(parents=True, exist_ok=True)
            encoding = {var: {'zlib': True, 'complevel': 4, '_FillValue': np.float32(1e20)} for var in final_climatology.data_vars}
            final_climatology.to_netcdf(output_final_file, encoding=encoding, mode='w')
            logging.info(f"Final climatology for {period_name} saved: {output_final_file}")

    except Exception as e:
        logging.error(f"Error averaging yearly climatologies for period {period_name}: {e}", exc_info=True)

# --- Main Script Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate May-Sep monthly-hourly mean climatologies for specific ERA5 periods.")
    parser.add_argument("--plev_dir", type=Path, default='/data/reloclim/normal/INTERACT/ERA5/pressure_levels/',
                        help="Directory containing raw ERA5 pressure level files")
    parser.add_argument("--surf_dir", type=Path, default="/data/reloclim/normal/INTERACT/ERA5/surface/",
                        help="Directory containing raw ERA5 surface files")
    parser.add_argument("--output_dir", type=Path, default="/home/dkn/climatology/ERA5/",
                        help="Base directory to save the output climatology NetCDF files")
    parser.add_argument("--ncores", type=int, default=20,
                        help="Number of cores for dask parallel processing (threads scheduler)")
    parser.add_argument("--part", type=str, required=True, help="Either part1, part2 or both.")
    args = parser.parse_args()

    dask.config.set(scheduler='threads', num_workers=args.ncores)
    logging.info(f"Using Dask with {args.ncores} thread workers for May-Sep climatologies.")

    output_dir_yearly_base = args.output_dir / "yearly_intermediates_may_sep"
    output_dir_yearly_base.mkdir(parents=True, exist_ok=True)

    if args.part == "part1" or args.part == "both":
        # --- Part 1: Generate Yearly May-Sep Climatologies (1996-2009) ---
        logging.info(f"--- PART 1: Generating Yearly May-Sep Climatologies ({INTERMEDIATE_START_YEAR}-{INTERMEDIATE_END_YEAR}) ---")
        for year_to_process in range(INTERMEDIATE_START_YEAR, INTERMEDIATE_END_YEAR + 1):
            if PLEV_VARS:
                logging.info(f"-- Processing PLEV variables for year {year_to_process} (May-Sep) --")
                output_dir_yearly_plev = output_dir_yearly_base / "plev"
                for var_name in PLEV_VARS:
                    calculate_single_year_may_sep_climatology(
                        data_dir=args.plev_dir,
                        file_pattern_template=PLEV_PATTERN,
                        variable_name=var_name,
                        year=year_to_process,
                        output_dir_yearly=output_dir_yearly_plev / var_name,
                        prefix=f"plev_{var_name}"
                    )
            if SURF_VARS:
                logging.info(f"-- Processing SURF variables for year {year_to_process} (May-Sep) --")
                output_dir_yearly_surf = output_dir_yearly_base / "surf"
                for var_name in SURF_VARS:
                    calculate_single_year_may_sep_climatology(
                        data_dir=args.surf_dir,
                        file_pattern_template=SURF_PATTERN,
                        variable_name=var_name,
                        year=year_to_process,
                        output_dir_yearly=output_dir_yearly_surf / var_name,
                        prefix=f"surf_{var_name}"
                    )
        logging.info(f"--- All yearly May-Sep climatology calculations ({INTERMEDIATE_START_YEAR}-{INTERMEDIATE_END_YEAR}) finished. ---")

    elif args.part == "part2" or args.part == "both":
        # --- Part 2: Average Yearly Climatologies for 'historical' and 'evaluation' periods ---
        logging.info("--- PART 2: Averaging Yearly Climatologies for Defined Periods ---")

        for period_name, period_details in PERIODS.items():
            start_p, end_p = period_details["start"], period_details["end"]
            logging.info(f"--- Processing final climatology for PERIOD: {period_name} ({start_p}-{end_p}) ---")

            if PLEV_VARS:
                for var_name in PLEV_VARS:
                    logging.info(f"-- Averaging yearly May-Sep files for PLEV variable: {var_name}, Period: {period_name} --")
                    input_dir_for_var_avg = output_dir_yearly_base / "plev" / var_name
                    output_final_plev_file = args.output_dir / f"era5_plev_{var_name}_clim_may_sep_{period_name}_{start_p}-{end_p}.nc"
                    average_yearly_climatologies_for_period(
                        input_dir_yearly=input_dir_for_var_avg,
                        glob_pattern_yearly=f"plev_{var_name}_*_may_sep_monthly_hourly.nc",
                        output_final_file=output_final_plev_file,
                        period_name=period_name,
                        start_year_period=start_p,
                        end_year_period=end_p
                    )
            if SURF_VARS:
                for var_name in SURF_VARS:
                    logging.info(f"-- Averaging yearly May-Sep files for SURF variable: {var_name}, Period: {period_name} --")
                    input_dir_for_var_avg = output_dir_yearly_base / "surf" / var_name
                    output_final_surf_file = args.output_dir / f"era5_surf_{var_name}_clim_may_sep_{period_name}_{start_p}-{end_p}.nc"
                    average_yearly_climatologies_for_period(
                        input_dir_yearly=input_dir_for_var_avg,
                        glob_pattern_yearly=f"surf_{var_name}_*_may_sep_monthly_hourly.nc",
                        output_final_file=output_final_surf_file,
                        period_name=period_name,
                        start_year_period=start_p,
                        end_year_period=end_p
                    )

        logging.info("--- Yearly Climatology averaging finished. ---")
    else:
        print("--part should either be <part1>, <part2> or <both>")