#!/usr/bin/env python3
"""
Calculate monthly-hourly mean climatologies from ERA5 data for specific
periods (historical: 1996-2005, evaluation: 2000-2009) and months (May-September).
Calculates theta_e from instantaneous t and q before time averaging.

Uses a robust year-by-year incremental approach.
Part 1: Calculates monthly-hourly means for May-September for each individual year
        in the range and saves them. This includes raw ERA5 variables and derived
        variables like theta_e.
Part 2: Averages these yearly means to produce the long-term climatologies for the
        defined periods.

This approach is more robust to interruptions and manages memory better.

Usage:
    python climatologies.py \\
        --plev_dir /path/to/era5/pressure_levels/ \\
        --surf_dir /path/to/era5/surface/ \\
        --output_dir ./climatology_output_custom/ \\
        [--ncores 4] --part <part1|part2|both>

Note: The script internally defines the overall year range for intermediate files
and the specific months (May-September).
"""

import argparse
import logging
import sys
import os
from typing import List, Optional
from pathlib import Path
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
import dask
import pandas as pd # For timestamps
from calc_atmospheric_variables import calculate_theta_e_on_levels

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Variables to process
PLEV_VARS_RAW = ['z', 't', 'q', 'u', 'v', 'w']  # Raw variables from ERA5 plev files
PLEV_VARS_DERIVED_FOR_CLIM = ['theta_e']      # Derived variables for climatology
PLEV_VARS_ALL_FOR_CLIM = PLEV_VARS_RAW + PLEV_VARS_DERIVED_FOR_CLIM # All plev vars for final clim

SURF_VARS = ['tp'] # Surface variables

# Filename patterns
PLEV_PATTERN = "{year}-{month:02d}_NA.nc"
SURF_PATTERN = "{year}_NA.nc" # Assumed to be annual files for surface

# Domain for consistency
DOMAIN_LAT_MIN, DOMAIN_LAT_MAX = 25, 65 
DOMAIN_LON_MIN, DOMAIN_LON_MAX = -20, 43 

# Months to process (May to September)
MONTHS_TO_PROCESS = list(range(5, 10))

# Overall year range for generating intermediate files
INTERMEDIATE_START_YEAR = 1991
INTERMEDIATE_END_YEAR = 2020 

# Definitions for final climatology periods
PERIODS = {
    "historical": {"start": 1991, "end": 2020}
}

# --- Helper Functions ---
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

def preprocess_era5(ds: xr.Dataset) -> xr.Dataset: # Modified from original file
    """Apply standard preprocessing: fix coords, reorder lat, select domain."""
    ds = fix_lat_lon_names(ds) 
    ds = reorder_lat(ds) 
    lat_coord_name = 'latitude' if 'latitude' in ds.coords else 'lat'
    lon_coord_name = 'longitude' if 'longitude' in ds.coords else 'lon'

    if lat_coord_name in ds.coords and lon_coord_name in ds.coords:
        # Ensure domain selection is robust for different coordinate setups
        # Assuming DOMAIN_LAT_MIN/MAX etc. are defined globally
        ds = ds.sel({lat_coord_name: slice(DOMAIN_LAT_MIN, DOMAIN_LAT_MAX),
                     lon_coord_name: slice(DOMAIN_LON_MIN, DOMAIN_LON_MAX)})
    else:
        logging.warning("Latitude or Longitude coordinates not found for domain selection.")

    if 'time' in ds.coords and ds['time'].dtype != 'datetime64[ns]': # From original file
        try:
            logging.debug("Attempting to convert time coordinate to datetime64[ns]")
            if hasattr(ds['time'].values[0], 'strftime'): 
                 times_pd = ds.indexes['time'].to_datetimeindex(unsafe=True)
                 ds['time'] = ("time", times_pd.values, ds.time.attrs)
            else:
                 ds['time'] = ds['time'].astype('datetime64[ns]')
        except Exception as e:
            logging.warning(f"Could not convert time coordinate to datetime64[ns]: {e}")
    return ds

# --- Part 1: Data Loading and Yearly Climatology Generation ---

def load_era5_data_for_year(
    data_dir: Path,
    file_pattern_template: str,
    variables_to_load: List[str],
    year: int
) -> Optional[xr.Dataset]:
    """
    Loads and concatenates ERA5 data for specified variables and a single year,
    for the months defined in MONTHS_TO_PROCESS.
    """
    logging.debug(f"Loading {variables_to_load} for year {year} (Months: {MONTHS_TO_PROCESS})...")
    paths_for_year_months = []

    if "{month:02d}" in file_pattern_template: # Monthly files
        for month in MONTHS_TO_PROCESS:
            fpath_str = file_pattern_template.format(year=year, month=month)
            fpath = data_dir / fpath_str
            if fpath.exists():
                paths_for_year_months.append(str(fpath))
            else:
                logging.debug(f"File not found: {fpath}")
    elif "{year}" in file_pattern_template: # Annual file
        fpath_str = file_pattern_template.format(year=year)
        fpath = data_dir / fpath_str
        if fpath.exists():
            paths_for_year_months.append(str(fpath))
        else:
            logging.debug(f"File not found: {fpath}")
    else:
        logging.error(f"Invalid file_pattern_template for loading: {file_pattern_template}")
        return None
        
    if not paths_for_year_months:
        logging.warning(f"No input files found for variables {variables_to_load}, year {year}, months {MONTHS_TO_PROCESS}. Cannot load.")
        return None

    logging.info(f"  Found {len(paths_for_year_months)} files for year {year} to load {variables_to_load}.")

    datasets_for_year = []
    for p_str in paths_for_year_months:
        try:
            with xr.open_dataset(p_str, chunks={'time': 'auto'}, decode_times=False) as ds_single_raw: # Changed chunk to auto
                ds_single = xr.decode_cf(ds_single_raw, decode_times=True)
                
                # If the input file is annual, select only May-Sep here
                if "{month:02d}" not in file_pattern_template: # i.e., it's an annual file
                    ds_single = ds_single.sel(time=ds_single.time.dt.month.isin(MONTHS_TO_PROCESS))
                    if ds_single.time.size == 0:
                        logging.debug(f"No May-Sep data in annual file {p_str} for year {year}. Skipping this file instance.")
                        continue
                
                # Select only the variables needed for this load, if they exist in the file
                vars_in_file_to_load = [v for v in variables_to_load if v in ds_single.data_vars]
                if not vars_in_file_to_load:
                    logging.debug(f"None of the requested variables {variables_to_load} found in {p_str}. Skipping file for these vars.")
                    continue

                ds_processed = preprocess_era5(ds_single[vars_in_file_to_load])
                if 'time' in ds_processed.coords and ds_processed.time.size > 0:
                    datasets_for_year.append(ds_processed.copy(deep=False)) # Use copy(deep=False)
                else:
                    logging.warning(f"Time missing or empty after preprocessing {p_str}. Skipping.")
        except Exception as e:
            logging.error(f"Error opening/preprocessing {p_str} for year {year}, vars {variables_to_load}: {e}", exc_info=True)

    if not datasets_for_year:
        logging.warning(f"No valid datasets loaded for vars {variables_to_load}, year {year} after attempting to load files.")
        return None

    try:
        # Concatenate all monthly datasets for the year
        ds_year_combined = xr.concat(datasets_for_year, dim='time', coords='minimal', data_vars='minimal', compat='override', join='override')
        return ds_year_combined
    except Exception as e:
        logging.error(f"Error concatenating datasets for year {year}, vars {variables_to_load}: {e}", exc_info=True)
        return None
    finally:
        for ds_item in datasets_for_year: # Ensure individual datasets are closed
            ds_item.close()


def generate_yearly_monthly_hourly_mean(
    input_ds: xr.Dataset,
    variable_name: str, # The specific variable in input_ds to process (e.g. "z" or "theta_e")
    year: int,
    output_dir_for_var: Path, # e.g. .../plev/z/ or .../plev/theta_e/
    output_file_prefix: str   # e.g. "plev_z" or "plev_theta_e"
):
    """
    Calculates monthly-hourly means for a given variable in an input dataset for a single year.
    Saves the result to a NetCDF file.
    """
    logging.info(f"  Generating May-Sep monthly-hourly mean for {output_file_prefix} (var: {variable_name}), year {year}...")
    # Construct output filename using the prefix which already contains var type and name
    output_yearly_file = output_dir_for_var / f"{output_file_prefix}_{year}_may_sep_monthly_hourly.nc"

    if output_yearly_file.exists():
        logging.info(f"  Yearly climatology {output_yearly_file.name} already exists. Skipping.")
        return

    if variable_name not in input_ds.data_vars:
        logging.error(f"  Variable '{variable_name}' not in provided input_ds for {output_file_prefix}, year {year}. Skipping.")
        return
    if 'time' not in input_ds.coords or input_ds.time.size == 0:
        logging.warning(f"  No time data in input_ds for {output_file_prefix}, var {variable_name}, year {year}. Skipping.")
        return

    ds_to_process = input_ds[[variable_name]] # Process only the target variable

    # Define chunks for processing
    chunks_spec = {'time': 'auto'}
    for dim_name in ['latitude', 'longitude', 'level']:
        if dim_name in ds_to_process.dims:
            chunks_spec[dim_name] = 'auto'
    ds_to_process = ds_to_process.chunk(chunks_spec)

    try:
        logging.debug(f"    Calculating means for {variable_name}, year {year}...")
        with ProgressBar():
            climatology_year = ds_to_process.astype(np.float32).groupby('time.month').apply(
                lambda x: x.groupby('time.hour').mean(dim='time', skipna=True)
            ).compute()
        
        # Validate months if needed
        expected_months_dataarray = xr.DataArray(MONTHS_TO_PROCESS, dims='month', name='month')
        if not climatology_year.month.equals(expected_months_dataarray):
             # This can happen if some months had no data at all. The groupby will only include existing months.
             logging.warning(f"    Month coordinate in yearly climatology for {year}, var {variable_name} ({climatology_year.month.values}) does not exactly match expected {MONTHS_TO_PROCESS}. This is okay if some months had no data.")

        climatology_year = climatology_year.assign_coords(year=year)
        climatology_year = climatology_year.expand_dims('year')
        
        # Add attributes from the original variable if possible (e.g., from input_ds)
        if variable_name in input_ds:
            climatology_year[variable_name].attrs = input_ds[variable_name].attrs.copy()
            original_long_name = input_ds[variable_name].attrs.get('long_name', variable_name)
            climatology_year[variable_name].attrs['long_name'] = f"Yearly mean (May-Sep) of {original_long_name}"

        output_dir_for_var.mkdir(parents=True, exist_ok=True)
        encoding = {vn: {'zlib': True, 'complevel': 4, '_FillValue': np.float32(1e20)} for vn in climatology_year.data_vars}
        climatology_year.to_netcdf(output_yearly_file, encoding=encoding, mode='w')
        logging.info(f"  Saved yearly May-Sep climatology: {output_yearly_file.name}")

    except Exception as e:
        logging.error(f"Error during climatology calculation/saving for {output_file_prefix}, var {variable_name}, year {year}: {e}", exc_info=True)
    finally:
        # The input_ds is managed by the calling loop in main() for closure
        pass


# --- Part 2: Function to average yearly climatologies for a specific period ---
# This function remains largely the same as in the original file
# but will be called with theta_e as a variable too.
def average_yearly_climatologies_for_period(
    input_dir_yearly_var: Path, # e.g. .../plev/z/
    glob_pattern_yearly: str, 
    output_final_file: Path,
    period_name: str,
    start_year_period: int,
    end_year_period: int
):
    logging.info(f"Averaging yearly May-Sep climatologies for PERIOD: {period_name} ({start_year_period}-{end_year_period})")
    logging.info(f"  Source dir for averaging: {input_dir_yearly_var} using pattern {glob_pattern_yearly}")

    all_yearly_files = sorted(list(input_dir_yearly_var.glob(glob_pattern_yearly)))
    files_for_period = []
    for fpath in all_yearly_files:
        try:
            # Extract year from filename, e.g., plev_z_1996_may_sep_monthly_hourly.nc
            # Example: plev_theta_e_1991_may_sep_monthly_hourly.nc -> parts: plev, theta, e, 1991, ...
            # A more robust way might be to use regex if filename parts vary.
            # Assuming format: prefix_YYYY_may_sep_monthly_hourly.nc
            filename_parts = fpath.name.split('_')
            year_str = ""
            for part in reversed(filename_parts): # Search for year from the end
                if part.isdigit() and len(part) == 4:
                    year_str = part
                    break
            if not year_str:
                raise ValueError("Year not found or not 4 digits.")

            file_year = int(year_str)
            if start_year_period <= file_year <= end_year_period:
                files_for_period.append(fpath)
        except (IndexError, ValueError) as e:
            logging.warning(f"Could not parse year from filename {fpath.name}: {e}. Skipping this file.")

    if not files_for_period:
        logging.error(f"No yearly climatology files found for period {period_name} ({start_year_period}-{end_year_period}) in {input_dir_yearly_var} with pattern {glob_pattern_yearly}")
        return

    logging.info(f"  Found {len(files_for_period)} yearly files for period {period_name} to average.")

    try:
        # It's crucial that yearly files have consistent coordinates for month, hour, level etc.
        with xr.open_mfdataset(files_for_period, concat_dim="year", combine="nested",
                               chunks={'month': 'auto', 'hour': 'auto', 'level': 'auto'},
                               decode_times=False) as ds_period_years: # decode_times=False if 'year' coord is simple int

            logging.info(f"  Calculating final mean across years for period {period_name}...")
            with ProgressBar():
                final_climatology = ds_period_years.mean(dim='year', skipna=True).compute()

            final_climatology.attrs['title'] = f"Mean May-Sep Monthly-Hourly Climatology ({period_name.capitalize()} Period: {start_year_period}-{end_year_period})"
            final_climatology.attrs['months_included'] = "May, June, July, August, September"
            final_climatology.attrs['source_processing_method'] = "Yearly incremental averaging (May-Sep)"
            final_climatology.attrs['history'] = f"Created by climatologies.py on {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S %Z')}"
            try: # Copy attributes from a sample yearly file
                with xr.open_dataset(files_for_period[0]) as sample_ds:
                    for var_name_attr in final_climatology.data_vars:
                        if var_name_attr in sample_ds.data_vars:
                            final_climatology[var_name_attr].attrs = sample_ds[var_name_attr].attrs.copy()
                            # Update long_name to reflect it's a climatological mean
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
        description="Calculate May-Sep monthly-hourly mean climatologies for specific ERA5 periods. Includes theta_e calculation.")
    parser.add_argument("--plev_dir", type=Path, default='/data/reloclim/normal/INTERACT/ERA5/pressure_levels/',
                        help="Directory containing raw ERA5 pressure level files")
    parser.add_argument("--surf_dir", type=Path, default="/data/reloclim/normal/INTERACT/ERA5/surface/",
                        help="Directory containing raw ERA5 surface files")
    parser.add_argument("--output_dir", type=Path, default="/home/dkn/climatology/ERA5/",
                        help="Base directory to save the output climatology NetCDF files")
    parser.add_argument("--ncores", type=int, default=4, # Default from original
                        help="Number of cores for dask parallel processing (threads scheduler)")
    parser.add_argument("--part", type=str, default="both", choices=['clim', 'clim_mean', 'both'], 
                        help="Specify which part of the script to run: 'part1' (yearly files), 'part2' (final period means), or 'both'.")
    args = parser.parse_args()

    dask.config.set(scheduler='threads', num_workers=args.ncores)
    logging.info(f"Using Dask with {args.ncores} thread workers for May-Sep climatologies.")

    output_dir_yearly_base = args.output_dir / "yearly_intermediates_may_sep"
    output_dir_yearly_base.mkdir(parents=True, exist_ok=True)

    if args.part in ["clim", "both"]:
        logging.info(f"--- PART 1: Generating Yearly May-Sep Climatologies ({INTERMEDIATE_START_YEAR}-{INTERMEDIATE_END_YEAR}) ---")
        for year_to_process in range(INTERMEDIATE_START_YEAR, INTERMEDIATE_END_YEAR + 1):
            logging.info(f"===== Processing Year: {year_to_process} =====")
            
            # --- Process RAW PLEV variables ---
            if PLEV_VARS_RAW:
                logging.info(f"-- Processing RAW PLEV variables for year {year_to_process} --")
                for var_name in PLEV_VARS_RAW:
                    yearly_raw_var_ds = load_era5_data_for_year(
                        data_dir=args.plev_dir,
                        file_pattern_template=PLEV_PATTERN,
                        variables_to_load=[var_name],
                        year=year_to_process
                    )
                    if yearly_raw_var_ds:
                        output_dir_for_var = output_dir_yearly_base / "plev" / var_name
                        generate_yearly_monthly_hourly_mean(
                            input_ds=yearly_raw_var_ds,
                            variable_name=var_name,
                            year=year_to_process,
                            output_dir_for_var=output_dir_for_var,
                            output_file_prefix=f"plev_{var_name}"
                        )
                        yearly_raw_var_ds.close()
                    else:
                        logging.warning(f"Skipping yearly climatology for {var_name}, year {year_to_process} due to load failure.")
            
            # --- Process DERIVED PLEV variables (theta_e) ---
            if PLEV_VARS_DERIVED_FOR_CLIM:
                logging.info(f"-- Processing DERIVED PLEV variables for year {year_to_process} --")
                # For theta_e, we need 't' and 'q'
                tq_data_for_year = load_era5_data_for_year(
                    data_dir=args.plev_dir,
                    file_pattern_template=PLEV_PATTERN,
                    variables_to_load=['t', 'q'], # Load t and q together
                    year=year_to_process
                )

                if tq_data_for_year and 't' in tq_data_for_year and 'q' in tq_data_for_year:
                    logging.info(f"  Calculating instantaneous theta_e for year {year_to_process}...")
                    theta_e_instantaneous_ds = calculate_theta_e_on_levels(tq_data_for_year.compute())  # Make sure to not give a dask array to the theta_e calculation
                    tq_data_for_year.close() 

                    if theta_e_instantaneous_ds and 'theta_e' in theta_e_instantaneous_ds:
                        var_name_theta_e = 'theta_e'
                        output_dir_for_theta_e = output_dir_yearly_base / "plev" / var_name_theta_e
                        generate_yearly_monthly_hourly_mean(
                            input_ds=theta_e_instantaneous_ds,
                            variable_name=var_name_theta_e,
                            year=year_to_process,
                            output_dir_for_var=output_dir_for_theta_e,
                            output_file_prefix=f"plev_{var_name_theta_e}"
                        )
                        theta_e_instantaneous_ds.close()
                    else:
                        logging.warning(f"  Theta_e calculation failed or yielded no data for year {year_to_process}.")
                else:
                    logging.warning(f"  Could not load 't' and 'q' data for year {year_to_process} to calculate theta_e. Skipping theta_e for this year.")
                    if tq_data_for_year: tq_data_for_year.close()

            # --- Process SURF variables ---
            if SURF_VARS:
                logging.info(f"-- Processing SURF variables for year {year_to_process} --")
                for var_name in SURF_VARS:
                    yearly_surf_var_ds = load_era5_data_for_year(
                        data_dir=args.surf_dir,
                        file_pattern_template=SURF_PATTERN, 
                        variables_to_load=[var_name],
                        year=year_to_process
                    )
                    if yearly_surf_var_ds:
                        output_dir_for_var = output_dir_yearly_base / "surf" / var_name
                        generate_yearly_monthly_hourly_mean(
                            input_ds=yearly_surf_var_ds,
                            variable_name=var_name,
                            year=year_to_process,
                            output_dir_for_var=output_dir_for_var,
                            output_file_prefix=f"surf_{var_name}"
                        )
                        yearly_surf_var_ds.close()
                    else:
                        logging.warning(f"Skipping yearly climatology for SURF {var_name}, year {year_to_process} due to load failure.")
        logging.info(f"--- All yearly May-Sep climatology calculations ({INTERMEDIATE_START_YEAR}-{INTERMEDIATE_END_YEAR}) finished. ---")

    if args.part in ["clim_mean", "both"]:
        logging.info("--- PART 2: Averaging Yearly Climatologies for Defined Periods ---")
        for period_name, period_details in PERIODS.items():
            start_p, end_p = period_details["start"], period_details["end"]
            logging.info(f"--- Processing final climatology for PERIOD: {period_name} ({start_p}-{end_p}) ---")

            # Average PLEV variables (now includes theta_e)
            if PLEV_VARS_ALL_FOR_CLIM: # Use the list that includes raw and derived (theta_e)
                for var_name in PLEV_VARS_ALL_FOR_CLIM:
                    logging.info(f"-- Averaging yearly May-Sep files for PLEV variable: {var_name}, Period: {period_name} --")
                    input_dir_for_var_avg = output_dir_yearly_base / "plev" / var_name
                    output_final_plev_file = args.output_dir / f"era5_plev_{var_name}_clim_may_sep_{period_name}_{start_p}-{end_p}.nc"
                    average_yearly_climatologies_for_period(
                        input_dir_yearly_var=input_dir_for_var_avg,
                        glob_pattern_yearly=f"plev_{var_name}_*_may_sep_monthly_hourly.nc", # Should match theta_e files too
                        output_final_file=output_final_plev_file,
                        period_name=period_name,
                        start_year_period=start_p,
                        end_year_period=end_p
                    )
            # Average SURF variables
            if SURF_VARS:
                for var_name in SURF_VARS:
                    logging.info(f"-- Averaging yearly May-Sep files for SURF variable: {var_name}, Period: {period_name} --")
                    input_dir_for_var_avg = output_dir_yearly_base / "surf" / var_name
                    output_final_surf_file = args.output_dir / f"era5_surf_{var_name}_clim_may_sep_{period_name}_{start_p}-{end_p}.nc"
                    average_yearly_climatologies_for_period(
                        input_dir_yearly_var=input_dir_for_var_avg,
                        glob_pattern_yearly=f"surf_{var_name}_*_may_sep_monthly_hourly.nc",
                        output_final_file=output_final_surf_file,
                        period_name=period_name,
                        start_year_period=start_p,
                        end_year_period=end_p
                    )
        logging.info("--- Yearly Climatology averaging finished. ---")
    
    if args.part not in ["part1", "part2", "both"]:
        print("--part argument must be 'part1', 'part2', or 'both'")
        sys.exit(1)