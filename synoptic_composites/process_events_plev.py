#!/usr/bin/env python3
"""
Process individual ERA5 events at multiple pressure levels for later use in
storm-motion-aligned composites.

This script reads event times from a CSV file, calculates a set of derived
variables at specified pressure levels, applies a spectral filter to separate
the fields into synoptic and meso-scale components, and saves all processed
individual events into two separate NetCDF files.

This script does NOT calculate mean composites or handle climatology data.
Theta-e is calculated using an external, user-provided module.

Output Files:
- events_plev_synoptic_{...}.nc
- events_plev_meso_{...}.nc

Output Data Structure:
- Variables: 4D (time, level, latitude, longitude)
- Metadata: 1D variables aligned with the 'time' dimension (e.g., event_weather_type)
  to identify the composite group for each event.

Author: David Kneidinger
Based on previous composite scripts.
Date: 2025-06-16
"""

import sys
import argparse
from pathlib import Path
import logging
import warnings
import pandas as pd
import numpy as np
import xarray as xr
from typing import List, Dict, Any, Optional

# MetPy for dynamic calculations and constants
from metpy.calc import (
    potential_vorticity_baroclinic,
    vorticity,
    lat_lon_grid_deltas,
    potential_temperature,
    divergence as mp_divergence
)
from metpy.units import units
from metpy.constants import earth_avg_radius

# FFT for spectral filtering
from numpy.fft import fft2, ifft2, fftfreq

# --- IMPORT FROM EXTERNAL MODULE as requested ---
from calc_atmospheric_variables import calculate_theta_e_on_levels


# --- Configuration ---
DOMAIN_LAT = (25, 65)
DOMAIN_LON = (-20, 43)
TARGET_MONTHS = [5, 6, 7, 8, 9]  # MJJAS
PERIODS = {
    "historical": {"start": 1991, "end": 2020, "name": "historical"}
}

# Define the derived variables to be computed
DERIVED_VARIABLES = [
    'z', 't', 'q', 'u', 'v', 'w', 'theta_e'
]

# --- Helper Functions ---
def get_era5_file(era5_dir: Path, year: int, month: int) -> Path:
    """Construct the ERA5 monthly filename."""
    fname = f"{year}-{month:02d}_NA.nc"
    return era5_dir / fname

def standardize_ds(ds: xr.Dataset) -> xr.Dataset:
    """Rename coords to latitude/longitude, sort latitude, subset domain."""
    if 'lat' in ds.coords:
        ds = ds.rename({'lat': 'latitude'})
    if 'lon' in ds.coords:
        ds = ds.rename({'lon': 'longitude'})

    if ds['latitude'].size > 1 and ds['latitude'].values[0] > ds['latitude'].values[-1]:
        ds = ds.reindex(latitude=list(np.sort(ds['latitude'].values)))

    return ds.sel(latitude=slice(*DOMAIN_LAT), longitude=slice(*DOMAIN_LON))

def create_offset_cols(df: pd.DataFrame) -> Dict[int, str]:
    """Extract time offset columns from the DataFrame."""
    offset_cols = {}
    for col in df.columns:
        if col.startswith("time_minus"):
            offset_cols[-int(col.replace("time_minus", "").replace("h", ""))] = col
        elif col.startswith("time_plus"):
            offset_cols[int(col.replace("time_plus", "").replace("h", ""))] = col
        elif col == "time_0h":
            offset_cols[0] = col
    return offset_cols

# --- Scale Separation and Variable Calculation ---

def synoptic_scale_filter(data_array: xr.DataArray) -> xr.DataArray:
    """
    Apply a 2D Gaussian low-pass filter. Operates on a 2D unitless DataArray.
    """
    if data_array.ndim != 2: raise ValueError("Input data_array must be 2-dimensional.")
    ny, nx = data_array.shape
    R = earth_avg_radius.to('m').m
    dlon_deg = np.abs(data_array.longitude.diff('longitude').mean().item())
    dlat_deg = np.abs(data_array.latitude.diff('latitude').mean().item())
    mean_lat_rad = np.deg2rad(data_array.latitude.mean().item())
    dx = R * np.cos(mean_lat_rad) * np.deg2rad(dlon_deg)
    dy = R * np.deg2rad(dlat_deg)
    kx = 2 * np.pi * fftfreq(nx, d=dx)
    ky = 2 * np.pi * fftfreq(ny, d=dy)
    kxx, kyy = np.meshgrid(kx, ky)
    k_magnitude = np.sqrt(kxx**2 + kyy**2)
    lambda_half_m, epsilon = 1000 * 1000, 1e-9
    k_half = (2 * np.pi) / lambda_half_m
    filter_mask = np.exp(-np.log(2) * (k_magnitude / (k_half + epsilon))**2)
    nan_mask = data_array.isnull()
    filled_data_values = data_array.fillna(0).values
    data_fft = fft2(filled_data_values)
    filtered_fft = data_fft * filter_mask
    filtered_data_values = np.real(ifft2(filtered_fft))
    filtered_da = xr.DataArray(filtered_data_values, coords=data_array.coords, dims=data_array.dims, name=data_array.name, attrs=data_array.attrs)
    return filtered_da.where(~nan_mask)

def apply_scale_separation_plev(ds_derived: xr.Dataset) -> (xr.Dataset, xr.Dataset):
    """
    Separates each variable in a multi-level dataset into synoptic and meso-scale components.
    Handles 4D (time, level, lat, lon) DataArrays by looping.
    """
    synoptic_vars = {}
    meso_vars = {}

    for var_name in ds_derived.data_vars:
        original_da_4d = ds_derived[var_name]

        synoptic_level_slices = []
        meso_level_slices = []

        # Loop through each pressure level
        for level_val in original_da_4d.level:
            original_da_3d = original_da_4d.sel(level=level_val)

            synoptic_time_slices = []
            meso_time_slices = []

            # Loop through each time step
            for t_step in original_da_3d.time:
                time_slice = original_da_3d.sel(time=t_step)
                time_slice_plain = time_slice.metpy.dequantify()
                synoptic_slice_plain = synoptic_scale_filter(time_slice_plain)
                meso_slice = time_slice_plain - synoptic_slice_plain
                synoptic_time_slices.append(synoptic_slice_plain)
                meso_time_slices.append(meso_slice)

            # Reconstruct the 3D array for this level
            if synoptic_time_slices:
                synoptic_level_slices.append(xr.concat(synoptic_time_slices, dim='time').assign_coords(level=level_val))
                meso_level_slices.append(xr.concat(meso_time_slices, dim='time').assign_coords(level=level_val))

        # Reconstruct the 4D array for this variable
        if synoptic_level_slices:
            synoptic_da = xr.concat(synoptic_level_slices, dim='level').transpose('time', 'level', 'latitude', 'longitude')
            meso_da = xr.concat(meso_level_slices, dim='level').transpose('time', 'level', 'latitude', 'longitude')

            synoptic_da.attrs.update(original_da_4d.attrs)
            meso_da.attrs.update(original_da_4d.attrs)

            synoptic_vars[f"{var_name}_synoptic"] = synoptic_da
            meso_vars[f"{var_name}_meso"] = meso_da

    return xr.Dataset(synoptic_vars), xr.Dataset(meso_vars)


def calculate_all_derived_variables_plev(ds_events: xr.Dataset) -> xr.Dataset:
    """
    Prepares the dataset for scale separation. This involves calculating
    theta_e using the external module and returning it along with the base variables.
    """
    # Select just the base variables to start with
    base_vars = {var: ds_events[var] for var in ['z', 't', 'q', 'u', 'v', 'w'] if var in ds_events}

    # Calculate Theta-e using the new multi-level external function
    if all(k in ds_events for k in ['t', 'q', 'level']):
        # The function expects a Dataset with t and q
        input_for_theta_e = ds_events[['t', 'q']]

        # Call the new function. It returns a Dataset containing 'theta_e'.
        theta_e_dataset = calculate_theta_e_on_levels(input_for_theta_e)

        # Extract the DataArray and add it to our dictionary of variables.
        # The external function is expected to return a plain (unitless) array,
        # which fits our established workflow.
        base_vars['theta_e'] = theta_e_dataset['theta_e']

    return xr.Dataset(base_vars)

def save_events_plev(
    out_path: Path,
    events_ds: xr.Dataset,
    period_details: Dict[str, Any],
    scale_type: str
):
    """Saves the individual processed multi-level events to a NetCDF file."""
    events_ds.attrs.update({
        'description': f"MJJAS individual {scale_type}-scale events for derived pressure-level variables.",
        'period_name': period_details['name'],
        'period_start_year': period_details['start'],
        'period_end_year': period_details['end'],
        'history': f"Created by process_events_plev.py on {pd.Timestamp.now(tz='UTC')}"
    })
    encoding_options = {v: {'zlib': True, 'complevel': 4, '_FillValue': np.float32(1e20)} for v in events_ds.data_vars}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    events_ds.to_netcdf(out_path, encoding=encoding_options)
    logging.info(f"Wrote individual {scale_type} events to {out_path}")

# --- Main Script ---
def main():
    parser = argparse.ArgumentParser(description="Process individual multi-level ERA5 events and save synoptic/meso components.")
    # --- ARGUMENTS WITH DEFAULTS from original script ---
    parser.add_argument("--era5_dir", type=Path, default="/data/reloclim/normal/INTERACT/ERA5/pressure_levels/", help="Directory containing ERA5 monthly pressure level files")
    parser.add_argument("--comp_csv_base", type=Path, default="/nas/home/dkn/Desktop/MoCCA/composites/scripts/synoptic_composites/csv/composite_", help="Base path for composite CSV files (e.g., './csv/composite_')")
    parser.add_argument("--output_dir", type=Path, default="/home/dkn/composites/ERA5/", help="Output directory")

    parser.add_argument("--period", choices=list(PERIODS.keys()), default="historical", help="Period to process")
    parser.add_argument("--region", type=str, required=True, help="Region identifier for CSV/output")
    parser.add_argument("--levels", type=str, default="850,500,250", help="Comma-separated pressure levels in hPa")
    parser.add_argument("--time_offsets", default="-12,0,12", help="Comma-separated time offsets in hours")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--noMCS", action="store_true", help="Use noMCS event CSV instead of MCS")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    logging.info(f"--- Starting Individual Event Processing Script for Pressure Levels ---")
    logging.info(f"Run arguments: {args}")

    levels_parsed = [int(l.strip()) for l in args.levels.split(',')]
    period_info = PERIODS[args.period]
    suffix = "_nomcs.csv" if args.noMCS else "_mcs.csv"
    csv_path = Path(f"{args.comp_csv_base}{args.region}{suffix}")

    if not csv_path.exists():
        logging.error(f"FATAL: Event CSV file not found: {csv_path}")
        sys.exit(1)

    df_all_events = pd.read_csv(csv_path, parse_dates=[0])
    df_all_events.columns = ['datetime' if i == 0 else c for i, c in enumerate(df_all_events.columns)]
    df_all_events['datetime'] = df_all_events['datetime'].dt.round('H')
    df_all_events['year_of_event'] = df_all_events['datetime'].dt.year
    df_all_events['month_of_event'] = df_all_events['datetime'].dt.month

    df_filtered_events = df_all_events[
        df_all_events['year_of_event'].between(period_info['start'], period_info['end']) &
        df_all_events['month_of_event'].isin(TARGET_MONTHS)
    ].copy()

    if df_filtered_events.empty:
        logging.warning(f"No events found in {csv_path} for period {period_info['name']} and MJJAS months. Exiting.")
        sys.exit(0)

    offset_column_map = create_offset_cols(df_filtered_events)
    selected_time_offsets = sorted([int(x) for x in args.time_offsets.split(',')])
    unique_wts = sorted(df_filtered_events['wt'].unique()) if 'wt' in df_filtered_events else []
    weather_types_to_process = [0] + [wt for wt in unique_wts if wt != 0]

    all_events_list = []

    for wt_value in weather_types_to_process:
        logging.info(f"Processing Weather Type (WT) = {wt_value}")
        df_wt = df_filtered_events if wt_value == 0 else df_filtered_events[df_filtered_events['wt'] == wt_value]
        if df_wt.empty: continue

        for offset_value in selected_time_offsets:
            actual_time_column = offset_column_map.get(offset_value)
            if not actual_time_column: continue

            df_offset = df_wt.dropna(subset=[actual_time_column]).copy()
            if df_offset.empty: continue
            df_offset[actual_time_column] = pd.to_datetime(df_offset[actual_time_column])

            logging.info(f"  Processing Offset = {offset_value}h...")

            for target_composite_month in TARGET_MONTHS:
                logging.debug(f"    Target Composite Month: {target_composite_month}")

                final_datetimes_to_load = pd.DatetimeIndex(
                    df_offset[df_offset['month_of_event'] == target_composite_month][actual_time_column]
                )
                if final_datetimes_to_load.empty: continue

                datetimes_grouped_by_era5_file = {}
                for dt in final_datetimes_to_load:
                    file_key = (dt.year, dt.month)
                    datetimes_grouped_by_era5_file.setdefault(file_key, []).append(dt)

                for (year, month), datetimes_in_file in datetimes_grouped_by_era5_file.items():
                    era5_file_path = get_era5_file(args.era5_dir, year, month)
                    if not era5_file_path.exists():
                        logging.warning(f"ERA5 file not found, skipping: {era5_file_path}")
                        continue

                    logging.debug(f"      Opening {era5_file_path} for {len(datetimes_in_file)} datetimes.")
                    with xr.open_dataset(era5_file_path) as raw_ds:
                        ds = standardize_ds(raw_ds).load()

                    event_data = ds.sel(time=pd.DatetimeIndex(datetimes_in_file), method='nearest', tolerance=pd.Timedelta('1H'))
                    _, unique_indices = np.unique(event_data.time, return_index=True)
                    event_data = event_data.isel(time=unique_indices)

                    if event_data.time.size == 0: continue

                    event_data_plev = event_data.sel(level=levels_parsed)

                    derived_vars = calculate_all_derived_variables_plev(event_data_plev)

                    n_events = derived_vars.dims.get('time', 0)
                    if n_events > 0:
                        event_wt = xr.DataArray(np.full(n_events, wt_value), dims="time", coords={"time": derived_vars.time})
                        event_offset = xr.DataArray(np.full(n_events, offset_value), dims="time", coords={"time": derived_vars.time})
                        event_month = xr.DataArray(np.full(n_events, target_composite_month), dims="time", coords={"time": derived_vars.time})

                        derived_vars['event_weather_type'] = event_wt
                        derived_vars['event_time_offset'] = event_offset
                        derived_vars['event_target_month'] = event_month

                        all_events_list.append(derived_vars)

    if not all_events_list:
        logging.error("No events were processed successfully. No output files will be created.")
        sys.exit(0)

    logging.info("Concatenating all individual synoptic events...")
    final_events_ds = xr.concat(all_events_list, dim="time")

    output_suffix_base = "_nomcs" if args.noMCS else ""

    output_syn_path = args.output_dir / f"events_plev_{args.region}_{period_info['name']}{output_suffix_base}.nc"
    save_events_plev(output_syn_path, final_events_ds, period_info, 'synoptic')

    logging.info("Processing completed successfully.")

if __name__ == "__main__":
    main()