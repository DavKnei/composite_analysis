#!/usr/bin/env python3
"""
Compute JJA ERA5 composites for derived single-level variables
(upper-level jet, divergence, PV, shear, moisture-flux convergence,
low-level convergence), stratified by weather type and time offset.
Refactored to load monthly data into memory, use direct MetPy calls,
and avoid try-except blocks in core processing.
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

# MetPy for dynamic calculations
from metpy.calc import (
    potential_vorticity_baroclinic,
    lat_lon_grid_deltas,
    potential_temperature,
    divergence as mp_divergence
)
from metpy.units import units

# Domain & months
DOMAIN_LAT = (20, 55)
DOMAIN_LON = (-20, 40)
TARGET_MONTHS = [6, 7, 8]  # JJA
PERIODS = {
    "historical": {"start": 1996, "end": 2005, "name": "historical"},
    "evaluation": {"start": 2000, "end": 2009, "name": "evaluation"}
}

# Define the derived variables to be computed
DERIVED_VARIABLES = [
    'jet_speed_250', 'div_250', 'pv_500',
    'shear_250_850', 'mfc_850', 'conv_850'
]

def get_era5_file(era5_dir: Path, year: int, month: int) -> Path:
    """Construct the ERA5 monthly filename (e.g., "2005-08_NA.nc")."""
    fname = f"{year}-{month:02d}_NA.nc"
    return era5_dir / fname


def standardize_ds(ds: xr.Dataset) -> xr.Dataset:
    """Rename coords to latitude/longitude, sort latitude, subset domain."""
    if 'lat' in ds.coords:
        ds = ds.rename({'lat': 'latitude'})
    if 'lon' in ds.coords:
        ds = ds.rename({'lon': 'longitude'})
    
    # Ensure latitude is sorted ascending
    if ds['latitude'].size > 1 and ds['latitude'].values[0] > ds['latitude'].values[-1]:
        # Ensure we are passing a 1D array / list to reindex
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

# --- Derived Variable Calculation Functions ---

def _calculate_divergence_base(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """
    Base helper to calculate divergence using MetPy with explicit dx, dy.
    Iterates over the time dimension to avoid indexing errors with 3D u/v and 2D dx/dy.
    """
    u_q = u.metpy.quantify()
    v_q = v.metpy.quantify()

    # Calculate dx and dy from coordinates. These are 2D.
    # If u.longitude and u.latitude are DataArrays, dx_spacing and dy_spacing will also be DataArrays.
    dx_spacing, dy_spacing = lat_lon_grid_deltas(u.longitude, u.latitude)

    divergence_slices = []
    
    # Check if 'time' dimension exists
    if 'time' in u_q.dims:
        for t_idx in range(u_q.time.size):
            u_slice = u_q.isel(time=t_idx)  # Becomes 2D (lat, lon)
            v_slice = v_q.isel(time=t_idx)  # Becomes 2D (lat, lon)
            
            # Now, all inputs to mp_divergence effectively describe a 2D field
            div_slice = mp_divergence(u_slice, v_slice, dx=dx_spacing, dy=dy_spacing)
            divergence_slices.append(div_slice)
        
        # Concatenate slices along the time dimension
        if divergence_slices:
            # Preserve original time coordinate if possible
            time_coord = u_q.time
            # If divergence_slices are pint.Quantities wrapping xr.DataArray, xr.concat should work
            # If they are bare Quantities, we might need to handle units more manually or ensure
            # they are DataArrays before concat. MetPy functions usually return DataArray if input is DataArray.
            combined_div_q = xr.concat(divergence_slices, dim=time_coord)
            # Ensure the final DataArray has the correct name if needed, though downstream processing renames it.
            # combined_div_q = combined_div_q.rename(u.name + "_divergence") # Or a generic name
        else: # Should not happen if u_q has a time dimension and size > 0
            # Create an empty or NaN array with the expected dimensions if no time slices
            # This case needs careful handling based on what an empty result should look like
            # For now, assume time.size > 0 if 'time' is in dims.
            # If u_q.time.size can be 0, this needs an appropriate empty DataArray structure.
            # However, if u_q.time.size is 0, the loop won't run, and combined_div_q will be undefined.
            # Let's prepare for it:
             if not u_q.time.size: # if time dimension exists but is empty
                # Construct an empty/NaN DataArray with the expected spatial shape but empty time
                expected_spatial_shape = u_q.isel(time=slice(0)).shape[1:] # lat, lon shape
                empty_shape = (0,) + expected_spatial_shape
                # Get the unit of divergence, e.g., 1/seconds
                # This is a bit tricky without knowing the exact output unit beforehand
                # Defaulting to NaN DataArray without units, dequantify will handle it or raise error
                # Or, calculate divergence once with dummy 2D data to get units, if really needed here.
                # For simplicity, let's assume dequantify handles it or it's okay to be unitless if empty.
                combined_div_q = xr.DataArray(np.full(empty_shape, np.nan), 
                                            coords={'time': [], 
                                                    'latitude': u.latitude, 
                                                    'longitude': u.longitude}, # use original coords
                                            dims=('time',) + u_q.dims[1:]) 
             else: # This means divergence_slices was empty but u_q.time.size > 0, indicates an issue
                 raise ValueError("Divergence calculation resulted in no slices despite time dimension existing.")


    else: # Input u_q is already 2D (no 'time' dimension)
        # This case might occur if the input `ds_events` only has one time step and it was squeezed.
        # Or if the functions are ever called with purely 2D spatial data.
        combined_div_q = mp_divergence(u_q, v_q, dx=dx_spacing, dy=dy_spacing)

    return combined_div_q.metpy.dequantify()


def calculate_jet_speed_250(ds_events: xr.Dataset) -> xr.DataArray:
    u250 = ds_events.u.sel(level=250)
    v250 = ds_events.v.sel(level=250)
    jet_speed = np.hypot(u250, v250).drop_vars('level', errors='ignore')
    jet_speed.attrs.update({'units': 'm s-1', 'long_name': 'Wind speed at 250 hPa'})
    return jet_speed.rename("jet_speed_250")

def calculate_div_250(ds_events: xr.Dataset) -> xr.DataArray:
    u250 = ds_events.u.sel(level=250)
    v250 = ds_events.v.sel(level=250)
    div = _calculate_divergence_base(u250, v250).drop_vars('level', errors='ignore')
    div.attrs.update({'units': 's-1', 'long_name': 'Divergence of horizontal wind at 250 hPa'})
    return div.rename("div_250")

def calculate_conv_850(ds_events: xr.Dataset) -> xr.DataArray:
    u850 = ds_events.u.sel(level=850)
    v850 = ds_events.v.sel(level=850)
    conv = (_calculate_divergence_base(u850, v850) * -1.0).drop_vars('level', errors='ignore')
    conv.attrs.update({'units': 's-1', 'long_name': 'Convergence of horizontal wind at 850 hPa'})
    return conv.rename("conv_850")

def calculate_mfc_850(ds_events: xr.Dataset) -> xr.DataArray:
    u850 = ds_events.u.sel(level=850)
    v850 = ds_events.v.sel(level=850)
    q850 = ds_events.q.sel(level=850) 
    
    uq850 = (u850 * q850).rename("uq850")
    vq850 = (v850 * q850).rename("vq850")
    
    mfc = (_calculate_divergence_base(uq850, vq850) * -1.0).drop_vars('level', errors='ignore')
    mfc.attrs.update({'units': 'kg kg-1 s-1', 'long_name': 'Moisture flux convergence at 850 hPa'})
    return mfc.rename("mfc_850")

def calculate_shear_250_850(ds_events: xr.Dataset) -> xr.DataArray:
    u250 = ds_events.u.sel(level=250)
    v250 = ds_events.v.sel(level=250)
    u850 = ds_events.u.sel(level=850)
    v850 = ds_events.v.sel(level=850)
    
    shear = np.hypot(u250 - u850, v250 - v850)
    shear.attrs.update({'units': 'm s-1', 'long_name': 'Magnitude of vector wind shear between 250 hPa and 850 hPa'})
    return shear.rename("shear_250_850")

def calculate_pv_500(ds_events: xr.Dataset) -> xr.DataArray:
    ds_q = ds_events.metpy.quantify()
    theta = potential_temperature(ds_q.level, ds_q.t) 
    
    pv_baroclinic = potential_vorticity_baroclinic(
        theta, ds_q.level, ds_q.u, ds_q.v, latitude=ds_q.latitude
    )
    
    pv_500hpa = (pv_baroclinic.sel(level=500 * units.hPa).metpy.dequantify() * 1e6)
    pv_500hpa = pv_500hpa.drop_vars('level', errors='ignore')
    pv_500hpa.attrs.update({'units': 'PVU', 'long_name': 'Potential Vorticity at 500 hPa'})
    return pv_500hpa.rename("pv_500")


def calculate_all_derived_variables(ds_events: xr.Dataset) -> xr.Dataset:
    """Calculates all derived variables for the given event dataset."""
    # Creating an empty dataset and adding variables one by one ensures
    # that if a calculation fails, it fails before assignment.
    derived_ds_dict = {}
    derived_ds_dict['jet_speed_250'] = calculate_jet_speed_250(ds_events)
    derived_ds_dict['div_250'] = calculate_div_250(ds_events)
    derived_ds_dict['conv_850'] = calculate_conv_850(ds_events)
    derived_ds_dict['mfc_850'] = calculate_mfc_850(ds_events)
    derived_ds_dict['shear_250_850'] = calculate_shear_250_850(ds_events)
    derived_ds_dict['pv_500'] = calculate_pv_500(ds_events)
    return xr.Dataset(derived_ds_dict, coords=ds_events.coords)


def save_composites(
    out_path: Path,
    composite_means_dict: Dict[str, xr.DataArray], 
    event_counts_da: xr.DataArray, 
    wts_list: List[int],
    target_months_list: List[int],
    offs_list: List[int],
    lat_values: np.ndarray,
    lon_values: np.ndarray,
    period_details: Dict[str, Any]
):
    """Saves the computed composites to a NetCDF file."""
    
    ds_output_vars = {**composite_means_dict, "event_count": event_counts_da}

    ds_to_save = xr.Dataset(
        ds_output_vars,
        coords={
            'wt': wts_list,
            'month': target_months_list,
            'off': offs_list,
            'latitude': lat_values,
            'longitude': lon_values
        }
    )

    ds_to_save.attrs.update({
        'description': (
            "JJA composites of derived single-level variables."
        ),
        'period_name': period_details['name'],
        'period_start_year': period_details['start'],
        'period_end_year': period_details['end'],
        'history': f"Created on {pd.Timestamp.now(tz='UTC')}"
    })

    encoding_options = {v: {'zlib': True, 'complevel': 4, '_FillValue': np.float32(1e20)} for v in ds_to_save.data_vars}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_to_save.to_netcdf(out_path, encoding=encoding_options)
    logging.info(f"Wrote composites to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute JJA single-level derived composites.")
    parser.add_argument("--data_dir", type=Path, default="/data/reloclim/normal/INTERACT/ERA5/pressure_levels", help="Directory with ERA5 monthly files")
    parser.add_argument("--period", choices=list(PERIODS.keys()), default="evaluation", help="Period to process")
    parser.add_argument("--wt_csv_base", default="./csv/composite_", help="Base path for weather type CSVs")
    parser.add_argument("--region", required=True, help="Region identifier for CSV/output")
    parser.add_argument("--output_dir", type=Path, default="/home/dkn/composites/ERA5/", help="Output directory")
    parser.add_argument("--time_offsets", default="-12,0,12", help="Comma-separated time offsets in hours")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--noMCS", action="store_true", help="Use noMCS event CSV")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    warnings.filterwarnings("ignore", category=FutureWarning)

    period_info = PERIODS[args.period]
    suffix = "_nomcs" if args.noMCS else "_mcs"
    csv_path = Path(f"{args.wt_csv_base}{args.region}{suffix}.csv")
    
    if not csv_path.exists():
        logging.error(f"FATAL: Event CSV file not found: {csv_path}")
        sys.exit(1) # Critical error, exit
        
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
        logging.warning(f"No events found in {csv_path} for period {period_info['name']} and JJA months. Output may be empty or reflect no data.")

    offset_column_map = create_offset_cols(df_filtered_events)
    selected_time_offsets = sorted([int(x) for x in args.time_offsets.split(',')])
    
    unique_wts = sorted(df_filtered_events['wt'].unique()) if not df_filtered_events.empty else []
    # Ensure WT 0 (all events) is included, typically first.
    weather_types_to_process = [0] + [wt for wt in unique_wts if wt != 0]
    if not unique_wts and 0 not in weather_types_to_process: # Handle case where df is empty from start
         weather_types_to_process = [0]

    # Initialize accumulators for sums and counts
    # Grid coordinates (lat, lon) will be determined from the first successfully processed ERA5 file.
    lat_coord_values: Optional[np.ndarray] = None
    lon_coord_values: Optional[np.ndarray] = None
    # Using dictionaries to store sum DataArrays, keyed by variable name
    global_sum_accumulators: Dict[str, xr.DataArray] = {}
    global_event_counts_accumulator: Optional[xr.DataArray] = None
    is_first_file_processed = False

    for wt_value in weather_types_to_process:
        logging.info(f"Processing Weather Type (WT) = {wt_value}")
        
        current_wt_df = df_filtered_events if wt_value == 0 else df_filtered_events[df_filtered_events['wt'] == wt_value]

        if current_wt_df.empty:
            logging.info(f"  No events for WT={wt_value} in the filtered data. Skipping.")
            continue
        
        for offset_value in selected_time_offsets:
            # Determine the column in the DataFrame that holds the datetimes for this specific offset
            actual_time_column = offset_column_map.get(offset_value, 'datetime')
            
            if actual_time_column not in current_wt_df.columns:
                logging.warning(f"  Offset column '{actual_time_column}' for offset {offset_value}h not found. Skipping WT={wt_value}, offset={offset_value}h.")
                continue

            # Series of datetimes (for this offset) that need to be loaded from ERA5 files
            # NaNs are dropped, as they represent events where the offset time is not applicable/available
            datetimes_for_offset = current_wt_df.dropna(subset=[actual_time_column])[actual_time_column]
            
            if datetimes_for_offset.empty:
                logging.info(f"  Offset {offset_value}h (column: {actual_time_column}): No valid event datetimes after NaN drop. Skipping.")
                continue
            
            # Get the original JJA month of these events to correctly assign them to a composite month
            original_jja_months = current_wt_df.loc[datetimes_for_offset.index, 'month_of_event']
            logging.info(f"  Processing Offset = {offset_value}h. Total potential datetimes to load: {len(datetimes_for_offset)}")

            for target_composite_month in TARGET_MONTHS: # e.g., June, July, August
                logging.debug(f"    Target JJA Composite Month: {target_composite_month}")

                # Filter the offset datetimes: only those whose *original event month* matches the current target_composite_month
                final_datetimes_to_load_for_cell = pd.DatetimeIndex(
                    datetimes_for_offset[original_jja_months == target_composite_month]
                )
                
                if final_datetimes_to_load_for_cell.empty:
                    logging.debug(f"      No events for WT={wt_value}, Offset={offset_value}h, contributing to Target Composite Month={target_composite_month}.")
                    continue
                
                # Group these datetimes by the actual (year, month) they fall into, as this dictates ERA5 file names
                datetimes_grouped_by_era5_file = {}
                for dt in final_datetimes_to_load_for_cell:
                    file_key = (dt.year, dt.month)
                    datetimes_grouped_by_era5_file.setdefault(file_key, []).append(dt)
                
                # Accumulators for the current composite cell (wt, offset, target_composite_month)
                cell_sum_dataarrays: Optional[Dict[str, xr.DataArray]] = None
                cell_total_event_timesteps: int = 0 

                for (era5_file_year, era5_file_month), datetimes_in_this_file in datetimes_grouped_by_era5_file.items():
                    era5_file_path = get_era5_file(args.data_dir, era5_file_year, era5_file_month)
                    logging.debug(f"        Opening {era5_file_path} for {len(datetimes_in_this_file)} datetimes.")

                    # If file doesn't exist, this will raise FileNotFoundError and script will halt
                    with xr.open_dataset(era5_file_path) as raw_monthly_ds:
                        standardized_ds = standardize_ds(raw_monthly_ds)
                        loaded_monthly_ds = standardized_ds.load() # Load into memory
                            
                    if not is_first_file_processed: # First successful file determines grid and initializes global accumulators
                        lat_coord_values = loaded_monthly_ds.latitude.values.copy()
                        lon_coord_values = loaded_monthly_ds.longitude.values.copy()
                        
                        coord_spec = {
                            'wt': weather_types_to_process, 'month': TARGET_MONTHS, 'off': selected_time_offsets,
                            'latitude': lat_coord_values, 'longitude': lon_coord_values
                        }
                        count_coord_spec = {
                             'wt': weather_types_to_process, 'month': TARGET_MONTHS, 'off': selected_time_offsets
                        }
                        for var_name in DERIVED_VARIABLES:
                            global_sum_accumulators[var_name] = xr.DataArray(
                                np.zeros((len(weather_types_to_process), len(TARGET_MONTHS), len(selected_time_offsets),
                                         len(lat_coord_values), len(lon_coord_values)), dtype=np.float32),
                                coords=coord_spec, dims=('wt', 'month', 'off', 'latitude', 'longitude')
                            )
                        global_event_counts_accumulator = xr.DataArray(
                            np.zeros((len(weather_types_to_process), len(TARGET_MONTHS), len(selected_time_offsets)), dtype=np.int32),
                            coords=count_coord_spec, dims=('wt', 'month', 'off')
                        )
                        is_first_file_processed = True
                    else: # Check for grid consistency with subsequent files
                        if not (np.array_equal(loaded_monthly_ds.latitude.values, lat_coord_values) and \
                                np.array_equal(loaded_monthly_ds.longitude.values, lon_coord_values)):
                            logging.error(f"FATAL: Grid mismatch in {era5_file_path}. Expected lat/lon like first processed file. Aborting.")
                            sys.exit(1) # Critical error, exit

                    # Select the specific event datetimes from this loaded monthly file
                    event_data_from_file = loaded_monthly_ds.sel(
                        time=pd.DatetimeIndex(datetimes_in_this_file), 
                        method='nearest', tolerance=pd.Timedelta('30M')
                    )

                    if event_data_from_file.time.size == 0:
                        logging.debug(f"          No matching time steps in {era5_file_path} after nearest time selection.")
                        continue # Skip to next file if no relevant times found
                    
                    # Calculate all derived variables for these selected event timesteps
                    derived_variables_for_events = calculate_all_derived_variables(event_data_from_file)
                    
                    # Sum the derived variables over the 'time' dimension for this file's events
                    sums_for_file = derived_variables_for_events.sum(dim='time', skipna=True)
                    
                    # Accumulate sums for the current composite cell
                    if cell_sum_dataarrays is None: # First batch of data for this cell
                        cell_sum_dataarrays = {v_name: sums_for_file[v_name].copy() for v_name in DERIVED_VARIABLES}
                    else:
                        for v_name in DERIVED_VARIABLES:
                            cell_sum_dataarrays[v_name] += sums_for_file[v_name]
                    
                    cell_total_event_timesteps += event_data_from_file.time.size
                
                # After processing all ERA5 files for this specific composite cell (wt, offset, target_composite_month)
                if cell_sum_dataarrays is not None and is_first_file_processed: # Ensure data was processed and global accumulators exist
                    # Get integer indices for assignment into global accumulators
                    wt_idx = weather_types_to_process.index(wt_value)
                    month_idx = TARGET_MONTHS.index(target_composite_month)
                    offset_idx = selected_time_offsets.index(offset_value)

                    for var_name in DERIVED_VARIABLES:
                        # Add this cell's sums to the corresponding slice in the global sum accumulator
                        # Fill NaN with 0 before adding to avoid NaN propagation if a file had all NaNs for a variable
                        global_sum_accumulators[var_name][wt_idx, month_idx, offset_idx, :, :] += cell_sum_dataarrays[var_name].fillna(0)
                    
                    # Add this cell's event count to the global event count accumulator
                    if global_event_counts_accumulator is not None: # Should be initialized if is_first_file_processed
                        global_event_counts_accumulator[wt_idx, month_idx, offset_idx] += cell_total_event_timesteps

    # --- End of main processing loops ---

    if not is_first_file_processed:
        logging.error("No ERA5 data files were successfully processed. Cannot calculate composites or save output.")
        if df_filtered_events.empty :
             logging.info("This is likely because no events were found in the input CSV for the specified period/months.")
        sys.exit(1) # Cannot proceed without grid information or any data

    # Calculate final means from the global sums and counts
    final_composite_means = {}
    for var_name in DERIVED_VARIABLES:
        # Ensure division by zero results in NaN, not an error or warning during division
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_da = global_sum_accumulators[var_name] / global_event_counts_accumulator
        # Explicitly set to NaN where count is zero
        if global_event_counts_accumulator is not None:
             final_composite_means[f"{var_name}_mean"] = mean_da.where(global_event_counts_accumulator > 0)
        else: # Should not happen if is_first_file_processed is true
             final_composite_means[f"{var_name}_mean"] = mean_da 


    output_filename = args.output_dir / f"composite_single_level_{args.region}_{period_info['name']}{suffix}.nc"
    
    # Ensure lat/lon values are available for saving
    if lat_coord_values is None or lon_coord_values is None or global_event_counts_accumulator is None:
        logging.error("FATAL: Grid coordinates or event counts are missing before saving. This indicates a severe issue in processing logic.")
        sys.exit(1)

    save_composites(
        output_filename,
        final_composite_means,
        global_event_counts_accumulator,
        weather_types_to_process,
        TARGET_MONTHS,
        selected_time_offsets,
        lat_coord_values,
        lon_coord_values,
        period_info
    )
    logging.info("Processing completed.")

if __name__ == "__main__":
    main()