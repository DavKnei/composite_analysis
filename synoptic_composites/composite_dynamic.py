#!/usr/bin/env python3
"""
Compute MJJAS ERA5 composites for derived dynamic variables
(upper-level jet, divergence, PV, shear, moisture-flux convergence,
low-level convergence), stratified by weather type and time offset.
This version includes a spectral filter to separate fields into
synoptic and meso-scale components. It saves both the mean composites
and the individual event data in separate files.
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


# Domain EUR & months
DOMAIN_LAT = (25, 65)
DOMAIN_LON = (-20, 43)
TARGET_MONTHS = [5, 6, 7, 8, 9]  # MJJAS
PERIODS = {
    "historical": {"start": 1991, "end": 2020, "name": "historical"}
}

# Define the derived variables to be computed
DERIVED_VARIABLES = [
    'jet_speed_250', 'div_250', 'pv_500',
    'shear_500_850', 'mfc_850', 'conv_850', 'rv_500', 'rv_250'
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

# --- Spectral Filter and Scale Separation ---
def synoptic_scale_filter(data_array: xr.DataArray) -> xr.DataArray:
    """
    Apply a 2D Gaussian low-pass filter for synoptic scale separation.
    This version operates on and returns unitless DataArrays.
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

    filtered_da = xr.DataArray(
        filtered_data_values,
        coords=data_array.coords,
        dims=data_array.dims,
        name=data_array.name,
        attrs=data_array.attrs
    )

    return filtered_da.where(~nan_mask)

def apply_scale_separation(ds_derived: xr.Dataset) -> (xr.Dataset, xr.Dataset):
    """
    Separates each variable in the dataset into synoptic and meso-scale components.
    """
    synoptic_vars = {}
    meso_vars = {}
    event_dim = 'event'

    for var_name in ds_derived.data_vars:
        original_da = ds_derived[var_name]

        synoptic_slices = []
        meso_slices = []

        for event_idx in original_da[event_dim]:
            event_slice = original_da.sel({event_dim: event_idx})

            event_slice_plain = event_slice.metpy.dequantify()

            synoptic_slice_plain = synoptic_scale_filter(event_slice_plain)
            meso_slice_plain = event_slice_plain - synoptic_slice_plain

            synoptic_slices.append(synoptic_slice_plain)
            meso_slices.append(meso_slice_plain)

        synoptic_da = xr.concat(synoptic_slices, dim=event_dim)
        meso_da = xr.concat(meso_slices, dim=event_dim)

        synoptic_da.attrs.update(original_da.attrs)
        meso_da.attrs.update(original_da.attrs)

        synoptic_vars[f"{var_name}_synoptic"] = synoptic_da.rename(f"{var_name}_synoptic")
        meso_vars[f"{var_name}_meso"] = meso_da.rename(f"{var_name}_meso")

    return xr.Dataset(synoptic_vars), xr.Dataset(meso_vars)


# --- Derived Variable Calculation Functions ---

def _calculate_divergence_base(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    u_q = u.metpy.quantify()
    v_q = v.metpy.dequantify().metpy.quantify()
    dx_spacing, dy_spacing = lat_lon_grid_deltas(u.longitude, u.latitude)
    divergence_slices = []

    event_dim = 'event' if 'event' in u_q.dims else 'time'

    if event_dim in u_q.dims:
        for i in range(u_q[event_dim].size):
            u_slice = u_q.isel({event_dim: i})
            v_slice = v_q.isel({event_dim: i})
            div_slice = mp_divergence(u_slice, v_slice, dx=dx_spacing, dy=dy_spacing)
            divergence_slices.append(div_slice)

        if divergence_slices:
            event_coord = u_q[event_dim]
            combined_div_q = xr.concat(divergence_slices, dim=event_coord)
        else:
             if not u_q[event_dim].size:
                expected_spatial_shape = u_q.isel({event_dim: slice(0)}).shape[1:]
                empty_shape = (0,) + expected_spatial_shape
                empty_coords = {event_dim: [], 'latitude': u.latitude, 'longitude': u.longitude}
                empty_dims = (event_dim,) + u_q.dims[1:]
                combined_div_q = xr.DataArray(np.full(empty_shape, np.nan),
                                            coords=empty_coords,
                                            dims=empty_dims)
             else:
                 raise ValueError("Divergence calculation resulted in no slices despite event dimension existing.")
    else:
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

def calculate_shear_500_850(ds_events: xr.Dataset) -> xr.DataArray:
    u500 = ds_events.u.sel(level=500)
    v500 = ds_events.v.sel(level=500)
    u850 = ds_events.u.sel(level=850)
    v850 = ds_events.v.sel(level=850)
    shear = np.hypot(u500 - u850, v500 - v850)
    shear.attrs.update({'units': 'm s-1', 'long_name': 'Magnitude of vector wind shear between 500 hPa and 850 hPa'})
    return shear.rename("shear_500_850").drop_vars('level', errors='ignore')

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

def calculate_rv_500(ds_events: xr.Dataset) -> xr.DataArray:
    u500 = ds_events.u.sel(level=500)
    v500 = ds_events.v.sel(level=500)
    rel_vorticity = vorticity(u500, v500).metpy.dequantify()
    rel_vorticity.attrs.update({'units': 's-1', 'long_name': 'Relative Vorticity at 500 hPa'})
    return rel_vorticity.rename("rv_500").drop_vars('level', errors='ignore')

def calculate_rv_250(ds_events: xr.Dataset) -> xr.DataArray:
    u250 = ds_events.u.sel(level=250)
    v250 = ds_events.v.sel(level=250)
    rel_vorticity = vorticity(u250, v250).metpy.dequantify()
    rel_vorticity.attrs.update({'units': 's-1', 'long_name': 'Relative Vorticity at 250 hPa'})
    return rel_vorticity.rename("rv_250").drop_vars('level', errors='ignore')

def calculate_all_derived_variables(ds_events: xr.Dataset) -> xr.Dataset:
    """Calculates all derived variables for the given event dataset."""
    derived_ds_dict = {}
    derived_ds_dict['jet_speed_250'] = calculate_jet_speed_250(ds_events)
    derived_ds_dict['div_250'] = calculate_div_250(ds_events)
    derived_ds_dict['conv_850'] = calculate_conv_850(ds_events)
    derived_ds_dict['mfc_850'] = calculate_mfc_850(ds_events)
    derived_ds_dict['shear_500_850'] = calculate_shear_500_850(ds_events)
    derived_ds_dict['pv_500'] = calculate_pv_500(ds_events)
    derived_ds_dict['rv_500'] = calculate_rv_500(ds_events)
    derived_ds_dict['rv_250'] = calculate_rv_250(ds_events)

    new_coords = {c: ds_events.coords[c] for c in ds_events.coords if c != 'level'}
    return xr.Dataset(derived_ds_dict, coords=new_coords)


def save_composites(
    out_path: Path,
    composite_means_dict: Dict[str, xr.DataArray],
    event_counts_da: xr.DataArray,
    wts_list: List[int],
    target_months_list: List[int],
    offs_list: List[int],
    lat_values: np.ndarray,
    lon_values: np.ndarray,
    period_details: Dict[str, Any],
    scale_type: str
):
    """Saves the computed mean composites to a NetCDF file."""
    ds_output_vars = {**composite_means_dict, "event_count": event_counts_da}
    ds_to_save = xr.Dataset(
        ds_output_vars,
        coords={
            'weather_type': wts_list,
            'month': target_months_list,
            'time_diff': offs_list,
            'latitude': lat_values,
            'longitude': lon_values
        }
    )
    ds_to_save.attrs.update({
        'description': f"MJJAS {scale_type}-scale mean composites of derived single-level variables.",
        'period_name': period_details['name'],
        'period_start_year': period_details['start'],
        'period_end_year': period_details['end'],
        'history': f"Created on {pd.Timestamp.now(tz='UTC')}"
    })
    encoding_options = {v: {'zlib': True, 'complevel': 4, '_FillValue': np.float32(1e20)} for v in ds_to_save.data_vars}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_to_save.to_netcdf(out_path, encoding=encoding_options)
    logging.info(f"Wrote {scale_type} mean composites to {out_path}")


def save_events(
    out_path: Path,
    events_ds: xr.Dataset,
    period_details: Dict[str, Any],
    scale_type: str
):
    """
    Saves the final, concatenated events dataset to a NetCDF file.
    This function now only writes a new file, it does not append.
    """
    events_ds.attrs.update({
        'description': f"MJJAS individual {scale_type}-scale events for derived single-level variables.",
        'period_name': period_details['name'],
        'period_start_year': period_details['start'],
        'period_end_year': period_details['end'],
        'history': f"Created on {pd.Timestamp.now(tz='UTC')}"
    })

    encoding_options = {v: {'zlib': True, 'complevel': 4, '_FillValue': np.float32(1e20)} for v in events_ds.data_vars}
    
    # Add explicit encoding for event and track_number coordinates to prevent
    # them from being misinterpreted as time variables by xarray.
    encoding_options['event'] = {'dtype': 'int64', '_FillValue': None}
    encoding_options['track_number'] = {'dtype': 'int64', '_FillValue': None}

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the file in one go. 'w' mode will overwrite if the file exists.
    events_ds.to_netcdf(out_path, mode='w', encoding=encoding_options, unlimited_dims=['event'])
    logging.info(f"Wrote all {events_ds.sizes['event']} individual {scale_type} events to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute MJJAS single-level derived composites with scale separation.")
    parser.add_argument("--data_dir", type=Path, default="/data/reloclim/normal/INTERACT/ERA5/pressure_levels", help="Directory with ERA5 monthly files")
    parser.add_argument("--period", choices=list(PERIODS.keys()), default="historical", help="Period to process. Default: historical")
    parser.add_argument("--wt_csv_base", default="/nas/home/dkn/Desktop/MoCCA/composites/scripts/synoptic_composites/csv/composite_", help="Base path for weather type CSVs")
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
        sys.exit(1)

    # --- MODIFIED: Robustly read the CSV with specified dtypes ---
    # Define the expected data types for numeric columns to prevent pandas
    # from incorrectly converting them to datetime objects. This is the root fix.
    dtype_mapping = {
        'center_lat': float,
        'center_lon': float,
        'track_number': float,  # Read as float to handle potential NaNs
        'total_precip': float,
        'area': float,
        'wt': float,            # Read as float to handle potential NaNs
    }
    # Read all other columns as objects to prevent unintended type inference
    df_all_events = pd.read_csv(csv_path, parse_dates=[0], dtype=dtype_mapping)

    if 'track_number' not in df_all_events.columns:
        logging.error(f"FATAL: 'track_number' column not found in {csv_path}. This column is required.")
        sys.exit(1)
        
    # Now, safely convert integer columns, filling any missing values appropriately
    df_all_events['track_number'] = df_all_events['track_number'].fillna(-1).astype(int)
    df_all_events['wt'] = df_all_events['wt'].fillna(0).astype(int)

    # Convert all time-related columns to datetime, coercing errors
    for col in df_all_events.columns:
        if 'time' in col:
            df_all_events[col] = pd.to_datetime(df_all_events[col], errors='coerce')
            df_all_events[col] = df_all_events[col].dt.round('H')
    # --- END MODIFIED SECTION ---

    #df_all_events.columns = ['datetime' if i == 0 else c for i, c in enumerate(df_all_events.columns)]
    df_all_events['datetime'] = df_all_events['time_0h'].dt.round('H')
    df_all_events['year_of_event'] = df_all_events['datetime'].dt.year
    df_all_events['month_of_event'] = df_all_events['datetime'].dt.month

    df_filtered_events = df_all_events[
        df_all_events['year_of_event'].between(period_info['start'], period_info['end']) &
        df_all_events['month_of_event'].isin(TARGET_MONTHS)
    ].copy()

    if df_filtered_events.empty:
        logging.warning(f"No events found in {csv_path} for period {period_info['name']} and MJJAS months. Output may be empty or reflect no data.")

    offset_column_map = create_offset_cols(df_filtered_events)
    selected_time_offsets = sorted([int(x) for x in args.time_offsets.split(',')])

    unique_wts = sorted(df_filtered_events['wt'].unique()) if not df_filtered_events.empty else []
    weather_types_to_process = [0] + [wt for wt in unique_wts if wt != 0]
    if not unique_wts and 0 not in weather_types_to_process:
         weather_types_to_process = [0]

    lat_coord_values: Optional[np.ndarray] = None
    lon_coord_values: Optional[np.ndarray] = None
    global_synoptic_sum_accumulators: Dict[str, xr.DataArray] = {}
    global_meso_sum_accumulators: Dict[str, xr.DataArray] = {}
    global_event_counts_accumulator: Optional[xr.DataArray] = None
    is_first_file_processed = False

    all_synoptic_events = []
    all_meso_events = []

    output_suffix_base = "_nomcs" if args.noMCS else ""

    # --- MODIFIED: Simplified and corrected main processing loop ---
    for wt_value in weather_types_to_process:
        logging.info(f"Processing Weather Type (WT) = {wt_value}")
        current_wt_df = df_filtered_events if wt_value == 0 else df_filtered_events[df_filtered_events['wt'] == wt_value]

        if current_wt_df.empty:
            logging.info(f"  No events for WT={wt_value} in the filtered data. Skipping.")
            continue

        for offset_value in selected_time_offsets:
            actual_time_column = offset_column_map.get(offset_value)

            if actual_time_column is None or actual_time_column not in current_wt_df.columns:
                logging.warning(f"  Offset column for offset {offset_value}h not found in CSV. Skipping.")
                continue

            events_for_offset = current_wt_df.dropna(subset=[actual_time_column])
            
            if events_for_offset.empty:
                logging.info(f"  Offset {offset_value}h: No valid event datetimes. Skipping.")
                continue

            logging.info(f"  Processing Offset = {offset_value}h. Total potential events: {len(events_for_offset)}")

            for target_composite_month in TARGET_MONTHS:
                logging.debug(f"    Target Composite Month: {target_composite_month}")
                
                final_events_to_load_for_cell = events_for_offset[
                    events_for_offset['month_of_event'] == target_composite_month
                ]

                if final_events_to_load_for_cell.empty:
                    continue

                events_grouped_by_era5_file = {}
                for _, event_row in final_events_to_load_for_cell.iterrows():
                    dt = event_row[actual_time_column]
                    track_num = event_row['track_number']
                    file_key = (dt.year, dt.month)
                    events_grouped_by_era5_file.setdefault(file_key, []).append((dt, track_num))

                # This logic from here onwards is unchanged.
                # It now operates on a clean, consistent group of events.
                cell_synoptic_sums: Optional[Dict[str, xr.DataArray]] = None
                cell_meso_sums: Optional[Dict[str, xr.DataArray]] = None
                cell_total_event_timesteps: int = 0

                for (era5_file_year, era5_file_month), events_in_this_file in events_grouped_by_era5_file.items():
                    era5_file_path = get_era5_file(args.data_dir, era5_file_year, era5_file_month)
                    logging.debug(f"        Opening {era5_file_path} for {len(events_in_this_file)} events.")

                    unique_dts_to_load = np.unique(pd.DatetimeIndex(list(set(dt for dt, tn in events_in_this_file))).sort_values())

                    if len(unique_dts_to_load) == 0:
                        continue

                    with xr.open_dataset(era5_file_path) as raw_monthly_ds:
                        standardized_ds = standardize_ds(raw_monthly_ds)
                        loaded_monthly_ds = standardized_ds.sel(
                            time=unique_dts_to_load,
                            method='nearest', tolerance=pd.Timedelta('30M')
                        ).load()

                    if not is_first_file_processed:
                        lat_coord_values = loaded_monthly_ds.latitude.values.copy()
                        lon_coord_values = loaded_monthly_ds.longitude.values.copy()

                        coord_spec = {'weather_type': weather_types_to_process, 'month': TARGET_MONTHS, 'time_diff': selected_time_offsets,
                                      'latitude': lat_coord_values, 'longitude': lon_coord_values}
                        count_coord_spec = {'weather_type': weather_types_to_process, 'month': TARGET_MONTHS, 'time_diff': selected_time_offsets}

                        dims_4d = ('weather_type', 'month', 'time_diff', 'latitude', 'longitude')
                        shape_4d = (len(weather_types_to_process), len(TARGET_MONTHS), len(selected_time_offsets),
                                  len(lat_coord_values), len(lon_coord_values))

                        for var_name in DERIVED_VARIABLES:
                            global_synoptic_sum_accumulators[var_name] = xr.DataArray(np.zeros(shape_4d, dtype=np.float32), coords=coord_spec, dims=dims_4d)
                            global_meso_sum_accumulators[var_name] = xr.DataArray(np.zeros(shape_4d, dtype=np.float32), coords=coord_spec, dims=dims_4d)

                        global_event_counts_accumulator = xr.DataArray(
                            np.zeros((len(weather_types_to_process), len(TARGET_MONTHS), len(selected_time_offsets)), dtype=np.int32),
                            coords=count_coord_spec, dims=('weather_type', 'month', 'time_diff')
                        )
                        is_first_file_processed = True
                    else:
                        if not (np.array_equal(loaded_monthly_ds.latitude.values, lat_coord_values) and \
                                np.array_equal(loaded_monthly_ds.longitude.values, lon_coord_values)):
                            logging.error(f"FATAL: Grid mismatch in {era5_file_path}. Aborting.")
                            sys.exit(1)

                    event_slices = []
                    track_numbers = []
                    for dt, track_num in events_in_this_file:
                        try:
                            actual_time_slice = loaded_monthly_ds.sel(time=dt, method='nearest', tolerance=pd.Timedelta('30M'))
                            event_slices.append(actual_time_slice)
                            track_numbers.append(track_num)
                        except:
                            breakpoint()
                            logging.warning(f"Could not find time {dt} in {era5_file_path} even with nearest selection. Skipping event.")
                            continue


                    if not event_slices:
                        logging.debug(f"No matching time steps in {era5_file_path} after nearest time selection.")
                        continue

                    event_data_from_file = xr.concat(event_slices, dim=pd.Index(range(len(event_slices)), name='event'))
                    event_data_from_file = event_data_from_file.assign_coords(track_number=('event', track_numbers))

                    if event_data_from_file.sizes['event'] == 0:
                        continue

                    derived_variables_for_events = calculate_all_derived_variables(event_data_from_file)
                    synoptic_vars, meso_vars = apply_scale_separation(derived_variables_for_events)

                    synoptic_sums_for_file = synoptic_vars.sum(dim='event', skipna=True)
                    meso_sums_for_file = meso_vars.sum(dim='event', skipna=True)

                    if cell_synoptic_sums is None:
                        cell_synoptic_sums = {v.replace('_synoptic', ''): synoptic_sums_for_file[v].copy() for v in synoptic_sums_for_file.data_vars}
                        cell_meso_sums = {v.replace('_meso', ''): meso_sums_for_file[v].copy() for v in meso_sums_for_file.data_vars}
                    else:
                        for v_name in DERIVED_VARIABLES:
                            cell_synoptic_sums[v_name] += synoptic_sums_for_file[f"{v_name}_synoptic"]
                            cell_meso_sums[v_name] += meso_sums_for_file[f"{v_name}_meso"]

                    cell_total_event_timesteps += event_data_from_file.sizes['event']

                    n_events_in_chunk = synoptic_vars.dims['event']
                    if n_events_in_chunk > 0 and not args.noMCS:
                        event_coords = {"event": synoptic_vars.event}
                        event_wt = xr.DataArray(np.full(n_events_in_chunk, wt_value), dims="event", coords=event_coords)
                        event_offset = xr.DataArray(np.full(n_events_in_chunk, offset_value), dims="event", coords=event_coords)
                        event_month = xr.DataArray(np.full(n_events_in_chunk, target_composite_month), dims="event", coords=event_coords)

                        synoptic_vars['event_weather_type'] = event_wt
                        synoptic_vars['event_time_offset'] = event_offset
                        synoptic_vars['event_target_month'] = event_month
                        all_synoptic_events.append(synoptic_vars)

                        meso_vars['event_weather_type'] = event_wt
                        meso_vars['event_time_offset'] = event_offset
                        meso_vars['event_target_month'] = event_month
                        all_meso_events.append(meso_vars)

                if cell_synoptic_sums is not None and is_first_file_processed:
                    wt_idx = weather_types_to_process.index(wt_value)
                    month_idx = TARGET_MONTHS.index(target_composite_month)
                    offset_idx = selected_time_offsets.index(offset_value)

                    for var_name in DERIVED_VARIABLES:
                        global_synoptic_sum_accumulators[var_name][wt_idx, month_idx, offset_idx, :, :] += cell_synoptic_sums[var_name].fillna(0)
                        global_meso_sum_accumulators[var_name][wt_idx, month_idx, offset_idx, :, :] += cell_meso_sums[var_name].fillna(0)

                    if global_event_counts_accumulator is not None:
                        global_event_counts_accumulator[wt_idx, month_idx, offset_idx] += cell_total_event_timesteps

    if not is_first_file_processed:
        logging.error("No data processed. Cannot create output files.")
        if df_filtered_events.empty:
             logging.info("This is likely because no events were found in the input CSV for the specified period/months.")
        sys.exit(1)

    # --- Calculate and Save Final Mean Composites ---
    if lat_coord_values is not None and lon_coord_values is not None and global_event_counts_accumulator is not None:
        final_synoptic_means = {}
        final_meso_means = {}
        for var_name in DERIVED_VARIABLES:
            with np.errstate(divide='ignore', invalid='ignore'):
                synoptic_mean_da = global_synoptic_sum_accumulators[var_name] / global_event_counts_accumulator
                meso_mean_da = global_meso_sum_accumulators[var_name] / global_event_counts_accumulator
            final_synoptic_means[f"{var_name}_mean"] = synoptic_mean_da.where(global_event_counts_accumulator > 0)
            final_meso_means[f"{var_name}_mean"] = meso_mean_da.where(global_event_counts_accumulator > 0)

        output_synoptic_filename = args.output_dir / f"composite_dynamic_synoptic_{args.region}_{period_info['name']}{output_suffix_base}.nc"
        save_composites(output_synoptic_filename, final_synoptic_means, global_event_counts_accumulator,
                        weather_types_to_process, TARGET_MONTHS, selected_time_offsets,
                        lat_coord_values, lon_coord_values, period_info, scale_type='synoptic')

        if not args.noMCS:
            output_meso_filename = args.output_dir / f"composite_dynamic_meso_{args.region}_{period_info['name']}{output_suffix_base}.nc"
            save_composites(output_meso_filename, final_meso_means, global_event_counts_accumulator,
                            weather_types_to_process, TARGET_MONTHS, selected_time_offsets,
                            lat_coord_values, lon_coord_values, period_info, scale_type='meso')
    else:
        logging.warning("Global accumulators not fully initialized. Skipping saving of mean composites.")

    # Save all collected individual events at the end of the script.
    if not args.noMCS:
        if all_synoptic_events:
            logging.info(f"Concatenating {len(all_synoptic_events)} synoptic event chunks...")
            final_synoptic_ds = xr.concat(all_synoptic_events, dim='event')
            output_synoptic_events_filename = args.output_dir / f"events_synoptic_{args.region}_{period_info['name']}{output_suffix_base}.nc"
            save_events(output_synoptic_events_filename, final_synoptic_ds, period_info, 'synoptic')
        else:
            logging.warning("No individual synoptic events were processed to save.")

        if all_meso_events:
            logging.info(f"Concatenating {len(all_meso_events)} meso event chunks...")
            final_meso_ds = xr.concat(all_meso_events, dim='event')
            output_meso_events_filename = args.output_dir / f"events_meso_{args.region}_{period_info['name']}{output_suffix_base}.nc"
            save_events(output_meso_events_filename, final_meso_ds, period_info, 'meso')
        else:
            logging.warning("No individual meso events were processed to save.")


    logging.info("Processing completed.")

if __name__ == "__main__":
    main()
