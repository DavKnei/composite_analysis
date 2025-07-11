""" 1631307 nohup
Script to extract motion-centered data cutouts for Mesoscale Convective System (MCS)
events from ERA5 data.

- The script now checks for the existence of the total-field and meso-field files
  independently. This allows it to create missing meso files even if the total
  field file from a previous run already exists.
- Corrects the end-of-run summary to accurately report the work done in the
  current session, rather than counting all files on disk.
"""
import pandas as pd
import xarray as xr
import numpy as np
import pyproj
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
import warnings
from metpy.units import units
import metpy.calc as mpcalc
from metpy.package_tools import Exporter
from metpy.units import process_units
from metpy.xarray import preprocess_and_wrap
from scipy.fft import set_backend, fftfreq, fft2, ifft2
import pyfftw

# Enable the pyfftw cache for speed
pyfftw.interfaces.cache.enable()
# Set the backend for all subsequent fft calls
set_backend(pyfftw.interfaces.scipy_fft)


warnings.filterwarnings("ignore", message="Slicing is producing a large chunk.")


# --- CONFIGURATION AND CONSTANTS ---
DEFAULT_CSV_PATH = "./csv/event_centered_composite.csv"
# DEFAULT_CSV_PATH = "./csv/debug.csv"
DEFAULT_ERA5_PATH = "/reloclim/dkn/data/ERA5/pressure_levels/merged_files"

# Define the common grid based on the peer-reviewed script
BOX_SIZE_KM = 400 * 2  # Using a larger box (800km)
GRID_RESOLUTION_KM = 25
NUM_GRID_POINTS = int(BOX_SIZE_KM / GRID_RESOLUTION_KM) + 1
RELATIVE_COORDS_KM = np.linspace(-BOX_SIZE_KM / 2, BOX_SIZE_KM / 2, NUM_GRID_POINTS)

# Wavelength where the filter response starts to roll off from 1.0.
HIGH_PASS_WL_KM = 2000.0  # Corresponds to the 'top' of the curve.
# Wavelength parameter that controls the steepness of the roll-off.
LOW_PASS_WL_KM = 200.0

exporter = Exporter(globals())


@exporter.export
@preprocess_and_wrap(wrap_like='t')
@process_units({'t': '[temperature]'}, '[pressure]')
def saturation_vapor_pressure(t):
    """
    Calculates saturation vapor pressure according to IFS Documentation Cy47r3.
    """

    # parameters
    t_ice = 250.16  # K
    t_0 = 273.16  # K
    a_1_w = 611.21  # Pa
    a_3_w = 17.502
    a_4_w = 32.19  # K
    a_1_i = 611.21  # Pa
    a_3_i = 22.587
    a_4_i = -0.7  # K

    # saturation vapor pressure (Tetens formula)
    alpha = np.empty(t.shape)
    svp_w = a_1_w * np.exp(a_3_w * (t - t_0) / (t - a_4_w))
    svp_i = a_1_i * np.exp(a_3_i * (t - t_0) / (t + a_4_i))
    alpha[t <= t_ice] = 0
    alpha[t >= t_0] = 1
    alpha[(t_ice < t) & (t < t_0)] = (
        (t[(t_ice < t) & (t < t_0)] - t_ice) / (t_0 - t_ice)
    ) ** 2
    svp = alpha * svp_w + (1 - alpha) * svp_i

    return svp


mpcalc.saturation_vapor_pressure = saturation_vapor_pressure

def assign_units_from_attrs(ds):
    """
    Loops through a dataset, reads the 'units' attribute string, and
    rebuilds the DataArray with unit-aware data (a pint.Quantity).
    """
    for var_name, da in ds.data_vars.items():
        # Proceed only if the variable has a 'units' attribute and is not already unit-aware
        if 'units' in da.attrs and not hasattr(da.data, 'units'):
            unit_str = da.attrs['units']
            try:
                unit_obj = units(unit_str)
                
                # --- THIS IS THE CORRECT METHOD ---
                # 1. Create a pint.Quantity by multiplying the numpy array by the unit object.
                quantity = da.values * unit_obj
                
                # 2. Replace the variable in the dataset with a new DataArray containing this quantity.
                ds[var_name] = xr.DataArray(
                    data=quantity,
                    coords=da.coords,
                    dims=da.dims,
                    attrs=da.attrs
                )

            except Exception as e:
                logging.warning(
                    f"Could not assign unit for '{var_name}'. "
                    f"Unrecognized unit string: '{unit_str}'. Error: {e}"
                )
    return ds


def spectral_filter_2d(
    data_array: xr.DataArray, high_pass_wl_km: float, low_pass_wl_km: float, axes=(-2, -1)
) -> xr.DataArray:
    """
    Applies the 2D Gaussian low-pass filter using a vectorized approach to handle
    2D or 3D DataArrays efficiently. This function is designed to isolate the
    synoptic-scale component of a field.

    Args:
        data_array: The 2D or 3D xr.DataArray to filter.
        high_pass_wl_km: The wavelength (in km) where the filter response
                         starts to roll off from 1.
        low_pass_wl_km: The wavelength (in km) that controls the steepness
                        of the filter roll-off.
        axes: The dimensions over which to apply the 2D FFT, defaulting to
              the last two dimensions.
    """
    if "latitude" not in data_array.dims or "longitude" not in data_array.dims:
        raise ValueError(
            "Input DataArray must have 'latitude' and 'longitude' dimensions."
        )

    # Use the specified axes to get the shape for the 2D grid
    ny, nx = data_array.shape[axes[0]], data_array.shape[axes[1]]
    lat_dim, lon_dim = "latitude", "longitude"

    # Earth radius in meters
    R = 6371000.0

    # Calculate grid spacing in meters from the data array's coordinates
    dlon_deg = np.abs(data_array[lon_dim].diff(dim=lon_dim).mean().item())
    dlat_deg = np.abs(data_array[lat_dim].diff(dim=lat_dim).mean().item())
    mean_lat_rad = np.deg2rad(data_array[lat_dim].mean().item())
    dx = R * np.cos(mean_lat_rad) * np.deg2rad(dlon_deg)
    dy = R * np.deg2rad(dlat_deg)

    # Calculate wavenumbers for the FFT
    kx = 2 * np.pi * fftfreq(nx, d=dx)
    ky = 2 * np.pi * fftfreq(ny, d=dy)
    kxx, kyy = np.meshgrid(kx, ky)
    k_magnitude = np.sqrt(kxx**2 + kyy**2)

    # Convert wavelength parameters from km to meters
    LP_m = low_pass_wl_km * 1000.0
    HP_m = high_pass_wl_km * 1000.0

    # Wavenumber corresponding to the start of the roll-off
    k_hp_m = (2 * np.pi) / HP_m

    # Calculate the 2D filter mask based on wavenumber
    exponent = -2 * ((LP_m / 2.0) * (k_magnitude - k_hp_m)) ** 2
    filter_mask = np.where(k_magnitude < k_hp_m, 1.0, np.exp(exponent))

    # Apply the filter using FFT along the specified axes
    nan_mask = data_array.isnull()
    filled_data = data_array.fillna(0).values

    # Perform FFT, apply the filter, and perform inverse FFT
    # The 2D filter_mask is automatically broadcast to the shape of data_fft
    data_fft = fft2(filled_data, axes=axes)
    filtered_fft = data_fft * filter_mask
    filtered_data_values = np.real(ifft2(filtered_fft, axes=axes))

    # Reconstruct the DataArray, preserving original metadata
    filtered_da = xr.DataArray(
        filtered_data_values,
        coords=data_array.coords,
        dims=data_array.dims,
        attrs=data_array.attrs,
    )
    # Re-apply the NaN mask to restore original missing values
    return filtered_da.where(~nan_mask)


def perform_scale_separation_on_slice(ds_slice: xr.Dataset) -> xr.Dataset:
    """
    Applies scale separation to a full dataset slice using a vectorized approach.
    Returns the mesoscale and synoptic datasets.
    """
    synoptic_ds_list = []
    vars_to_filter = [
        v for v, da in ds_slice.data_vars.items()
        if "latitude" in da.dims and "longitude" in da.dims
    ]

    for var_name in vars_to_filter:
        original_da = ds_slice[var_name]
        is_3d = "pressure_level" in original_da.dims

        # This vectorized approach handles both 2D and 3D DataArrays
        if original_da.ndim in [2, 3]:
            # Apply the filter across the last two dimensions (lat, lon)
            # This avoids the slow Python loop over pressure levels.
            synoptic_da = spectral_filter_2d(
                original_da, HIGH_PASS_WL_KM, LOW_PASS_WL_KM, axes=(-2, -1)
            )
            synoptic_ds_list.append(synoptic_da.rename(var_name))
        else:
            logging.info(
                f"Skipping filtering for variable '{var_name}' with unexpected dimensions: {original_da.dims}"
            )

    if not synoptic_ds_list:
        logging.warning(
            "No suitable variables were found/filtered for scale separation."
        )
        return xr.Dataset(attrs=ds_slice.attrs)

    # Create the synoptic dataset
    ds_synoptic = xr.merge(synoptic_ds_list)
    ds_synoptic.attrs = ds_slice.attrs.copy()
    ds_synoptic.attrs["processing_level"] = "synoptic_scale"
    ds_synoptic.attrs[
        "scale_separation_method"
    ] = f"2D Gaussian low-pass filter (HP: {HIGH_PASS_WL_KM} km, LP: {LOW_PASS_WL_KM} km)"

    ds_meso = ds_slice - ds_synoptic

    ds_meso.attrs = ds_slice.attrs.copy()
    ds_meso.attrs["processing_level"] = "mesoscale_perturbation"
    ds_meso.attrs[
        "scale_separation_method"
    ] = f"2D Gaussian filter (HP: {HIGH_PASS_WL_KM} km, LP: {LOW_PASS_WL_KM} km)"
    ds_meso.attrs[
        "note"
    ] = "Mesoscale field derived by filtering on the original lat/lon grid before centering."

    # Arithmetic can drop attributes. This loop copies them back from the original source.
    for var in ds_meso.data_vars:
        if var in ds_slice:
            ds_meso[var].attrs = ds_slice[var].attrs.copy()
    for var in ds_synoptic.data_vars:
        if var in ds_slice:
            ds_synoptic[var].attrs = ds_slice[var].attrs.copy()

    return ds_meso, ds_synoptic


def center_grid_projection_optimized(
    ds_event_source,
    event_center_lat,
    event_center_lon,
    target_x_km_coords,
    target_y_km_coords,
    source_lat_coord_name="latitude",
    source_lon_coord_name="longitude",
):
    """
    Optimized function to align and regrid a source dataset to a NORTH-UP
    grid centered on an event. It uses a direct one-step interpolation for
    maximum performance.
    """
    source_crs = pyproj.CRS("EPSG:4326")
    target_crs_aeqd = pyproj.CRS(
        f"+proj=aeqd +lat_0={event_center_lat} +lon_0={event_center_lon} +ellps=sphere +units=m"
    )
    transformer_target_to_source = pyproj.Transformer.from_crs(
        target_crs_aeqd, source_crs, always_xy=True
    )
    target_x_m_coords, target_y_m_coords = (
        target_x_km_coords * 1000.0,
        target_y_km_coords * 1000.0,
    )
    target_xx_m, target_yy_m = np.meshgrid(target_x_m_coords, target_y_m_coords)
    lons_to_sample, lats_to_sample = transformer_target_to_source.transform(
        target_xx_m, target_yy_m
    )
    lons_da = xr.DataArray(
        lons_to_sample,
        dims=("y_relative_km", "x_relative_km"),
        coords={
            "y_relative_km": target_y_km_coords,
            "x_relative_km": target_x_km_coords,
        },
    )
    lats_da = xr.DataArray(
        lats_to_sample,
        dims=("y_relative_km", "x_relative_km"),
        coords={
            "y_relative_km": target_y_km_coords,
            "x_relative_km": target_x_km_coords,
        },
    )
    ds_final_centered = ds_event_source.interp(
        {source_lon_coord_name: lons_da, source_lat_coord_name: lats_da},
        method="linear",
    )
    return ds_final_centered


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract ERA5 data for MCS events and perform scale separation."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=DEFAULT_CSV_PATH,
        help="Path to the MCS event index CSV file.",
    )
    parser.add_argument(
        "--era5_path",
        type=str,
        default=DEFAULT_ERA5_PATH,
        help="Base directory where merged ERA5 NetCDF files are stored.",
    )
    parser.add_argument(
        "--temp_dir",
        type=str,
        default="/home/dkn/mesocomposites/ERA5/temp_files/",
        help="Directory for temporary total-field event files.",
    )
    return parser.parse_args()


def reorder_lat(ds: xr.Dataset) -> xr.Dataset:
    """Ensure latitude is in ascending order."""
    if (
        "latitude" in ds.coords
        and ds["latitude"].values.size > 1
        and ds["latitude"].values[0] > ds["latitude"].values[-1]
    ):
        ds = ds.reindex({"latitude": list(reversed(ds["latitude"]))})
    return ds


def create_event_dataset(args):
    """Main function to process MCS events and extract data."""
    logging.info("Step 1: Loading and filtering MCS event data...")
    events_df = pd.read_csv(args.csv_path)
    events_df["datetime"] = pd.to_datetime(events_df["datetime"]).dt.round("H")
    events_df["year_month"] = events_df["datetime"].dt.strftime("%Y-%m")

    TEMP_DIR_BASE = Path(args.temp_dir)
    TEMP_DIR_SYNOPTIC = TEMP_DIR_BASE / "synoptic"
    TEMP_DIR_MESO = TEMP_DIR_BASE / "meso"
    TEMP_DIR_SYNOPTIC.mkdir(parents=True, exist_ok=True)
    TEMP_DIR_MESO.mkdir(parents=True, exist_ok=True)
    logging.info(f"Synoptic-field files will be stored in: {TEMP_DIR_SYNOPTIC}")
    logging.info(f"Mesoscale perturbation files will be stored in: {TEMP_DIR_MESO}")

    monthly_groups = events_df.groupby("year_month")

    events_processed_or_updated = 0
    events_skipped = 0

    logging.info(f"Step 3 & 4: Processing {len(monthly_groups)} months...")
    for year_month, month_group_df in tqdm(monthly_groups, desc="Total Progress"):
        era5_file = Path(args.era5_path) / f"{year_month}.nc"
        if not era5_file.exists():
            logging.warning(
                f"Merged file not found: {era5_file}. Skipping all {len(month_group_df)} events for month {year_month}."
            )
            events_skipped += len(month_group_df)
            continue

        ds_month = xr.open_dataset(
            era5_file, chunks={"latitude": 240, "longitude": 240}
        )
        ds_month = reorder_lat(ds_month)

        events_by_time = month_group_df.groupby("datetime")

        for event_time, time_group_df in tqdm(
            events_by_time, desc=f"Processing {year_month}", leave=False
        ):

            # Before loading any data, check if all required output files for this timestamp already exist.
            track_numbers_for_time = time_group_df["track_number"]
            all_files_exist = all(
                (
                    TEMP_DIR_SYNOPTIC / f"event_{tn}.nc"
                ).exists()  # Check for the synoptic file
                and (
                    TEMP_DIR_MESO / f"event_{tn}.nc"
                ).exists()  # Check for the meso file
                for tn in track_numbers_for_time
            )

            if all_files_exist:
                logging.debug(
                    f"All {len(track_numbers_for_time)} event files for {event_time} already exist. Skipping computation."
                )
                events_skipped += len(track_numbers_for_time)
                continue  # Proceed to the next event_time

            try:
                ds_time_slice = ds_month.sel(valid_time=event_time).load()
                logging.info(f"Performing scale separation for {event_time}...")

                ds_meso_slice, ds_synoptic_slice = perform_scale_separation_on_slice(
                    ds_time_slice
                )

                ds_meso_slice = assign_units_from_attrs(ds_meso_slice)
                ds_synoptic_slice = assign_units_from_attrs(ds_synoptic_slice)

                ds_total_slice = (
                    (ds_meso_slice + ds_synoptic_slice)
                    .metpy.assign_crs(
                        grid_mapping_name="latitude_longitude", earth_radius=6371229.0
                    )
                    .metpy.parse_cf()
                )
                for var in ds_total_slice.data_vars:  # Add units back to ds_total_slice
                    if var in ds_time_slice:
                        ds_total_slice[var].attrs = ds_time_slice[var].attrs.copy()
                
                ds_total_slice = assign_units_from_attrs(ds_total_slice)

                # Calculate equivilant potential temperature
                # STEP 1: Calculate dewpoint from pressure, temperature, and specific humidity.
                dewpoint_total = mpcalc.dewpoint_from_specific_humidity(
                    ds_total_slice['pressure_level'], ds_total_slice['t'], ds_total_slice['q']
                )

                # STEP 2: Now use the calculated dewpoint to get equivalent potential temperature.
                theta_e_total = mpcalc.equivalent_potential_temperature(
                    ds_total_slice['pressure_level'], ds_total_slice['t'], dewpoint_total
                )

                # Calculate relative humidity with monkey patches function, similar to ERA5
                rh_total = mpcalc.relative_humidity_from_specific_humidity(
                    ds_total_slice["pressure_level"],
                    ds_total_slice["t"],
                    ds_total_slice["q"],
                )

                # Add CRS to the mesoscale dataset before calculating vorticity
                ds_meso_slice = ds_meso_slice.metpy.assign_crs(
                    grid_mapping_name="latitude_longitude", earth_radius=6371229.0
                ).metpy.parse_cf()

                # Calculate relative vorticity from MESOSCALE winds
                vort_meso = mpcalc.vorticity(ds_meso_slice["u"], ds_meso_slice["v"])

                ds_meso_slice["theta_e"] = theta_e_total.assign_attrs(
                    units="K",
                    long_name="Equivalent Potential Temperature",
                    note="Calculated from total (synoptic + mesoscale) fields.",
                )
                ds_meso_slice["rh"] = rh_total.assign_attrs(
                    units="percent",
                    long_name="Relative Humidity",
                    note="Calculated from total (synoptic + mesoscale) fields.",
                )
                ds_meso_slice["rel_vort_meso"] = vort_meso.assign_attrs(
                    units="s**-1",
                    long_name="Mesoscale Relative Vorticity",
                    note="Calculated from mesoscale wind fields.",
                )

                for _, event in time_group_df.iterrows():
                    try:
                        track_number = event["track_number"]
                        # Define paths for the new synoptic and existing meso files
                        synoptic_temp_file_path = (
                            TEMP_DIR_SYNOPTIC / f"event_{track_number}.nc"
                        )
                        meso_temp_file_path = TEMP_DIR_MESO / f"event_{track_number}.nc"

                        # Check if both files already exist
                        if (
                            synoptic_temp_file_path.exists()
                            and meso_temp_file_path.exists()
                        ):
                            continue

                        events_processed_or_updated += 1

                        # --- 1. Center and save the SYNOPTIC field (if it doesn't exist) ---
                        if not synoptic_temp_file_path.exists():
                            centered_synoptic_ds = center_grid_projection_optimized(
                                ds_synoptic_slice,  # Use the new synoptic slice
                                event["center_lat"],
                                event["center_lon"],
                                RELATIVE_COORDS_KM,
                                RELATIVE_COORDS_KM,
                            )
                            temp_synoptic_ds = centered_synoptic_ds.expand_dims(
                                "track_number"
                            ).assign_coords(
                                {
                                    "track_number": ("track_number", [track_number]),
                                    "event_datetime": ("track_number", [event_time]),
                                    "event_center_lat": (
                                        "track_number",
                                        [event["center_lat"]],
                                    ),
                                    "event_center_lon": (
                                        "track_number",
                                        [event["center_lon"]],
                                    ),
                                }
                            )

                            if 'metpy_crs' in temp_synoptic_ds:
                                temp_synoptic_ds = temp_synoptic_ds.drop_vars('metpy_crs')
                            # Save the synoptic file
                            temp_synoptic_ds.to_netcdf(synoptic_temp_file_path)

                        # --- 2. Center and save the MESOSCALE field (if it doesn't exist) ---
                        if not meso_temp_file_path.exists():
                            centered_meso_ds = center_grid_projection_optimized(
                                ds_meso_slice,  # Use the meso slice
                                event["center_lat"],
                                event["center_lon"],
                                RELATIVE_COORDS_KM,
                                RELATIVE_COORDS_KM,
                            )
                            temp_meso_ds = centered_meso_ds.expand_dims(
                                "track_number"
                            ).assign_coords(
                                {
                                    "track_number": ("track_number", [track_number]),
                                    "event_datetime": ("track_number", [event_time]),
                                    "event_center_lat": (
                                        "track_number",
                                        [event["center_lat"]],
                                    ),
                                    "event_center_lon": (
                                        "track_number",
                                        [event["center_lon"]],
                                    ),
                                }
                            )

                            if 'metpy_crs' in temp_meso_ds:
                                temp_meso_ds = temp_meso_ds.drop_vars('metpy_crs')

                            encoding = {
                                var: {"zlib": True, "complevel": 5}
                                for var in temp_meso_ds.data_vars
                            }
                            # Save the mesoscale file
                            temp_meso_ds.to_netcdf(
                                meso_temp_file_path, encoding=encoding
                            )

                    except Exception as e_inner:
                        logging.warning(
                            f"Skipped saving temp file for event at {event_time} (Track: {event.get('track_number', 'N/A')}). Reason: {e_inner}"
                        )

            except KeyError:
                logging.warning(
                    f"Time coordinate 'valid_time={event_time}' not found in file for {year_month}. Skipping {len(time_group_df)} events."
                )
            except Exception as e_outer:
                logging.warning(
                    f"Could not process time slice {event_time}. Skipping {len(time_group_df)} events. Reason: {e_outer}"
                )

    # At the end of your script, update the summary message:
    logging.info("=" * 50)
    logging.info(f"Synoptic files are in: {TEMP_DIR_SYNOPTIC}")
    logging.info(f"Mesoscale files are in: {TEMP_DIR_MESO}")
    logging.info("=" * 50)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_arguments()
    create_event_dataset(args)
