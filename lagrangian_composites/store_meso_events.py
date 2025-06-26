"""
Script to extract motion-centered data cutouts for Mesoscale Convective System (MCS)
events from ERA5 data.

VERSION 7: Implements a highly optimized one-step interpolation for the
projection-based centering. This dramatically improves performance by avoiding
a costly intermediate regridding step.
"""
import pandas as pd
import xarray as xr
import numpy as np
import pyproj
import metpy.xarray
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
import time

# --- CONFIGURATION AND CONSTANTS ---
DEFAULT_CSV_PATH = "../csv/mcs_EUR_filtered_by_origin_and_region.csv"
DEFAULT_ERA5_PATH = "/reloclim/dkn/data/ERA5/pressure_levels/merged_files"
DEFAULT_OUTPUT_PATH = "/home/dkn/mesocomposites/ERA5/mcs_events_data_for_composites.nc"
MONTHS_TO_INCLUDE = [6, 7, 8]  # for now only use JJA TODO: discuss with Douglas

# Define the common grid based on the peer-reviewed script
BOX_SIZE_KM = 400 * 2 # Using a larger box (800km)
GRID_RESOLUTION_KM = 25
NUM_GRID_POINTS = int(BOX_SIZE_KM / GRID_RESOLUTION_KM) + 1
RELATIVE_COORDS_KM = np.linspace(-BOX_SIZE_KM / 2, BOX_SIZE_KM / 2, NUM_GRID_POINTS)


def center_grid_projection_optimized(ds_event_source, event_center_lat, event_center_lon,
                                     target_x_km_coords, target_y_km_coords,
                                     source_lat_coord_name='latitude',
                                     source_lon_coord_name='longitude'):
    """
    Optimized function to align and regrid a source dataset to a new NORTH-UP
    grid centered on an event. It uses a direct one-step interpolation for
    maximum performance.
    """
    # Define the source CRS (standard lat/lon)
    source_crs = pyproj.CRS("EPSG:4326") # WGS 84
    
    # Define the target CRS: an Azimuthal Equidistant projection centered on the event
    target_crs_aeqd = pyproj.CRS(f"+proj=aeqd +lat_0={event_center_lat} +lon_0={event_center_lon} +ellps=sphere +units=m")

    # Create a transformer to go from our target projection (in meters) back to the source lat/lon
    transformer_target_to_source = pyproj.Transformer.from_crs(target_crs_aeqd, source_crs, always_xy=True)

    # Create a meshgrid of our simple target coordinates (in meters)
    target_x_m_coords, target_y_m_coords = target_x_km_coords * 1000.0, target_y_km_coords * 1000.0
    target_xx_m, target_yy_m = np.meshgrid(target_x_m_coords, target_y_m_coords)

    # Use the transformer to find out where each point of our target grid falls
    # in the original latitude/longitude coordinate system.
    lons_to_sample, lats_to_sample = transformer_target_to_source.transform(target_xx_m, target_yy_m)

    # Create DataArrays for the sample points, which xarray's .interp() needs.
    # The dimensions and coordinates must match our final desired output grid.
    lons_da = xr.DataArray(
        lons_to_sample, dims=('y_relative_km', 'x_relative_km'),
        coords={'y_relative_km': target_y_km_coords, 'x_relative_km': target_x_km_coords}
    )
    lats_da = xr.DataArray(
        lats_to_sample, dims=('y_relative_km', 'x_relative_km'),
        coords={'y_relative_km': target_y_km_coords, 'x_relative_km': target_x_km_coords}
    )

    # Perform the single, direct interpolation from the source grid to our target grid.
    ds_final_centered = ds_event_source.interp(
        {source_lon_coord_name: lons_da, source_lat_coord_name: lats_da},
        method="linear"
    )
        
    return ds_final_centered


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Extract ERA5 data for MCS events from pre-merged files.")
    parser.add_argument('--csv_path', type=str, default=DEFAULT_CSV_PATH, help='Path to the MCS event index CSV file.')
    parser.add_argument('--era5_path', type=str, default=DEFAULT_ERA5_PATH, help='Base directory where merged ERA5 NetCDF files are stored.')
    parser.add_argument('--output_path', type=str, default=DEFAULT_OUTPUT_PATH, help='Path for the final output NetCDF file.')
    return parser.parse_args()

def reorder_lat(ds: xr.Dataset) -> xr.Dataset:
    """Ensure latitude is in ascending order."""
    lat_coord_name = 'latitude' if 'latitude' in ds.coords else 'lat'
    if lat_coord_name in ds.coords and ds[lat_coord_name].values.size > 1 and ds[lat_coord_name].values[0] > ds[lat_coord_name].values[-1]:
        ds = ds.reindex({lat_coord_name: list(reversed(ds[lat_coord_name]))})
    return ds

def create_event_dataset(args):
    """Main function to process MCS events and extract data."""
    logging.info("Step 1: Loading and filtering MCS event data...")
    events_df = pd.read_csv(args.csv_path)
    events_df['datetime'] = pd.to_datetime(events_df['datetime']).dt.round('H')
    
    initial_count = len(events_df)
    events_df = events_df[events_df['datetime'].dt.month.isin(MONTHS_TO_INCLUDE)].copy()
    if events_df.empty:
        logging.warning("No events found in the specified months. Exiting.")
        return
    
    events_df['year_month'] = events_df['datetime'].dt.strftime('%Y-%m')
    logging.info(f"Filtered events from {initial_count} to {len(events_df)} based on months: {MONTHS_TO_INCLUDE}")

    logging.info("Step 2: Grouping events by month for processing...")
    monthly_groups = events_df.groupby('year_month')
    all_centered_events = []
    all_event_metadata = []

    logging.info(f"Step 3 & 4: Processing {len(monthly_groups)} months...")
    for year_month, month_group_df in tqdm(monthly_groups, desc="Total Progress"):
        era5_file = Path(args.era5_path) / f"{year_month}.nc"
        if not era5_file.exists():
            logging.warning(f"Merged file not found: {era5_file}. Skipping all {len(month_group_df)} events for month {year_month}.")
            continue
        ds_month = xr.open_dataset(era5_file, chunks={'latitude': 240, 'longitude': 240})
        ds_month = reorder_lat(ds_month)
        
        events_by_time = month_group_df.groupby('datetime')
        
        for event_time, time_group_df in tqdm(events_by_time, desc=f"Processing {year_month}", leave=False):
            try:
                ds_time_slice = ds_month.sel(valid_time=event_time).load()
                
                for _, event in time_group_df.iterrows():
                    try:
                        # Call the new, highly optimized centering function
                        centered_ds = center_grid_projection_optimized(
                            ds_time_slice,
                            event['center_lat'],
                            event['center_lon'],
                            RELATIVE_COORDS_KM,
                            RELATIVE_COORDS_KM,
                        )

                        all_centered_events.append(centered_ds)
                        all_event_metadata.append({
                            'event_datetime': event_time, 'track_number': event['track_number'],
                            'event_center_lat': event['center_lat'], 'event_center_lon': event['center_lon'],
                        })
                    except Exception as e_inner:
                        logging.warning(f"Skipped centering for event at {event_time}. Reason: {e_inner}")

            except Exception as e_outer:
                logging.warning(f"Could not load data for time slice {event_time}. Skipping {len(time_group_df)} events. Reason: {e_outer}")
    
    if not all_centered_events:
        logging.warning("No events were successfully processed. No output file will be created.")
    else:
        logging.info("Step 5: Combining all processed events into a single dataset...")
        final_ds = xr.concat(all_centered_events, dim='event_index')
        meta_df = pd.DataFrame(all_event_metadata)
        final_ds = final_ds.assign_coords({
            'event_index': ('event_index', np.arange(len(all_centered_events))),
            'track_number': ('event_index', meta_df['track_number'].values),
            'event_datetime': ('event_index', meta_df['event_datetime'].values),
            'event_center_lat': ('event_index', meta_df['event_center_lat'].values),
            'event_center_lon': ('event_index', meta_df['event_center_lon'].values),
        })
        logging.info(f"Final dataset created with {len(final_ds.event_index)} events.")
        
        # Remove the problematic metpy_crs coordinate before saving if it exists
        if 'metpy_crs' in final_ds.coords:
            final_ds = final_ds.drop_vars('metpy_crs', errors='ignore')
        
        encoding = {var: {'zlib': True, 'complevel': 5} for var in final_ds.data_vars}
        logging.info(f"Saving final dataset to: {args.output_path}")
        final_ds.to_netcdf(args.output_path, encoding=encoding, mode='w')
        logging.info("✅ Success! Output file saved.")

    total_attempted = len(events_df)
    total_processed = len(all_centered_events)
    total_skipped = total_attempted - total_processed
    logging.info("\n" + "="*50 + "\nProcessing Summary\n" + "="*50)
    logging.info(f"Total events attempted: {total_attempted}")
    logging.info(f"Successfully processed and saved: {total_processed}")
    logging.info(f"Skipped due to errors: {total_skipped}")
    logging.info("="*50)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = parse_arguments()
    create_event_dataset(args)