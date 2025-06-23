import pandas as pd
import xarray as xr
import numpy as np
import math
import pyproj
import argparse
import yaml
import metpy
import os

# --- Configuration Constants (from original script) ---
# These define the target grid for the composite.
BOX_SIZE_KM = 400
GRID_RESOLUTION_KM = 25
NUM_GRID_POINTS = int(BOX_SIZE_KM / GRID_RESOLUTION_KM) + 1
RELATIVE_COORDS_KM = np.linspace(-BOX_SIZE_KM / 2, BOX_SIZE_KM / 2, NUM_GRID_POINTS)

# --- Projection-based Alignment Function (EXACTLY as in mesocomp_surface_precipitation.py) ---
# This function is preserved to ensure methodological consistency.
def align_grid_projection(ds_event_source, event_center_lat, event_center_lon,
                          event_motion_angle_deg,
                          target_x_km_coords, target_y_km_coords,
                          crs_proj,
                          source_lat_coord_name='latitude',
                          source_lon_coord_name='longitude'):
    """
    Aligns and regrids a source dataset to a new grid centered on an event,
    rotated according to the event's motion.
    
    This implementation is identical to the one used in peer-reviewed analysis
    to ensure consistency.
    """
    current_alignment_angle_deg = event_motion_angle_deg if not pd.isna(event_motion_angle_deg) else 0.0

    # Assign CRS using MetPy based on the projection info.
    if crs_proj == 'latitude_longitude':
        ds_event_source = ds_event_source.metpy.assign_crs(grid_mapping_name=crs_proj)
    elif crs_proj == 'rotated_latitude_longitude':
        if 'rotated_pole' in list(ds_event_source.data_vars.keys()):
            ds_event_source = ds_event_source.metpy.assign_crs(
                grid_mapping_name=ds_event_source.rotated_pole.grid_mapping_name,
                grid_north_pole_latitude=ds_event_source.rotated_pole.grid_north_pole_latitude,
                grid_north_pole_longitude=ds_event_source.rotated_pole.grid_north_pole_longitude
            )
        else:
            ds_event_source = ds_event_source.metpy.assign_crs(
                grid_mapping_name='rotated_latitude_longitude',
                grid_north_pole_latitude=39.25,
                grid_north_pole_longitude=-162.
            )
    
    # Robustly get the source CRS from the first data variable.
    first_var = list(ds_event_source.data_vars.keys())[0]
    source_crs = ds_event_source[first_var].metpy.pyproj_crs

    # Step 1: Create an intermediate Azimuthal Equidistant (AEQD) projection centered on the event.
    crs_aeqd_event_north_up = pyproj.CRS(
        f"+proj=aeqd +lat_0={event_center_lat} +lon_0={event_center_lon} +ellps=sphere +units=m"
    )
    transformer_aeqd_to_source = pyproj.Transformer.from_crs(
        crs_aeqd_event_north_up, source_crs, always_xy=True
    )

    # Define a large intermediate grid in meters.
    interm_y_m = np.arange(-1000.e3, 1000.1e3, 5.e3)
    interm_x_m = np.arange(-1000.e3, 1000.1e3, 5.e3)
    interm_xx_m, interm_yy_m = np.meshgrid(interm_x_m, interm_y_m)

    # Transform the intermediate grid points back to the source CRS (lat/lon).
    interm_lons_src_crs, interm_lats_src_crs = transformer_aeqd_to_source.transform(interm_xx_m, interm_yy_m)

    lons_for_interm_interp_da = xr.DataArray(
        interm_lons_src_crs, dims=('y_interm', 'x_interm'),
        coords={'y_interm': interm_y_m, 'x_interm': interm_x_m}
    )
    lats_for_interm_interp_da = xr.DataArray(
        interm_lats_src_crs, dims=('y_interm', 'x_interm'),
        coords={'y_interm': interm_y_m, 'x_interm': interm_x_m}
    )

    print(f"    Align Step 1: Interpolating source data to event-centered AEQD grid...")
    # Interpolate the source data onto this intermediate AEQD grid.
    ds_on_aeqd_north_up = ds_event_source.interp(
        {source_lon_coord_name: lons_for_interm_interp_da, source_lat_coord_name: lats_for_interm_interp_da},
        method="linear"
    )

    # Step 2: Define the final target grid and rotate it according to motion angle.
    target_x_m_coords = target_x_km_coords * 1000.0
    target_y_m_coords = target_y_km_coords * 1000.0
    target_xx_m, target_yy_m = np.meshgrid(target_x_m_coords, target_y_m_coords)

    alpha_rad = math.radians(current_alignment_angle_deg)
    
    # Apply rotation matrix to the target grid coordinates.
    x_sample_points_on_aeqd_north_up = target_xx_m * math.cos(alpha_rad) + target_yy_m * math.sin(alpha_rad)
    y_sample_points_on_aeqd_north_up = -target_xx_m * math.sin(alpha_rad) + target_yy_m * math.cos(alpha_rad)

    print(f"    Align Step 2: Interpolating from AEQD to final motion-aligned grid...")
    # Interpolate from the intermediate AEQD grid to the final rotated target grid.
    ds_final_aligned = ds_on_aeqd_north_up.interp(
        x_interm=xr.DataArray(x_sample_points_on_aeqd_north_up,
                              dims=('y_relative', 'x_relative'),
                              coords={'y_relative': target_y_km_coords, 'x_relative': target_x_km_coords}),
        y_interm=xr.DataArray(y_sample_points_on_aeqd_north_up,
                              dims=('y_relative', 'x_relative'),
                              coords={'y_relative': target_y_km_coords, 'x_relative': target_x_km_coords}),
        method="linear"
    )
    
    # Clean up intermediate coordinates that may have been carried over.
    coords_to_drop = [c for c in ['x_interm', 'y_interm'] if c in ds_final_aligned.coords]
    if coords_to_drop:
        ds_final_aligned = ds_final_aligned.drop_vars(coords_to_drop, errors='ignore')

    return ds_final_aligned

# --- Main Script ---
def main():
    parser = argparse.ArgumentParser(description="Generate mesoscale composites from pre-calculated event data.")
    parser.add_argument("--region", required=True, help="Region specifier (e.g., Alps, Balcan), used for input/output file naming.")
    parser.add_argument("--input_dir", default="/home/dkn/composites/ERA5", help="Directory containing the pre-calculated event NetCDF files.")
    parser.add_argument("--csv_dir", default="./csv", help="Directory containing the MCS initiation CSV files.")
    parser.add_argument("--output_dir", default="/home/dkn/composites/ERA5/composites", help="Directory to save the final composite NetCDF files.")
    args = parser.parse_args()

    # --- File Path Construction ---
    csv_file = os.path.join(args.csv_dir, f"mcs_initiation_dates_angles_{args.region}.csv")
    input_nc_file = os.path.join(args.input_dir, f"events_dynamic_meso_{args.region}_historical.nc")
    output_composite_file = os.path.join(args.output_dir, f"mesocomp_dynamic_{args.region}.nc")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"--- Starting Composite Generation for Region: {args.region} ---")

    # --- Load Input Data ---
    if not os.path.exists(csv_file):
        print(f"ERROR: MCS metadata file not found at: {csv_file}")
        return
    print(f"Loading MCS initiation data from {csv_file}...")
    mcs_df = pd.read_csv(csv_file)
    mcs_df['datetime'] = pd.to_datetime(mcs_df['datetime'])

    if not os.path.exists(input_nc_file):
        print(f"ERROR: Pre-calculated event data not found at: {input_nc_file}")
        return
    print(f"Loading pre-calculated event data from {input_nc_file}...")
    ds_full_raw = xr.open_dataset(input_nc_file)
    breakpoint()
    # --- Create a unique index to work around duplicate timestamps ---
    # 1. Check if the number of events in the CSV matches the number of time slices in the NetCDF
    if len(mcs_df) != len(ds_full_raw.time):
        print(f"FATAL ERROR: Mismatch in event counts.")
        print(f"  CSV file '{csv_file}' has {len(mcs_df)} events.")
        print(f"  NetCDF file '{input_nc_file}' has {len(ds_full_raw.time)} time slices.")
        print("  The script cannot proceed because the order-based matching assumption is invalid.")
        return

    # 2. Rename the problematic 'time' dimension to a generic 'event_index'
    ds_full = ds_full_raw.rename({'time': 'event_index'})
    
    # 3. Assign a new, unique coordinate for this dimension (0, 1, 2, ...)
    ds_full = ds_full.assign_coords(event_index=np.arange(len(ds_full.event_index)))
    
    # 4. Re-attach the original time values as a non-indexing coordinate for reference
    ds_full = ds_full.assign_coords(time=('event_index', ds_full_raw.time.values))
    print("Successfully re-indexed dataset to use unique 'event_index' for selection.")
    
    all_aligned_events = []
    
    print(f"\nProcessing {len(mcs_df)} events for region '{args.region}'...")
    # --- Main Event Loop ---
    # Use .iterrows() to get the integer index 'idx' which corresponds to our new 'event_index'
    for idx, event in mcs_df.iterrows():
        event_time = event['datetime']
        center_lat = event['center_lat']
        center_lon = event['center_lon']
        track_number = event['track_number']
        event_angle_degrees = event['angle_of_movement']

        print(f"\nProcessing Event: Index #{idx}, Track #{track_number}, Time {event_time}")

        current_alignment_angle = event_angle_degrees if not pd.isna(event_angle_degrees) else 0.0
        if pd.isna(event_angle_degrees):
             print(f"    Angle is NaN. Aligning North-up relative to event center.")
        else:
            print(f"    Aligning with motion angle {current_alignment_angle:.2f} degrees.")
        
        # Select the event data using its unique integer position (index).
        # If this fails (e.g., index out of bounds), the script will crash as requested.
        ds_event_source = ds_full.isel(event_index=idx)

        # The align_grid_projection function will process all data variables in the dataset.
        aligned_ds = align_grid_projection(
            ds_event_source,
            center_lat,
            center_lon,
            current_alignment_angle,
            RELATIVE_COORDS_KM,
            RELATIVE_COORDS_KM,
            crs_proj='latitude_longitude', # The CRS of our pre-calculated files
            source_lat_coord_name='latitude',
            source_lon_coord_name='longitude'
        )

        if aligned_ds:
            # Add a coordinate to identify the event after concatenation
            aligned_ds = aligned_ds.assign_coords(event_track_number=track_number)
            all_aligned_events.append(aligned_ds)
            print(f"    --> Successfully processed and aligned event {track_number}.")
        else:
            # This case might indicate an issue in the align_grid_projection function
            # where it returns None or an empty dataset. The script will likely
            # crash later if all events result in this.
            print(f"    WARNING: Data alignment failed for event {track_number}.")

    # --- Aggregating and Saving Results ---
    print("\n-------------------------------------------------")
    if not all_aligned_events:
        print("No events were successfully processed. No output file will be created.")
        print("Script finished.")
        return
        
    print(f"Aggregating {len(all_aligned_events)} successfully processed events...")
    
    # Use the track number as the value for the new 'event' dimension
    event_dim_coord_val = [ds.event_track_number.item() for ds in all_aligned_events]
    event_dim_index = pd.Index(event_dim_coord_val, name="event")
    
    # Drop the temporary coordinate before concatenation
    snapshots_to_concat = [ds.drop_vars("event_track_number", errors='ignore') for ds in all_aligned_events]
    
    # Concatenate all individual aligned events into a single DataArray along the new 'event' dimension
    combined_ds = xr.concat(snapshots_to_concat, dim=event_dim_index)

    # Rename the relative coordinates for clarity in the final output
    combined_ds = combined_ds.rename({
        'y_relative': 'distance_along_motion_km',
        'x_relative': 'distance_across_motion_km'
    })

    # Calculate the composite by taking the mean across all events
    composite_ds = combined_ds.mean(dim='event')
    
    # Clean up metadata before saving
    if 'metpy_crs' in composite_ds.coords:
         composite_ds = composite_ds.drop_vars('metpy_crs', errors='ignore')

    # Save the final composite to a NetCDF file.
    # If this fails (e.g., permissions, disk space), the script will crash as requested.
    composite_ds.to_netcdf(output_composite_file)
    print(f"\nSUCCESS: Composite for region '{args.region}' saved to:")
    print(f"  {output_composite_file}")

    # --- Optional: Save all aligned events before averaging ---
    # Uncomment the following lines if you want a file with all individual aligned events,
    # which is useful for variance analysis.
    # all_events_output_file = os.path.join(args.output_dir, f"all_events_aligned_dynamic_{args.region}.nc")
    # combined_ds.to_netcdf(all_events_output_file)
    # print(f"\nAll aligned events saved to: {all_events_output_file}")
    # -----------------------------------------------------------

    print("\nScript finished.")

if __name__ == "__main__":
    main()