import pandas as pd
import xarray as xr
import numpy as np
import math
import argparse
import os
from pathlib import Path
from tqdm import tqdm

# --- Configuration Constants ---
BOX_SIZE_KM = 400
GRID_RESOLUTION_KM = 25
NUM_GRID_POINTS = int(BOX_SIZE_KM / GRID_RESOLUTION_KM) + 1
RELATIVE_COORDS_KM = np.linspace(-BOX_SIZE_KM / 2, BOX_SIZE_KM / 2, NUM_GRID_POINTS)

def rotate_centered_grid(ds_centered_source, event_motion_angle_deg,
                         target_x_km_coords, target_y_km_coords):
    """
    Rotates an already centered dataset according to the event's motion.
    (This function is correct and remains unchanged)
    """
    current_alignment_angle_deg = event_motion_angle_deg if not pd.isna(event_motion_angle_deg) else 0.0

    ds_centered_source = ds_centered_source.rename({
        'x_relative_km': 'x_interm',
        'y_relative_km': 'y_interm'
    })
    
    target_x_m_coords = target_x_km_coords * 1000.0
    target_y_m_coords = target_y_km_coords * 1000.0
    target_xx_m, target_yy_m = np.meshgrid(target_x_m_coords, target_y_m_coords)

    alpha_rad = math.radians(current_alignment_angle_deg)
    
    x_sample_points = target_xx_m * math.cos(alpha_rad) + target_yy_m * math.sin(alpha_rad)
    y_sample_points = -target_xx_m * math.sin(alpha_rad) + target_yy_m * math.cos(alpha_rad)

    x_sample_points_km = x_sample_points / 1000.0
    y_sample_points_km = y_sample_points / 1000.0
    
    ds_final_rotated = ds_centered_source.interp(
        x_interm=xr.DataArray(x_sample_points_km,
                              dims=('y_relative', 'x_relative'),
                              coords={'y_relative': target_y_km_coords, 'x_relative': target_x_km_coords}),
        y_interm=xr.DataArray(y_sample_points_km,
                              dims=('y_relative', 'x_relative'),
                              coords={'y_relative': target_y_km_coords, 'x_relative': target_x_km_coords}),
        method="linear"
    )
    
    return ds_final_rotated

def main():
    parser = argparse.ArgumentParser(description="Generate mesoscale composites from pre-centered event data.")
    parser.add_argument("--csv_path", default="./csv/meso_composite_events.csv", help="Path to the MCS events CSV file containing angles of movement.")
    parser.add_argument("--temp_dir", default="/home/dkn/mesocomposites/ERA5/temp_files/", help="Directory containing the pre-centered (non-rotated) event NetCDF files.")
    parser.add_argument("--temp_dir_rotated", default="/home/dkn/mesocomposites/ERA5/temp_files_rotated/", help="Directory to save the rotated temporary event files.")
    parser.add_argument("--output_dir", default="/home/dkn/mesocomposites/ERA5/", help="Directory to save the final composite NetCDF files.")
    args = parser.parse_args()

   # --- Setup Directories and Load Angle Data ---
    input_dir = Path(args.temp_dir)
    rotated_temp_dir = Path(args.temp_dir_rotated)
    output_dir = Path(args.output_dir)
    rotated_temp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_rotated_events_file = output_dir / "meso_rotated_events.nc"
    output_composite_file = output_dir / "mesocomp_thermodynamic_composite.nc"

    print("--- Starting Rotation and Composite Generation ---")
    
    try:
        print(f"Loading angles from {args.csv_path}...")
        # Load the CSV and set track_number as the index for fast lookups
        events_df = pd.read_csv(args.csv_path).set_index('track_number')
    except (FileNotFoundError, KeyError) as e:
        print(f"ERROR: Could not load or parse the required CSV file. Please check the path and format. Details: {e}")
        return

    # --- Main Processing Loop ---
    event_files = sorted(list(input_dir.glob("event_*.nc")))
    if not event_files:
        print(f"WARNING: No event files found in {input_dir}. Exiting.")
        return

    print(f"Found {len(event_files)} event files to process.")
    pbar = tqdm(event_files, desc="Rotating Events")
    for event_file_path in pbar:
        try:
            track_number = int(event_file_path.stem.split('_')[-1])
            rotated_file_path = rotated_temp_dir / f"{event_file_path.stem}_rotated.nc"

            if rotated_file_path.exists():
                continue

            # Look up the angle from the DataFrame
            event_angle_degrees = events_df.loc[track_number, 'angle_of_movement']

            with xr.open_dataset(event_file_path) as ds_event_source:
                rotated_ds = rotate_centered_grid(
                    ds_event_source,
                    event_angle_degrees,
                    RELATIVE_COORDS_KM,
                    RELATIVE_COORDS_KM
                )
                if rotated_ds:
                    # Add metadata to the rotated file
                    rotated_ds = rotated_ds.assign_coords(
                        event_datetime=ds_event_source.event_datetime,
                        angle_of_movement=event_angle_degrees
                    )
                    rotated_ds.to_netcdf(rotated_file_path)

        except KeyError:
            print(f"\nWARNING: Could not find angle for Track #{track_number} in the CSV file. Skipping.")
            continue
        except Exception as e:
            print(f"\nERROR: Failed to process {event_file_path.name}. Reason: {e}")

    # --- Final Aggregation Step ---
    print("\n-------------------------------------------------")
    rotated_files_list = sorted(list(rotated_temp_dir.glob("*_rotated.nc")))
    if not rotated_files_list:
        print("No events were successfully rotated. No output file will be created.")
        return
        
    print(f"Aggregating {len(rotated_files_list)} successfully rotated events...")
    
    combined_ds = xr.open_mfdataset(
        rotated_files_list,
        concat_dim="track_number",
        combine="nested"
    ).chunk({'track_number': 100})

    combined_ds = combined_ds.rename({
        'y_relative': 'distance_along_motion_km',
        'x_relative': 'distance_across_motion_km'
    })

    print(f"\nCalculating the final composite (mean of all events: n={len(combined_ds.track_number)})...")
    composite_ds = combined_ds.mean(dim='track_number', keep_attrs=True)
    
    composite_ds.to_netcdf(output_composite_file)
    print(f"\nSUCCESS: Final composite saved to:\n  {output_composite_file}")
    
    print("\nScript finished.")

if __name__ == "__main__":
    main()