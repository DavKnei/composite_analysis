import pandas as pd
import numpy as np
import math
import argparse
import yaml # For reading the regions YAML file (install with: pip install PyYAML)
from pathlib import Path # For easier path manipulation

# --- Configuration Constants ---
KM_PER_DEG_LAT = 111.0
KM_PER_DEG_LON_EQUATOR = 111.320
ZERO_MOVEMENT_PLACEHOLDER = -999.0  # Special value for atan2(0,0) cases before filling

# --- Helper Function: get_displacement_km ---
def get_displacement_km(p_start, p_end):
    """Calculates the cartesian displacement in kilometers between two lat/lon points."""
    lat_s, lon_s = p_start['center_lat'], p_start['center_lon']
    lat_e, lon_e = p_end['center_lat'], p_end['center_lon']
    delta_lat = lat_e - lat_s
    delta_lon = lon_e - lon_s
    avg_lat = (lat_s + lat_e) / 2.0
    dy_km = delta_lat * KM_PER_DEG_LAT
    dx_km = delta_lon * KM_PER_DEG_LON_EQUATOR * math.cos(math.radians(avg_lat))
    return dx_km, dy_km

# --- Function to get MCS initiation times ---
def extract_mcs_initiation_data(input_csv_with_angles_path, output_csv_initiation_path):
    """
    Reads a CSV of track data, extracts the first chronological entry for each track,
    and saves it to a new CSV file.
    """
    print(f"  Extracting initiation data from: {Path(input_csv_with_angles_path).name}")
    df_angles = pd.read_csv(input_csv_with_angles_path, parse_dates=["datetime"])

    if 'angle_of_movement' not in df_angles.columns:
        print(f"  Warning: 'angle_of_movement' column not found. Adding NaN column.")
        df_angles['angle_of_movement'] = np.nan
        
    initiation_df = df_angles.sort_values(by=['track_number', 'datetime']).groupby("track_number", as_index=False).first()
    initiation_df = initiation_df.sort_values("datetime")
    
    Path(output_csv_initiation_path).parent.mkdir(parents=True, exist_ok=True)
    initiation_df.to_csv(output_csv_initiation_path, index=False, float_format='%.3f')
    print(f"  Saved {len(initiation_df)} MCS initiation entries to {Path(output_csv_initiation_path).name}")
    return initiation_df

# --- Main Angle Calculation Function ---
def main():
    """
    Main function to load MCS data, calculate movement angles, and then filter/save
    the data for tracks that initiate within specified regions and timeframes.
    """
    parser = argparse.ArgumentParser(description="Calculate MCS movement angles and filter tracks by initiation location and time.")
    parser.add_argument("--input_csv", default="../csv/mcs_EUR_index.csv", help="Input CSV file with raw MCS track data.")
    parser.add_argument("--output_dir", default="./csv", help="Output directory for all processed regional CSV files.")
    parser.add_argument("--regions_file", default="../regions.yaml", help="YAML file defining all geographic regions to be processed.")
    args = parser.parse_args()

    # 1. Load Full Track Data
    print(f"Loading full track data from {args.input_csv}...")
    try:
        df_full = pd.read_csv(args.input_csv)
    except FileNotFoundError:
        print(f"ERROR: Input CSV file '{args.input_csv}' not found.")
        return
        
    df_full['datetime'] = pd.to_datetime(df_full['datetime'])
    df_full['angle_of_movement'] = np.nan

    # Filter to remove single-point tracks before any processing.
    print("Checking for and removing single-point tracks...")
    track_counts = df_full['track_number'].value_counts()
    single_point_tracks = track_counts[track_counts == 1].index
    
    if not single_point_tracks.empty:
        print(f"  Identified {len(single_point_tracks)} single-point tracks that will be removed.")
        df_full = df_full[~df_full['track_number'].isin(single_point_tracks)].copy()
        print(f"  Dataframe size after removal: {len(df_full)} rows.")
    else:
        print("  No single-point tracks found.")

    # 2. Calculate Angles for ALL remaining valid tracks
    grouped_tracks = df_full.groupby('track_number')
    processed_tracks_count = 0
    print(f"Processing {len(grouped_tracks)} total multi-point track IDs...")

    all_calculated_angles_series_for_df_full = []

    for track_id, group_df in grouped_tracks:
        track_segment_df = group_df.sort_values(by='datetime').reset_index(drop=True)
        L = len(track_segment_df)
        current_segment_angles_values = [np.nan] * L 

        if L >= 2:
            for i in range(L):
                avg_dx_km, avg_dy_km = 0.0, 0.0
                if i == 0: 
                    dx1, dy1 = get_displacement_km(track_segment_df.iloc[0], track_segment_df.iloc[1])
                    if L >= 3: 
                        dx2, dy2 = get_displacement_km(track_segment_df.iloc[1], track_segment_df.iloc[2])
                        avg_dx_km = (dx1 + dx2) / 2.0
                        avg_dy_km = (dy1 + dy2) / 2.0
                    else:
                        avg_dx_km, avg_dy_km = dx1, dy1
                elif i == L - 1:
                    dx2, dy2 = get_displacement_km(track_segment_df.iloc[L-2], track_segment_df.iloc[L-1])
                    if L >= 3:
                        dx1, dy1 = get_displacement_km(track_segment_df.iloc[L-3], track_segment_df.iloc[L-2])
                        avg_dx_km = (dx1 + dx2) / 2.0
                        avg_dy_km = (dy1 + dy2) / 2.0
                    else:
                        avg_dx_km, avg_dy_km = dx2, dy2
                else:
                    dx1, dy1 = get_displacement_km(track_segment_df.iloc[i-1], track_segment_df.iloc[i])
                    dx2, dy2 = get_displacement_km(track_segment_df.iloc[i], track_segment_df.iloc[i+1])
                    avg_dx_km = (dx1 + dx2) / 2.0
                    avg_dy_km = (dy1 + dy2) / 2.0

                if avg_dx_km == 0.0 and avg_dy_km == 0.0:
                    current_segment_angles_values[i] = ZERO_MOVEMENT_PLACEHOLDER
                else:
                    angle_rad = math.atan2(avg_dx_km, avg_dy_km)
                    angle_deg = math.degrees(angle_rad)
                    current_segment_angles_values[i] = (angle_deg + 360) % 360
            
            for j in range(L - 2, -1, -1): 
                if current_segment_angles_values[j] == ZERO_MOVEMENT_PLACEHOLDER:
                    next_angle = current_segment_angles_values[j+1]
                    use_prev = False
                    if j > 0:
                        prev_angle_val = current_segment_angles_values[j-1]
                        if prev_angle_val != ZERO_MOVEMENT_PLACEHOLDER and not pd.isna(prev_angle_val):
                            use_prev = True
                            prev_angle = prev_angle_val
                    
                    if next_angle != ZERO_MOVEMENT_PLACEHOLDER and not pd.isna(next_angle):
                        current_segment_angles_values[j] = next_angle
                    elif use_prev:
                        current_segment_angles_values[j] = prev_angle 
                    else: 
                        current_segment_angles_values[j] = -1 
            
            temp_final_angles = []
            for idx_k, angle_val in enumerate(current_segment_angles_values):
                if angle_val == ZERO_MOVEMENT_PLACEHOLDER or angle_val == -1:
                    temp_final_angles.append(0.0)
                else:
                    temp_final_angles.append(angle_val)
            current_segment_angles_values = temp_final_angles

        original_indices = group_df.sort_values(by='datetime').index
        all_calculated_angles_series_for_df_full.append(pd.Series(current_segment_angles_values, index=original_indices))
        
        processed_tracks_count += 1
        if processed_tracks_count % 500 == 0:
            print(f"  Calculated angles for {processed_tracks_count} tracks...")

    print(f"Finished angle calculation for all {processed_tracks_count} tracks.")

    if all_calculated_angles_series_for_df_full:
        for series_with_orig_indices in all_calculated_angles_series_for_df_full:
            df_full['angle_of_movement'].update(series_with_orig_indices)

    # 3. Post-calculation: Filter by initiation location and save files
    print("\n--- Filtering tracks based on initiation location ---")
    
    # Step 1: Find the initiation (first chronological point) for EVERY track.
    print("Finding all track initiation points...")
    df_full_sorted = df_full.sort_values(['track_number', 'datetime'])
    df_initiations = df_full_sorted.groupby('track_number', as_index=False).first()
    print(f"  Found {len(df_initiations)} unique tracks in the original dataset.")

    # Step 2: From the initiation points, keep only those that occurred in MJJAS.
    df_initiations_mjas = df_initiations[df_initiations['datetime'].dt.month.isin([5, 6, 7, 8, 9])].copy()
    print(f"  Found {len(df_initiations_mjas)} tracks that initiated in MJJAS.")

    # Load region definitions
    try:
        with open(args.regions_file, 'r') as f:
            regions_data = yaml.safe_load(f)
        print(f"Loaded {len(regions_data)} regions from {args.regions_file}")
    except FileNotFoundError:
        print(f"ERROR: Regions file '{args.regions_file}' not found. Cannot create regional files.")
        return

    # Loop through each defined region to perform the filtering
    for region_name, region_bounds in regions_data.items():
        print(f"\nProcessing region: {region_name}")

        # Step 3: Filter the MJJAS initiations to find which ones are inside THIS region's bounds.
        region_initiations = df_initiations_mjas[
            (df_initiations_mjas['center_lon'] >= region_bounds['lon_min']) &
            (df_initiations_mjas['center_lon'] <= region_bounds['lon_max']) &
            (df_initiations_mjas['center_lat'] >= region_bounds['lat_min']) &
            (df_initiations_mjas['center_lat'] <= region_bounds['lat_max'])
        ]

        # Step 4: Get the unique track numbers that initiated in this region.
        tracks_initiated_in_region = region_initiations['track_number'].unique()

        if len(tracks_initiated_in_region) == 0:
            print(f"  No tracks found that initiated in '{region_name}' during MJJAS. Skipping.")
            continue
        
        print(f"  Found {len(tracks_initiated_in_region)} tracks that initiated in this region during MJJAS.")

        # Step 5: Go back to the original df_full and get the ENTIRE history for these valid tracks.
        df_region_output = df_full[df_full['track_number'].isin(tracks_initiated_in_region)].copy()
        
        # Sort the final output for consistency
        df_region_output = df_region_output.sort_values(by=['track_number', 'datetime'])
        
        # Define output paths for this specific region
        output_csv_all_steps_path = Path(args.output_dir) / f"mcs_index_with_angles_{region_name}.csv"
        output_csv_initiation_path = Path(args.output_dir) / f"mcs_initiation_dates_angles_{region_name}.csv"

        # Save the filtered data (full track histories) for the region
        Path(output_csv_all_steps_path).parent.mkdir(parents=True, exist_ok=True)
        df_region_output.to_csv(output_csv_all_steps_path, index=False, float_format='%.3f')
        print(f"  Saved {len(df_region_output)} total track points to {Path(output_csv_all_steps_path).name}")

        # This function is now correctly used, as the input CSV contains only tracks that initiated in the region.
        extract_mcs_initiation_data(output_csv_all_steps_path, output_csv_initiation_path)

    print("\nScript finished successfully.")


if __name__ == "__main__":
    main()