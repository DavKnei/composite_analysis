"""
Pre-processing script to filter an MCS events CSV file based on geographic regions.

This script reads a main CSV file of MCS events and a YAML file defining
named geographical bounding boxes. It performs a two-step filtering process:

1.  It identifies all MCS tracks that **originate** (have their first recorded point)
    within any of the defined subregions.
2.  From that set of tracks, it keeps only the subsequent grid points that also
    fall within any of the defined regions.

The output is a new, smaller CSV file intended to be used as a more efficient
input for subsequent data processing scripts.

Example Usage:
# Run with default paths
python 02a_filter_get_mcs_dates.py

# Specify custom paths
python 02a_filter_get_mcs_dates.py \
    --input_csv /path/to/mcs_EUR_index.csv \
    --regions_yaml /path/to/my_regions.yaml \
    --output_csv /path/to/filtered_events.csv
"""
import pandas as pd
import yaml
import argparse
import logging
from pathlib import Path

# --- CONFIGURATION DEFAULTS ---
DEFAULT_INPUT_CSV = "./csv/mcs_EUR_index.csv"
DEFAULT_REGIONS_YAML = "./regions.yaml"
# Updated the default output filename to reflect the new filtering logic
DEFAULT_OUTPUT_CSV = "./csv/mcs_EUR_filtered_by_origin_and_region.csv"

# --- COLUMN NAME CONFIGURATION ---
# Names of the relevant columns in the input CSV
TRACK_ID_COL = 'track_number'
DATETIME_COL = 'datetime'
LAT_COL = 'center_lat'
LON_COL = 'center_lon'

def parse_arguments():
    """Parses command-line arguments for the script."""
    parser = argparse.ArgumentParser(
        description="Filter MCS events by origination and location within geographic regions."
    )
    parser.add_argument(
        '--input_csv', type=str, default=DEFAULT_INPUT_CSV,
        help='Path to the input MCS events CSV file.'
    )
    parser.add_argument(
        '--regions_yaml', type=str, default=DEFAULT_REGIONS_YAML,
        help='Path to the YAML file defining the geographic regions.'
    )
    parser.add_argument(
        '--output_csv', type=str, default=DEFAULT_OUTPUT_CSV,
        help='Path to save the filtered output CSV file.'
    )
    return parser.parse_args()

def filter_events(args):
    """
    Loads events and regions, filters by origination and location, and saves the result.
    """
    # --- Step 1: Load the region definitions from the YAML file ---
    logging.info(f"Loading region definitions from: {args.regions_yaml}")
    try:
        with open(args.regions_yaml, 'r') as f:
            regions = yaml.safe_load(f)
        logging.info(f"Successfully loaded {len(regions)} regions: {', '.join(regions.keys())}")
    except Exception as e:
        logging.error(f"FATAL: Could not read or parse the YAML file. Error: {e}")
        return

    # --- Step 2: Load the main events CSV file ---
    logging.info(f"Loading events data from: {args.input_csv}")
    try:
        # Convert datetime column to datetime objects on load
        events_df = pd.read_csv(
            args.input_csv,
            parse_dates=[DATETIME_COL]
        )
    except FileNotFoundError:
        logging.error(f"FATAL: The input CSV file was not found at {args.input_csv}")
        return
    except ValueError:
        logging.error(f"FATAL: Could not parse dates in the '{DATETIME_COL}' column. Ensure it's a valid format.")
        return

    initial_count = len(events_df)
    logging.info(f"Loaded {initial_count} total event points.")

    # --- Step 3: Identify tracks that ORIGINATE in a specified region ---
    logging.info("Identifying tracks that originate within the specified regions...")

    # Sort by track number and time to ensure the 'first' entry is the genesis point
    events_df = events_df.sort_values(by=[TRACK_ID_COL, DATETIME_COL])
    
    # Get the first point of each track
    genesis_points_df = events_df.groupby(TRACK_ID_COL).first().reset_index()
    logging.info(f"Found {len(genesis_points_df)} unique tracks.")

    # Create a boolean mask for genesis points within any region
    genesis_mask = pd.Series(False, index=genesis_points_df.index)
    for region_name, bounds in regions.items():
        region_mask = (
            (genesis_points_df[LON_COL] >= bounds['lon_min']) &
            (genesis_points_df[LON_COL] <= bounds['lon_max']) &
            (genesis_points_df[LAT_COL] >= bounds['lat_min']) &
            (genesis_points_df[LAT_COL] <= bounds['lat_max'])
        )
        genesis_mask |= region_mask
    
    # Get the list of track numbers that originated in the regions
    originating_track_ids = genesis_points_df[genesis_mask][TRACK_ID_COL]
    logging.info(f"Found {len(originating_track_ids)} tracks that originated in the regions.")

    # --- Step 4: Filter the main dataframe to keep only those tracks ---
    tracks_in_region_df = events_df[events_df[TRACK_ID_COL].isin(originating_track_ids)].copy()
    
    count_after_origin_filter = len(tracks_in_region_df)
    logging.info(f"Kept {count_after_origin_filter} event points from these tracks.")

    # --- Step 5: Now, filter these points to keep only those WITHIN the regions ---
    logging.info("Filtering points to keep only those within the regions...")
    
    # Create the final combined mask for all points
    final_mask = pd.Series(False, index=tracks_in_region_df.index)
    for region_name, bounds in regions.items():
        region_mask = (
            (tracks_in_region_df[LON_COL] >= bounds['lon_min']) &
            (tracks_in_region_df[LON_COL] <= bounds['lon_max']) &
            (tracks_in_region_df[LAT_COL] >= bounds['lat_min']) &
            (tracks_in_region_df[LAT_COL] <= bounds['lat_max'])
        )
        final_mask |= region_mask

    # Apply the final mask
    filtered_df = tracks_in_region_df[final_mask]
    final_count = len(filtered_df)
    logging.info(f"Filtering complete. Kept {final_count} final event points out of {initial_count}.")

    # --- Step 6: Save the new, smaller DataFrame to a new file ---
    output_path = Path(args.output_csv)
    # Ensure the parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving filtered data to: {output_path}")
    filtered_df.to_csv(output_path, index=False)
    logging.info("✅ Success! Filtered CSV file saved.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = parse_arguments()
    filter_events(args)