#!/usr/bin/env python3
"""
Filter MCS initiation times for a specific subregion and produce a composite CSV file.

For MCS events:
  - Reads a CSV file with MCS initiation times (with columns including datetime, center_lat, center_lon, track_number, total_precip, area)
  - Selects the first appearance of each track (i.e. initiation)
  - Filters these events to retain only those initiated within a specified subregion.
  - For each event, creates additional time columns for user-specified offsets (e.g. -6h, -3h, 0h, 3h, 6h).
  - Merges in the corresponding weather_type from an existing weather_types_{region}.csv file based on the initiation time.
  - Saves the resulting DataFrame as a CSV.

Usage:
  python 03_filter_composite_times.py --mcs_csv mcs_initiation.csv --region southern_alps --times -6,-3,0,3,6 --window 48 --output_prefix composite_

Author: David Kneidinger (updated)
Date: 2025-03-26
"""

import argparse
import pandas as pd
import numpy as np
from datetime import timedelta

# Define subregion boundaries
SUBREGIONS = {
    'western_alps': {'lon_min': 3,   'lon_max': 8,   'lat_min': 43, 'lat_max': 49},
    'southern_alps': {'lon_min': 8,   'lon_max': 13,  'lat_min': 43, 'lat_max': 46},
    'dinaric_alps':  {'lon_min': 13,  'lon_max': 20,  'lat_min': 42, 'lat_max': 46},
    'eastern_alps':  {'lon_min': 8,   'lon_max': 17,  'lat_min': 46, 'lat_max': 49}
}

def filter_by_region(df, region_bounds, lat_col='center_lat', lon_col='center_lon'):
    """Return rows where the center coordinates are within the specified region bounds."""
    return df[
        (df[lat_col] >= region_bounds['lat_min']) &
        (df[lat_col] <= region_bounds['lat_max']) &
        (df[lon_col] >= region_bounds['lon_min']) &
        (df[lon_col] <= region_bounds['lon_max'])
    ].copy()

def select_initiated_tracks(df):
    """
    Select the first appearance (initiation) for each track.
    Assumes 'datetime' is present.
    """
    df['datetime'] = pd.to_datetime(df['datetime'])
    df_sorted = df.sort_values(by='datetime')
    return df_sorted.drop_duplicates(subset='track_number', keep='first')

def create_offset_times(df, offset_hours):
    """Create new columns for each time offset based on the 'datetime' column."""
    df['datetime'] = pd.to_datetime(df['datetime'])
    for off in offset_hours:
        if off < 0:
            colname = f"time_minus{abs(off)}h"
        elif off == 0:
            colname = "time_0h"
        else:
            colname = f"time_plus{off}h"
        df[colname] = df['datetime'] + timedelta(hours=off)
    return df

def main():
    parser = argparse.ArgumentParser(
        description="Filter MCS initiation times for a specific subregion and produce a composite CSV file."
    )
    parser.add_argument("--mcs_csv", type=str, default="synoptic_composites/csv/mcs_initiation_dates_exp_GAR.csv",
                        help="Input CSV file with MCS initiation times")
    parser.add_argument("--weather_type_csv", type=str, default="./weather_typing/csv/GWT_Z500_",
                        help="Input CSV file with weather types")
    parser.add_argument("--region", type=str, required=True,
                        help="Subregion to consider (e.g., western_alps, southern_alps, dinaric_alps, eastern_alps)")
    parser.add_argument("--times", type=str, default="-12,-6,0,6,12",
                        help="Comma-separated list of time offsets in hours (e.g., -12,-6,0,6,12)")
    parser.add_argument("--output_prefix", type=str, default="synoptic_composites/csv/composite_",
                        help="Prefix for output CSV files")
    args = parser.parse_args()
    
    region_key = args.region
    if region_key not in SUBREGIONS:
        print(f"Error: region '{region_key}' is not defined. Available regions: {list(SUBREGIONS.keys())}")
        return
    region_bounds = SUBREGIONS[region_key]
    
    # Parse offsets into a list of integers
    offset_hours = [int(x) for x in args.times.split(",")]
    
    # Read the MCS initiation CSV, select first appearances, and filter by region
    mcs_df = pd.read_csv(args.mcs_csv, parse_dates=['datetime'])
    mcs_initiated = select_initiated_tracks(mcs_df)
    mcs_region = filter_by_region(mcs_initiated, region_bounds)
    if mcs_region.empty:
        print(f"No MCS events initiated in region {region_key} found in {args.mcs_csv}.")
        return
    else:
        print(f"Found {len(mcs_region)} MCS events initiated in region {region_key}.")
    
    # Create offset time columns for MCS events
    mcs_region = create_offset_times(mcs_region, offset_hours)
    
    # Merge weather_type from the weather_types CSV.
    # It is assumed that there is a file named "{weather_types_csv}{region_key}.csv" with at least columns: "date" and "weather_type"
    weather_csv = f"{args.weather_type_csv}{region_key}_ncl10.csv" 
    try:
        wt_df = pd.read_csv(weather_csv, parse_dates=['datetime'])
        wt_df = wt_df[['datetime', 'wt']]
        wt_df = wt_df.set_index('datetime')
        # Merge using the initiation time (time_0h) and the weather type date.
        mcs_region = mcs_region.set_index('datetime')
        mcs_region = pd.merge(mcs_region, wt_df, left_on='time_0h', right_on='datetime', how='outer')
        mcs_region = mcs_region.dropna(axis=0)
        print(f"Merged weather type data from {weather_csv}.")
    except Exception as e:
        print(f"Warning: Could not load weather types file {weather_csv} ({e}). Weather type column will be set to NaN.")
        mcs_region['wt'] = np.nan

    # Order time columns in increasing order
    time_cols = []
    for off in sorted(offset_hours):
        if off < 0:
            time_cols.append(f"time_minus{abs(off)}h")
        elif off == 0:
            time_cols.append("time_0h")
        else:
            time_cols.append(f"time_plus{off}h")
    
    # Include weather_type in the output columns.
    required_cols = time_cols + ["center_lat", "center_lon", "track_number", "total_precip", "area", "wt"]
    mcs_out = mcs_region[required_cols].copy()
    mcs_out_filename = f"{args.output_prefix}{region_key}_mcs.csv"
    mcs_out.to_csv(mcs_out_filename, index=False)
    print(f"Saved composite MCS times to {mcs_out_filename}")

if __name__ == "__main__":
    main()
