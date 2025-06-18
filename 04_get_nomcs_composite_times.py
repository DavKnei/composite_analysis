"""
Identify 'noMCS' datetimes for a specific subregion and produce a composite CSV file.

This script now includes functionality to perform stratified random sampling on the
'noMCS' datetimes, selecting a fixed number of samples for each weather type.
This is useful for creating balanced composites and reducing autocorrelation.

'noMCS' times are defined as datetimes where no MCS center was located within the
specified region during a surrounding time window (e.g., +/- 48 hours). The analysis
is constrained to the time period available in the weather type file.

Steps:
  - Loads the weather type CSV to define the analysis time window.
  - Reads a full MCS index CSV file.
  - Filters MCS occurrences based on a specified subregion.
  - Generates a complete timeline based on the weather type data's period.
  - Identifies datetimes in the timeline that were *not* affected by MCS activity.
  - Merges 'noMCS' times with weather types using an inner join to ensure all
    datetimes have a corresponding weather type.
  - **NEW**: Randomly samples a specified number of 'noMCS' times for each weather type.
  - Saves the resulting sampled 'noMCS' datetimes and weather types as a CSV.

Usage:
  python 04_get_nomcs_composite_times.py --input <full_mcs_index.csv> --region <region_name> --output <output_nomcs.csv> [--samples_per_wt <number>]

Example with Sampling:
  python 04_get_nomcs_composite_times.py --input ./csv/mcs_EUR_index.csv \
                               --region Alps \
                               --output ./csv/composite_eastern_alps_nomcs_sampled.csv \
                               --weather_type_csv ./weather_typing/csv/GWT_Z500_ \
                               --window_hours 48 \
                               --samples_per_wt 100

Author: David Kneidinger
Date: 2025-05-12
Revised: 2025-06-18
"""

import argparse
import pandas as pd
import numpy as np
from datetime import timedelta
import sys
import yaml


def filter_by_region(df, region_bounds, lat_col="center_lat", lon_col="center_lon"):
    """Return rows where the center coordinates are within the specified region bounds."""
    # Function copied from 03_filter_composite_times.py
    return df[
        (df[lat_col] >= region_bounds["lat_min"])
        & (df[lat_col] <= region_bounds["lat_max"])
        & (df[lon_col] >= region_bounds["lon_min"])
        & (df[lon_col] <= region_bounds["lon_max"])
    ].copy()


def main():
    parser = argparse.ArgumentParser(
        description="Identify and sample 'noMCS' times for a specific subregion."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./csv/mcs_EUR_index.csv",
        help="Input CSV file with ALL MCS track timesteps (e.g., mcs_EUR_index.csv)",
    )
    parser.add_argument(
        "--region", type=str, required=True, help="Subregion to consider (e.g., Alps)"
    )
    parser.add_argument(
        "--weather_type_csv",
        type=str,
        default="./weather_typing/csv/GWT_Z500_",
        help="Base path/prefix for input CSV file with weather types (e.g., ./weather_typing/csv/GWT_Z500_)",
    )
    parser.add_argument(
        "--window_hours",
        type=int,
        default=48,
        help="Number of hours +/- around an MCS event to exclude (default: 48)",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="1h",
        help="Frequency for the complete timeline (e.g., '30min', 'H')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./synoptic_composites/csv/composite_",
        help="Output CSV file path for 'noMCS' times",
    )
    parser.add_argument(
        "--samples_per_wt",
        type=int,
        default=300,
        help="Number of random samples to take per weather type. If not set, all 'noMCS' times are saved.",
    )

    args, _ = parser.parse_known_args()

    # --- Validate Region ---
    region_key = args.region
    with open("regions.yaml", "r") as f:
        regions_data = yaml.safe_load(f)
    if args.region not in regions_data:
        print(f"ERROR: Region '{args.region}' not found in regions.yaml.")
        return
    region_bounds = regions_data[args.region]
    print(f"Processing for region: {args.region} {region_bounds}")

    # --- Load Weather Types to Define Time Window ---
    weather_csv = f"{args.weather_type_csv}{region_key}_ncl10.csv"
    try:
        wt_df = pd.read_csv(weather_csv, parse_dates=["datetime"])
        wt_col = "wt" if "wt" in wt_df.columns else "weather_type"
        if wt_col not in wt_df.columns:
             raise ValueError("No 'wt' or 'weather_type' column found in weather type file.")
        if wt_col != 'wt':
            wt_df.rename(columns={wt_col: 'wt'}, inplace=True)
        print(f"Loaded weather types from {weather_csv} to define time window.")
    except FileNotFoundError:
        print(f"ERROR: Weather types file not found at {weather_csv}. This file is required to define the analysis timeline.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Could not load or process weather types from {weather_csv}: {e}")
        sys.exit(1)


    # --- Load and Filter MCS Data ---
    try:
        mcs_df_full = pd.read_csv(args.input, parse_dates=["datetime"])
        print(f"Loaded full MCS index from {args.input}")
    except FileNotFoundError:
        print(f"Error: Input MCS file not found at {args.input}")
        sys.exit(1)

    required_cols = ["datetime", "center_lat", "center_lon"]
    if not all(col in mcs_df_full.columns for col in required_cols):
        print(f"Error: Input MCS file must contain columns: {required_cols}")
        sys.exit(1)

    mcs_df_region = filter_by_region(mcs_df_full, region_bounds)
    if mcs_df_region.empty:
        print(f"No MCS events found within the region '{region_key}'.")
    else:
        print(f"Found {len(mcs_df_region)} MCS timesteps within region '{region_key}'.")

    mcs_region_times = pd.to_datetime(mcs_df_region["datetime"].unique())

    # --- Generate Full Timeline Based on Weather Type File ---
    min_time = wt_df["datetime"].min()
    max_time = wt_df["datetime"].max()
    if pd.isna(min_time) or pd.isna(max_time):
        print("Error: Could not determine time range from weather type data.")
        sys.exit(1)

    print(f"Generating timeline based on weather types from {min_time} to {max_time}")
    full_timeline = pd.date_range(start=min_time, end=max_time, freq=args.freq)

    # --- Identify MCS-Affected Times ---
    print(f"Identifying times affected by MCS within +/- {args.window_hours} hours...")
    is_nomcs = pd.Series(True, index=full_timeline)
    window_delta = timedelta(hours=args.window_hours)

    if len(mcs_region_times) > 0:
        timeline_numeric = full_timeline.values.astype(np.int64)
        for mcs_time in mcs_region_times:
            start_window = mcs_time - window_delta
            end_window = mcs_time + window_delta
            start_idx = np.searchsorted(timeline_numeric, start_window.value, side="left")
            end_idx = np.searchsorted(timeline_numeric, end_window.value, side="right")
            if start_idx < end_idx:
                is_nomcs.iloc[start_idx:end_idx] = False
        affected_count = (~is_nomcs).sum()
        print(f"Marked {affected_count} unique time steps as MCS-affected.")
    else:
        print("No MCS times found in the region, all timeline steps are potential 'noMCS'.")

    # --- Extract NoMCS Times and Merge with Weather Types ---
    nomcs_times = full_timeline[is_nomcs]
    nomcs_df = pd.DataFrame({"datetime": nomcs_times})
    print(f"Identified {len(nomcs_df)} potential 'noMCS' datetimes in timeline.")

    # Use an INNER join to ensure we only keep times that exist in the weather type file.
    # This correctly handles cases where the wt_df is sparse (e.g., only summer months).
    output_df = pd.merge(nomcs_df, wt_df, on="datetime", how="inner")
    print(f"Found {len(output_df)} 'noMCS' datetimes with a valid weather type.")


    # --- Perform Stratified Random Sampling ---
    if args.samples_per_wt is not None and not output_df.empty:
        print(f"Performing stratified random sampling: {args.samples_per_wt} samples per weather type.")
        
        # Drop rows where weather type is NaN, as we can't stratify them (inner join should prevent this)
        clean_df = output_df.dropna(subset=['wt']).copy()
        
        # Ensure wt is integer type for grouping
        clean_df['wt'] = clean_df['wt'].astype(int)

        # Group by weather type and sample
        sampled_df = clean_df.groupby('wt', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), args.samples_per_wt), random_state=1)
        )
        
        # Sort by datetime for a clean output file
        output_df = sampled_df.sort_values(by="datetime").reset_index(drop=True)
        
        print(f"Sampled down to {len(output_df)} total 'noMCS' times.")

    # --- Save Final DataFrame ---
    if output_df.empty:
        print("Warning: No 'noMCS' times found after filtering and merging. No output file will be created.")
    else:
        # Ensure standard column order
        output_df = output_df[["datetime", "wt"]]
        output_filename = f"{args.output}{region_key}_nomcs.csv"
        output_df.to_csv(
            output_filename,
            index=False,
            date_format="%Y-%m-%d %H:%M:%S",
        )
        print(f"Saved {len(output_df)} 'noMCS' times to {output_filename}")


if __name__ == "__main__":
    main()