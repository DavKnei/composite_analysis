"""
Identify 'noMCS' datetimes for a specific subregion and produce a composite CSV file.

'noMCS' times are defined as datetimes where no MCS center was located within the
specified region during a surrounding time window (e.g., +/- 48 hours).

Steps:
  - Reads a full MCS index CSV file (containing all timesteps of MCS tracks).
  - Filters MCS occurrences based on a specified subregion.
  - Determines the set of all datetimes affected by MCS presence within the region,
    including a buffer window before and after each MCS occurrence.
  - Generates a complete timeline for the dataset's period at a given frequency.
  - Identifies datetimes in the complete timeline that were *not* affected by MCS activity
    within the region and its buffer window.
  - Merges in the corresponding weather_type from an existing weather_types_{region}.csv file.
  - Saves the resulting 'noMCS' datetimes and weather types as a CSV.

Usage:
  python 04_get_nomcs_times.py --input <full_mcs_index.csv> --region <region_name> --output <output_nomcs.csv>

Example:
  python 04_get_nomcs_times.py --input ./csv/mcs_EUR_index.csv \
                               --region Alps \
                               --output ./csv/composite_eastern_alps_nomcs.csv \
                               --weather_type_csv ./weather_typing/csv/GWT_Z500_ \
                               --window_hours 48 \
                               --freq 30min

Author: David Kneidinger 
Date: 2025-05-12
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
        description="Identify 'noMCS' times for a specific subregion."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./csv/mcs_EUR_index.csv",  # Default from 02_get_composite_dates.py
        help="Input CSV file with ALL MCS track timesteps (e.g., mcs_EUR_index.csv)",
    )
    parser.add_argument(
        "--region", type=str, required=True, help=f"Subregion to consider (e.g., Alps"
    )
    parser.add_argument(
        "--weather_type_csv",
        type=str,
        default="./weather_typing/csv/GWT_Z500_",  # Default from 03_filter_composite_times.py
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
        default="30min",  # Default from 02_get_composite_dates.py
        help="Frequency for the complete timeline (e.g., '30min', 'H')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./csv/composite_",
        help="Output CSV file path for 'noMCS' times",
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

    # --- Load and Filter MCS Data ---
    try:
        mcs_df_full = pd.read_csv(args.input, parse_dates=["datetime"])
        print(f"Loaded full MCS index from {args.input}")
    except FileNotFoundError:
        print(f"Error: Input MCS file not found at {args.input}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading MCS file {args.input}: {e}")
        sys.exit(1)

    # Check required columns
    required_cols = ["datetime", "center_lat", "center_lon"]
    if not all(col in mcs_df_full.columns for col in required_cols):
        print(f"Error: Input MCS file must contain columns: {required_cols}")
        sys.exit(1)

    # Filter MCS events by the specified region
    mcs_df_region = filter_by_region(mcs_df_full, region_bounds)
    if mcs_df_region.empty:
        print(f"No MCS events found within the region '{region_key}'.")
        # If no MCS in region, potentially all times are noMCS, but let's proceed
        # cautiously and perhaps require at least some MCS activity in the full dataset.
        if mcs_df_full.empty:
            print("Error: The input MCS file contains no data at all.")
            sys.exit(1)
        # Decide how to proceed: treat all times as noMCS or exit?
        # For now, we'll proceed assuming the goal is to find times truly *without* MCS influence
        # even if the specific region had zero MCS events recorded within it.
        # The script will correctly identify all times as 'noMCS' if mcs_region_times is empty.

    else:
        print(f"Found {len(mcs_df_region)} MCS timesteps within region '{region_key}'.")

    # Get unique datetimes when MCS was present *in the region*
    mcs_region_times = mcs_df_region["datetime"].unique()
    mcs_region_times = pd.to_datetime(
        mcs_region_times
    )  # Ensure they are datetime objects

    # --- Generate Full Timeline ---
    min_time = mcs_df_full["datetime"].min()
    max_time = mcs_df_full["datetime"].max()
    if pd.isna(min_time) or pd.isna(max_time):
        print(
            "Error: Could not determine time range from MCS data (min/max is NaN). Check input file."
        )
        sys.exit(1)

    print(f"Generating full timeline from {min_time} to {max_time}")
    # Ensure frequency is valid pandas offset alias
    try:
        full_timeline = pd.date_range(start=min_time, end=max_time, freq="1h")
    except ValueError as e:
        print(
            f"Error: Invalid frequency string '{args.freq}'. Please use a valid pandas offset alias (e.g., '30min', 'H', 'D'). Error: {e}"
        )
        sys.exit(1)

    if full_timeline.empty:
        print(
            f"Error: Generated timeline is empty. Check frequency ('{args.freq}') and date range ({min_time} to {max_time})."
        )
        sys.exit(1)

    # --- Identify MCS-Affected Times ---
    print(f"Identifying times affected by MCS within +/- {args.window_hours} hours...")
    # Create a boolean mask, initially all True (potential noMCS)
    is_nomcs = pd.Series(True, index=full_timeline)

    # Convert window to timedelta
    window_delta = timedelta(hours=args.window_hours)

    if (
        len(mcs_region_times) > 0
    ):  # Only need to mark if there were MCS times in the region
        # Using searchsorted for potentially better performance than iteration over is_nomcs index
        timeline_numeric = full_timeline.values.astype(
            np.int64
        )  # Convert timeline to numeric for faster search

        affected_count = 0
        marked_indices = 0
        for mcs_time in mcs_region_times:
            start_window = mcs_time - window_delta
            end_window = mcs_time + window_delta

            # Convert window times to nanoseconds since epoch for comparison with timeline_numeric
            start_window_ns = start_window.value
            end_window_ns = end_window.value

            # Find indices in the full timeline that fall within the window
            # searchsorted requires the array (timeline_numeric) to be sorted, which date_range guarantees
            start_idx = np.searchsorted(timeline_numeric, start_window_ns, side="left")
            end_idx = np.searchsorted(
                timeline_numeric, end_window_ns, side="right"
            )  # Use 'right' to include end time if it matches exactly

            if start_idx < end_idx:  # Ensure there are indices within the window
                # Mark the range as False (not noMCS)
                # Ensure indices are within the bounds of the Series
                start_idx = max(0, start_idx)
                end_idx = min(len(is_nomcs), end_idx)
                if start_idx < end_idx:  # Check again after clamping indices
                    is_nomcs.iloc[start_idx:end_idx] = False
                    # marked_indices count is less reliable now if windows overlap heavily
                    # We will count the final number of False values instead

        # Calculate how many unique time steps were marked False
        affected_count = (~is_nomcs).sum()
        print(
            f"Marked {affected_count} unique time steps as MCS-affected (including windows)."
        )
    else:
        print(
            "No MCS times found in the region, so all timeline steps are considered potential 'noMCS'."
        )

    # --- Extract NoMCS Times ---
    nomcs_times = full_timeline[is_nomcs]
    if nomcs_times.empty:
        print("Warning: No 'noMCS' times found within the specified criteria.")
        # Create empty DataFrame with correct columns for consistency
        nomcs_df = pd.DataFrame(
            {
                "datetime": pd.Series(dtype="datetime64[ns]"),
                "wt": pd.Series(dtype="object"),
            }
        )
    else:
        print(f"Identified {len(nomcs_times)} 'noMCS' datetimes.")
        nomcs_df = pd.DataFrame({"datetime": nomcs_times})
        nomcs_df["wt"] = np.nan  # Initialize weather type column

    # --- Merge Weather Types ---
    # This block only runs if nomcs_df is not empty initially, or is created empty above
    weather_csv = f"{args.weather_type_csv}{region_key}_ncl10.csv"  # Using naming convention from 03_...
    wt_col_name_found = None  # Track if we found 'wt' or 'weather_type'
    try:
        wt_df = pd.read_csv(weather_csv, parse_dates=["datetime"])
        # Select only necessary columns, assuming 'wt' or 'weather_type' is the column name
        if "wt" in wt_df.columns:
            wt_col_name_found = "wt"
        elif "weather_type" in wt_df.columns:
            wt_col_name_found = "weather_type"
        else:
            print(
                f"Warning: Could not find standard weather type column ('wt' or 'weather_type') in {weather_csv}."
            )
            # Proceed without weather types, 'wt' column already initialized to NaN

        if wt_col_name_found:
            wt_df_filtered = wt_df[["datetime", wt_col_name_found]].copy()
            # Rename to 'wt' for consistency if needed
            if wt_col_name_found != "wt":
                wt_df_filtered = wt_df_filtered.rename(
                    columns={wt_col_name_found: "wt"}
                )

            # Merge based on the 'datetime' column
            if not nomcs_df.empty:  # Only merge if there are noMCS times
                # Preserve original 'wt' column temporarily if it exists from initialization
                if "wt" in nomcs_df.columns:
                    nomcs_df = nomcs_df.drop(columns=["wt"])
                nomcs_df = pd.merge(nomcs_df, wt_df_filtered, on="datetime", how="left")
                print(f"Merged weather type data from {weather_csv}.")
                # Check how many merges were successful
                missing_wt = nomcs_df["wt"].isnull().sum()
                if missing_wt > 0:
                    print(
                        f"Warning: {missing_wt} out of {len(nomcs_df)} 'noMCS' times could not be matched with a weather type in {weather_csv}."
                    )
                # Fill any remaining NaNs that might result from the merge if needed (though how='left' handles this)
                # nomcs_df['wt'].fillna('Unknown', inplace=True) # Example if you want to replace NaNs
            else:
                # If nomcs_df was initially empty, it still is, but we know the wt file exists.
                print(
                    f"Weather type file {weather_csv} loaded, but no 'noMCS' times to merge with."
                )
                # Ensure 'wt' column exists if nomcs_df is empty
                if "wt" not in nomcs_df.columns:
                    nomcs_df["wt"] = pd.Series(dtype="object")

        else:
            # If no weather type column found, ensure 'wt' column exists with NaNs
            if "wt" not in nomcs_df.columns:
                nomcs_df["wt"] = np.nan
            elif nomcs_df.empty:  # Ensure column exists even if dataframe is empty
                nomcs_df["wt"] = pd.Series(dtype="object")

            print(
                f"No weather type column found in {weather_csv}, 'wt' column filled with NaN."
            )

    except FileNotFoundError:
        print(
            f"Warning: Weather types file not found at {weather_csv}. 'wt' column will be NaN."
        )
        # Ensure 'wt' column exists even if file not found
        if "wt" not in nomcs_df.columns:
            nomcs_df["wt"] = np.nan
        elif nomcs_df.empty:  # Ensure column exists even if dataframe is empty
            nomcs_df["wt"] = pd.Series(dtype="object")

    except Exception as e:
        print(
            f"Warning: Could not load or merge weather types from {weather_csv} ({e}). 'wt' column will be NaN."
        )
        # Ensure 'wt' column exists even if merge failed
        if "wt" not in nomcs_df.columns:
            nomcs_df["wt"] = np.nan
        elif nomcs_df.empty:  # Ensure column exists even if dataframe is empty
            nomcs_df["wt"] = pd.Series(dtype="object")

    # Reorder columns if needed, ensure 'wt' column exists
    if "datetime" not in nomcs_df.columns:
        # This case should ideally not happen if logic above is correct
        nomcs_df["datetime"] = pd.Series(dtype="datetime64[ns]")
    if "wt" not in nomcs_df.columns:
        nomcs_df["wt"] = np.nan  # Should be redundant now, but safe check

    # Ensure standard column order
    output_df = nomcs_df[["datetime", "wt"]]

    # Use a specific datetime format for output consistency
    output_df.to_csv(
        f"{args.output}{region_key}_nomcs.csv",
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )
    print(f"Saved {len(output_df)} 'noMCS' times to {args.output}")


if __name__ == "__main__":
    main()
