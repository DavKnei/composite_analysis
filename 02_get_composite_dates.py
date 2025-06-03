#!/usr/bin/env python3
""" Script to calculate MCS initiation times from an MCS index CSV file. 
For each unique MCS (using pcptracknumber), 
the script determines the initiation time (the earliest datetime) and saves these as mcs_initiation.csv. 

Usage: python 02_get_composite_date.py --input mcs_EUR_index.csv

Author: David Kneidinger: 2025-03-25 """

import argparse
import pandas as pd


def get_mcs_initiation_times(input_csv, output_csv):
    # Read the CSV file into a pandas DataFrame.
    df = pd.read_csv(input_csv, parse_dates=["datetime"])

    # Group by track_number and take the minimum datetime (initiation time) for each track.
    initiation_df = df.groupby("track_number", as_index=False).agg(
        {
            "datetime": "min",
            "center_lat": "first",
            "center_lon": "first",
            "total_precip": "first",
            "area": "first",
        }
    )
    initiation_df = initiation_df.sort_values("datetime")
    initiation_df.to_csv(output_csv, index=False)

    print(f"Saved {len(initiation_df)} MCS initiation times to {output_csv}")
    return initiation_df


def main():
    parser = argparse.ArgumentParser(
        description="Calculate MCS initiation from an MCS index CSV file."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./csv/mcs_EUR_index.csv",
        help="Input CSV file with MCS events",
    )
    parser.add_argument(
        "--init_output",
        type=str,
        default="./csv/mcs_initiation_dates_EUR.csv",
        help="Output CSV file for MCS initiation times",
    )
    args = parser.parse_args()

    # Get the initiation times.
    initiation_df = get_mcs_initiation_times(args.input, args.init_output)


if __name__ == "__main__":
    main()
