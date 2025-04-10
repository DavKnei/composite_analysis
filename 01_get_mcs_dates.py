#!/usr/bin/env python3
"""
Script to extract MCS events that are fully inside the Greater Alpine Region (GAR)
from netCDF track files. For each qualifying track (using the variable
cloudtracknumber_nomergesplit), the script calculates:

    The datetime (parsed from the filename)
    The precipitation‐weighted center point (lat, lon)
    The total precipitation (sum over all grid cells in the track)
    The area (number of grid cells, later convertible to physical area)
    The unique track number (cloudtracknumber_nomergesplit)

Results are saved into a CSV file.

The script supports parallel processing using multiprocessing (default 32 cores)
or serial processing for debugging.

Usage:
    python extract_mcs_in_gar.py [--ncores N] [--serial]

Author: Your Name
Date: YYYY-MM-DD
"""

import os
import glob
import argparse
from datetime import datetime
import numpy as np
import netCDF4 as nc
import csv
from multiprocessing import Pool

# Define the Greater Alpine Region boundaries
GAR_LON_MIN = -10.0
GAR_LON_MAX = 25.0
GAR_LAT_MIN = 35.0
GAR_LAT_MAX = 52.0


def parse_datetime_from_filename(filename):
    """
    Parse datetime from filename of the form: mcstrack_YYYYMMDD_HHMM.nc
    Returns a datetime object.
    """
    base = os.path.basename(filename)
    # Remove extension and split by underscore: ["mcstrack", "YYYYMMDD", "HHMM"]
    parts = base.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format: {filename}")
    date_str = parts[1]
    time_str = parts[2].split(".")[0]  # remove .nc extension
    dt_str = date_str + time_str  # e.g. "200101110730"
    return datetime.strptime(dt_str, "%Y%m%d%H%M")


def process_file(file_path):
    """
    Process a single netCDF file.
    For each unique MCS track (using cloudtracknumber_nomergesplit),
    check if all grid points for that track are inside GAR.
    If yes, compute the precipitation-weighted center, total precipitation,
    and area (number of grid cells).

    Returns a list of dictionaries (one per track) with:
      datetime, center_lat, center_lon, track_number, total_precip, area.
    """
    results = []
    try:
        ds = nc.Dataset(file_path, "r")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return results

    # Check that required variables exist
    required_vars = [
        "cloudtracknumber_nomergesplit",
        "precipitation",
        "latitude",
        "longitude",
    ]
    if not all(var in ds.variables for var in required_vars):
        print(f"File {file_path} does not contain required variables.")
        ds.close()
        return results

    # Read variables as numpy arrays
    track_num = ds.variables["cloudtracknumber_nomergesplit"][:]
    precip = ds.variables["precipitation"][:]
    lat = ds.variables["latitude"][:]
    lon = ds.variables["longitude"][:]
    ds.close()

    # Flatten arrays assuming each grid cell is an independent observation with a track label
    track_num = np.ravel(track_num)
    precip = np.ravel(precip)
    lat = np.ravel(lat)
    lon = np.ravel(lon)

    # Get unique track numbers; assume non-zero are valid
    unique_tracks = np.unique(track_num)
    unique_tracks = unique_tracks[unique_tracks != 0]  # filter out zeros

    count = 0
    for t in unique_tracks:
        
        mask = track_num == t
        if not np.any(mask):
            continue

        track_lats = lat[mask]
        track_lons = lon[mask]
        track_precip = precip[mask]

        # Check if the entire track footprint is inside GAR
        if (
            track_lats.min() >= GAR_LAT_MIN
            and track_lats.max() <= GAR_LAT_MAX
            and track_lons.min() >= GAR_LON_MIN
            and track_lons.max() <= GAR_LON_MAX
        ):

            # Compute precipitation-weighted center point
            total_precip = np.sum(track_precip)
            if total_precip > 0:
                center_lat = np.sum(track_lats * track_precip) / total_precip
                center_lon = np.sum(track_lons * track_precip) / total_precip
            else:
                center_lat = np.mean(track_lats)
                center_lon = np.mean(track_lons)

            area = int(np.sum(mask))  # number of grid cells

            # Parse the datetime from the filename
            dt = parse_datetime_from_filename(file_path)

            results.append(
                {
                    "datetime": dt.strftime("%Y-%m-%d %H:%M"),
                    "center_lat": center_lat,
                    "center_lon": center_lon,
                    "track_number": int(t),
                    "total_precip": total_precip,
                    "area": area,
                }
            )
            print("Processed:", dt.strftime("%Y-%m-%d %H:%M"))
    return results


def get_all_files(root_dir):
    """
    Walk through the root directory and return a list of all .nc files.
    """
    pattern = os.path.join(root_dir, "**", "*.nc")
    files = glob.glob(pattern, recursive=True)
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Extract MCS events fully inside the Greater Alpine Region from track files."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="/data/reloclim/backup/MCS_database/raw_data/mcstracking",
        help="Root directory containing MCS track files (organized by YYYY/MM/)",
    )
    parser.add_argument(
        "--output", type=str, default="./csv/mcs_exp_GAR_index.csv", help="Output CSV file name"
    )
    parser.add_argument(
        "--ncores",
        type=int,
        default=32,
        help="Number of cores to use for multiprocessing",
    )
    parser.add_argument(
        "--serial", action="store_true", help="Run processing serially for debugging"
    )
    args = parser.parse_args()

    files = get_all_files(args.root)
    print(f"Found {len(files)} files to process.")

    all_results = []
    if args.serial:
        for f in files:
            res = process_file(f)
            all_results.extend(res)
            
    else:
        with Pool(processes=args.ncores) as pool:
            results = pool.map(process_file, files)
            # Flatten the list of lists
            for res in results:
                all_results.extend(res)
    all_results.sort(key=lambda x: x["datetime"])
    print(f"Total qualifying MCS events found: {len(all_results)}")

    # Save to CSV
    fieldnames = [
        "datetime",
        "center_lat",
        "center_lon",
        "track_number",
        "total_precip",
        "area",
    ]
    with open(args.output, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
