"""
Samples noMCS datetimes based on the hourly distribution of MCS initiations.

This script performs the following steps:
1. Reads an MCS initiation CSV file (e.g., mcs_initiation_dates_exp_GAR.csv),
   which is assumed to contain only the initiation datetime for each MCS.
2. Filters these MCS initiation events by a specified subregion.
3. Calculates the normalized hourly distribution of these MCS initiation events.
4. Plots and saves this distribution as a PNG image.
5. Reads a noMCS CSV file (output from 04_get_nomcs_datetimes.py, e.g., <prefix><region>_nomcs.csv),
   which contains hourly datetimes for periods without MCS activity.
6. For each unique day in the noMCS file, it randomly samples one hourly timestamp.
   This sampling is weighted by the previously calculated MCS initiation hour distribution.
7. Saves the resulting filtered list of noMCS datetimes (one per original noMCS day)
   to a new CSV file (e.g., <output_dir><region>_nomcs_filtered.csv).

This helps in creating comparable composite datasets for noMCS periods by ensuring
the diurnal cycle of sampled noMCS times reflects that of MCS initiations.

Usage:
  python sample_noMCS_hours.py --mcs_input_file <path_to_mcs_initiation_data.csv> \
                               --nomcs_input_dir_prefix <path_and_prefix_for_region_nomcs.csv> \
                               --output_dir <path_to_output_directory> \
                               --region <region_name>

Example:
  python sample_noMCS_hours.py --region eastern_alps
  (using default file paths)
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Define subregion boundaries (consistent with 04_get_nomcs_datetimes.py)
SUBREGIONS = {
    'western_alps': {'lon_min': 3,   'lon_max': 8,   'lat_min': 43, 'lat_max': 49},
    'southern_alps': {'lon_min': 8,   'lon_max': 13,  'lat_min': 43, 'lat_max': 46},
    'dinaric_alps':  {'lon_min': 13,  'lon_max': 20,  'lat_min': 42, 'lat_max': 46},
    'eastern_alps':  {'lon_min': 8,   'lon_max': 17,  'lat_min': 46, 'lat_max': 49}
}

def filter_by_region(df, region_bounds, lat_col='center_lat', lon_col='center_lon'):
    """Filters a DataFrame for rows where coordinates are within specified region bounds."""
    return df[
        (df[lat_col] >= region_bounds['lat_min']) &
        (df[lat_col] <= region_bounds['lat_max']) &
        (df[lon_col] >= region_bounds['lon_min']) &
        (df[lon_col] <= region_bounds['lon_max'])
    ].copy()

def main():
    parser = argparse.ArgumentParser(
        description="Sample noMCS datetimes based on MCS initiation hour distribution."
    )
    parser.add_argument(
        "--mcs_input_file",
        type=str,
        default="./synoptic_composites/csv/mcs_initiation_dates_exp_GAR.csv",
        help="Input CSV file with MCS initiation datetimes (e.g., mcs_initiation_dates_exp_GAR.csv)"
    )
    parser.add_argument(
        "--nomcs_input_dir_prefix", # Renamed for clarity
        type=str,
        default="./synoptic_composites/csv/composite_",
        help="Input directory path and filename prefix for the <region>_nomcs.csv files (e.g., ./synoptic_composites/csv/composite_)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./synoptic_composites/csv/",
        help="Directory to save the filtered noMCS CSV and the distribution plot"
    )
    parser.add_argument(
        "--region",
        type=str,
        required=True,
        choices=SUBREGIONS.keys(),
        help=f"Subregion to process (e.g., {', '.join(SUBREGIONS.keys())})"
    )
    args = parser.parse_args()

    # --- Validate Region ---
    region_key = args.region
    if region_key not in SUBREGIONS:
        print(f"Error: Region '{region_key}' is not defined. Available regions: {list(SUBREGIONS.keys())}")
        sys.exit(1)
    region_bounds = SUBREGIONS[region_key]
    print(f"Processing for region: {region_key}")

    # --- Create Output Directory if it doesn't exist ---
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    # --- 1. Load MCS Initiation Data and Calculate Hour Distribution ---
    try:
        mcs_initiation_df_full = pd.read_csv(args.mcs_input_file, parse_dates=['datetime'])
        print(f"Loaded MCS initiation data from {args.mcs_input_file}")
    except FileNotFoundError:
        print(f"Error: MCS initiation input file not found at {args.mcs_input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading MCS initiation file {args.mcs_input_file}: {e}")
        sys.exit(1)

    # Check required columns for MCS initiation data
    required_mcs_cols = ['datetime', 'center_lat', 'center_lon']
    if not all(col in mcs_initiation_df_full.columns for col in required_mcs_cols):
        print(f"Error: MCS initiation input file must contain columns: {required_mcs_cols}")
        sys.exit(1)

    # Filter MCS initiations by region
    mcs_initiations_region = filter_by_region(mcs_initiation_df_full, region_bounds)
    
    if mcs_initiations_region.empty:
        print(f"No MCS initiation events found within the region '{region_key}' in {args.mcs_input_file}.")
        # Create an empty filtered file and plot for consistency
        nomcs_filtered_output_file = os.path.join(args.output_dir, f"{region_key}_nomcs_filtered.csv")
        pd.DataFrame(columns=['datetime', 'wt']).to_csv(nomcs_filtered_output_file, index=False, date_format='%Y-%m-%d %H:%M:%S')
        print(f"Created empty filtered noMCS file: {nomcs_filtered_output_file} as no MCS initiations were found in the region.")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'No MCS initiation events found for region: {region_key}',
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title(f'MCS Initiation Hour Distribution - {region_key.replace("_", " ").title()}')
        plot_output_file = os.path.join(args.output_dir, f"{region_key}_mcs_initiation_hour_distribution.png")
        plt.savefig(plot_output_file)
        print(f"Saved plot indicating no data to: {plot_output_file}")
        plt.close(fig)
        sys.exit(0)

    print(f"Found {len(mcs_initiations_region)} MCS initiation events in region '{region_key}'.")

    # Calculate hour of initiation
    mcs_initiations_region['initiation_hour'] = mcs_initiations_region['datetime'].dt.hour

    # Calculate hourly distribution (frequency)
    hourly_counts = mcs_initiations_region['initiation_hour'].value_counts().sort_index()

    # Ensure all hours from 0 to 23 are present, fill with 0 if not
    for hour in range(24):
        if hour not in hourly_counts:
            hourly_counts[hour] = 0
    hourly_counts = hourly_counts.sort_index()

    # Normalize to get probabilities
    if hourly_counts.sum() == 0: # Should be caught by mcs_initiations_region.empty check, but as a safeguard
        print(f"Error: Sum of hourly counts is zero, cannot create distribution for {region_key}.")
        # Create empty plot and exit
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'MCS initiation events found but sum to zero counts for region: {region_key}',
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title(f'MCS Initiation Hour Distribution - {region_key.replace("_", " ").title()}')
        plot_output_file = os.path.join(args.output_dir, f"{region_key}_mcs_initiation_hour_distribution.png")
        plt.savefig(plot_output_file)
        print(f"Saved plot indicating data error to: {plot_output_file}")
        plt.close(fig)
        sys.exit(1)
        
    hourly_distribution = hourly_counts / hourly_counts.sum()
    print("MCS Initiation Hour Distribution (Probabilities):")
    print(hourly_distribution)

    # --- 2. Plot and Save the Distribution ---
    fig, ax = plt.subplots(figsize=(12, 7))
    hourly_distribution.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
    ax.set_title(f'MCS Initiation Hour Distribution - {region_key.replace("_", " ").title()}', fontsize=16)
    ax.set_xlabel('Hour of Day (UTC)', fontsize=14)
    ax.set_ylabel('Probability', fontsize=14)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot_output_file = os.path.join(args.output_dir, f"{region_key}_mcs_initiation_hour_distribution.png")
    plt.savefig(plot_output_file)
    print(f"Saved MCS initiation hour distribution plot to: {plot_output_file}")
    plt.close(fig)

    # --- 3. Load noMCS Data ---
    # Construct noMCS input filename using the directory and prefix
    nomcs_input_file = f"{args.nomcs_input_dir_prefix}{region_key}_nomcs.csv"
    try:
        nomcs_df = pd.read_csv(nomcs_input_file, parse_dates=['datetime'])
        print(f"Loaded noMCS data from: {nomcs_input_file}")
    except FileNotFoundError:
        print(f"Error: noMCS input file not found at {nomcs_input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading noMCS file {nomcs_input_file}: {e}")
        sys.exit(1)

    if nomcs_df.empty:
        print(f"No noMCS datetimes found in {nomcs_input_file}. Output will be empty.")
        nomcs_filtered_output_file = os.path.join(args.output_dir, f"{region_key}_nomcs_filtered.csv")
        pd.DataFrame(columns=['datetime', 'wt']).to_csv(nomcs_filtered_output_file, index=False, date_format='%Y-%m-%d %H:%M:%S')
        print(f"Saved empty filtered noMCS file to: {nomcs_filtered_output_file}")
        sys.exit(0)

    if 'wt' not in nomcs_df.columns:
        nomcs_df['wt'] = np.nan # Ensure 'wt' column exists

    # --- 4. Sample noMCS Datetimes ---
    nomcs_df['date'] = nomcs_df['datetime'].dt.date
    grouped_nomcs_days = nomcs_df.groupby('date')
    sampled_nomcs_records = []
 
    sampling_hours = np.array(hourly_distribution.index)
    sampling_probabilities = np.array(hourly_distribution.values)
    sampling_probabilities /= np.sum(sampling_probabilities) # Ensure sum to 1

    for date_obj, group in grouped_nomcs_days: # date_obj is a datetime.date object
        available_hours_on_day = group['datetime'].dt.hour.unique()
        possible_sample_hours = [h for h in sampling_hours if h in available_hours_on_day]

        if not possible_sample_hours:
            print(f"Warning: No available noMCS hours on {date_obj.strftime('%Y-%m-%d')} match the MCS initiation hour profile. Skipping this day.")
            continue

        current_day_probabilities = hourly_distribution[hourly_distribution.index.isin(possible_sample_hours)]

        selected_hour = -1 # Initialize to an invalid hour

        if current_day_probabilities.sum() == 0:
            if possible_sample_hours:
                 selected_hour = np.random.choice(possible_sample_hours)
                 print(f"Note: For day {date_obj.strftime('%Y-%m-%d')}, MCS initiation hours had 0 probability for available noMCS hours. Sampling uniformly from available hours. Selected: {selected_hour:02d}h")
            else:
                 print(f"Critical Warning: No hours to sample from for day {date_obj.strftime('%Y-%m-%d')} after all checks. Skipping.")
                 continue
        else:
            current_day_probabilities_normalized = current_day_probabilities / current_day_probabilities.sum()
            selected_hour = np.random.choice(
                np.array(current_day_probabilities_normalized.index),
                p=np.array(current_day_probabilities_normalized.values)
            )
        
        # Ensure selected_hour is valid before proceeding
        if selected_hour == -1 : # Should not be reached if logic above is correct
            print(f"Error selecting hour for day {date_obj.strftime('%Y-%m-%d')}. Skipping.")
            continue

        selected_record = group[group['datetime'].dt.hour == selected_hour].iloc[0]
        sampled_nomcs_records.append(selected_record)

    if not sampled_nomcs_records:
        print("No noMCS records were sampled. This might happen if noMCS days/hours "
              "could not be matched with the MCS initiation hour distribution or if no noMCS data was available.")
        nomcs_filtered_output_file = os.path.join(args.output_dir, f"{region_key}_nomcs_filtered.csv")
        pd.DataFrame(columns=['datetime', 'wt']).to_csv(nomcs_filtered_output_file, index=False, date_format='%Y-%m-%d %H:%M:%S')
        print(f"Saved empty filtered noMCS file to: {nomcs_filtered_output_file}")
        sys.exit(0)

    output_df = pd.DataFrame(sampled_nomcs_records)
    output_columns = ['datetime']
    if 'wt' in output_df.columns:
        output_columns.append('wt')
    # Ensure correct column order and drop the temporary 'date' column
    output_df = output_df[output_columns]


    # --- 5. Save Filtered noMCS Data ---
    nomcs_filtered_output_file = os.path.join(args.output_dir, f"{region_key}_nomcs_filtered.csv")
    output_df.to_csv(nomcs_filtered_output_file, index=False, date_format='%Y-%m-%d %H:%M:%S')
    print(f"Saved {len(output_df)} sampled noMCS datetimes to: {nomcs_filtered_output_file}")

if __name__ == "__main__":
    main()
