#!/usr/bin/env python3
import subprocess
import os

# Define the target directory for composite files
TARGET_DIR = "/home/dkn/composites/ERA5"

# Define the list of regions
# Ensure "eastern_alps" is included if it's one of your target regions,
# as it appears in your example filenames.
regions = ["eastern_alps", "southern_alps", "western_alps", "dinaric_alps"]

# Loop over each region
for region in regions:
    print(f"\n=== Processing region: {region} ===")

    # --- Preparatory scripts ---
    # These are run first. If they also produce uniquely named files in TARGET_DIR
    # that should be checked, similar logic as for composite_tasks could be applied.
    # For now, they run for each region as per the original script's structure.
    preparatory_commands_info = [
        {
            "command": ["python", "03_filter_composite_times.py", "--region", region],
            "desc": "Filter composite times"
        },
        {
            "command": ["python", "04_get_nomcs_datetimes.py", "--region", region],
            "desc": "Get noMCS datetimes"
        }
    ]

    print("\n--- Running preparatory scripts ---")
    preparatory_scripts_failed = False
    for prep_task in preparatory_commands_info:
        cmd = prep_task["command"]
        print(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Command failed: {' '.join(cmd)}\nError: {e}\nStdout: {e.stdout}\nStderr: {e.stderr}")
            preparatory_scripts_failed = True
            break # Stop preparatory scripts for this region on failure
    
    if preparatory_scripts_failed:
        print(f"Skipping composite generation for region {region} due to preparatory script failure.")
        continue # Move to the next region

    # --- Composite generation scripts (with file checking) ---
    print("\n--- Running composite generation scripts ---")
    composite_tasks = [
        {
            "script": "./synoptic_composites/composite_pressure_levels.py",
            "args": [],
            "output_file_pattern": "composite_plev_{region}_wt_clim_evaluation.nc",
        },
        {
            "script": "./synoptic_composites/composite_pressure_levels.py",
            "args": ["--noMCS"],
            "output_file_pattern": "composite_plev_{region}_wt_clim_evaluation_nomcs.nc",
        },
        {
            "script": "./synoptic_composites/composite_surface.py",
            "args": [],
            "output_file_pattern": "composite_surface_{region}_msl_wt_clim_evaluation.nc",
        },
        {
            "script": "./synoptic_composites/composite_surface.py",
            "args": ["--noMCS"],
            "output_file_pattern": "composite_surface_{region}_msl_wt_clim_evaluation_nomcs.nc",
        },
        {
            "script": "./synoptic_composites/composite_surface_precipitation.py",
            "args": [],
            # Note: Based on 'composite_surface_eastern_alps_precipitation_wt.nc'.
            # If your script produces '..._wt_evaluation.nc', adjust pattern below.
            "output_file_pattern": "composite_surface_{region}_precipitation_wt.nc",
            # Alternative: "composite_surface_{region}_precipitation_wt_evaluation.nc",
        },
        {
            "script": "./synoptic_composites/composite_surface_precipitation.py",
            "args": ["--noMCS"],
            "output_file_pattern": "composite_surface_{region}_precipitation_wt_evaluation_nomcs.nc",
        },
        {
            "script": "./synoptic_composites/composite_single_level.py",
            "args": [], # Assumes default is MCS if --noMCS is not present
            "output_file_pattern": "composite_single_level_{region}_evaluation_mcs.nc",
        },
        {
            "script": "./synoptic_composites/composite_single_level.py",
            "args": ["--noMCS"],
            "output_file_pattern": "composite_single_level_{region}_evaluation_nomcs.nc",
        },
    ]

    region_composites_failed = False
    for task in composite_tasks:
        cmd_list = ["python", task["script"], "--region", region] + task["args"]
        cmd_str = " ".join(cmd_list)

        expected_filename = task["output_file_pattern"].format(region=region)
        full_output_path = os.path.join(TARGET_DIR, expected_filename)

        print(f"Checking for: {full_output_path}")
        if os.path.exists(full_output_path):
            print(f"Skipping: {cmd_str}")
            print(f"Reason: Output file {full_output_path} already exists.")
        else:
            print(f"Running: {cmd_str}")
            print(f"Reason: Output file {full_output_path} not found.")
            try:
                # Using capture_output and text=True for better error reporting if needed
                result = subprocess.run(cmd_list, check=True, capture_output=True, text=True)
                # print(f"Successfully ran: {cmd_str}\nStdout: {result.stdout}") # Optional: print stdout
            except subprocess.CalledProcessError as e:
                print(f"Command failed: {cmd_str}\nError: {e}\nStdout: {e.stdout}\nStderr: {e.stderr}")
                region_composites_failed = True
                break # Stop processing other composite tasks for this region if one fails
    
    if region_composites_failed:
        print(f"Finished processing region: {region} with errors during composite generation.")
        # exit(1) # Uncomment to stop all processing on any error
    else:
        print(f"Finished processing region: {region} successfully.")

print("\nAll regions processed.")