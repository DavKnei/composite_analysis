#!/usr/bin/env python3
import subprocess

# Define the list of regions
regions = ["southern_alps", "western_alps", "dinaric_alps"]

# Loop over each region
for region in regions:
    print(f"\n=== Processing region: {region} ===")
    
    # List of commands to run for the given region.
    commands = [
        ["python", "03_filter_composite_times.py", "--region", region],
        ["python", "04_get_nomcs_datetimes.py", "--region", region],
        ["python", "./synoptic_composites/composite_pressure_levels.py",  "--region", region],
        ["python", "./synoptic_composites/composite_pressure_levels.py",  "--region", region, "--noMCS"],
        ["python", "./synoptic_composites/composite_surface.py", "--region", region],
        ["python", "./synoptic_composites/composite_surface.py", "--region", region, "--noMCS"],
        ["python", "./synoptic_composites/composite_surface_precipitation.py", "--region", region],
        ["python", "./synoptic_composites/composite_surface_precipitation.py", "--region", region, "--noMCS"],
    ]
    
    # Execute each command sequentially
    for cmd in commands:
        print("Running command:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Command failed: {' '.join(cmd)}\nError: {e}")
            # Optionally: exit or continue with the next region
            exit(1)
    
    print(f"Finished processing region: {region}")
print("\nAll regions processed successfully.")