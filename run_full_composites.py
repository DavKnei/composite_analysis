#!/usr/bin/env python3
import subprocess

# Define the list of regions
regions = ["southern_alps", "eastern_alps", "western_alps", "dinaric_alps"]

# Loop over each region
for region in regions:
    print(f"\n=== Processing region: {region} ===")
    
    # List of commands to run for the given region.
    # Note: Only the first command requires the region parameter.
    commands = [
        #["python", "lamb_weathertypes_msl.py", "--region", region],
        ["python", "03_filter_composite_times.py", "--region", region],
        ["python", "create_composites_pressure_levels_wt.py",  "--region", region],
        ["python", "create_composites_surface_flex.py",  "--region", region]
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
subprocess.run(["python", "run_comp_plots.py"])
print("All composites plotted")