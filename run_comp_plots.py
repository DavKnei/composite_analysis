#!/usr/bin/env python3

import subprocess

# Dictionary of regions and corresponding weather types
regions_and_weather_types = {
    "dinaric_alps": [0, 6, 9],
    "eastern_alps": [0, 4, 5, 6, 7, 9, 10, 16],
    "southern_alps": [0, 4, 5, 6, 9, 10, ],
    "western_alps": [0, 1, 2, 7, 9, 10]
}

# Loop over each region and weather type and run the script
for region, weather_types in regions_and_weather_types.items():
    for wt in weather_types:
        cmd = [
            "python",
            "plot_composites_full.py",
            "--region", region,
            "--weather_type", str(wt)
        ]
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd)