#!/usr/bin/env python3
"""
Plot pre-filtered (anomaly) OR raw ERA5 fields for every MCS-launch time,
coloured by weather-type. Optionally also store the mean field of every
weather-type (flag --add_mean, on by default).

Dynamically selects input data based on the --method argument and the
--plot_raw_fields flag. Assumes NetCDF files exist with predictable names
(e.g., ..._{region}.nc for raw, ..._filtered_{region}.nc for filtered).

Plot extent and contour levels are automatically determined from the input data.

* Weather-type table : …/weather_types/{method}_{region}_ncl{ncl}.csv
* MCS launch times   : …/composite_{region}_mcs.csv
* Input Data         : Defined in PLOT_SPECS, constructed using --datadir
                       and --plot_raw_fields flag.

Output → …/figures/{raw|filtered}/{region}/{method}/
"""

import argparse
import logging
import sys
from pathlib import Path
from multiprocessing import Pool
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs, cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker # For level generation

# --- Configuration: Map Method Name to Plotting Variables & Base File Patterns ---
# Structure: method_name: ( (var1_name, base_var1_pattern), (var2_name, base_var2_pattern) or None )
# Base patterns should contain '{region}' but NOT '_filtered'. This will be added dynamically.
# Ensure var names match those in the NetCDF files.
PLOT_SPECS: Dict[str, Tuple[Tuple[str, str], Optional[Tuple[str, str]]]] = {
    # --- Methods using MSL and Z500 ---
    "CAP_MSLZ500": (
        ("msl", "slp_daily_2001_2020_{region}.nc"),
        ("z500", "z500_daily_2001_2020_{region}.nc")
    ),
    # --- Methods using Z850 and Z500 ---
    "CAP_Z850Z500": (
        ("z850", "z850_daily_2001_2020_{region}.nc"),
        ("z500", "z500_daily_2001_2020_{region}.nc")
    ),
    # --- Single-level methods (Examples - add others as needed) ---
    "CAP_Z500": (
        ("z500", "z500_daily_2001_2020_{region}.nc"),
        None
    ),
    "CAP_Z850": (
        ("z850", "z850_daily_2001_2020_{region}.nc"),
        None
    ),
    "CAP_MSL": (
        ("msl", "slp_daily_2001_2020_{region}.nc"),
        None
    ),
     "GWT_MSL": (
        ("msl", "slp_daily_2001_2020_{region}.nc"),
        None
    ),
     "GWT_Z500": (
        ("z500", "z500_daily_2001_2020_{region}.nc"),
        None
     ),
      "GWTWS": ( # GWTWS uses raw MSL and Z500 according to docs
        ("msl", "slp_daily_2001_2020_{region}.nc"), # Should NOT be filtered for GWTWS
        ("z500", "z500_daily_2001_2020_{region}.nc"),# Should NOT be filtered for GWTWS
     ),
     "GWT_Z850": (
        ("z850", "z850_daily_2001_2020_{region}.nc"),
        None
    ),
     "LIT": ( # Uses SLP
        ("msl", "slp_daily_2001_2020_{region}.nc"),
        None
    ),
     "JCT": ( # Uses SLP
        ("msl", "slp_daily_2001_2020_{region}.nc"),
        None
    ),
    # Add other methods here
}

# ───────────── CLI ──────────────────────────────────────────────────────────
p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# --- Core Arguments ---
p.add_argument("--method", required=True, choices=PLOT_SPECS.keys(),
               help="Weather typing method name (must exist in PLOT_SPECS)")
p.add_argument("--region", required=True, help="Region name (e.g., eastern_alps)")
p.add_argument("--ncl", type=int, default=9, help="Number of classes")
# --- Input Data Arguments ---
p.add_argument("--datadir", type=Path, default="/home/dkn/ERA5",
               help="Directory containing the NetCDF files defined in PLOT_SPECS")
p.add_argument("--plot_raw_fields", action="store_true", default=False,
               help="Plot raw fields instead of filtered/anomaly fields.")
# --- Plotting Arguments ---
p.add_argument("--cores", type=int, default=32, help="Number of cores for parallel plotting")
p.add_argument("--serial", action="store_true", help="Run plotting serially")
p.add_argument("--add_mean", dest="add_mean", action="store_true",
               help="Also create mean composite per weather-type")
p.add_argument("--individual", dest="individual", action="store_true", help="Plot individual MCS datetimes.")
p.add_argument("--no-add-mean", dest="add_mean", action="store_false")
p.set_defaults(add_mean=True, individual=False)
args   = p.parse_args()

method = args.method
region = args.region
ncl    = args.ncl
is_raw = args.plot_raw_fields # Boolean flag for raw vs filtered

# --- Get Variables and File Patterns from PLOT_SPECS ---
try:
    spec_var1, spec_var2 = PLOT_SPECS[method]
    VAR1_NAME, pattern1_base = spec_var1

    VAR2_NAME, pattern2_base = (None, None)
    if spec_var2:
        VAR2_NAME, pattern2_base = spec_var2
except KeyError:
    sys.exit(f"Error: Method '{method}' not found in PLOT_SPECS configuration.")
except Exception as e:
    sys.exit(f"Error processing PLOT_SPECS for method '{method}': {e}")

# --- Construct Filenames based on raw/filtered flag ---
def get_filepath(base_pattern: str, is_raw_flag: bool, region_name: str) -> Path:
    """Constructs the full filepath, adding '_filtered' if needed."""
    p = Path(base_pattern.format(region=region_name))
    if not is_raw_flag:
        # Expect filtered file: insert '_filtered' before the suffix
        expected_file = p.with_stem(p.stem + "_filtered")
    else:
        # Expect raw file
        expected_file = p
    return args.datadir / expected_file

# Special handling for GWTWS which always uses raw data according to docs
if method == "GWTWS":
    logging.info("GWTWS method selected, forcing use of raw input files.")
    is_raw = True # Override flag for GWTWS

FILE_VAR1 = get_filepath(pattern1_base, is_raw, region)
FILE_VAR2 = get_filepath(pattern2_base, is_raw, region) if pattern2_base else None

logging.info(f"Plotting method: {method} ({'Raw' if is_raw else 'Filtered'} fields)")
logging.info(f"  Variable 1: {VAR1_NAME} from {FILE_VAR1}")
if VAR2_NAME and FILE_VAR2:
    logging.info(f"  Variable 2: {VAR2_NAME} from {FILE_VAR2}")

# ───────────── files / paths ────────────────────────────────────────────────
# --- Weather Type and MCS files ---
WT_CSV = Path(f"/nas/home/dkn/Desktop/MoCCA/composites/scripts/preprocess/"
              f"weather_types/{method}_{region}_ncl{ncl}.csv")
MCS_CSV = Path(f"/nas/home/dkn/Desktop/MoCCA/composites/scripts/csv/"
               f"composite_{region}_mcs.csv")

# --- Output Directory ---
out_subdir = "raw" if is_raw else "filtered"
out_root = Path("/nas/home/dkn/Desktop/MoCCA/composites/figures/weather_types")
OUT_DIR  = out_root / out_subdir / region / method # Add raw/filtered and method
OUT_DIR.mkdir(parents=True, exist_ok=True)
logging.info(f"Output directory: {OUT_DIR}")

# --- Check Input Files Exist ---
input_files_ok = True
if not FILE_VAR1.exists():
    logging.error(f"Input file for Variable 1 not found: {FILE_VAR1}")
    input_files_ok = False
if FILE_VAR2 and not FILE_VAR2.exists():
    logging.warning(f"Input file for Variable 2 ({VAR2_NAME}) not found: {FILE_VAR2}. Will plot Variable 1 only.")
    # Allow plotting Var1 only if Var2 is missing
    FILE_VAR2 = None
    VAR2_NAME = None

if not input_files_ok:
     sys.exit("Essential input file(s) are missing. Exiting.")


OPEN_KW = dict(chunks="auto", decode_times=True, engine="netcdf4")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")

# --- Determine Plot Extent Dynamically ---
plot_extent = None
try:
    # Open the first primary variable file to get coordinates
    logging.info(f"Determining plot extent from: {FILE_VAR1}")
    with xr.open_dataset(FILE_VAR1, **OPEN_KW) as ds_extent:
        # Find coordinate names (more robust than assuming 'lat', 'lon')
        lon_coord_name = None
        lat_coord_name = None
        for coord_name in ds_extent.coords:
            if 'lon' in coord_name.lower():
                lon_coord_name = coord_name
            elif 'lat' in coord_name.lower():
                lat_coord_name = coord_name
        if not lon_coord_name or not lat_coord_name:
             raise ValueError("Could not automatically determine latitude/longitude coordinate names.")

        lon_min = float(ds_extent[lon_coord_name].min())
        lon_max = float(ds_extent[lon_coord_name].max())
        lat_min = float(ds_extent[lat_coord_name].min())
        lat_max = float(ds_extent[lat_coord_name].max())
        plot_extent = [lon_min, lon_max, lat_min, lat_max]
        logging.info(f"Determined plot extent from data: {plot_extent}")
except Exception as e:
    logging.error(f"Could not determine plot extent from {FILE_VAR1}: {e}")
    sys.exit("Exiting due to failure in reading data extent.")


# ───────────── helpers & caching ────────────────────────────────────────────
# Cache for loaded datasets to avoid reopening files repeatedly
_data_cache = {}

def load_fields(dt: pd.Timestamp,
                 file1: Path, var1_name: str,
                 file2: Optional[Path], var2_name: Optional[str]
                 ) -> Tuple[Optional[xr.DataArray], Optional[xr.DataArray]]:
    """
    Return fields for the given datetime from specified files.
    Uses a simple cache based on filename. Handles different variable names.
    Performs necessary unit conversions.
    """
    dt = pd.to_datetime(dt)
    data1 = None
    data2 = None

    # --- Load Variable 1 ---
    if file1:
        if file1 not in _data_cache:
            logging.debug(f"Loading data from: {file1}")
            try:
                 _data_cache[file1] = xr.open_dataset(file1, **OPEN_KW)
            except Exception as e:
                 logging.error(f"Error loading {file1}: {e}")
                 _data_cache[file1] = None # Mark as failed

        if _data_cache[file1] is not None:
            try:
                # Check if variable exists in the dataset
                if var1_name not in _data_cache[file1]:
                     raise KeyError(f"Variable '{var1_name}' not found in {file1}. Available: {list(_data_cache[file1].data_vars)}")
                data1 = _data_cache[file1][var1_name].sel(time=dt, method="nearest")

                # --- Unit Conversion / Level Selection for Variable 1 (if needed) ---
                units1 = data1.attrs.get('units', '').lower()
                # Convert MSL from Pa to hPa if needed
                if var1_name == 'msl' and 'pa' in units1 and 'hpa' not in units1:
                     logging.debug(f"Converting {var1_name} units Pa -> hPa for {dt}")
                     data1 = data1 / 100.0
                     data1.attrs['units'] = 'hPa' # Update units attribute
                # Convert Z from m^2/s^2 to m if needed
                elif 'z' in var1_name.lower() and ('m2 s-2' in units1 or var1_name == 'z'):
                     if 'm' not in data1.attrs.get('units', '').lower():
                         logging.debug(f"Converting {var1_name} units m^2/s^2 -> m for {dt}")
                         data1 = data1 / 9.80665 # Standard gravity
                         data1.attrs['units'] = 'm'

            except Exception as e:
                logging.warning(f"Could not select/process time {dt} for {var1_name} from {file1}: {e}")
                data1 = None # Ensure data is None if selection fails

    # --- Load Variable 2 ---
    if file2 and var2_name: # Only proceed if file2 and var2_name are defined
        if file2 not in _data_cache:
            logging.debug(f"Loading data from: {file2}")
            try:
                _data_cache[file2] = xr.open_dataset(file2, **OPEN_KW)
            except Exception as e:
                 logging.error(f"Error loading {file2}: {e}")
                 _data_cache[file2] = None # Mark as failed

        if _data_cache[file2] is not None:
            try:
                # Check if variable exists in the dataset
                if var2_name not in _data_cache[file2]:
                    raise KeyError(f"Variable '{var2_name}' not found in {file2}. Available: {list(_data_cache[file2].data_vars)}")

                # Check for level dimension if variable is geopotential height
                if 'z' in var2_name.lower() and 'level' in _data_cache[file2][var2_name].dims:
                     level_to_select = 500 # Default Z500
                     if var2_name == 'z850': level_to_select = 850
                     elif var2_name == 'z500': level_to_select = 500
                     # Add more elifs if other 'z' levels are used
                     data2 = _data_cache[file2][var2_name].sel(level=level_to_select, time=dt, method="nearest")
                else:
                     # Assume level dimension was removed or not present
                     data2 = _data_cache[file2][var2_name].sel(time=dt, method="nearest")

                # --- Unit Conversion for Variable 2 (if needed) ---
                units2 = data2.attrs.get('units', '').lower()
                if 'z' in var2_name.lower() and ('m2 s-2' in units2 or var2_name == 'z'):
                     if 'm' not in data2.attrs.get('units', '').lower():
                         logging.debug(f"Converting {var2_name} units m^2/s^2 -> m for {dt}")
                         data2 = data2 / 9.80665 # Standard gravity
                         data2.attrs['units'] = 'm'

            except Exception as e:
                logging.warning(f"Could not select/process time {dt} for {var2_name} from {file2}: {e}")
                data2 = None # Ensure data is None if selection fails

    return data1, data2


def setup_axis(ax, extent):
    """Configure map projection and features using provided extent."""
    if extent is None:
        logging.warning("Plot extent is None in setup_axis, using global.")
        ax.set_global() # Fallback if extent couldn't be determined
    else:
        # Add a small buffer to the extent for better visualization
        lon_buffer = (extent[1] - extent[0]) * 0.02
        lat_buffer = (extent[3] - extent[2]) * 0.02
        buffered_extent = [extent[0] - lon_buffer, extent[1] + lon_buffer,
                           extent[2] - lat_buffer, extent[3] + lat_buffer]
        # Ensure buffer doesn't push lat beyond +/- 90
        buffered_extent[2] = max(-90, buffered_extent[2])
        buffered_extent[3] = min(90, buffered_extent[3])
        ax.set_extent(buffered_extent, ccrs.PlateCarree())
    ax.coastlines("10m", lw=.6, color='gray') # Use gray for better contrast
    ax.add_feature(cfeature.BORDERS, lw=.4, edgecolor='gray')

def calculate_levels(data_array: xr.DataArray, num_levels: int = 11, centered: bool = True, step: Optional[float] = None) -> np.ndarray:
    """
    Calculates contour levels dynamically based on data range. Handles NaNs.
    """
    if data_array is None or data_array.size == 0 or data_array.isnull().all():
        logging.warning("Cannot calculate levels: Data is None, empty, or all NaN.")
        return np.array([]) # Return empty if no valid data

    # Compute min/max, handling potential all-NaN slices
    min_val = float(data_array.min(skipna=True).compute())
    max_val = float(data_array.max(skipna=True).compute())

    if not np.isfinite(min_val) or not np.isfinite(max_val):
        logging.warning("Could not determine valid min/max for levels (all NaN?). Returning empty levels.")
        return np.array([])

    if np.isclose(min_val, max_val):
        # Handle constant field case: create a small range around the value
        logging.debug(f"Data is nearly constant ({min_val}). Creating small level range.")
        center_val = min_val
        spread = max(abs(center_val * 0.1), 0.1) # 10% spread or at least 0.1
        min_val = center_val - spread
        max_val = center_val + spread
        centered = False # Don't force centering if data was constant

    if centered:
        max_abs = max(abs(min_val), abs(max_val))
        if max_abs == 0: # Handle case where data is zero everywhere
             levels = np.linspace(-0.1, 0.1, 3) # Small range around zero
        elif step:
            # Ensure range covers [-max_abs, max_abs] with the given step
            limit = np.ceil(max_abs / step) * step
            levels = np.arange(-limit, limit + step/2, step)
            # Ensure 0 is included if step allows
            if 0 not in levels and np.sign(levels.min()) != np.sign(levels.max()):
                 levels = np.sort(np.append(levels, 0))
        else:
            # Use MaxNLocator for nice symmetric levels around 0
            locator = mticker.MaxNLocator(nbins=num_levels, symmetric=True)
            levels = locator.tick_values(-max_abs, max_abs)
    else: # Not centered (e.g., raw data)
        if step:
             levels = np.arange(np.floor(min_val/step)*step, (np.ceil(max_val/step)+1)*step, step)
        else:
            # Use MaxNLocator for non-centered levels
            locator = mticker.MaxNLocator(nbins=num_levels)
            levels = locator.tick_values(min_val, max_val)

    # Ensure levels are unique and sorted
    levels = np.unique(levels)
    # Ensure at least two levels exist for contouring
    if len(levels) < 2:
         logging.warning(f"Only {len(levels)} unique level(s) calculated. Creating default range.")
         levels = np.linspace(min_val, max_val, 3) # Fallback to 3 levels

    logging.debug(f"Calculated levels: min={min_val:.2f}, max={max_val:.2f}, levels={levels}")
    return levels


# ───────────── plotting per event ───────────────────────────────────────────
def plot_one(row_tuple):
    """
    Plots a single MCS launch event.
    Accepts a tuple: (row, var1_name, file1, var2_name, file2, plot_extent, is_raw_flag)
    """
    row, var1_name, file1, var2_name, file2, current_plot_extent, is_raw_flag = row_tuple # Unpack tuple
    dt = pd.to_datetime(row.datetime);  wt = int(row.wt)
    plot_type_str = "raw" if is_raw_flag else "filtered"
    fname = OUT_DIR / f"wt{wt:02d}_{method}_{region}_{dt:%Y%m%d%H}_{plot_type_str}.png"
    if fname.exists():
        return

    data1, data2 = load_fields(dt, file1, var1_name, file2, var2_name)

    # Check if data loading failed
    if data1 is None and data2 is None:
        logging.error(f"Failed to load any data for time {dt}. Skipping plot.")
        return
    elif data1 is None and var2_name: # Check if var2 was expected
        logging.warning(f"{var1_name} data missing for time {dt}. Plotting {var2_name} only.")
    elif data2 is None and var2_name: # Check if var2 was expected
        logging.warning(f"{var2_name} data missing for time {dt}. Plotting {var1_name} only.")

    proj = ccrs.PlateCarree();  fig = plt.figure(figsize=(10,6))
    ax = plt.axes(projection=proj); setup_axis(ax, current_plot_extent) # Pass extent

    # --- Plotting Variable 1 (Contour Fill) ---
    cf = None # Initialize contour fill object
    if data1 is not None:
        var1_upper = var1_name.upper()
        # Find coordinate names dynamically
        lon_coord_name1 = next((c for c in data1.coords if 'lon' in c.lower()), None)
        lat_coord_name1 = next((c for c in data1.coords if 'lat' in c.lower()), None)
        if not lon_coord_name1 or not lat_coord_name1:
            logging.error(f"Could not find lat/lon coordinates for variable 1 ({var1_name}) at {dt}. Skipping plot.")
            plt.close(fig)
            return

        # Adapt colormap and centering based on raw/filtered
        cmap = "coolwarm" if is_raw_flag else "bwr"
        centered_levels = not is_raw_flag
        units_label = data1.attrs.get('units', '')
        plot_data1 = data1

        # Specific handling for known variables
        if 'msl' in var1_name.lower() or 'slp' in var1_name.lower():
            units_label = "hPa"
            plot_data1 = data1 # Assumes already in hPa
        elif 'z' in var1_name.lower(): # Covers z500, z850 etc.
            units_label = "m"
            plot_data1 = data1 # Assumes already in meters
        else: # Default fallback for unknown variables
            logging.warning(f"Using default cmap/centering for unknown variable 1: {var1_name}")
            cmap = "viridis"
            centered_levels = False

        # Calculate dynamic levels
        levels1 = calculate_levels(plot_data1, num_levels=13, centered=centered_levels)

        if len(levels1) > 0:
            cf = ax.contourf(plot_data1[lon_coord_name1], plot_data1[lat_coord_name1], plot_data1,
                             levels=levels1, cmap=cmap,
                             extend="both", transform=proj)
            cb = fig.colorbar(cf, orientation="horizontal", pad=0.03, aspect=40, shrink=0.8)
            label_suffix = "filtered anomaly" if not is_raw_flag else "raw"
            cb.set_label(f"{var1_upper} {label_suffix} [{units_label}]")
        else:
            logging.warning(f"Could not generate levels for {var1_name} at {dt}. Skipping contour fill.")

    else:
         ax.set_title(f"WT {wt} - {dt:%Y-%m-%d %H:%M} ({var1_name} data missing)", fontsize=10)


    # --- Plotting Variable 2 (Contours) ---
    cz = None # Initialize contour object
    if data2 is not None and var2_name is not None: # Ensure var2 was expected and loaded
        var2_upper = var2_name.upper()
        # Find coordinate names dynamically
        lon_coord_name2 = next((c for c in data2.coords if 'lon' in c.lower()), None)
        lat_coord_name2 = next((c for c in data2.coords if 'lat' in c.lower()), None)
        if not lon_coord_name2 or not lat_coord_name2:
            logging.error(f"Could not find lat/lon coordinates for variable 2 ({var2_name}) at {dt}. Skipping contour plot.")
        else:
            plot_data2 = data2 # Assume units already handled in load_filtered_fields
            # Contours are usually not centered, even for anomalies, unless specified
            centered_levels2 = False
            # Calculate dynamic levels for contours (maybe fewer levels)
            levels2 = calculate_levels(plot_data2, num_levels=11, centered=centered_levels2)

            if len(levels2) > 0:
                cz = ax.contour(plot_data2[lon_coord_name2], plot_data2[lat_coord_name2], plot_data2,
                                levels=levels2, colors="k", linewidths=1, transform=proj)
                if len(levels2) > 1: # Check if clabel can be drawn
                     ax.clabel(cz, fmt="%.0f", inline=True, fontsize=8)
            else:
                 logging.warning(f"Could not generate levels for {var2_name} at {dt}. Skipping contours.")

            # Add title only if Var1 is missing
            if data1 is None:
                ax.set_title(f"WT {wt} - {dt:%Y-%m-%d %H:%M} ({var2_name} data missing)", fontsize=10)

    # General title if both fields plotted or only Var1 plotted
    if data1 is not None:
         title_suffix = "(Raw Data)" if is_raw_flag else "(Filtered Data)"
         title_str = f"Weather Type {wt} - {dt:%Y-%m-%d %H:%M} ({method} - {title_suffix})"
         ax.set_title(title_str, fontsize=10)
    # Case where only Var2 is plotted is handled above


    try:
        fig.savefig(fname, dpi=150, bbox_inches="tight");
        logging.info("saved %s", fname.name)
    except Exception as e:
        logging.error(f"Failed to save figure {fname.name}: {e}")
    finally:
        plt.close(fig) # Ensure figure is closed

# ───────────── mean composite per WT  (optional) ────────────────────────────
def mean_composite(wt: int, dts: List[pd.Timestamp],
                   var1_name: str, file1: Path,
                   var2_name: Optional[str], file2: Optional[Path],
                   current_plot_extent: List[float], is_raw_flag: bool): # Pass extent and raw flag
    """Calculates and plots the mean composite for a weather type."""
    plot_type_str = "raw" if is_raw_flag else "filtered"
    out = OUT_DIR / f"mean_wt{wt:02d}_{method}_{region}_{plot_type_str}.png"
    if out.exists():
        return

    data1_sum, data2_sum = None, None
    count = 0
    logging.info(f"Calculating mean composite for WT {wt} ({len(dts)} events)...")
    # Store first valid data arrays to get coordinates later
    first_data1, first_data2 = None, None

    for dt in dts:
        data1, data2 = load_fields(dt, file1, var1_name, file2, var2_name)

        # Determine if this event should be included (depends on whether we need 1 or 2 vars)
        include_event = False
        if var2_name is None: # Only need var1
            if data1 is not None:
                include_event = True
                if first_data1 is None: first_data1 = data1 # Store first valid data1
        else: # Need both var1 and var2
            if data1 is not None and data2 is not None:
                include_event = True
                if first_data1 is None: first_data1 = data1
                if first_data2 is None: first_data2 = data2 # Store first valid data2

        if include_event:
            # --- Accumulate Var 1 ---
            if data1 is not None:
                if data1_sum is None:
                    data1_sum = data1.copy(data=data1.data.astype("float64"))
                elif data1_sum.shape == data1.shape:
                    data1_sum.data += data1.data.astype("float64")
                else:
                    logging.warning(f"Shape mismatch for {var1_name} in WT {wt} at time {dt}. Skipping accumulation.")
                    include_event = False # Don't count if shapes mismatch

            # --- Accumulate Var 2 ---
            if data2 is not None and var2_name is not None: # Only if var2 is expected and loaded
                if data2_sum is None:
                     data2_sum = data2.copy(data=data2.data.astype("float64"))
                elif data2_sum.shape == data2.shape:
                    data2_sum.data += data2.data.astype("float64")
                else:
                    logging.warning(f"Shape mismatch for {var2_name} in WT {wt} at time {dt}. Skipping accumulation.")
                    include_event = False # Don't count if shapes mismatch

            # Increment count only if accumulation was successful for required vars
            if include_event:
                count += 1
        else:
            logging.warning(f"Skipping event {dt} for WT {wt} mean due to missing required data.")


    if count == 0:
        logging.error(f"No valid events found to calculate mean for WT {wt}. Skipping composite plot.")
        return
    if first_data1 is None and first_data2 is None:
         logging.error(f"Could not get coordinate information for WT {wt}. Skipping composite plot.")
         return

    # --- Calculate Mean ---
    data1_mean = data1_sum / count if data1_sum is not None else None
    data2_mean = data2_sum / count if data2_sum is not None else None

    # --- Plot Mean Composite ---
    proj = ccrs.PlateCarree();  fig = plt.figure(figsize=(10,6))
    ax = plt.axes(projection=proj);  setup_axis(ax, current_plot_extent) # Pass extent

    # --- Plotting Mean Variable 1 (Contour Fill) ---
    cf = None
    if data1_mean is not None:
        var1_upper = var1_name.upper()
        # Find coordinate names dynamically from the first valid data array
        lon_coord_name1 = None
        lat_coord_name1 = None
        ref_data_for_coords1 = first_data1 if first_data1 is not None else data1_mean # Fallback
        for coord_name in ref_data_for_coords1.coords:
            if 'lon' in coord_name.lower(): lon_coord_name1 = coord_name
            elif 'lat' in coord_name.lower(): lat_coord_name1 = coord_name
        if not lon_coord_name1 or not lat_coord_name1:
             logging.error(f"Could not find lat/lon coordinates for mean variable 1 ({var1_name}). Skipping fill plot.")
             data1_mean = None # Prevent plotting
        else:
            cmap = "coolwarm" if is_raw_flag else "bwr"
            centered_levels = not is_raw_flag
            units_label = data1_mean.attrs.get('units', '')
            plot_data1_mean = data1_mean

            if 'msl' in var1_name.lower() or 'slp' in var1_name.lower():
                units_label = "hPa"
                if not is_raw_flag: cmap = "bwr" # Ensure diverging for anomaly
                else: cmap = "coolwarm" # Sequential for raw
            elif 'z' in var1_name.lower():
                units_label = "m"
                if not is_raw_flag: cmap = "bwr"
                else: cmap = "viridis" # Sequential for raw Z
            else: # Default fallback
                logging.warning(f"Using default cmap/centering for mean of unknown variable 1: {var1_name}")
                cmap = "viridis"
                centered_levels = False

            # Calculate dynamic levels for the mean
            levels1 = calculate_levels(plot_data1_mean, num_levels=13, centered=centered_levels)

            if len(levels1) > 0:
                cf = ax.contourf(plot_data1_mean[lon_coord_name1], plot_data1_mean[lat_coord_name1], plot_data1_mean,
                                 levels=levels1, cmap=cmap,
                                 extend="both", transform=proj)
                cb = fig.colorbar(cf, orientation="horizontal", pad=0.03, aspect=40, shrink=0.8)
                label_suffix = "filtered anomaly" if not is_raw_flag else "raw"
                cb.set_label(f"Mean {var1_upper} {label_suffix} [{units_label}]")
            else:
                logging.warning(f"Could not generate levels for mean {var1_name}. Skipping contour fill.")


    # --- Plotting Mean Variable 2 (Contours) ---
    cz = None
    if data2_mean is not None and var2_name is not None:
        var2_upper = var2_name.upper()
        # Find coordinate names dynamically from the first valid data array
        lon_coord_name2 = None
        lat_coord_name2 = None
        ref_data_for_coords2 = first_data2 if first_data2 is not None else data2_mean # Fallback
        for coord_name in ref_data_for_coords2.coords:
             if 'lon' in coord_name.lower(): lon_coord_name2 = coord_name
             elif 'lat' in coord_name.lower(): lat_coord_name2 = coord_name
        if not lon_coord_name2 or not lat_coord_name2:
             logging.error(f"Could not find lat/lon coordinates for mean variable 2 ({var2_name}). Skipping contour plot.")
             data2_mean = None # Prevent plotting
        else:
            plot_data2_mean = data2_mean
            # Contours are usually not centered
            centered_levels2 = False
            # Calculate dynamic levels for the mean contours
            levels2 = calculate_levels(plot_data2_mean, num_levels=11, centered=centered_levels2)

            if len(levels2) > 0:
                cz = ax.contour(plot_data2_mean[lon_coord_name2], plot_data2_mean[lat_coord_name2], plot_data2_mean,
                                levels=levels2, colors="k", linewidths=1, transform=proj)
                if len(levels2) > 1: # Check if clabel can be drawn
                    ax.clabel(cz, fmt="%.0f", inline=True, fontsize=8)
            else:
                logging.warning(f"Could not generate levels for mean {var2_name}. Skipping contours.")


    # --- Set Title ---
    title_suffix = "(Raw Data)" if is_raw_flag else "(Filtered Data)"
    ax.set_title(f"Mean Weather Type {wt} (n={count}) - {method} {title_suffix}", fontsize=10)

    try:
        fig.savefig(out, dpi=150, bbox_inches="tight");
        logging.info("saved %s (mean composite)", out.name)
    except Exception as e:
         logging.error(f"Failed to save mean composite figure {out.name}: {e}")
    finally:
        plt.close(fig) # Ensure figure is closed


# ───────────── main ────────────────────────────────────────────────────────
def main():
    # --- Load Weather Type and MCS Data ---
    if not WT_CSV.exists():
        sys.exit(f"Weather type file not found: {WT_CSV}")
    if not MCS_CSV.exists():
        sys.exit(f"MCS file not found: {MCS_CSV}")

    wt_full  = pd.read_csv(WT_CSV,  parse_dates=["datetime"])
    mcs = pd.read_csv(MCS_CSV, parse_dates=["time_0h"])

    # --- Filter data (Example: JJA months and match times) ---
    # Filter mcs for June, July, and August
    mcs_JJA = mcs[mcs['time_0h'].dt.month.isin([6, 7, 8])]
    # Filter wt for datetimes that are in mcs_JJA
    # Use .copy() to avoid SettingWithCopyWarning later
    df = wt_full[wt_full['datetime'].isin(mcs_JJA['time_0h'])].copy()
    # Or, if you want all times from the WT file:
    # df = wt_full.copy() # Use copy here too for safety if modifying later
    logging.info(f"Loaded {len(wt_full)} weather type entries from {WT_CSV.name}.")
    logging.info(f"Processing {len(df)} events for region '{region}', method '{method}' (Filtered by MCS JJA times)")


    # --- Prepare data for plotting ---
    # Ensure 'wt' column is integer (do this *after* filtering and copying)
    try:
        # Use .loc to ensure modification happens on the DataFrame directly
        df.loc[:, 'wt'] = df['wt'].astype(int)
    except KeyError:
        sys.exit(f"Column 'wt' not found in {WT_CSV}")
    except ValueError:
         sys.exit(f"Column 'wt' in {WT_CSV} contains non-integer values.")

    # Create list of tuples for Pool.map: (row, var1_name, file1, var2_name, file2, plot_extent, is_raw)
    map_args = [(r, VAR1_NAME, FILE_VAR1, VAR2_NAME, FILE_VAR2, plot_extent, is_raw)
                for _, r in df.iterrows()]

    # --- Plot individual events ---
    if args.individual:
        if args.serial or args.cores == 1:
            logging.info("Running individual plotting serially...")
            for map_arg_tuple in map_args:
                plot_one(map_arg_tuple)
        else:
            logging.info(f"Running individual plotting in parallel with {args.cores} cores...")
            # Use Pool context manager
            with Pool(args.cores) as P:
                # Use imap_unordered for potentially better progress reporting/memory use
                results = list(P.imap_unordered(plot_one, map_args))
    else:
        logging.info("Skipping individual plots (--individual not specified).")


    # --- Plot mean composites ---
    if args.add_mean:
        logging.info("Calculating and plotting mean composites...")
        # Group by weather type ID
        for wt_id, grp in df.groupby("wt"):
            # Pass the list/array of datetime objects and file/var info and extent and raw flag
            mean_composite(int(wt_id), grp['datetime'].to_list(),
                           VAR1_NAME, FILE_VAR1, VAR2_NAME, FILE_VAR2,
                           plot_extent, is_raw) # Pass extent and raw flag
    else:
        logging.info("Skipping mean composite plots (--add_mean not specified or --no-add-mean used).")

    logging.info("Plotting finished.")
    # Clear cache if desired (optional)
    _data_cache.clear()


if __name__ == "__main__":
    main()
