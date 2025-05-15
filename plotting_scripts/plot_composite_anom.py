#!/usr/bin/env python3
"""
Plot composites of ANOMALIES (mean - climatology mean) for surface and
pressure level variables, stratified by weather type, month, and time offset,
mimicking the layout of plot_composites_full.py (excluding precipitation).

Reads pre-computed composite NetCDF files containing both the mean composite
field (<var>_mean) and the mean climatology field (<var>_clim_mean) for
each group.

Layout (2x2):
  Top-Left:    Theta-e anomaly @ 850hPa (-12h) + Z500 anom contours + Wind anom vectors
  Bottom-Left: Theta-e anomaly @ 850hPa (  0h) + Z500 anom contours + Wind anom vectors
  Top-Right:   Specific Humidity (q) anomaly @ 850hPa (0h) + Z500 anom contours
  Bottom-Right:Temperature (t) anomaly @ 850hPa (0h) + Z500 anom contours

Usage:
    python plot_composites_anom.py --region eastern_alps \\
        --comp_dir /home/dkn/composites/ERA5 \\
        --output_dir ./plots_composites_anom/ \\
        --period evaluation \\
        [--weather_types 1,2,3] [--time_offsets -12,0] # Ensure -12 and 0 are included

Author: David Kneidinger
Date: 2025-05-08
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle # For subregion box
from matplotlib.gridspec import GridSpec
from typing import Tuple, Optional, Dict, List, Any

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

VARS_TO_CALC_ANOM = {
    'msl': None, 'z': 500, 'theta_e': 850, 'q': 850,
    't': 850, 'u': 850, 'v': 850
}

def create_custom_cmap():
    colors_list = [
        (0.0, 'darkblue'), (0.2, 'blue'), (0.45, 'white'),
        (0.7, 'orange'), (1.0, 'darkred')
    ]
    return mcolors.LinearSegmentedColormap.from_list("red_white_blue", colors_list, N=256)

ANOM_PLOT_CONFIG = {
    'msl': {'cmap': 'RdBu_r', 'unit': 'Pa'},
    'z500': {'cmap': 'RdBu_r', 'unit': 'm'}, # Fixed for Z contours
    'theta_e850': {'cmap': create_custom_cmap(), 'unit': 'K'},
    'q850': {'cmap': 'BrBG', 'unit': 'g/kg'}, # Symmetric for q850
    't850': {'cmap': 'coolwarm', 'unit': 'K'},
}

PERIODS = {
    "historical": {"start": 1996, "end": 2005, "name_in_file": "historical"},
    "evaluation": {"start": 2000, "end": 2009, "name_in_file": "evaluation"}
}
WIND_LEVEL = 850
WIND_STRIDE = 8
TARGET_MONTHS = [6, 7, 8] # JJA
MAP_PROJ = ccrs.PlateCarree()
MAP_EXTENT = [-10, 25, 35, 52]
SUBREGIONS = {
    'western_alps': {'lon_min': 3, 'lon_max': 8, 'lat_min': 43, 'lat_max': 49},
    'southern_alps': {'lon_min': 8, 'lon_max': 13, 'lat_min': 43, 'lat_max': 46},
    'dinaric_alps': {'lon_min': 13, 'lon_max': 20, 'lat_min': 42, 'lat_max': 46},
    'eastern_alps': {'lon_min': 8, 'lon_max': 17, 'lat_min': 46, 'lat_max': 49}
}
REQUIRED_OFFSETS = [-12, 0]

def get_composite_filenames(comp_dir: Path, region: str, period: str) -> Tuple[Path, Path]:
    plev_file = comp_dir / f"composite_plev_{region}_wt_clim_{period}.nc"
    surf_file = comp_dir / f"composite_surface_{region}_msl_wt_clim_{period}.nc"
    return plev_file, surf_file

def calculate_anomaly(ds: xr.Dataset, var_name: str) -> Optional[xr.DataArray]:
    mean_var, clim_mean_var = f"{var_name}_mean", f"{var_name}_clim_mean"
    if mean_var in ds and clim_mean_var in ds:
        anomaly = (ds[mean_var] - ds[clim_mean_var]).load()
        anomaly.attrs['long_name'] = f"{ds[mean_var].attrs.get('long_name', var_name)} Anomaly"
        anomaly.attrs['units'] = ds[mean_var].attrs.get('units', 'unknown')
        if var_name == 'z':
            anomaly = anomaly / 9.80665
            anomaly.attrs.update({'long_name': "Geopotential Height Anomaly", 'units': 'm'})
        elif var_name == 'q':
            anomaly = anomaly * 1000.0
            anomaly.attrs.update({'long_name': "Specific Humidity Anomaly", 'units': 'g/kg'})
        elif var_name == 't':
             anomaly.attrs.update({'long_name': "Temperature Anomaly", 'units': 'K'})
        return anomaly
    logging.warning(f"Fields for anomaly of '{var_name}' not found."); return None

def add_subregion_box(ax: plt.Axes, region: str):
    if region in SUBREGIONS:
        b = SUBREGIONS[region]
        rect = Rectangle((b['lon_min'], b['lat_min']), b['lon_max']-b['lon_min'], b['lat_max']-b['lat_min'],
                         lw=1.5, ec='red', fc='none', transform=MAP_PROJ, zorder=10)
        ax.add_patch(rect)
    else: logging.warning(f"Subregion '{region}' undefined.");

def lamb_weather_type_label(wt: int) -> str:
    if wt == 0: return "All"
    return {1:"W", 2:"SW", 3:"NW", 4:"N", 5:"NE", 6:"E", 7:"SE", 8:"S", 9:"C", 10:"A"}.get(wt, f"Unk WT{wt}")


# MODIFIED: Removed count_data, count_z500, count_wind parameters
def plot_anomaly_panel(ax: plt.Axes, lon: np.ndarray, lat: np.ndarray,
                       field_anom: xr.DataArray, 
                       z500_anom: Optional[xr.DataArray],
                       wind_u_anom: Optional[xr.DataArray], wind_v_anom: Optional[xr.DataArray],
                       wind_stride: int, region: str,
                       plot_config_key: str, 
                       title_base: str):

    current_plot_config = ANOM_PLOT_CONFIG[plot_config_key]

    if 'var_name_for_debug' not in current_plot_config:
        current_plot_config['var_name_for_debug'] = plot_config_key

    title = title_base
    logging.debug(f"Plotting panel: {title} with config key: {plot_config_key}")
    cmap = current_plot_config['cmap']
    
    # MODIFIED: No more masking. Plot the raw anomaly data.
    # field_anom_masked = field_anom.where(count_data >= 5) # REMOVED
    field_to_plot = field_anom # Use directly
    
    field_min = np.floor(np.min(field_anom))
    field_max = np.ceil(np.max(field_anom))
    if np.abs(field_max) > np.abs(field_min):
        field_min = -field_max
    else:
        field_max = -field_min
    
    delta = np.ceil((field_max - field_min)/ 12)
    levels = np.arange(field_min, field_max + delta, delta)
    if len(levels) < 7:
        levels = np.arange(field_min, field_max + 0.5, 0.5)
    
    norm = mcolors.BoundaryNorm(levels, ncolors=plt.get_cmap(cmap).N, clip=False)
    cf = ax.contourf(lon, lat, field_to_plot, levels=levels, cmap=cmap, norm=norm, extend='both', transform=MAP_PROJ)

    if z500_anom is not None:
        # z500_masked = z500_anom.where(count_z500 >= 5) # REMOVED
        z500_to_plot = z500_anom # Use directly
        z_min = np.floor(np.min(z500_anom))
        z_max = np.ceil(np.max(z500_anom))
        delta_z = np.ceil((z_max - z_min)/ 11)
        z500_levels = np.arange(z_min, z_max, delta_z)

        cs = ax.contour(lon, lat, z500_to_plot, levels=z500_levels, colors='black', linewidths=0.8, linestyles='solid', transform=MAP_PROJ)
        ax.clabel(cs, z500_levels, inline=True, fontsize=8, fmt='%d')

    if wind_u_anom is not None and wind_v_anom is not None:
        # u_anom_masked = wind_u_anom.where(count_wind >= 5) # REMOVED
        # v_anom_masked = wind_v_anom.where(count_wind >= 5) # REMOVED
        u_to_plot = wind_u_anom # Use directly
        v_to_plot = wind_v_anom # Use directly

        lon_sub, lat_sub = lon[::wind_stride], lat[::wind_stride]
        u_sub, v_sub = u_to_plot[::wind_stride, ::wind_stride], v_to_plot[::wind_stride, ::wind_stride]
        if np.any(np.isfinite(u_sub) & np.isfinite(v_sub)):
            qk = ax.quiver(lon_sub, lat_sub, u_sub.data, v_sub.data, scale=None, angles='xy', color='dimgray',
                           width=0.006, headwidth=4, headlength=6, headaxislength=5.5,
                           minshaft=1.5, minlength=2, transform=MAP_PROJ, pivot='mid', zorder=5)
            ref_wind_val = 2.0
            ax.quiverkey(qk, X=0.85, Y=-0.08, U=ref_wind_val, label=f'{ref_wind_val:.0f} m/s',
                         labelpos='E', coordinates='axes', fontproperties={'size': 8})
        else: logging.debug(f"Skipping wind vectors for panel '{title}' (no valid data after subsampling).")

    ax.coastlines(resolution='50m', color='black', linewidth=0.5, zorder=4)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, zorder=4)
    gl = ax.gridlines(draw_labels=True, lw=0.5, color='gray', alpha=0.5, ls='--')
    gl.top_labels=False; gl.right_labels=False
    gl.xlabel_style={'size':8}; gl.ylabel_style={'size':8}
    add_subregion_box(ax, region)
    ax.set_title(title, fontsize=9)
    ax.set_extent(MAP_EXTENT, crs=MAP_PROJ)
    return cf

def main():
    parser = argparse.ArgumentParser(description="Plot 2x2 JJA composite anomalies.")
    parser.add_argument("--region",type=str,required=True)
    parser.add_argument("--comp_dir",type=Path,default='/home/dkn/composites/ERA5/')
    parser.add_argument("--output_dir",type=Path,default='./plots_composites_anom/')
    parser.add_argument("--period",type=str,default="evaluation",choices=PERIODS.keys())
    parser.add_argument("--weather_types",type=str,default="0,1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--time_offsets",type=str,default="-12,0")
    parser.add_argument("--debug",action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format='%(asctime)s-%(levelname)s-%(funcName)s:%(lineno)d - %(message)s', force=True)
    logging.info(f"--- Starting Composite Anomaly Plotting --- Args: {args}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        wts_to_plot = [int(wt.strip()) for wt in args.weather_types.split(',')]
        available_offsets_arg = [int(off.strip()) for off in args.time_offsets.split(',')]
        if not set(REQUIRED_OFFSETS).issubset(set(available_offsets_arg)):
             logging.warning(f"Required offsets {REQUIRED_OFFSETS} not in arg {available_offsets_arg}.")
    except ValueError as e: logging.error(f"Invalid format in args: {e}"); sys.exit(1)

    plev_comp_file, surf_comp_file = get_composite_filenames(args.comp_dir, args.region, args.period)
    ds_plev, ds_surf = None, None
    try:
        ds_plev = xr.open_dataset(plev_comp_file); ds_surf = xr.open_dataset(surf_comp_file)
        for ds_name, ds_obj in [("plev", ds_plev), ("surf", ds_surf)]:
            for coord in ['weather_type', 'month']:
                if coord not in ds_obj.coords: raise ValueError(f"'{coord}' missing in {ds_name} file.")
            if 'event_count' not in ds_obj: raise ValueError(f"'event_count' missing in {ds_name} file.") # Still needed for suptitle N
        available_wts_data = ds_plev.weather_type.data
        available_offs_data = ds_plev.time_diff.data
        wts_to_plot = [wt for wt in wts_to_plot if wt in available_wts_data]
        if not all(ro in available_offs_data for ro in REQUIRED_OFFSETS):
            raise ValueError(f"Required time_offsets {REQUIRED_OFFSETS} not in data: {available_offs_data}")
        if not wts_to_plot: raise ValueError("No requested WTs found in data.")
    except Exception as e: logging.error(f"Data loading/checking: {e}", exc_info=True); sys.exit(1)

    anomalies = {} # MODIFIED: Removed counts_for_masking dictionary
    lat, lon = ds_plev.latitude.data, ds_plev.longitude.data
    for var, L_val in VARS_TO_CALC_ANOM.items():
        ds_src = ds_plev if L_val is not None else ds_surf
        anom = calculate_anomaly(ds_src, var)
        k = f"{var}{L_val}" if L_val is not None else var
        if anom is not None:
            anomalies[k] = anom.sel(level=L_val, method="nearest") if L_val is not None else anom
            # MODIFIED: No longer storing counts_for_masking here

    required_keys = ['z500', 'theta_e850', 'q850', 't850', f'u{WIND_LEVEL}', f'v{WIND_LEVEL}']
    if not all(k in anomalies for k in required_keys): # MODIFIED: Check only anomalies
        logging.error(f"Missing critical anomalies. Exiting."); sys.exit(1)

    logging.info("Generating anomaly plots...")
    for wt in wts_to_plot:
        wt_label_str = lamb_weather_type_label(wt)
        n_jja_for_suptitle = 0
        try:
            event_count_months_arr = ds_surf['event_count'].sel(
                weather_type=wt, month=TARGET_MONTHS, time_diff=0 ).data
            current_sum = 0
            if event_count_months_arr.ndim == 3 and event_count_months_arr.shape[0] == len(TARGET_MONTHS):
                for i in range(len(TARGET_MONTHS)): 
                    current_sum += event_count_months_arr[i, 0, 0] 
                n_jja_for_suptitle = int(round(current_sum))
            else:
                logging.warning(f"Unexpected shape for event_count_months_arr WT {wt}: {event_count_months_arr.shape}.")
                n_jja_for_suptitle = -1 
        except Exception as e_count_user:
            logging.error(f"Error calculating N for suptitle WT {wt}: {e_count_user}", exc_info=True)
            n_jja_for_suptitle = -99 

        logging.info(f"  Plotting WT: {wt} ({wt_label_str}), N_JJA_sum_at_point_td0={n_jja_for_suptitle}")

        fig = plt.figure(figsize=(16, 10.5), constrained_layout=True)
        gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.05], hspace=0.25, wspace=0.15)
        ax0 = fig.add_subplot(gs[0, 0], projection=MAP_PROJ)
        ax1 = fig.add_subplot(gs[0, 1], projection=MAP_PROJ)
        ax2 = fig.add_subplot(gs[1, 0], projection=MAP_PROJ)
        ax3 = fig.add_subplot(gs[1, 1], projection=MAP_PROJ)
        axs_list = [ax0, ax1, ax2, ax3]
        cax_theta_e = fig.add_subplot(gs[2, 0])

        fig.suptitle(f'Composite Anomalies - Region: {args.region.replace("_"," ").title()}, WT: {wt_label_str} (N={n_jja_for_suptitle}), Period: {args.period}', fontsize=13)
        plot_handles = {}
        data_for_wt = {} # This will now only store anomaly data, not counts for masking
        try:
            for off in REQUIRED_OFFSETS:
                data_for_wt[off] = {}
                # MODIFIED: Removed panel-specific count preparation
                data_for_wt[off]['theta_e850'] = anomalies['theta_e850'].sel(weather_type=wt, time_diff=off, month=TARGET_MONTHS).mean(dim="month")
                data_for_wt[off]['z500'] = anomalies['z500'].sel(weather_type=wt, time_diff=off, month=TARGET_MONTHS).mean(dim="month")
                data_for_wt[off][f'u{WIND_LEVEL}'] = anomalies[f'u{WIND_LEVEL}'].sel(weather_type=wt, time_diff=off, month=TARGET_MONTHS).mean(dim="month")
                data_for_wt[off][f'v{WIND_LEVEL}'] = anomalies[f'v{WIND_LEVEL}'].sel(weather_type=wt, time_diff=off, month=TARGET_MONTHS).mean(dim="month")
                if off == 0:
                    data_for_wt[off]['q850'] = anomalies['q850'].sel(weather_type=wt, time_diff=off, month=TARGET_MONTHS).mean(dim="month")
                    data_for_wt[off]['t850'] = anomalies['t850'].sel(weather_type=wt, time_diff=off, month=TARGET_MONTHS).mean(dim="month")
        except Exception as e_sel:
            logging.error(f"Panel data prep failed for WT={wt}: {e_sel}", exc_info=True); plt.close(fig); continue

        off_m12, off_0 = -12, 0
        title_sfx_m12, title_sfx_0 = f"(JJA Avg, Offset={off_m12}h)", f"(JJA Avg, Offset={off_0}h)"
        plot_key_th = 'theta_e850'
        # MODIFIED: Call plot_anomaly_panel without count arguments
        plot_handles[plot_key_th] = plot_anomaly_panel(axs_list[0], lon, lat, 
                                     data_for_wt[off_m12]['theta_e850'], 
                                     data_for_wt[off_m12]['z500'], 
                                     data_for_wt[off_m12][f'u{WIND_LEVEL}'], data_for_wt[off_m12][f'v{WIND_LEVEL}'], 
                                     WIND_STRIDE, args.region, plot_key_th, f"Theta-e Anomaly @ {WIND_LEVEL}hPa {title_sfx_m12}")
        cf_th0 = plot_anomaly_panel(axs_list[2], lon, lat, 
                                     data_for_wt[off_0]['theta_e850'], 
                                     data_for_wt[off_0]['z500'], 
                                     data_for_wt[off_0][f'u{WIND_LEVEL}'], data_for_wt[off_0][f'v{WIND_LEVEL}'], 
                                     WIND_STRIDE, args.region, plot_key_th, f"Theta-e Anomaly @ {WIND_LEVEL}hPa {title_sfx_0}")
        if plot_handles.get(plot_key_th) is None : plot_handles[plot_key_th] = cf_th0
        
        plot_key_q = 'q850' 
        plot_handles[plot_key_q] = plot_anomaly_panel(axs_list[1], lon, lat, 
                                     data_for_wt[off_0]['q850'], 
                                     data_for_wt[off_0]['z500'], 
                                     data_for_wt[off_0][f'u{WIND_LEVEL}'], data_for_wt[off_0][f'v{WIND_LEVEL}'], 
                                     WIND_STRIDE, args.region, plot_key_q, f"Spec. Hum. Anomaly @ 850hPa {title_sfx_0}")
        
        plot_key_t = 't850'
        plot_handles[plot_key_t] = plot_anomaly_panel(axs_list[3], lon, lat, 
                                     data_for_wt[off_0]['t850'], 
                                     data_for_wt[off_0]['z500'], 
                                     None, None, # No wind vectors for this panel
                                     WIND_STRIDE, args.region, plot_key_t, f"Temp. Anomaly @ 850hPa {title_sfx_0}")

        if plot_handles.get(plot_key_q):
             cbar_q = fig.colorbar(plot_handles[plot_key_q], ax=axs_list[1], orientation='vertical', pad=0.02, aspect=20)
             cbar_q.set_label(f"{ANOM_PLOT_CONFIG[plot_key_q]['unit']}", fontsize=9); cbar_q.ax.tick_params(labelsize=8)
        if plot_handles.get(plot_key_t):
             cbar_t = fig.colorbar(plot_handles[plot_key_t], ax=axs_list[3], orientation='vertical', pad=0.02, aspect=20)
             cbar_t.set_label(f"{ANOM_PLOT_CONFIG[plot_key_t]['unit']}", fontsize=9); cbar_t.ax.tick_params(labelsize=8)
        if plot_handles.get(plot_key_th): 
            cbar_th = fig.colorbar(plot_handles[plot_key_th], cax=cax_theta_e, orientation='horizontal')
            cbar_th.set_label(f"Theta-e Anomaly ({ANOM_PLOT_CONFIG[plot_key_th]['unit']})", fontsize=9); cbar_th.ax.tick_params(labelsize=8)
        
        plot_filename = args.output_dir / f"composite_anomalies_{args.region}_WT{wt}_{args.period}.png"
        logging.info(f"  Saving plot: {plot_filename}")
        try: plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        except Exception as e_save: logging.error(f"Error saving plot {plot_filename}: {e_save}", exc_info=True)
        plt.close(fig)

    if ds_plev: ds_plev.close()
    if ds_surf: ds_surf.close()
    logging.info("--- Plotting complete. ---")

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning, module="metpy")
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    warnings.filterwarnings("ignore", message=".*invalid value encountered in cast*")
    main()