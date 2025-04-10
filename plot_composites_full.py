"""
Plot multidimensional ERA5 composites for MCS environments with 500 hPa geopotential overlay,
including additional panels for precipitation and temperature.

This script reads two composite netCDF files:
  1. The pressure-level composite file (e.g. composite_multidim_{region}.nc) that contains fields:
     θe_mean, u_mean, v_mean, q_mean, t_mean, z_mean, and corresponding count variables.
     This file is subset by weather type (via --weather_type) and averaged over the selected months.
  2. The precipitation composite file (e.g. composite_surface_{region}_precipitation.nc) that contains:
     precipitation_mean.
     
The figure is arranged in 2 rows × 4 columns:
  Top row:
    - Panel 0: θe at -12h (with wind vectors) from the chosen pressure level,
               with overlay of 500 hPa geopotential.
    - Panel 1: θe at 0h (same overlay).
    - Panel 2: θe at +12h (same overlay).
    - Panel 3: Specific humidity (q_mean, converted to g/kg) at 0h, with overlay.
  Bottom row:
    - Panel 4: Precipitation at -12h (from IMERG composites) using RdYlGn_r colormap.
    - Panel 5: Precipitation at 0h.
    - Panel 6: Precipitation at +12h.
    - Panel 7: Temperature (t_mean) at 0h (converted to °C) using coolwarm colormap with 500 hPa geopotential overlay.
    
Colorbars:
  - A common horizontal colorbar is added below the top row for the θe panels (smaller height).
  - A common horizontal colorbar is added below the bottom row for the precipitation panels (smaller height).
  - Vertical colorbars are added to the right for specific humidity (top row) and temperature (bottom row).

Usage example:
  python plot_composites_500z.py --base_dir ./data --region southern_alps \
      --comp_file_multidim composite_multidim_ --precip_comp_file composite_surface_{region}_precipitation.nc \
      --weather_type 0 --months 6,7,8 --pressure_level 850 --time_offsets -12,0,12 \
      --plot_field mean --output_prefix composite_southern_alps_JJA.png

Author: David Kneidinger (updated)
Date: 2025-03-27
"""

import argparse
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle
import warnings

# Define subregions (for drawing the red rectangle)
SUBREGIONS = {
    'western_alps': {'lon_min': 3, 'lon_max': 8, 'lat_min': 43, 'lat_max': 49},
    'southern_alps': {'lon_min': 8, 'lon_max': 13, 'lat_min': 43, 'lat_max': 46},
    'dinaric_alps': {'lon_min': 13, 'lon_max': 20, 'lat_min': 42, 'lat_max': 46},
    'eastern_alps': {'lon_min': 8, 'lon_max': 17, 'lat_min': 46, 'lat_max': 49}
}

def create_custom_cmap():
    """Create a custom red–white–blue colormap for θe."""
    colors_list = [
        (0.0, 'darkblue'),
        (0.2, 'blue'),
        (0.45, 'white'),
        (0.7, 'orange'),
        (1.0, 'darkred')
    ]
    return mcolors.LinearSegmentedColormap.from_list("red_white_blue", colors_list, N=256)

def subsample_vectors(u, v, step=5):
    """Subsample the u and v arrays for quiver plotting."""
    ny, nx = u.shape
    y_inds = np.arange(0, ny, step)
    x_inds = np.arange(0, nx, step)
    return u[np.ix_(y_inds, x_inds)], v[np.ix_(y_inds, x_inds)], y_inds, x_inds

def plot_panel(ax, lons, lats, field, z, u, v, region_bounds, title, field_label, cmap, norm=None):
    """
    Plot a composite panel on ax.
    
    Parameters:
      ax: The matplotlib axis.
      lons, lats: 2D arrays of grid coordinates.
      field: 2D array for filled contours.
      z: 2D array for contour lines (if provided; otherwise skipped).
      u, v: 2D arrays for wind vectors (if provided).
      region_bounds: dict with subregion bounds.
      title: subplot title.
      field_label: e.g. "θe (K)", "q (g/kg)", "Precip (mm)", "Temp (°C)".
      cmap: colormap.
      norm: normalization (if provided, e.g. for precipitation).
    """
    ax.set_extent([-10, 25, 35, 52], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    
    if norm is None:
        fmin = np.nanmin(field)
        fmax = np.nanmax(field)
        # Use different levels for θe and Temperature.
        if field_label.startswith("θe"):
            vmin = np.floor(fmin / 5) * 5
            vmax = np.ceil(fmax / 5) * 5
            clevs = np.arange(vmin, vmax + 1, 3)
        elif field_label.startswith("Temp"):
            # For temperature in °C, use a 2°C step.
            vmin = np.floor(fmin / 2) * 2
            vmax = np.ceil(fmax / 2) * 2
            clevs = np.arange(vmin, vmax + 1, 4)
        elif field_label.startswith("q"):
            vmin = np.floor(fmin*1000)/1000
            vmax = np.ceil(fmax*1000)/1000
            clevs = np.linspace(vmin, vmax, 1)
        else:
            vmin = np.nanmin(field)
            vmax = np.nanmax(field)
            clevs = np.linspace(vmin, vmax, 5)
    else:
        clevs = norm.boundaries

    cf = ax.contourf(lons, lats, field, levels=clevs, cmap=cmap, norm=norm,
                     extend='both', transform=ccrs.PlateCarree())
    
    if z is not None and np.ndim(z) == 2:
        cz = ax.contour(lons, lats, z, colors='k', linewidths=1, transform=ccrs.PlateCarree())
        ax.clabel(cz, inline=True, fontsize=8, fmt="%.0f")
    
    if u is not None and v is not None:
        u_sub, v_sub, y_inds, x_inds = subsample_vectors(u, v, step=5)
        qv = ax.quiver(lons[np.ix_(y_inds, x_inds)],
                       lats[np.ix_(y_inds, x_inds)],
                       u_sub, v_sub, scale=200, width=0.002, transform=ccrs.PlateCarree())
        ax.quiverkey(qv, 0.85, -0.05, 5, "5 m/s", labelpos='E', coordinates='axes', fontproperties={'size':8})
    
    rect = Rectangle((region_bounds['lon_min'], region_bounds['lat_min']),
                     region_bounds['lon_max'] - region_bounds['lon_min'],
                     region_bounds['lat_max'] - region_bounds['lat_min'],
                     linewidth=2, edgecolor='red', facecolor='none', transform=ccrs.PlateCarree())
    ax.add_patch(rect)
    
    ax.set_title(title, fontsize=10)
    return cf

def lamb_weather_type_label(wt: int) -> str:
    """
    Convert a simplified Lamb weather type integer (1-9) into a descriptive label.
    Alpine simplified weather types according to Table 2 in DOI: 10.1029/2020JD032824.
    
    Mapping:
      1: Northerly (N, NE, AN) – typically wet in the northwest, drier in the southeast
      2: Easterly (E, SE, AE, ASE) – associated with North Sea high; dry especially in the northeast
      3: Southerly (S, SW, CS, CSW) – typically wet in the west and dry in the east
      4: Westerly (W, NW, CN, CNW) – from central European low; widespread, mostly westward
      5: Weak southerly (C, CW) – Alpine low; wet in the southwest, drier in the northeast
      6: Weak southerly (A, ANE) – Alpine high; dry in the southeast
      7: Easterly (CNE, CE, CSE) – associated with a Genoa low; wet in the southwest, dry in the northeast
      8: South westerly (AS, ASW) – Atlantic low; dry, particularly in the east
      9: Westerly (AW, ANW) – Scandinavian low; wet in the northwest, drier in the southeast
    """
    if wt == 0:
        return "All weather types"
    
    mapping = {
        1: "Northerly (wet NW, drier SE)",
        2: "Easterly (dry, especially NE)",
        3: "Southerly (wet in W, dry in E)",
        4: "Westerly (widespread, mostly W)",
        5: "Weak southerly (Alpine low, wet SW)",
        6: "Weak southerly (Alpine high, dry SE)",
        7: "Easterly (Genoa low, wet SW, dry NE)",
        8: "South westerly (Atlantic low, dry in E)",
        9: "Westerly (Scandinavian low, wet NW)"
    }
    return mapping.get(wt, "Unknown")

def lamb_weather_type_mapping(wt: int) -> str:
    """
    Map the normal Lamb weather type (1 to 27) to an abbreviation.
    
    Mapping:
      1-8: Basic directions ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
      9: "C" (Cyclonic)
      10: "A" (Anticyclonic)
      11-18: Cyclonic hybrids, labeled as "C-" plus the corresponding direction (e.g., 11 -> "C-N")
      19-26: Anticyclonic hybrids, labeled as "A-" plus the corresponding direction (e.g., 19 -> "A-N")
      27: "U" (Unclassified)
    """
    directions = {1: "N", 2: "NE", 3: "E", 4: "SE", 5: "S", 6: "SW", 7: "W", 8: "NW"}
    if 1 <= wt <= 8:
        return directions[wt]
    elif wt == 9:
        return "C"
    elif wt == 10:
        return "A"
    elif 11 <= wt <= 18:
        idx = wt - 10  # gives 1 through 8
        return "C-" + directions[idx]
    elif 19 <= wt <= 26:
        idx = wt - 18  # gives 1 through 8
        return "A-" + directions[idx]
    elif wt == 27:
        return "U"
    else:
        return "Unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Plot multidimensional ERA5 composites for a given subregion with 500 hPa geopotential overlay, including precipitation and temperature panels."
    )
    parser.add_argument("--base_dir", type=str, default="/data/reloclim/normal/MoCCA/composites/composite_files/",
                        help="Base directory containing composite netCDF files")
    parser.add_argument("--region", type=str, required=True,
                        help="Subregion to plot (e.g., western_alps, southern_alps, dinaric_alps, eastern_alps)")
    parser.add_argument("--comp_file_multidim", type=str, default="composite_multidim_",
                        help="Composite netCDF file base name for pressure-level data (e.g., composite_multidim_southern_alps.nc)")
    parser.add_argument("--precip_comp_file", type=str, 
                        default="composite_surface_{region}_precipitation_wt.nc",
                        help="Precipitation composite file (e.g., composite_surface_dinaric_alps_precipitation_wt.nc). Use {region} as placeholder.")
    parser.add_argument("--months", type=str, default="6,7,8",
                        help="Comma-separated list of months to include (default: 6,7,8 for JJA)")
    parser.add_argument("--pressure_level", type=int, default=850,
                        help="Pressure level to plot for θe, u, v, q, t (e.g., 850 hPa). Geopotential always from 500 hPa.")
    parser.add_argument("--plot_field", type=str, default="mean",
                        help="Field type to plot: 'mean' or 'anomaly' (default: mean)")
    parser.add_argument("--weather_type", type=int, default=0,
                        help="Lamb weather type to plot (0 for all events, 1-9 for individual types)")
    parser.add_argument("--time_offsets", type=str, default="-12,0,12",
                        help="Comma-separated list of time offsets to plot (default: -12,0,12)")
    parser.add_argument("--output_prefix", type=str, default="composite_",
                        help="Output filename prefix; full filename will be composite_{region}_{months}_wt{weather_type}.png")
    args = parser.parse_args()
    
    # Process months.
    months_list = [int(x.strip()) for x in args.months.split(",")]
    if sorted(months_list) == [6,7,8]:
        month_label = "JJA"
    else:
        month_label = "_".join(map(str, sorted(months_list)))
    
    # Build precipitation composite file path.
    precip_comp_file = os.path.join(args.base_dir, args.precip_comp_file.format(region=args.region))
    # Open the pressure-level composite file.
    comp_path_multi = os.path.join(args.base_dir, args.comp_file_multidim + f"{args.region}_wt.nc")
    if not os.path.exists(comp_path_multi):
        print(f"Pressure-level composite file not found: {comp_path_multi}")
        return
    ds_multi = xr.open_dataset(comp_path_multi)
    ds_multi = ds_multi.sel(weather_type=args.weather_type)
    ds_multi_sel = ds_multi.sel(month=months_list).mean(dim="month")
    # For fields (θe, u, v, q, t) use the chosen pressure level.
    ds_level = ds_multi_sel.sel(level=args.pressure_level, method="nearest")
    # For geopotential overlay, always use 500 hPa.
    ds_500 = ds_multi_sel.sel(level=500, method="nearest")
    
    # Parse time offsets.
    time_offsets = [int(x.strip()) for x in args.time_offsets.split(",")]
    time_diffs = ds_level.time_diff.values
    sort_idx = np.argsort(time_diffs)
    time_diffs = time_diffs[sort_idx]
    suffix = "_" + args.plot_field
    try:
        theta_e_all = ds_level["theta_e" + suffix].values[sort_idx, :, :]
        u_all = ds_level["u" + suffix].values[sort_idx, :, :]
        v_all = ds_level["v" + suffix].values[sort_idx, :, :]
        q_all = ds_level["q" + suffix].values[sort_idx, :, :]
        t_all = ds_level["t" + suffix].values[sort_idx, :, :]
    except Exception as e:
        print(f"Error extracting variables from pressure-level dataset: {e}")
        return
    try:
        z_all = ds_500["z" + suffix].values[sort_idx, :, :]
    except Exception as e:
        print(f"Error extracting geopotential from 500 hPa dataset: {e}")
        return
    
    # Extract the event count from theta_e_count at time_diff=0 and level=args.pressure_level.
    try:
        count_vals = ds_multi["theta_e_count"].sel(time_diff=0, level=args.pressure_level)
        count_val = np.sum(count_vals.sel(month=slice(months_list[0], months_list[-1])).values)
    except Exception as e:
        print(f"Error extracting count value: {e}")
        count_val = np.nan
    
    # Open precipitation composite file.
    if not os.path.exists(precip_comp_file):
        print(f"Precipitation composite file not found: {precip_comp_file}")
        return
    ds_precip = xr.open_dataset(precip_comp_file)
    ds_precip = ds_precip.sel(weather_type=args.weather_type)
    ds_precip_sel = ds_precip.sel(month=months_list).mean(dim="month")

    try:
        precip_all = ds_precip_sel["precipitation_mean"].values
    except Exception as e:
        print(f"Error extracting precipitation from file: {e}")
        return
    
    # Get lat/lon grid for pressure-level fields.
    lats = ds_level["latitude"].values
    lons = ds_level["longitude"].values
    if lats.ndim == 1 and lons.ndim == 1:
        lon2d, lat2d = np.meshgrid(lons, lats)
    else:
        lat2d = lats
        lon2d = lons
    # Get precipitation grid.
    lats_prec = ds_precip_sel["latitude"].values
    lons_prec = ds_precip_sel["longitude"].values
    if lats_prec.ndim == 1 and lons_prec.ndim == 1:
        lon2d_prec, lat2d_prec = np.meshgrid(lons_prec, lats_prec)
    else:
        lat2d_prec = lats_prec
        lon2d_prec = lons_prec
    
    ds_multi.close()
    ds_precip.close()
    
    fig, axs = plt.subplots(2, 4, figsize=(24, 8),
                            subplot_kw={'projection': ccrs.PlateCarree()})
    axs = axs.flatten()
    
    my_cmap = create_custom_cmap()
    
    # Build mapping from time_diff value to index.
    offset_dict = {int(val): np.where(time_diffs == val)[0][0] for val in time_diffs}
    desired_offsets = [-12, 0, 12]
    for off in desired_offsets:
        if off not in offset_dict:
            print(f"Time offset {off}h not found in composite data.")
            return
    
    # --- Top row ---
    # Panel 0: θe at -12h.
    idx = offset_dict[-12]
    title0 = f"{args.pressure_level} hPa θe (-12h), 500 hPa z overlay"
    cf0 = plot_panel(axs[0], lon2d, lat2d,
                     theta_e_all[idx, :, :],
                     z_all[idx, :, :],
                     u_all[idx, :, :],
                     v_all[idx, :, :],
                     SUBREGIONS[args.region],
                     title0,
                     "θe (K)",
                     my_cmap)
    # Panel 1: θe at 0h.
    idx = offset_dict[0]
    title1 = f"{args.pressure_level} hPa θe (0h), 500 hPa z overlay"
    cf1 = plot_panel(axs[1], lon2d, lat2d,
                     theta_e_all[idx, :, :],
                     z_all[idx, :, :],
                     u_all[idx, :, :],
                     v_all[idx, :, :],
                     SUBREGIONS[args.region],
                     title1,
                     "θe (K)",
                     my_cmap)
    # Panel 2: θe at +12h.
    idx = offset_dict[12]
    title2 = f"{args.pressure_level} hPa θe (+12h), 500 hPa z overlay"
    cf2 = plot_panel(axs[2], lon2d, lat2d,
                     theta_e_all[idx, :, :],
                     z_all[idx, :, :],
                     u_all[idx, :, :],
                     v_all[idx, :, :],
                     SUBREGIONS[args.region],
                     title2,
                     "θe (K)",
                     my_cmap)
    
    # Add a common horizontal colorbar below the top row for θe (smaller height).
    cbar_ax_top = fig.add_axes([0.1, 0.5, 0.5, 0.01])
    c_levels_theta = cf2.levels
    cbar_top = fig.colorbar(cf2, cax=cbar_ax_top, orientation='horizontal')
    cbar_top.set_label("Equivalent Potential Temperature (K)", fontsize=10)
    cbar_top.set_ticks(c_levels_theta)
    cbar_top.ax.set_xticklabels([f"{int(x)}" for x in c_levels_theta])
    
    # Panel 3: Specific humidity (q) at 0h.
    idx = offset_dict[0]
    ax_top3 = axs[3]
    q_cmap = plt.get_cmap("BrBG")
    q_min = np.floor(np.min(q_all*1000))
    q_max = np.ceil(np.max(q_all*1000))
    delta_q = np.ceil((q_max - q_min)/ 11)
    q_levels = np.arange(q_min, q_max, delta_q)

    q_norm = mcolors.BoundaryNorm(q_levels, ncolors=q_cmap.N, clip=False)

    title3 = f"{args.pressure_level} hPa q (0h), 500 hPa z overlay"
    cf3 = plot_panel(ax_top3, lon2d, lat2d,
                     q_all[idx, :, :]*1000,
                     z_all[idx, :, :],
                     None, None,
                     SUBREGIONS[args.region],
                     title3,
                     "q (g/kg)",
                     q_cmap,
                     norm=q_norm)
    # Add vertical colorbar for specific humidity to the right (top row).
    cbar_ax_q = fig.add_axes([0.76, 0.5, 0.2, 0.01])
    cbar_q = fig.colorbar(cf3, cax=cbar_ax_q, orientation='horizontal')
    cbar_q.set_label("Specific Humidity (g/kg)", fontsize=10)
    
    # --- Bottom row ---
    # For precipitation, use specified colormap and levels.
    prec_cmap = plt.cm.RdYlGn_r.copy()
    max_prec = np.round(np.round(np.max(precip_all), 1) + 0.1, 1)
    min_prec = np.round(np.percentile(precip_all, 96), 1)
    delta_prec = np.round((max_prec-min_prec)/7, 1)
    prec_levels_comp = np.arange(min_prec, max_prec, delta_prec)
    prec_levels_comp = [round(x,1) for x in prec_levels_comp]  # weird rounding error otherwise
    prec_norm = mcolors.BoundaryNorm(prec_levels_comp, ncolors=prec_cmap.N, clip=False)
    
    # Panel 4: Precipitation at -12h.
    idx_prec = np.where(ds_precip_sel.time_diff.values == -12)[0][0]
    ax_bot0 = axs[4]
    prec_field0 = np.ma.masked_less(precip_all[idx_prec, :, :], prec_levels_comp[0])
    title4 = "Precip (-12h)"
    cf4 = plot_panel(ax_bot0, lon2d_prec, lat2d_prec,
                     prec_field0,
                     None,
                     None, None,
                     SUBREGIONS[args.region],
                     title4,
                     "Precip (mm)",
                     prec_cmap,
                     norm=prec_norm)
    
    # Panel 5: Precipitation at 0h.
    idx_prec = np.where(ds_precip_sel.time_diff.values == 0)[0][0]
    ax_bot1 = axs[5]
    prec_field1 = np.ma.masked_less(precip_all[idx_prec, :, :], prec_levels_comp[0])
    title5 = "Precip (0h)"
    cf5 = plot_panel(ax_bot1, lon2d_prec, lat2d_prec,
                     prec_field1,
                     None,
                     None, None,
                     SUBREGIONS[args.region],
                     title5,
                     "Precip (mm)",
                     prec_cmap,
                     norm=prec_norm)
    
    # Panel 6: Precipitation at +12h.
    idx_prec = np.where(ds_precip_sel.time_diff.values == 12)[0][0]
    ax_bot2 = axs[6]
    prec_field2 = np.ma.masked_less(precip_all[idx_prec, :, :], prec_levels_comp[0])
    title6 = "Precip (+12h)"
    cf6 = plot_panel(ax_bot2, lon2d_prec, lat2d_prec,
                     prec_field2,
                     None,
                     None, None,
                     SUBREGIONS[args.region],
                     title6,
                     "Precip (mm)",
                     prec_cmap,
                     norm=prec_norm)
    
    # Add a common horizontal colorbar below the bottom row for precipitation (smaller height).
    cbar_ax_bot = fig.add_axes([0.1, 0.00, 0.5, 0.01])
    cbar_bot = fig.colorbar(cf4, cax=cbar_ax_bot, orientation='horizontal')
    cbar_bot.set_label("Precipitation (mm)", fontsize=10)
    cbar_bot.set_ticks(prec_levels_comp)
    cbar_bot.ax.set_xticklabels([f"{x}" for x in prec_levels_comp])
    
    # Panel 7: Temperature at 0h.
    idx = offset_dict[0]
    ax_bot3 = axs[7]
    t_cmap = plt.get_cmap("coolwarm")
   

    title7 = f"{args.pressure_level} hPa  Temp (0h), 500 hPa z overlay"
    # Convert temperature from Kelvin to Celsius.
    t_celsius = t_all[idx, :, :] - 273.15
    t_min = np.floor(np.min(t_celsius))
    t_max = np.ceil(np.max(t_celsius))
    delta_t = np.ceil((t_max - t_min)/ 11)
    t_levels = np.arange(t_min, t_max, delta_t)
    
    t_norm = mcolors.BoundaryNorm(t_levels, ncolors=t_cmap.N, clip=False)

    cf7 = plot_panel(ax_bot3, lon2d, lat2d,
                     t_celsius,
                     z_all[idx, :, :],
                     None, None,
                     SUBREGIONS[args.region],
                     title7,
                     "Temp (°C)",
                     t_cmap,
                     norm=t_norm)
    # Add vertical colorbar for temperature to the right (bottom row)
    cbar_ax_temp = fig.add_axes([0.76, 0.00, 0.2, 0.01])
    cbar_temp = fig.colorbar(cf7, cax=cbar_ax_temp, orientation='horizontal')
    cbar_temp.set_label(f"Temperature (°C)", fontsize=10)

    weather_type_str = lamb_weather_type_mapping(int(args.weather_type))
    plt.suptitle(
        f"{args.region.replace('_',' ').title()} ({month_label})\n"
        f"Weather Type: {weather_type_str} ({count_val:.0f} events)",
        fontsize=16
    )
    # Reduce white space between subplots.
    plt.subplots_adjust(top=0.95, bottom=0.005, left=0.03, right=0.97, hspace=0.0, wspace=0.05)
    out_file = args.output_prefix + args.region + '_' + month_label + f'_wt{args.weather_type}.png'
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    #plt.show()
    print(f"Saved figure: {out_file}")

if __name__ == '__main__':
    import matplotlib.colors as mcolors
    warnings.filterwarnings("ignore", message="Relative humidity >120%, ensure proper units.")
    main()