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
import yaml

# ====================================================================
# HELPER FUNCTIONS
# ====================================================================

def create_diverging_map(cmap_name: str, boundaries: list):
    """
    Creates a discrete, diverging colormap where the outermost colors
    are reserved for the 'extend' triangles.
    """
    if len(boundaries) < 2:
        raise ValueError("Boundaries list must contain at least 2 values.")
    n_inner_colors = len(boundaries) - 1
    n_total_colors = n_inner_colors + 2
    full_cmap = plt.get_cmap(cmap_name, n_total_colors)
    all_colors = full_cmap(np.linspace(0, 1, n_total_colors))
    inner_colors = all_colors[1:-1]
    inner_cmap = mcolors.ListedColormap(inner_colors, name=f"{cmap_name}_inner")
    under_color = all_colors[0]
    over_color = all_colors[-1]
    inner_cmap.set_under(under_color)
    inner_cmap.set_over(over_color)
    norm = mcolors.BoundaryNorm(boundaries, n_inner_colors)
    return inner_cmap, norm

def create_sequential_map(cmap_name: str, boundaries: list):
    """
    Creates a discrete, sequential colormap where the darkest color
    is reserved for the 'over' extend triangle.
    """
    if len(boundaries) < 2:
        raise ValueError("Boundaries list must contain at least 2 values.")
    n_inner_colors = len(boundaries) - 1
    n_total_colors = n_inner_colors + 1
    full_cmap = plt.get_cmap(cmap_name, n_total_colors)
    all_colors = full_cmap(np.linspace(0, 1, n_total_colors))
    inner_colors = all_colors[:-1]
    inner_cmap = mcolors.ListedColormap(inner_colors, name=f"{cmap_name}_inner")
    over_color = all_colors[-1]
    inner_cmap.set_over(over_color)
    norm = mcolors.BoundaryNorm(boundaries, n_inner_colors)
    return inner_cmap, norm

def plot_panel(ax, lons, lats, field, z, region_bounds, cmap, norm, extend='both', title=""):
    """
    Plot a composite panel on ax.
    """
    ax.set_extent([-10, 35, 35, 65], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    levels = norm.boundaries
    cf = ax.contourf(lons, lats, field, levels=levels, cmap=cmap, norm=norm, extend=extend, transform=ccrs.PlateCarree())
    if z is not None and np.ndim(z) == 2:
        cz = ax.contour(lons, lats, z, colors='k', linewidths=0.8, transform=ccrs.PlateCarree())
        ax.clabel(cz, inline=True, fontsize=6, fmt="%.0f")
    rect = Rectangle((region_bounds['lon_min'], region_bounds['lat_min']), region_bounds['lon_max'] - region_bounds['lon_min'], region_bounds['lat_max'] - region_bounds['lat_min'],
                     linewidth=1.5, edgecolor='red', facecolor='none', transform=ccrs.PlateCarree())
    ax.add_patch(rect)
    if title: ax.set_title(title, fontsize=10)
    return cf

# Other domain-specific functions
def calc_event_count(ds, months_list, pressure_level):
    event_count_months_arr = ds["event_count"].sel(time_diff=0, month=months_list, level=pressure_level).data
    current_sum = 0
    if event_count_months_arr.ndim == 3 and event_count_months_arr.shape[0] == len(months_list):
        for month in range(len(months_list)):
            current_sum += event_count_months_arr[month, 0, 0]
    return int(round(current_sum))

def lamb_weather_type_label(wt: int) -> str:
    if wt == 0: return "All"
    return {1:"W", 2:"SW", 3:"NW", 4:"N", 5:"NE", 6:"E", 7:"SE", 8:"S", 9:"C", 10:"A"}.get(wt, f"Unk WT{wt}")

def select_and_merge_weather_type(ds: xr.Dataset, wt_to_plot: int, merge_map: dict) -> tuple:
    wt_str_to_int = {lamb_weather_type_label(i): i for i in range(11)}
    original_label = lamb_weather_type_label(wt_to_plot)
    source_labels, plot_title = None, original_label
    for key, values in merge_map.items():
        if original_label in values:
            source_labels, plot_title = values, key
            break
    if source_labels is None: source_labels = [original_label]
    source_ints = sorted([wt_str_to_int[lbl] for lbl in source_labels if lbl in wt_str_to_int])
    filename_str = "_".join(map(str, source_ints))
    existing_source_ints = [i for i in source_ints if i in ds.weather_type.values]
    if not existing_source_ints: return None, None, None
    ds_slice = ds.sel(weather_type=existing_source_ints)
    total_events = ds_slice['event_count'].sum(dim='weather_type')
    if total_events.sum() == 0: return None, None, None
    new_vars = {}
    for var_name in ds.data_vars:
        if var_name == 'event_count': new_vars[var_name] = total_events
        else:
            weighted_sum = (ds_slice[var_name] * ds_slice['event_count']).sum(dim='weather_type')
            new_vars[var_name] = weighted_sum / total_events
    merged_ds = xr.Dataset(new_vars)
    return merged_ds, plot_title, filename_str

# ====================================================================
# MAIN SCRIPT
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="Plot a comparison of dynamic variables.")
    parser.add_argument("--base_dir", type=str, default="/home/dkn/composites/ERA5/", help="Base directory")
    parser.add_argument("--weather_type", type=int, required=True, help="Weather type to plot (0-10).")
    parser.add_argument("--output_dir", type=str, default="./figures/", help="Output directory")
    parser.add_argument("--regions_file", type=str, default="../regions.yaml", help="Regions YAML file.")
    args = parser.parse_args()

    weather_type_merge_map = {"N+NW": ["N", "NW"], "E+NE": ["E", "NE"], "S+SE": ["S", "SE"]}
    regions = ["Alps", "Balcan", "France", "Eastern_Europe"]
    conditions = ["mcs", "no_mcs"]
    months_list, month_label = [5, 6, 7, 8, 9], "MJJAS"
    time_offset, period = 0, "historical"
    pressure_level = 850

    with open(args.regions_file, 'r') as f: SUBREGIONS = yaml.safe_load(f)

    fig, axs = plt.subplots(len(regions) * len(conditions), 4, figsize=(20, 25),
                            subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)

    rv_cmap, rv_norm = create_diverging_map("RdBu_r", boundaries=[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    mfc_cmap, mfc_norm = create_diverging_map("BrBG", boundaries=[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    pv_cmap, pv_norm = create_sequential_map("viridis", boundaries=np.arange(0.25, 1.51, 0.25).tolist())
    shear_cmap, shear_norm = create_sequential_map("YlOrBr", boundaries=np.arange(5, 16, 2).tolist())

    colormaps = {
        "rel_vorticity": {"cmap": rv_cmap, "norm": rv_norm, "extend": "both"},
        "pot_vorticity": {"cmap": pv_cmap, "norm": pv_norm, "extend": "max"},
        "shear":         {"cmap": shear_cmap, "norm": shear_norm, "extend": "max"},
        "mfc":           {"cmap": mfc_cmap, "norm": mfc_norm, "extend": "both"}
    }
    
    cf_list = []
    final_plot_title, final_filename_suffix = "", str(args.weather_type)

    for i, region in enumerate(regions):
        for j, condition in enumerate(conditions):
            row_idx = i * 2 + j
            nomcs_suffix = "_nomcs" if condition == "no_mcs" else ""
            single_level_file = os.path.join(args.base_dir, f"composite_dynamic_synoptic_{region}_{period}{nomcs_suffix}.nc")
            plev_file = os.path.join(args.base_dir, f"composite_plev_{region}_wt_clim_{period}{nomcs_suffix}.nc")
    
            if not os.path.exists(single_level_file) or not os.path.exists(plev_file): continue

            with xr.open_dataset(single_level_file) as ds_sl_orig, xr.open_dataset(plev_file) as ds_plev_orig:
                ds_sl_sel, plot_title, filename_suffix = select_and_merge_weather_type(ds_sl_orig, args.weather_type, weather_type_merge_map)
                ds_plev_sel, _, _ = select_and_merge_weather_type(ds_plev_orig, args.weather_type, weather_type_merge_map)

            if ds_sl_sel is None:
                print(f"Skipping {region} - {condition}: No events for WT group.")
                axs[row_idx, 1].text(0.5, 0.5, "Data not found", ha='center', va='center', transform=axs[row_idx, 1].transAxes, fontsize=14)
                continue

            if not final_plot_title: final_plot_title, final_filename_suffix = plot_title, filename_suffix
            n_for_suptitle = calc_event_count(ds_plev_sel, months_list, pressure_level)
            
            ds_sl_sel = ds_sl_sel.sel(month=months_list, time_diff=time_offset, method="nearest").mean(dim="month")
            ds_plev_sel = ds_plev_sel.sel(month=months_list, time_diff=time_offset, method="nearest").mean(dim="month")
            
            z500 = ds_plev_sel.sel(level=500, method="nearest")["z_mean"]
            data_fields = [
                ds_sl_sel["rv_500_mean"] * 1e5,
                ds_sl_sel["pv_500_mean"],
                ds_sl_sel["shear_500_850_mean"],
                ds_sl_sel["mfc_850_mean"] * 1e8
            ]
           
            lons, lats = np.meshgrid(ds_sl_sel["longitude"], ds_sl_sel["latitude"])
            axs[row_idx, 0].text(-0.1, 0.5, f"{region.replace('_', ' ')}\n({condition})\nn={n_for_suptitle}",
                                va='center', ha='right', fontsize=12, transform=axs[row_idx, 0].transAxes)
            
            # --- CLEANER PLOTTING LOOP ---
            current_cf_list = []
            for k, var_name in enumerate(colormaps.keys()):
                cf = plot_panel(axs[row_idx, k], lons, lats, data_fields[k], z500, SUBREGIONS[region], **colormaps[var_name])
                current_cf_list.append(cf)
            
            if not cf_list: cf_list = current_cf_list

    # --- FINAL TITLES AND ROBUST COLORBARS ---
    title_props = [
        (r"Rel. Vorticity (500 hPa)", r"$10^{-5} s^{-1}$"),
        ("Pot. Vorticity (500 hPa)", "PVU"),
        ("Shear (500-850 hPa)", "m/s"),
        (r"Moisture Flux Conv. (850 hPa)", r"$10^{-8} kg m^{-2}s^{-1}$")
    ]
    
    if cf_list:
        # Loop through the columns to set titles and create aligned colorbars
        for k, (var_name, (title, cbar_label)) in enumerate(zip(colormaps.keys(), title_props)):
            # Set title for the top-most plot in each column
            axs[0, k].set_title(title, fontsize=12)
            
            # Create the colorbar, attached to the entire column of axes for perfect alignment
            cbar = fig.colorbar(
                cf_list[k],
                ax=axs[:, k], # Attach colorbar to all axes in this column
                orientation='horizontal',
                fraction=0.05, # Use a fraction of the original axes height
                pad=0.04, # Padding between the plot and the colorbar
                ticks=colormaps[var_name]['norm'].boundaries
            )
            cbar.set_label(cbar_label, fontsize=10)

    fig.suptitle(f"Dynamic Variable Comparison ({month_label}) - Weather Type: {final_plot_title}", fontsize=20)
    
    output_filename = os.path.join(args.output_dir, f"dynamic_composite_wt{final_filename_suffix}.png")
    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(output_filename, dpi=150)
    print(f"Saved figure: {output_filename}")

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    main()