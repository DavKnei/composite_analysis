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

def plot_panel(ax, lons, lats, field, z, region_bounds, cmap, norm, title=""):
    """
    Plot a composite panel on ax.
    """
    ax.set_extent([-10, 35, 35, 65], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    cf = ax.contourf(lons, lats, field, levels=norm.boundaries, cmap=cmap, norm=norm,
                     extend='both', transform=ccrs.PlateCarree())

    if z is not None and np.ndim(z) == 2:
        cz = ax.contour(lons, lats, z, colors='k', linewidths=0.8, transform=ccrs.PlateCarree())
        ax.clabel(cz, inline=True, fontsize=6, fmt="%.0f")

    rect = Rectangle((region_bounds['lon_min'], region_bounds['lat_min']),
                     region_bounds['lon_max'] - region_bounds['lon_min'],
                     region_bounds['lat_max'] - region_bounds['lat_min'],
                     linewidth=1.5, edgecolor='red', facecolor='none', transform=ccrs.PlateCarree())
    ax.add_patch(rect)

    if title:
        ax.set_title(title, fontsize=10)
    return cf

def lamb_weather_type_label(wt: int) -> str:
    if wt == 0: return "All"
    return {1:"W", 2:"SW", 3:"NW", 4:"N", 5:"NE", 6:"E", 7:"SE", 8:"S", 9:"C", 10:"A"}.get(wt, f"Unk WT{wt}")

def main():
    parser = argparse.ArgumentParser(
        description="Plot a comparison of dynamic variables for different regions."
    )
    parser.add_argument("--base_dir", type=str, default="/home/dkn/composites/ERA5/",
                        help="Base directory containing composite netCDF files")
    parser.add_argument("--weather_type", type=int, required=True,
                        help="Weather type to plot (0-10).")
    parser.add_argument("--output_dir", type=str, default="./figures/",
                        help="Output directory for the plot.")
    parser.add_argument("--regions_file", type=str, default="../regions.yaml",
                        help="Path to the regions YAML file.")
    args = parser.parse_args()

    # --- Fixed Parameters ---
    regions = ["Alps", "Balcan", "France", "Eastern_Europe"]
    conditions = ["mcs", "no_mcs"]
    months_list = [5, 6, 7, 8, 9]
    month_label = "MJJAS"
    time_offset = 0
    plot_field = "mean"
    period = "historical"

    if not os.path.exists(args.regions_file):
        print(f"ERROR: Regions file not found at {args.regions_file}")
        return

    with open(args.regions_file, 'r') as f:
        SUBREGIONS = yaml.safe_load(f)

    fig, axs = plt.subplots(len(regions) * len(conditions), 4, figsize=(20, 30),
                            subplot_kw={'projection': ccrs.PlateCarree()})

    # --- Define Colormaps and Normalizations for Dynamic Variables ---
    # Relative Vorticity (scaled by 1e5)
    rv_cmap = plt.get_cmap("RdBu_r")
    rv_levels = np.arange(-5, 5.1, 1)
    rv_norm = mcolors.BoundaryNorm(rv_levels, ncolors=rv_cmap.N, clip=True)
    
    # Potential Vorticity
    pv_cmap = plt.get_cmap("viridis")
    pv_levels = np.arange(0, 2.1, 0.25)
    pv_norm = mcolors.BoundaryNorm(pv_levels, ncolors=pv_cmap.N, clip=False)

    # Shear
    shear_cmap = plt.get_cmap("Reds")
    shear_levels = np.arange(0, 21, 2.5)
    shear_norm = mcolors.BoundaryNorm(shear_levels, ncolors=shear_cmap.N, clip=False)

    # Moisture Flux Convergence (scaled by 1e5)
    mfc_cmap = plt.get_cmap("BrBG")
    mfc_levels = np.arange(-0.01, 0.011, 0.005)
    mfc_norm = mcolors.BoundaryNorm(mfc_levels, ncolors=mfc_cmap.N, clip=True)

    cf_list = []

    for i, region in enumerate(regions):
        for j, condition in enumerate(conditions):
            row_idx = i * 2 + j
            
            nomcs_suffix = "_nomcs" if condition == "no_mcs" else ""
            
            # Define file paths
            single_level_file = os.path.join(args.base_dir, f"composite_single_level_{region}_{period}{nomcs_suffix}.nc")
            plev_file = os.path.join(args.base_dir, f"composite_plev_{region}_wt_clim_{period}{nomcs_suffix}.nc")
    
            if not os.path.exists(single_level_file) or not os.path.exists(plev_file):
                print(f"Skipping {region} - {condition} due to missing files.")
                for ax_idx in range(4):
                    ax = axs[row_idx, ax_idx]
                    ax.text(0.5, 0.5, f'Data not found for\n{region} ({condition})', ha='center', va='center', transform=ax.transAxes, fontsize=8)
                continue

            # Load datasets
            ds_sl = xr.open_dataset(single_level_file)
            ds_plev = xr.open_dataset(plev_file)
            
            # Clean and sort coordinates
            ds_sl = ds_sl.sortby('time_diff').dropna(dim='weather_type')
            ds_plev = ds_plev.sortby('time_diff').dropna(dim='weather_type')

            # Select data for the given weather type and time
            ds_sl_sel = ds_sl.sel(weather_type=args.weather_type, month=months_list, time_diff=time_offset, method="nearest").mean(dim="month")
            ds_plev_sel = ds_plev.sel(weather_type=args.weather_type, month=months_list, time_diff=time_offset, method="nearest").mean(dim="month")
            
            # Extract variables
            z500 = ds_plev_sel.sel(level=500, method="nearest")["z_mean"]
            rv = ds_sl_sel["rv_500_mean"] * 1e5  # Scale for plotting
            pv = ds_sl_sel["pv_500_mean"]
            shear = ds_sl_sel["shear_500_850_mean"]
            mfc = ds_sl_sel["mfc_850_mean"] * 1e5 # Scale for plotting

            lons, lats = np.meshgrid(ds_sl_sel["longitude"], ds_sl_sel["latitude"])
            
            axs[row_idx, 0].text(-0.1, 0.5, f"{region.replace('_', ' ')}\n({condition})",
                                va='center', ha='right', fontsize=12, transform=axs[row_idx, 0].transAxes)

            # --- Plotting ---
            cf0 = plot_panel(axs[row_idx, 0], lons, lats, rv, z500, SUBREGIONS[region], rv_cmap, rv_norm)
            cf1 = plot_panel(axs[row_idx, 1], lons, lats, pv, z500, SUBREGIONS[region], pv_cmap, pv_norm)
            cf2 = plot_panel(axs[row_idx, 2], lons, lats, shear, z500, SUBREGIONS[region], shear_cmap, shear_norm)
            cf3 = plot_panel(axs[row_idx, 3], lons, lats, mfc, z500, SUBREGIONS[region], mfc_cmap, mfc_norm)
            
            if i == 0 and j == 0:
                cf_list.extend([cf0, cf1, cf2, cf3])

            ds_sl.close()
            ds_plev.close()

    # --- Set Titles and Colorbars ---
    axs[0, 0].set_title(r"Rel. Vorticity (500 hPa) [$10^{-5} s^{-1}$]", fontsize=10)
    axs[0, 1].set_title("Pot. Vorticity (500 hPa) [PVU]", fontsize=10)
    axs[0, 2].set_title("Shear (500-850 hPa) [m/s]", fontsize=10)
    axs[0, 3].set_title(r"Moisture Flux Conv. (850 hPa) [$10^{-5} kg m^{-2}s^{-1}$]", fontsize=10)

    if cf_list:
        fig.subplots_adjust(bottom=0.1)
        cbar_ax0 = fig.add_axes([0.10, 0.05, 0.15, 0.01])
        cbar0 = fig.colorbar(cf_list[0], cax=cbar_ax0, orientation='horizontal')
        cbar0.set_label(r"$10^{-5} s^{-1}$")

        cbar_ax1 = fig.add_axes([0.31, 0.05, 0.15, 0.01])
        cbar1 = fig.colorbar(cf_list[1], cax=cbar_ax1, orientation='horizontal')
        cbar1.set_label("PVU")

        cbar_ax2 = fig.add_axes([0.52, 0.05, 0.15, 0.01])
        cbar2 = fig.colorbar(cf_list[2], cax=cbar_ax2, orientation='horizontal')
        cbar2.set_label("m/s")

        cbar_ax3 = fig.add_axes([0.73, 0.05, 0.15, 0.01])
        cbar3 = fig.colorbar(cf_list[3], cax=cbar_ax3, orientation='horizontal')
        cbar3.set_label(r"$10^{-5} kg m^{-2}s^{-1}$")

    weather_type_str = lamb_weather_type_label(args.weather_type)
    fig.suptitle(f"Dynamic Variable Comparison ({month_label}) - Weather Type: {weather_type_str}", fontsize=20)
    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.08, right=0.98, hspace=0.1, wspace=0.05)

    output_filename = os.path.join(args.output_dir, f"dynamic_composite_wt{args.weather_type}.png")
    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(output_filename, dpi=150, bbox_inches="tight")
    print(f"Saved figure: {output_filename}")

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
