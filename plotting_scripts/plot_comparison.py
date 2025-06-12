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

def calc_event_count(ds, wt, months_list, pressure_level):
    if wt in list(ds.weather_type):
        event_count_months_arr = ds["event_count"].sel(time_diff=0, weather_type=wt, month=months_list, level=pressure_level).data
        current_sum = 0
        if event_count_months_arr.ndim == 3 and event_count_months_arr.shape[0] == len(months_list):
            for month in range(len(months_list)): 
                current_sum += event_count_months_arr[month, 0, 0] 
            n = int(round(current_sum))
    else:
        n = np.nan
    return n

def subsample_vectors(u, v, step=5):
    """Subsample the u and v numpy arrays for quiver plotting."""
    ny, nx = u.shape
    y_inds = np.arange(0, ny, step)
    x_inds = np.arange(0, nx, step)
    return u[np.ix_(y_inds, x_inds)], v[np.ix_(y_inds, x_inds)], y_inds, x_inds

def plot_panel(ax, lons, lats, field, z, u, v, region_bounds, title, field_label, cmap, norm=None, show_vectors=False):
    """
    Plot a composite panel on ax.
    
    Parameters:
      u, v: xarray DataArrays for wind vectors.
    """
    ax.set_extent([-10, 35, 35, 60], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    if norm is None:
        if field_label.startswith("θe"):
            clevs = np.arange(290, 346, 5)
        elif field_label.startswith("Temp"):
            clevs = np.arange(-5, 31, 5)
        elif field_label.startswith("q"):
            clevs = np.arange(0, 13, 1)
        else:
            vmin = np.nanmin(field)
            vmax = np.nanmax(field)
            clevs = np.linspace(vmin, vmax, 11)
        norm = mcolors.BoundaryNorm(clevs, ncolors=cmap.N, clip=False)

    cf = ax.contourf(lons, lats, field, levels=norm.boundaries, cmap=cmap, norm=norm,
                     extend='both', transform=ccrs.PlateCarree())

    if z is not None and np.ndim(z) == 2:
        cz = ax.contour(lons, lats, z, colors='k', linewidths=0.8, transform=ccrs.PlateCarree())
        ax.clabel(cz, inline=True, fontsize=6, fmt="%.0f")

    if u is not None and v is not None and show_vectors:
        # Convert xarray DataArrays to numpy arrays before subsampling
        u_np, v_np = u.values, v.values
        u_sub, v_sub, y_inds, x_inds = subsample_vectors(u_np, v_np, step=8)
        
        ax.quiver(lons[np.ix_(y_inds, x_inds)],
                  lats[np.ix_(y_inds, x_inds)],
                  u_sub, v_sub, scale=200, width=0.003, transform=ccrs.PlateCarree())

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
        description="Plot a synoptic comparison of different regions for a given weather type."
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
    pressure_level = 850
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

    # --- Define Colormaps and Normalizations ---
    theta_e_cmap = create_custom_cmap()
    theta_e_levels = np.arange(290, 345, 5)
    theta_e_norm = mcolors.BoundaryNorm(theta_e_levels, ncolors=theta_e_cmap.N, clip=False)

    q_cmap = plt.get_cmap("BrBG")
    q_levels = np.arange(0, 13, 1)
    q_norm = mcolors.BoundaryNorm(q_levels, ncolors=q_cmap.N, clip=False)

    t_cmap = plt.get_cmap("coolwarm")
    t_levels = np.arange(-5, 31, 5)
    t_norm = mcolors.BoundaryNorm(t_levels, ncolors=t_cmap.N, clip=False)

    prec_cmap = plt.cm.RdYlGn_r.copy()
    prec_levels = np.arange(0.2, 5.2, 0.5)
    prec_norm = mcolors.BoundaryNorm(prec_levels, ncolors=prec_cmap.N, clip=False)

    cf_list = [] # To store the contour objects for colorbars

    for i, region in enumerate(regions):
        for j, condition in enumerate(conditions):
            row_idx = i * 2 + j

            nomcs_suffix = "_nomcs" if condition == "no_mcs" else ""
            plev_file = os.path.join(args.base_dir, f"composite_plev_{region}_wt_clim_{period}{nomcs_suffix}.nc")
            precip_file = os.path.join(args.base_dir, f"composite_surface_{region}_precipitation_wt_{period}{nomcs_suffix}.nc")
            
            if not os.path.exists(plev_file) or not os.path.exists(precip_file):
                print(f"Skipping {region} - {condition} due to missing files.")
                for ax_idx in range(4):
                    ax = axs[row_idx, ax_idx]
                    ax.text(0.5, 0.5, f'Data not found for\n{region} ({condition})', ha='center', va='center', transform=ax.transAxes, fontsize=8)
                continue

            ds_plev = xr.open_dataset(plev_file)
            ds_precip = xr.open_dataset(precip_file)
            ds_plev = ds_plev
            ds_precip = ds_precip
        
            # Extract the event count from theta_e_count at time_diff=0 and level=args.pressure_level.
            n_for_suptitle = calc_event_count(ds_plev, args.weather_type, months_list, pressure_level)
            if n_for_suptitle == np.nan:
                continue
    
            ds_plev_sel = ds_plev.sel(weather_type=args.weather_type, month=months_list, time_diff=time_offset).mean(dim="month")
            ds_precip_sel = ds_precip.sel(weather_type=args.weather_type, month=months_list, time_diff=time_offset).mean(dim="month")
            
            
            
            ds_level = ds_plev_sel.sel(level=pressure_level, method="nearest")
            ds_500 = ds_plev_sel.sel(level=500, method="nearest")
         
            suffix = "_" + plot_field
            theta_e = ds_level["theta_e" + suffix]
            u = ds_level["u" + suffix]
            v = ds_level["v" + suffix]
            q = ds_level["q" + suffix] * 1000
            t = ds_level["t" + suffix] - 273.15
            z500 = ds_500["z" + suffix]
            precip = ds_precip_sel["precipitation_mean"]

            lons, lats = np.meshgrid(ds_level["longitude"], ds_level["latitude"])
            lons_prec, lats_prec = np.meshgrid(ds_precip_sel["longitude"], ds_precip_sel["latitude"])

            axs[row_idx, 0].text(-0.1, 0.5, f"{region.replace('_', ' ')}\n({condition})\n n={n_for_suptitle}",
                                va='center', ha='right', fontsize=12, transform=axs[row_idx, 0].transAxes)

            cf0 = plot_panel(axs[row_idx, 0], lons, lats, theta_e, z500, u, v, SUBREGIONS[region], "", "θe (K)", theta_e_cmap, norm=theta_e_norm, show_vectors=True)
            cf1 = plot_panel(axs[row_idx, 1], lons, lats, q, z500, None, None, SUBREGIONS[region], "", "q (g/kg)", q_cmap, norm=q_norm)
            cf2 = plot_panel(axs[row_idx, 2], lons, lats, t, z500, None, None, SUBREGIONS[region], "", "Temp (°C)", t_cmap, norm=t_norm)
            precip_masked = np.ma.masked_less(precip, prec_levels[0])
            cf3 = plot_panel(axs[row_idx, 3], lons_prec, lats_prec, precip_masked, None, None, None, SUBREGIONS[region], "", "Precip (mm)", prec_cmap, norm=prec_norm)
            
            if i == 0 and j == 0:
                cf_list.extend([cf0, cf1, cf2, cf3])

            ds_plev.close()
            ds_precip.close()

    axs[0, 0].set_title(f"{pressure_level} hPa θe, 500 hPa Z", fontsize=12)
    axs[0, 1].set_title(f"{pressure_level} hPa q, 500 hPa Z", fontsize=12)
    axs[0, 2].set_title(f"{pressure_level} hPa T, 500 hPa Z", fontsize=12)
    axs[0, 3].set_title("Precipitation", fontsize=12)

    if cf_list:
        fig.subplots_adjust(bottom=0.1)
        cbar_ax0 = fig.add_axes([0.10, 0.05, 0.15, 0.01])
        cbar0 = fig.colorbar(cf_list[0], cax=cbar_ax0, orientation='horizontal')
        cbar0.set_label("Eq. Pot. Temp. (K)")

        cbar_ax1 = fig.add_axes([0.31, 0.05, 0.15, 0.01])
        cbar1 = fig.colorbar(cf_list[1], cax=cbar_ax1, orientation='horizontal')
        cbar1.set_label("Spec. Humidity (g/kg)")

        cbar_ax2 = fig.add_axes([0.52, 0.05, 0.15, 0.01])
        cbar2 = fig.colorbar(cf_list[2], cax=cbar_ax2, orientation='horizontal')
        cbar2.set_label("Temperature (°C)")

        cbar_ax3 = fig.add_axes([0.73, 0.05, 0.15, 0.01])
        cbar3 = fig.colorbar(cf_list[3], cax=cbar_ax3, orientation='horizontal')
        cbar3.set_label("Precipitation (mm)")

    weather_type_str = lamb_weather_type_label(args.weather_type)
    fig.suptitle(f"Synoptic Comparison ({month_label}) - Weather Type: {weather_type_str}", fontsize=20)
    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.08, right=0.98, hspace=0.1, wspace=0.05)

    output_filename = os.path.join(args.output_dir, f"synoptic_composite_wt{args.weather_type}.png")
    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(output_filename, dpi=150, bbox_inches="tight")
    print(f"Saved figure: {output_filename}")

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message="Relative humidity >120%, ensure proper units.")
    main()
