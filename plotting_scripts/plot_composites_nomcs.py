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

PERIODS = {
    "historical": {"start": 1996, "end": 2005, "name_in_file": "historical"},
    "evaluation": {"start": 2000, "end": 2009, "name_in_file": "evaluation"}
}

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


def plot_panel(ax, lons, lats, field, z, u, v, region_bounds,
               title, field_label, cmap, norm=None):
    """
    Plot a composite panel on ax.
    """
    ax.set_extent([-10, 25, 35, 52], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    # Determine contour levels if no norm provided
    if norm is None:
        fmin, fmax = np.nanmin(field), np.nanmax(field)
        if field_label.startswith("θe"):
            vmin = np.floor(fmin / 5) * 5
            vmax = np.ceil(fmax / 5) * 5
            clevs = np.arange(vmin, vmax + 1, 3)
        elif field_label.startswith("Temp"):
            vmin = np.floor(fmin / 2) * 2
            vmax = np.ceil(fmax / 2) * 2
            clevs = np.arange(vmin, vmax + 1, 4)
        elif field_label.startswith("q"):
            vmin = np.floor(fmin*1000)/1000
            vmax = np.ceil(fmax*1000)/1000
            clevs = np.linspace(vmin, vmax, 11)
        else:
            clevs = np.linspace(fmin, fmax, 5)
    else:
        clevs = norm.boundaries

    cf = ax.contourf(lons, lats, field, levels=clevs, cmap=cmap,
                     norm=norm, extend='both', transform=ccrs.PlateCarree())

    if z is not None:
        cz = ax.contour(lons, lats, z, colors='k', linewidths=1,
                        transform=ccrs.PlateCarree())
        ax.clabel(cz, inline=True, fontsize=8, fmt="%.0f")

    if u is not None and v is not None:
        u_sub, v_sub, y_inds, x_inds = subsample_vectors(u, v)
        qv = ax.quiver(lons[np.ix_(y_inds, x_inds)],
                       lats[np.ix_(y_inds, x_inds)],
                       u_sub, v_sub, scale=200, width=0.002,
                       transform=ccrs.PlateCarree())
        ax.quiverkey(qv, 0.85, -0.05, 5, "5 m/s",
                     labelpos='E', coordinates='axes', fontproperties={'size':8})

    rect = Rectangle((region_bounds['lon_min'], region_bounds['lat_min']),
                     region_bounds['lon_max'] - region_bounds['lon_min'],
                     region_bounds['lat_max'] - region_bounds['lat_min'],
                     linewidth=2, edgecolor='red', facecolor='none',
                     transform=ccrs.PlateCarree())
    ax.add_patch(rect)
    ax.set_title(title, fontsize=10)
    return cf


def lamb_weather_type_label(wt: int) -> str:
    if wt == 0:
        return "All"
    return {1:"W", 2:"SW", 3:"NW", 4:"N", 5:"NE", 6:"E", 7:"SE", 8:"S", 9:"C", 10:"A"}.get(wt, f"Unk WT{wt}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot multidimensional ERA5 composites for a given subregion with 500 hPa geopotential overlay, including precipitation and temperature panels."
    )
    parser.add_argument("--base_dir", type=str, default="/home/dkn/composites/ERA5/",
                        help="Base directory containing composite netCDF files")
    parser.add_argument("--region", type=str, required=True,
                        help="Subregion to plot (e.g., western_alps, southern_alps, dinaric_alps, eastern_alps)")
    parser.add_argument("--comp_file_multidim", type=str, default="composite_plev_",
                        help="Composite netCDF file base name for pressure-level data")
    parser.add_argument("--precip_comp_file", type=str,
                        default="composite_surface_{region}_precipitation_wt_evaluation_nomcs.nc",
                        help="Precipitation composite file (with {region} placeholder)")
    parser.add_argument("--months", type=str, default="6,7,8",
                        help="Comma-separated list of months to include (default: 6,7,8 for JJA)")
    parser.add_argument("--pressure_level", type=int, default=850,
                        help="Pressure level to plot for θe, u, v, q, t (e.g., 850 hPa)")
    parser.add_argument("--plot_field", type=str, default="mean", choices=["mean","anomaly"],
                        help="Field type to plot: 'mean' or 'anomaly'")
    parser.add_argument("--weather_types",type=str,default="0,1,2,3,4,5,6,7,8,9,10",
                        help="Comma-separated weather types to plot")
    parser.add_argument("--period",type=str,default="evaluation",choices=PERIODS.keys())
    parser.add_argument("--time_offsets", type=str, default="0",
                        help="Comma-separated list of time offsets to plot (default: 0)")
    parser.add_argument("--output_prefix", type=str, default="./figures/plots_composite_nomcs/composite_",
                        help="Output filename prefix")
    args = parser.parse_args()

    # Process months
    months_list = [int(x) for x in args.months.split(",")]
    month_label = "JJA" if sorted(months_list)==[6,7,8] else "_".join(map(str,sorted(months_list)))

    # Compose file paths
    comp_path_multi = os.path.join(args.base_dir,
        args.comp_file_multidim + f"{args.region}_wt_clim_{args.period}_nomcs.nc")
    precip_comp_file = os.path.join(
        args.base_dir,
        args.precip_comp_file.format(region=args.region)
    )
    if not os.path.exists(comp_path_multi) or not os.path.exists(precip_comp_file):
        print(f"Composite file(s) not found: {comp_path_multi} or {precip_comp_file}")
        return

    ds_multi = xr.open_dataset(comp_path_multi)
    ds_precip = xr.open_dataset(precip_comp_file)

    for weather_type in [int(w) for w in args.weather_types.split(',')]:
        if weather_type not in ds_multi.weather_type:
            continue
        wt_label = lamb_weather_type_label(weather_type)

        ds_m = ds_multi.sel(weather_type=weather_type).sel(month=months_list).mean(dim="month")
        ds_p = ds_precip.sel(weather_type=weather_type).sel(month=months_list).mean(dim="month")

        # Pressure-level and 500 hPa
        ds_level = ds_m.sel(level=args.pressure_level, method="nearest")
        ds_500 = ds_m.sel(level=500, method="nearest")
        event_count = ds_500['event_count'][:,:,:].mean()

        # Sort by time_diff and find index for 0h
        td = ds_level.time_diff.values
        sort_idx = np.argsort(td)
        idx0 = sort_idx[np.where(np.sort(td)==0)[0][0]]

        theta_e = ds_level[f"theta_e_{args.plot_field}"].values[sort_idx]
        u_all = ds_level[f"u_{args.plot_field}"].values[sort_idx]
        v_all = ds_level[f"v_{args.plot_field}"].values[sort_idx]
        q_all = ds_level[f"q_{args.plot_field}"].values[sort_idx]
        t_all = ds_level[f"t_{args.plot_field}"].values[sort_idx]
        z_all = ds_500[f"z_{args.plot_field}"].values[sort_idx]
        precip_all = ds_p["precipitation_mean"].values

        # Create lat/lon grids
        lons = ds_level.longitude.values
        lats = ds_level.latitude.values
        if lats.ndim == 1:
            lon2d, lat2d = np.meshgrid(lons, lats)
        else:
            lon2d, lat2d = lons, lats

        lons_p = ds_p.longitude.values
        lats_p = ds_p.latitude.values
        if lats_p.ndim == 1:
            lon2d_p, lat2d_p = np.meshgrid(lons_p, lats_p)
        else:
            lon2d_p, lat2d_p = lons_p, lats_p

        # Set up a 2x2 figure
        fig, axs = plt.subplots(2, 2, figsize=(16, 12),
                                subplot_kw={'projection': ccrs.PlateCarree()})
        axs = axs.flatten()


        # Panel 0: θe
        my_cmap = create_custom_cmap()
        theta_e_levels = np.arange(300, 340, 3)
        theta_e_norm = mcolors.BoundaryNorm(theta_e_levels, ncolors=my_cmap.N, clip=False)
        cf1 = plot_panel(
            axs[0], lon2d, lat2d,
            theta_e[idx0], z_all[idx0], u_all[idx0], v_all[idx0],
            SUBREGIONS[args.region],
            f"{args.pressure_level} hPa θe (0h), 500 hPa z overlay",
            "θe (K)", my_cmap, norm=theta_e_norm
        )
        # θe colorbar below
        cbar1 = fig.colorbar(cf1, ax=axs[0], orientation='horizontal', pad=0.07, fraction=0.02, aspect=50)
        cbar1.set_label("Equivalent Potential Temperature (K)")

        # Panel 1: specific humidity
        q_g = q_all[idx0] * 1000
        q_cmap = plt.get_cmap("BrBG")
        # q_levels = np.linspace(np.floor(q_g.min()), np.ceil(q_g.max()), 11)
        q_levels = np.arange(1, 12, 2)
        q_norm = mcolors.BoundaryNorm(q_levels, q_cmap.N)
        cf2 = plot_panel(
            axs[1], lon2d, lat2d,
            q_g, z_all[idx0], None, None,
            SUBREGIONS[args.region],
            f"{args.pressure_level} hPa q (0h), 500 hPa z overlay",
            "q (g/kg)", q_cmap, norm=q_norm
        )
        cbar2 = fig.colorbar(cf2, ax=axs[1], orientation='horizontal', pad=0.07, fraction=0.02, aspect=50)
        cbar2.set_label("Specific Humidity (g/kg)")

        # Panel 2: precipitation
        idx_p0 = np.where(ds_p.time_diff.values == 0)[0][0]
        prec_cmap = plt.cm.RdYlGn_r
        prec_levels = np.arange(0.2, 5, 1)
        p_norm = mcolors.BoundaryNorm(prec_levels, prec_cmap.N)
        cf3 = plot_panel(
            axs[2], lon2d_p, lat2d_p,
            np.ma.masked_less(precip_all[idx_p0], prec_levels[0]),
            None, None, None,
            SUBREGIONS[args.region],
            "Precip (0h)", "Precip (mm)", prec_cmap, norm=p_norm
        )
        cbar3 = fig.colorbar(cf3, ax=axs[2], orientation='horizontal', pad=0.07, fraction=0.02, aspect=50)
        cbar3.set_label("Precipitation (mm)")

        # Panel 3: temperature
        t_c = t_all[idx0] - 273.15
        t_cmap = plt.get_cmap("coolwarm")
        # t_levels = np.linspace(np.floor(t_c.min()), np.ceil(t_c.max()), 11)
        t_levels = np.arange(3, 31, 3)
        t_norm = mcolors.BoundaryNorm(t_levels, t_cmap.N)
        cf4 = plot_panel(
            axs[3], lon2d, lat2d,
            t_c, z_all[idx0], None, None,
            SUBREGIONS[args.region],
            f"{args.pressure_level} hPa Temp (0h), 500 hPa z overlay",
            "Temp (°C)", t_cmap, norm=t_norm
        )
        cbar4 = fig.colorbar(cf4, ax=axs[3], orientation='horizontal', pad=0.07, fraction=0.02, aspect=50)
        cbar4.set_label("Temperature (°C)")

        #plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.2)
        plt.tight_layout()

        plt.suptitle(
            f"No MCS: {args.region.replace('_',' ').title()} ({month_label})\n"
            f"Weather Type: {wt_label} ({event_count:.0f} events)".format(0),
            fontsize=16
        )

        out_file = f"{args.output_prefix}{args.region}_{month_label}_wt{weather_type}_nomcs.png"
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        print(f"Saved figure: {out_file}")

if __name__ == '__main__':
    warnings.filterwarnings("ignore", message="Relative humidity >120%, ensure proper units.")
    main()
