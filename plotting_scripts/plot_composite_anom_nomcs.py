#!/usr/bin/env python3
"""
Plot no-MCS composite anomalies for θe, specific humidity, and temperature
arranged in a single row of three panels.

Reads a composite NetCDF `composite_plev_{region}_wt_clim_{period}_nomcs.nc` containing
both mean and climatology fields, computes anomalies, and plots:

  [ θe_anom @850hPa ] [ q_anom  @850hPa ] [ t_anom  @850hPa ]

Each panel overlays 500 hPa geopotential-height anomaly contours; wind-anomaly
vectors remain only on the θe panel.

Usage:
    python plot_composites_anom_nomcs.py \
        --region eastern_alps \
        --comp_dir /home/dkn/composites/ERA5/ \
        --output_dir ./plots_composites_anom_nomcs/ \
        --period evaluation \
        [--weather_types 1,2,3]

Author: David Kneidinger
Date: 2025-05-15 (adapted for no-MCS)
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import warnings

# Constants
VARS_TO_CALC_ANOM = {
    'z': 500,
    'theta_e': 850,
    'q': 850,
    't': 850,
    'u': 850,
    'v': 850
}

def create_custom_cmap():
    colors_list = [
        (0.0, 'darkblue'), (0.2, 'blue'), (0.45, 'white'),
        (0.7, 'orange'), (1.0, 'darkred')
    ]
    return mcolors.LinearSegmentedColormap.from_list("red_white_blue", colors_list, N=256)

ANOM_PLOT_CONFIG = {
    'z500':    {'cmap': 'RdBu_r',    'unit': 'm'},
    'theta_e850': {'cmap': create_custom_cmap(), 'unit': 'K'},
    'q850':    {'cmap': 'BrBG',       'unit': 'g/kg'},
    't850':    {'cmap': 'coolwarm',   'unit': 'K'},
    'u850':    {'cmap': None,         'unit': 'm/s'},
    'v850':    {'cmap': None,         'unit': 'm/s'},
}

PERIODS = {
    "historical": {"start": 1996, "end": 2005},
    "evaluation": {"start": 2000, "end": 2009}
}
WIND_LEVEL = 850
WIND_STRIDE = 8
TARGET_MONTHS = [6, 7, 8]
MAP_PROJ = ccrs.PlateCarree()
MAP_EXTENT = [-10, 25, 35, 52]
SUBREGIONS = {
    'western_alps': {'lon_min': 3, 'lon_max': 8, 'lat_min': 43, 'lat_max': 49},
    'southern_alps':{'lon_min': 8, 'lon_max':13,'lat_min':43,'lat_max':46},
    'dinaric_alps': {'lon_min':13,'lon_max':20,'lat_min':42,'lat_max':46},
    'eastern_alps': {'lon_min': 8, 'lon_max':17,'lat_min':46,'lat_max':49}
}


def calculate_anomaly(ds: xr.Dataset, var: str) -> xr.DataArray:
    """
    Compute anomaly = mean - climatology mean for var.
    Converts units for z->m, q->g/kg.
    Returns DataArray with dims (weather_type, month, level?, lat, lon).
    """
    mean_var = f"{var}_mean"
    clim_var = f"{var}_clim_mean"
    if mean_var not in ds or clim_var not in ds:
        logging.error(f"Missing variables for anomaly: {mean_var} or {clim_var}")
        sys.exit(1)
    anom = ds[mean_var] - ds[clim_var]
    anom = anom.load()
    if var == 'z':
        anom = anom / 9.80665
        anom.attrs.update({'units':'m'})
    elif var == 'q':
        anom = anom * 1000.0
        anom.attrs.update({'units':'g/kg'})
    return anom


def add_subregion_box(ax, region: str):
    b = SUBREGIONS.get(region)
    if b:
        rect = Rectangle((b['lon_min'], b['lat_min']),
                         b['lon_max']-b['lon_min'], b['lat_max']-b['lat_min'],
                         lw=1.5, ec='red', fc='none', transform=MAP_PROJ)
        ax.add_patch(rect)


def lamb_weather_type_label(wt: int) -> str:
    if wt == 0: return "All"
    return {1:"W",2:"SW",3:"NW",4:"N",5:"NE",6:"E",7:"SE",8:"S",9:"C",10:"A"}.get(wt,f"WT{wt}")


def plot_anomaly_panel(ax, lon, lat, field_anom, z500_anom,
                       u_anom, v_anom, region: str,
                       config_key: str, title: str):
    """
    Plot anomaly filled contour + z500 contours + optional wind vectors.
    """
    cfg = ANOM_PLOT_CONFIG[config_key]
    cmap = cfg['cmap']
    # symmetric levels
    vmin = -max(abs(field_anom.min()), abs(field_anom.max()))
    vmax =  max(abs(field_anom.min()), abs(field_anom.max()))
    levels = np.linspace(vmin, vmax, 11)
    norm = mcolors.BoundaryNorm(levels, ncolors=256)
    cf = ax.contourf(lon, lat, field_anom,
                     levels=levels, cmap=cmap, norm=norm,
                     extend='both', transform=MAP_PROJ)
    # 500hPa overlay
    if z500_anom is not None:
        zmin = np.floor(z500_anom.min()); zmax = np.ceil(z500_anom.max())
        z_levs = np.linspace(zmin, zmax, 9)
        cs = ax.contour(lon, lat, z500_anom,
                        levels=z_levs, colors='k', linewidths=0.8,
                        transform=MAP_PROJ)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%d')
    # wind only on theta_e
    if u_anom is not None and v_anom is not None:
        # Subsample both dimensions of the 2D lon/lat grid
        lon_sub = lon[::WIND_STRIDE, ::WIND_STRIDE]
        lat_sub = lat[::WIND_STRIDE, ::WIND_STRIDE]
        u_sub = u_anom[::WIND_STRIDE, ::WIND_STRIDE]
        v_sub = v_anom[::WIND_STRIDE, ::WIND_STRIDE]
        qk = ax.quiver(lon_sub, lat_sub, u_sub, v_sub,
                       scale=None, angles='xy', width=0.003,
                       transform=MAP_PROJ, color='dimgray')
        ax.quiverkey(qk, X=0.85, Y=-0.1, U=2, label='2 m/s',
                     labelpos='E', coordinates='axes', fontproperties={'size':8})
    # map features
    ax.coastlines(resolution='50m', lw=0.5)
    ax.add_feature(cfeature.BORDERS, ls=':', lw=0.5)
    ax.set_extent(MAP_EXTENT, crs=MAP_PROJ)
    add_subregion_box(ax, region)
    ax.set_title(title, fontsize=10)
    return cf


def main():
    parser = argparse.ArgumentParser(description="Plot no-MCS composite anomalies (1x3 layout)")
    parser.add_argument("--region", type=str, required=True)
    parser.add_argument("--comp_dir", type=Path,
                        default=Path('/home/dkn/composites/ERA5/'))
    parser.add_argument("--output_dir", type=Path,
                        default=Path('./plots_composites_anom_nomcs/'))
    parser.add_argument("--period", type=str, default="evaluation",
                        choices=PERIODS.keys())
    parser.add_argument("--weather_types", type=str,
                        default="0,1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # load composite file
    comp_file = args.comp_dir / f"composite_plev_{args.region}_wt_clim_{args.period}_nomcs.nc"
    if not comp_file.exists():
        logging.error(f"Composite file not found: {comp_file}")
        sys.exit(1)
    ds = xr.open_dataset(comp_file).sel(time_diff=0)

    # compute anomalies once
    anomalies = {}
    for var, lvl in VARS_TO_CALC_ANOM.items():
        anom = calculate_anomaly(ds, var)
        key = f"{var}{lvl if lvl else ''}"
        if lvl is not None:
            anomalies[key] = anom.sel(level=lvl, method='nearest')
        else:
            anomalies[key] = anom
    # coords check
    wts = [int(w) for w in args.weather_types.split(',')]
    months = TARGET_MONTHS
    for wt in wts:
        if wt not in ds.weather_type.data:
            logging.warning(f"WT {wt} not in data, skipping.")
            continue
        # prepare 2D fields by averaging over months
        data_th = anomalies['theta_e850'].sel(weather_type=wt, month=months).mean(dim='month')
        data_q  = anomalies['q850'].sel(weather_type=wt, month=months).mean(dim='month')
        data_t  = anomalies['t850'].sel(weather_type=wt, month=months).mean(dim='month')
        data_z  = anomalies['z500'].sel(weather_type=wt, month=months).mean(dim='month')
        data_u  = anomalies['u850'].sel(weather_type=wt, month=months).mean(dim='month')
        data_v  = anomalies['v850'].sel(weather_type=wt, month=months).mean(dim='month')

        # lat/lon arrays
        lon = ds.longitude.values; lat = ds.latitude.values
        if lat.ndim==1:
            lon2d, lat2d = np.meshgrid(lon, lat)
        else:
            lon2d, lat2d = lon, lat

        # figure with GridSpec for panels + colorbars
        fig = plt.figure(figsize=(18,6))
        gs = GridSpec(2,3, figure=fig, height_ratios=[1,0.05], hspace=0.2)

        ax0 = fig.add_subplot(gs[0,0], projection=MAP_PROJ)
        ax1 = fig.add_subplot(gs[0,1], projection=MAP_PROJ)
        ax2 = fig.add_subplot(gs[0,2], projection=MAP_PROJ)
        cax0 = fig.add_subplot(gs[1,0])
        cax1 = fig.add_subplot(gs[1,1])
        cax2 = fig.add_subplot(gs[1,2])

        # plot each panel
        cf0 = plot_anomaly_panel(ax0, lon2d, lat2d, data_th, data_z, data_u, data_v,
                                 args.region, 'theta_e850', f"Theta-e Anom @ {WIND_LEVEL}hPa")
        cf1 = plot_anomaly_panel(ax1, lon2d, lat2d, data_q, data_z, None, None,
                                 args.region, 'q850',          f"Spec. Hum. Anom @ {WIND_LEVEL}hPa")
        cf2 = plot_anomaly_panel(ax2, lon2d, lat2d, data_t, data_z, None, None,
                                 args.region, 't850',          f"Temp Anom @ {WIND_LEVEL}hPa")

        # colorbars below each plot
        cb0 = fig.colorbar(cf0, cax=cax0, orientation='horizontal')
        cb0.set_label(f"θe Anomaly ({ANOM_PLOT_CONFIG['theta_e850']['unit']})")
        cb1 = fig.colorbar(cf1, cax=cax1, orientation='horizontal')
        cb1.set_label(f"q Anomaly ({ANOM_PLOT_CONFIG['q850']['unit']})")
        cb2 = fig.colorbar(cf2, cax=cax2, orientation='horizontal')
        cb2.set_label(f"T Anomaly ({ANOM_PLOT_CONFIG['t850']['unit']})")

        # suptitle and save
        wt_label = lamb_weather_type_label(wt)
        plt.suptitle(f"No-MCS Composite Anomalies — {args.region.replace('_',' ').title()}, WT: {wt_label}, Period: {args.period}")
        outfn = args.output_dir / f"composite_anom_nomcs_{args.region}_WT{wt}_{args.period}.png"
        plt.savefig(outfn, dpi=150, bbox_inches='tight')
        logging.info(f"Saved: {outfn}")
        plt.close(fig)

    ds.close()

if __name__ == '__main__':
    warnings.filterwarnings("ignore", message=".*invalid value encountered.*")
    main()
