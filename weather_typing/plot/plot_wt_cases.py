#!/usr/bin/env python3
"""
Plot ERA5 MSLP + 500-hPa height for every MCS-launch time,
colour-by weather-type.  Optionally also store the mean field of every
weather-type (flag --add_mean, on by default).

* Weather-type table : …/weather_types/{method}_{region}_ncl{ncl}.csv
* MCS launch times   : …/composite_{region}_mcs.csv
* ERA-5  SLP         : /data/reloclim/normal/INTERACT/ERA5/surface/slp_YYYY_NA.nc   (var msl)
           z500       : /data/reloclim/normal/INTERACT/ERA5/pressure_levels/YYYY-MM_NA.nc (var z)

Output → …/figures/weather_types/{raw|anom}/
"""

import argparse, logging
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs, cartopy.feature as cfeature
import matplotlib.pyplot as plt

# ───────────── CLI ──────────────────────────────────────────────────────────
p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument("--method", default="CAP_MSLZ500")
p.add_argument("--region", required=True)
p.add_argument("--ncl", type=int, default=27)
p.add_argument("--anomaly", action="store_true",
               help="plot 31-day Gaussian-high-pass anomalies")
p.add_argument("--cores", type=int, default=32)
p.add_argument("--serial", action="store_true")
p.add_argument("--extent", nargs=4, type=float,
               default=[-15, 30, 30, 55], metavar=("W","E","S","N"))  #[-15, 30, 30, 55] [-20, 10, 45, 70]
p.add_argument("--add_mean", dest="add_mean", action="store_true",
               help="also create mean composite per weather-type")
p.add_argument("--no-add-mean", dest="add_mean", action="store_false")
p.set_defaults(add_mean=True)
args   = p.parse_args()

method = args.method
region = args.region
ncl    = args.ncl

# ───────────── files / paths ────────────────────────────────────────────────
WT_CSV = Path(f"/nas/home/dkn/Desktop/MoCCA/composites/scripts/preprocess/"
              f"weather_types/{method}_{region}_ncl{ncl}.csv")
MCS_CSV = Path(f"/nas/home/dkn/Desktop/MoCCA/composites/scripts/csv/"
               f"composite_{region}_mcs.csv")

out_root = Path("/nas/home/dkn/Desktop/MoCCA/composites/figures/weather_types")
OUT_DIR  = out_root / ("anom" if args.anomaly else "raw") / region
OUT_DIR.mkdir(parents=True, exist_ok=True)

ERA5_SLP = Path("/data/reloclim/normal/INTERACT/ERA5/surface/slp_{year}_NA.nc")
ERA5_Z   = Path("/data/reloclim/normal/INTERACT/ERA5/pressure_levels/{year}-{month:02d}_NA.nc")

OPEN_KW = dict(chunks="auto", decode_times=True, engine="netcdf4")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")

# ───────────── helpers & caching ────────────────────────────────────────────
_clim_cache = {}   # {(var,year): DataArray}
def gaussian_running_mean31(da: xr.DataArray) -> xr.DataArray:
    """31-day centred Gaussian smoother (σ≈7.3 d) – pure xarray/numpy."""
    window, sigma = 31, 7.3
    w = np.exp(-0.5 * ((np.arange(window) - window//2) / sigma)**2)
    w /= w.sum()
    pad = window // 2
    padded = da.pad(time=(pad, pad), mode="reflect")
    smoothed = (padded
                .rolling(time=window, center=True)
                .construct("window")
                .dot(xr.DataArray(w, dims="window")))
    return smoothed.isel(time=slice(pad, -pad))

def lowfreq_field(var: str, dt: pd.Timestamp):
    """Return the 31-day Gaussian-smoothed field for *that* hour."""
    key = (var, dt.year)
    if key not in _clim_cache:
        if var == "msl":
            ds = xr.open_dataset(ERA5_SLP.with_name(f"slp_{dt.year}_NA.nc"),
                                 **OPEN_KW)
        else:
            ds = xr.open_dataset(ERA5_Z.with_name(
                     f"{dt.year}-{dt.month:02d}_NA.nc"), **OPEN_KW).sel(level=500)
        _clim_cache[key] = gaussian_running_mean31(ds[var])
    return _clim_cache[key].sel(time=dt, method="nearest")

def setup_axis(ax, extent):
    ax.set_extent(extent, ccrs.PlateCarree())
    ax.coastlines("10m", lw=.6)
    ax.add_feature(cfeature.BORDERS, lw=.4)

# ───────────── plotting per event ───────────────────────────────────────────
def load_fields(dt: pd.Timestamp):
    """Return (msl, z500) for the given datetime (already anomaly-adjusted if requested)."""
    dt = pd.to_datetime(dt)
    msl_ds = xr.open_dataset(ERA5_SLP.with_name(f"slp_{dt.year}_NA.nc"), **OPEN_KW)
    z_ds   = xr.open_dataset(ERA5_Z.with_name(f"{dt.year}-{dt.month:02d}_NA.nc"), **OPEN_KW)

    msl  = msl_ds["msl"].sel(time=dt, method="nearest")
    z500 = z_ds["z"].sel(level=500, time=dt, method="nearest") / 9.81

    if args.anomaly:
        msl  = msl  - lowfreq_field("msl", dt)
        z500 = z500 - lowfreq_field("z",   dt) / 9.81
    return msl, z500

def plot_one(row):
    dt = pd.to_datetime(row.datetime);  wt = int(row.wt)
    fname = OUT_DIR / f"wt{wt:02d}_{region}_{dt:%Y%m%d%H}_{'anom' if args.anomaly else 'raw'}.png"
    if fname.exists():
        return

    msl, z500 = load_fields(dt)

    proj = ccrs.PlateCarree();  fig = plt.figure(figsize=(10,6))
    ax = plt.axes(projection=proj);      setup_axis(ax, args.extent)

    if args.anomaly:
        cf = ax.contourf(msl.longitude, msl.latitude, msl/100.,
                         levels=np.arange(-12, 13, 2), cmap="bwr",
                         extend="both", transform=proj)
        cb_label = "MSLP anomaly [hPa]"
        cz_lev   = np.arange(-160, 161, 20)
    else:
        vmin, vmax = float(msl.min()/100.), float(msl.max()/100.)
        cf = ax.contourf(msl.longitude, msl.latitude, msl/100.,
                         levels=np.arange(np.floor(vmin), np.ceil(vmax)+1, 2),
                         cmap="coolwarm", extend="both", transform=proj)
        cb_label = "MSLP [hPa]"
        cz_lev   = np.arange(np.floor(z500.min()/40)*40,
                             np.ceil(z500.max()/40)*40+40, 40)

    fig.colorbar(cf, orientation="horizontal", pad=0.03,
                 aspect=40).set_label(cb_label)

    cz = ax.contour(z500.longitude, z500.latitude, z500,
                    levels=cz_lev, colors="k", linewidths=1, transform=proj)
    ax.clabel(cz, fmt="%.0f", inline=True, fontsize=8)

    fig.savefig(fname, dpi=150, bbox_inches="tight");  plt.close(fig)
    logging.info("saved %s", fname.name)

# ───────────── mean composite per WT  (optional) ────────────────────────────
def mean_composite(wt, dts):
    out = OUT_DIR / f"mean_wt{wt:02d}_{region}_{'anom' if args.anomaly else 'raw'}.png"
    if out.exists():
        return

    msl_sum, z_sum = None, None
    for dt in dts:
        msl, z500 = load_fields(dt)
        if msl_sum is None:
            msl_sum = msl.copy(data=msl.data.astype("float64"))
            z_sum   = z500.copy(data=z500.data.astype("float64"))
        else:
            msl_sum.data += msl.data
            z_sum.data   += z500.data
    msl_mean = msl_sum / len(dts)
    z_mean   = z_sum   / len(dts)
    
    # ▸ plot
    proj = ccrs.PlateCarree();  fig = plt.figure(figsize=(10,6))
    ax = plt.axes(projection=proj);  setup_axis(ax, args.extent)

    if args.anomaly:
        cf = ax.contourf(msl_mean.longitude, msl_mean.latitude, msl_mean/100.,
                         levels=np.arange(np.floor(np.min(msl_mean/100)), np.ceil(np.max(msl_mean/100)), 2), cmap="bwr",
                         extend="both", transform=proj)
        cb_label = "MSLP anomaly [hPa]"
        cz_lev   = np.arange(-160, 161, 20)
    else:
        vmin, vmax = float(msl_mean.min()/100.), float(msl_mean.max()/100.)
        cf = ax.contourf(msl_mean.longitude, msl_mean.latitude, msl_mean/100.,
                         levels=np.arange(np.floor(vmin), np.ceil(vmax)+1, 2),
                         cmap="coolwarm", extend="both", transform=proj)
        cb_label = "MSLP [hPa]"
        cz_lev   = np.arange(np.floor(z_mean.min()/40)*40,
                             np.ceil(z_mean.max()/40)*40+40, 40)

    fig.colorbar(cf, orientation="horizontal", pad=0.03,
                 aspect=40).set_label(cb_label)

    cz = ax.contour(z_mean.longitude, z_mean.latitude, z_mean,
                    levels=cz_lev, colors="k", linewidths=1, transform=proj)
    ax.clabel(cz, fmt="%.0f", inline=True, fontsize=8)
    fig.suptitle(f"n:{len(dts)}")

    fig.savefig(out, dpi=150, bbox_inches="tight");  plt.close(fig)
    logging.info("saved %s (mean composite)", out.name)

# ───────────── main ────────────────────────────────────────────────────────
def main():
    wt  = pd.read_csv(WT_CSV,  parse_dates=["datetime"])
    mcs = pd.read_csv(MCS_CSV, parse_dates=["time_0h"])
    # Filter mcs for June, July, and August
    mcs_JJA = mcs[mcs['time_0h'].dt.month.isin([6, 7, 8])]

    # Filter wt for datetimes that are in mcs_JJA
    df = wt[wt['datetime'].isin(mcs_JJA['time_0h'])]

    logging.info("%d MCS launch events", len(df))
    rows = [r for _, r in df.iterrows()]

    # individual plots (parallel)
    #if args.serial or args.cores == 1:
    #    for r in rows: plot_one(r)
    #else:
    #    with Pool(args.cores) as P: P.map(plot_one, rows)

    # mean composites --------------------------------------------------------
    if args.add_mean:
        for wt_id, grp in df.groupby("wt"):
            mean_composite(int(wt_id), grp.datetime.values)

if __name__ == "__main__":
    main()
