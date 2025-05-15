#!/usr/bin/env python3
"""
Compute JJA ERA5 composites for derived single-level variables
(upper-level jet, divergence, PV, shear, moisture-flux convergence,
low-level convergence), stratified by weather type and time offset,
using the exact same CLI as composite_surface.py.
"""
import sys
import argparse
from pathlib import Path
import logging
import warnings

import pandas as pd
import numpy as np
import xarray as xr
import dask
from multiprocessing import Pool
from typing import List, Dict, Any, Tuple

# MetPy for dynamic calculations
from metpy.calc import (
    potential_vorticity_baroclinic,
    lat_lon_grid_deltas,
    potential_temperature,
    divergence as mp_divergence
)
from metpy.units import units

# Domain & months
DOMAIN_LAT = (20, 55)
DOMAIN_LON = (-20, 40)
TARGET_MONTHS = [6, 7, 8]
PERIODS = {
    "historical": {"start": 1996, "end": 2005, "name": "historical"},
    "evaluation": {"start": 2000, "end": 2009, "name": "evaluation"}
}

def get_data_file(data_dir: Path, year: int, pattern: str) -> Path:
    return data_dir / pattern.format(year=year)

def standardize_ds(ds: xr.Dataset) -> xr.Dataset:
    """Rename coords to latitude/longitude, sort latitude, subset domain."""
    if 'lat' in ds.coords:
        ds = ds.rename({'lat': 'latitude'})
    if 'lon' in ds.coords:
        ds = ds.rename({'lon': 'longitude'})
    lat = ds['latitude']
    if lat[0] > lat[-1]:
        ds = ds.reindex(latitude=np.sort(lat))
    return ds.sel(latitude=slice(*DOMAIN_LAT), longitude=slice(*DOMAIN_LON))

def create_offset_cols(df: pd.DataFrame) -> Dict[int, str]:
    """Extract time offset columns from the DataFrame."""
    offset_cols = {}
    for col in df.columns:
        if col.startswith("time_minus"):
            h = int(col.replace("time_minus", "").replace("h", ""))
            offset_cols[-h] = col
        elif col.startswith("time_plus"):
            h = int(col.replace("time_plus", "").replace("h", ""))
            offset_cols[h] = col
        elif col == "time_0h":
            offset_cols[0] = col
    return offset_cols

def compute_derived(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute six derived fields from plev data:
      jet_speed_250  : wind speed at 250 hPa
      div_250        : horizontal divergence at 250 hPa
      pv_500         : Ertel PV at 500 hPa (PVU)
      shear_250_850  : bulk wind shear between 250 & 850 hPa
      mfc_850        : moisture-flux convergence at 850 hPa
      conv_850       : wind convergence at 850 hPa
    """
    # assign lat/lon for MetPy
    ds = ds.metpy.assign_latitude_longitude('latitude', 'longitude')
    # compute grid spacing
    dx, dy = lat_lon_grid_deltas(ds.longitude, ds.latitude)

    # 1) Jet speed at 250 hPa
    u250 = ds['u'].sel(level=250)
    v250 = ds['v'].sel(level=250)
    jet250 = np.sqrt(u250**2 + v250**2).rename("jet_speed_250")

    # 2) Horizontal divergence at 250 hPa
    div250 = mp_divergence(
        u250.metpy.quantify(), v250.metpy.quantify(),
        dx=dx, dy=dy,
        x_dim=-1, y_dim=-2,
        latitude=ds.latitude,
        longitude=ds.longitude
    ).metpy.dequantify().rename("div_250")

    # 3) Potential vorticity at 500 hPa
    p3d = ds['level'].metpy.quantify() * units.hPa
    theta = potential_temperature(p3d * 100, ds['t'] * units.kelvin)
    pv3d = potential_vorticity_baroclinic(
        pressure=p3d*100,
        potential_temperature=theta,
        u=ds['u'] * units('m/s'),
        v=ds['v'] * units('m/s'),
        dx=dx, dy=dy
    )
    pv500 = pv3d.sel(level=500 * units.hPa).to('PVU').rename("pv_500")

    # 4) Bulk shear between 250 & 850 hPa
    u850 = ds['u'].sel(level=850)
    v850 = ds['v'].sel(level=850)
    shear = np.sqrt((u250 - u850)**2 + (v250 - v850)**2).rename("shear_250_850")

    # 5) Moisture-flux convergence at 850 hPa
    q850 = ds['q'].sel(level=850)
    uq = (u850 * q850)
    vq = (v850 * q850)
    mfc850 = -mp_divergence(
        uq.metpy.quantify(), vq.metpy.quantify(),
        dx=dx, dy=dy,
        latitude=ds.latitude,
        longitude=ds.longitude
    ).metpy.dequantify().rename("mfc_850")

    # 6) Wind convergence at 850 hPa
    conv850 = -mp_divergence(
        u850.metpy.quantify(), v850.metpy.quantify(),
        dx=dx, dy=dy,
        latitude=ds.latitude,
        longitude=ds.longitude
    ).metpy.dequantify().rename("conv_850")

    return xr.Dataset({
        'jet_speed_250': jet250,
        'div_250':       div250,
        'pv_500':        pv500,
        'shear_250_850': shear,
        'mfc_850':       mfc850,
        'conv_850':      conv850
    })

def process_group(task: Tuple) -> Dict[str,Any]:
    """
    For one (year,month, times…) group: load data,
    compute derived variables, sum over time & count,
    and return sums + count + lat/lon.
    """
    year, month, times, data_dir, pattern = task
    sums: Dict[str, xr.DataArray] = {}
    count = None
    ds_file = get_data_file(data_dir, year, pattern)
    if not ds_file.exists():
        return {'sums':None,'count':None,'lat':None,'lon':None}

    ds = standardize_ds(xr.open_dataset(ds_file, chunks={'time':'auto'}))
    # select nearest in time within 1h
    ds_ev = ds.sel(time=times, method='nearest', tolerance=pd.Timedelta("1H"))
    if ds_ev.time.size == 0:
        return {'sums':None,'count':None,'lat':None,'lon':None}

    # compute derived vars
    ds_der = compute_derived(ds_ev)

    # for each var: sum over time
    for v in ds_der.data_vars:
        sums[v] = ds_der[v].sum(dim='time', skipna=True).compute()
    # count number of valid timesteps (same for all if no missing)
    count = ds_der['jet_speed_250'].notnull().sum(dim='time').compute()
    # lat/lon for output
    lat = ds_der.latitude.values
    lon = ds_der.longitude.values

    return {'sums':sums, 'count':count, 'lat':lat, 'lon':lon}

def combine(results: List[Dict[str,Any]]) -> Dict[str,Any]:
    """Combine a list of per-group dicts into a single sums+count."""
    combined: Dict[str,Any] = {'sums':{}, 'count':None, 'lat':None, 'lon':None}
    for res in results:
        if res['count'] is None: 
            continue
        if combined['count'] is None:
            combined['count'] = res['count']
            combined['sums']  = res['sums']
            combined['lat'], combined['lon'] = res['lat'], res['lon']
        else:
            combined['count'] += res['count']
            for v, da in res['sums'].items():
                combined['sums'][v] += da
    return combined

def save_composites(
    out: Path,
    results: Dict[int,Dict[int,Dict[int,Any]]],
    wts: List[int], months: List[int], offs: List[int],
    lat: np.ndarray, lon: np.ndarray,
    period: Dict[str,Any]
):
    """
    Build an xarray.Dataset with dims (wt,month,off,lat,lon)
    for each derived-var mean, plus event_count(wt,month,off),
    and write to NetCDF with compression.
    """
    derived_vars = [
        'jet_speed_250','div_250','pv_500',
        'shear_250_850','mfc_850','conv_850'
    ]
    nwt, nm, noff = len(wts), len(months), len(offs)
    ny, nx = len(lat), len(lon)

    # allocate arrays
    data = {}
    for v in derived_vars:
        data[v] = np.full((nwt,nm,noff,ny,nx), np.nan, np.float32)
    cnt_arr = np.zeros((nwt,nm,noff), np.int32)

    # fill
    for i, wt in enumerate(wts):
        for j, m in enumerate(months):
            for k, o in enumerate(offs):
                comp = results[wt][o][m]
                if comp['count'] is None:
                    continue
                cnt = comp['count'].values if hasattr(comp['count'],'values') else comp['count']
                cnt_arr[i,j,k] = cnt
                for v in derived_vars:
                    s = comp['sums'][v]
                    arr = s.values if hasattr(s,'values') else s
                    data[v][i,j,k] = arr / cnt

    # build dataset
    ds_out = xr.Dataset(
        {f"{v}_mean": (('wt','month','off','lat','lon'), data[v]) for v in derived_vars}
        | {"event_count": (('wt','month','off'), cnt_arr)},
        coords={
            'wt':    wts,
            'month': months,
            'off':   offs,
            'lat':   lat,
            'lon':   lon
        }
    )
    # metadata
    ds_out.attrs['description'] = (
        "JJA composites of derived single-level variables: "
        "jet_speed_250 (250 hPa), div_250 (250 hPa), pv_500 (500 hPa), "
        "shear_250_850 (250–850 hPa), mfc_850 (850 hPa moisture-flux convergence), "
        "conv_850 (850 hPa wind convergence)."
    )
    ds_out.attrs['history'] = f"Created on {pd.Timestamp.now(tz='UTC')}"
    # write
    enc = {v:{'zlib':True,'complevel':4,'_FillValue':1e20} for v in ds_out.data_vars}
    out.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(out, encoding=enc)
    logging.info(f"Wrote composites to {out}")

def main():
    parser = argparse.ArgumentParser(
        description="Compute JJA single-level derived composites for MCS events."
    )
    # **exactly the same** args as composite_surface.py
    parser.add_argument("--data_dir",    type=Path,
                        default="/data/reloclim/INTERACT/ERA5/levels/",
                        help="ERA5 multi-level input directory")
    parser.add_argument("--period",       choices=PERIODS, default="evaluation")
    parser.add_argument("--file_pattern", required=True,
                        help="e.g. 'era5_plev_{year}.nc'")
    parser.add_argument("--wt_csv_base",  default="./csv/composite_")
    parser.add_argument("--region",       required=True)
    parser.add_argument("--output_dir",   type=Path,
                        default="./composites/ERA5/derived/", help="Output directory")
    parser.add_argument("--ncores",       type=int, default=32)
    parser.add_argument("--serial",       action="store_true")
    parser.add_argument("--time_offsets", default="-12,0,12")
    parser.add_argument("--debug",        action="store_true")
    parser.add_argument("--noMCS",        action="store_true")
    args = parser.parse_args()

    # logging & dask
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    warnings.filterwarnings("ignore")
    dask.config.set({"array.slicing.split_large_chunks": True})

    period = PERIODS[args.period]

    # load CSV of events
    if args.noMCS:
        csv_path = Path(f"{args.wt_csv_base}{args.region}_nomcs.csv")
    else:
        csv_path = Path(f"{args.wt_csv_base}{args.region}_mcs.csv")
    df = pd.read_csv(csv_path, parse_dates=[0])
    # assume first datetime column is the base
    df.columns = ['datetime' if i==0 else c for i,c in enumerate(df.columns)]
    df['datetime'] = df['datetime'].dt.round('H')
    df['year'], df['month'] = df['datetime'].dt.year, df['datetime'].dt.month
    df = df[df['year'].between(period['start'],period['end'])
            & df['month'].isin(TARGET_MONTHS)]
    offset_cols = create_offset_cols(df)
    offs = sorted(int(x) for x in args.time_offsets.split(','))
    wts = sorted(df['wt'].unique())
    if 0 not in wts: wts.insert(0,0)

    # prepare result structure
    results = { wt:{ off:{ m:None for m in TARGET_MONTHS } for off in offs } for wt in wts }

    # loop weather types & offsets
    for wt in wts:
        sub = (df if wt==0 else df[df['wt']==wt])
        logging.info(f"Processing WT={wt}, events={len(sub)}")
        for off in offs:
            col = offset_cols.get(off, 'datetime')
            grp = sub.dropna(subset=[col])
            tasks = []
            for (yr,m), g in grp.groupby([grp[col].dt.year, grp['month']]):
                times = pd.DatetimeIndex(g[col].tolist())
                tasks.append((yr, m, times, args.data_dir, args.file_pattern))
            logging.info(f"  offset {off}h → {len(tasks)} files")
            if args.serial or args.ncores<=1:
                res = [ process_group(t) for t in tasks ]
            else:
                with Pool(args.ncores) as p:
                    res = p.map(process_group, tasks)
            # group by month
            by_month = { m:[] for m in TARGET_MONTHS }
            for (_yr,m,_times,_,_), r in zip(tasks, res):
                by_month[m].append(r)
            for m in TARGET_MONTHS:
                results[wt][off][m] = combine(by_month[m])

    # pick lat/lon from the first non-None result
    sample = next(
        res for wt in results.values()
               for off in wt.values()
               for res in off.values()
               if res['count'] is not None
    )
    lat, lon = sample['lat'], sample['lon']

    # save
    suffix = "_nomcs" if args.noMCS else ""
    out_file = args.output_dir / f"composite_derived_{args.region}_{period['name']}{suffix}.nc"
    save_composites(out_file, results, wts, TARGET_MONTHS, offs, lat, lon, period)

if __name__ == "__main__":
    main()