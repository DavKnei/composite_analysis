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


def get_era5_file(era5_dir: Path, year: int, month: int) -> Path:
    """Construct the ERA5 monthly filename (e.g., "2005-08_NA.nc")."""
    fname = f"{year}-{month:02d}_NA.nc"
    return era5_dir / fname


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
            offset_cols[-int(col.replace("time_minus", "").replace("h", ""))] = col
        elif col.startswith("time_plus"):
            offset_cols[int(col.replace("time_plus", "").replace("h", ""))] = col
        elif col == "time_0h":
            offset_cols[0] = col
    return offset_cols


def compute_derived(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute derived single-level fields and 500 hPa PV:
      - jet_speed_250
      - div_250
      - conv_850
      - mfc_850
      - shear_250_850
      - pv_500
    """
    # Load data fully to avoid chunking issues
    ds = ds.load()

    # Select fields at levels
    u250 = ds.u.sel(level=250)
    v250 = ds.v.sel(level=250)
    u850 = ds.u.sel(level=850)
    v850 = ds.v.sel(level=850)
    q850 = ds.q.sel(level=850)

    # 1) Jet speed at 250 hPa
    jet250 = np.hypot(u250, v250).rename("jet_speed_250")

    # 2) Divergence at 250 hPa using latitude/longitude for dx,dy
    div250 = mp_divergence(
        u250.metpy.quantify(),
        v250.metpy.quantify(),
        #latitude=ds.latitude,
        #longitude=ds.longitude
    ).metpy.dequantify().rename("div_250")

    # 3) Convergence at 850 hPa
    conv850 = mp_divergence(
        u850.metpy.quantify(),
        v850.metpy.quantify(),
        #latitude=ds.latitude,
        #longitude=ds.longitude
    ).metpy.dequantify().rename("conv_850") * -1

    # 4) Moisture-flux convergence at 850
    uq = (u850 * q850).rename("uq850")
    vq = (v850 * q850).rename("vq850")
    mfc850 = mp_divergence(
        uq.metpy.quantify(),
        vq.metpy.quantify(),
        #latitude=ds.latitude,
        #longitude=ds.longitude
    ).metpy.dequantify().rename("mfc_850") * -1

    # 5) Bulk shear 250-850 hPa
    shear = np.hypot(u250 - u850, v250 - v850).rename("shear_250_850")

    # 6) 500 hPa potential vorticity via apply_ufunc
    p3d = ds.level.metpy.quantify() * units.hPa
    theta = potential_temperature(p3d * 100, ds.t * units.kelvin)
    pv3d = xr.apply_ufunc(
        potential_vorticity_baroclinic,
        p3d * 100,
        theta,
        ds.u * units('m/s'),
        ds.v * units('m/s'),
        input_core_dims=[
            ['level'],
            ['time','level','latitude','longitude'],
            ['time','level','latitude','longitude'],
            ['time','level','latitude','longitude'],
        ],
        output_core_dims=[['time','level','latitude','longitude']],
        vectorize=True,
        dask='parallelized',
        dask_gufunc_kwargs={'allow_rechunk': True},
        kwargs={'latitude': ds.latitude},
        output_dtypes=[float],
    )

    # select & convert to PVU
    pv500 = (pv3d.sel(level=500*units.hPa) * 1e6).rename("pv_500")
    pv500.attrs["units"] = "PVU"

    # Build the dataset (all now have dims time,latitude,longitude)
    return xr.Dataset({
        "jet_speed_250": jet250.drop_vars('level'),
        "div_250":       div250.drop_vars('level'),
        "conv_850":      conv850.drop_vars('level'),
        "mfc_850":       mfc850.drop_vars('level'),
        "shear_250_850": shear,
        "pv_500":        pv500.drop_vars('level'),
    })


def process_group(task: Tuple) -> Dict[str, Any]:
    year, month, times, era5_dir = task
    ds_file = get_era5_file(era5_dir, year, month)
    if not ds_file.exists():
        return {'sums': None, 'count': None, 'lat': None, 'lon': None}

    ds = standardize_ds(xr.open_dataset(ds_file, chunks={'time':'auto'}))
    ds_ev = ds.sel(time=times, method='nearest', tolerance=pd.Timedelta('1H'))
    if ds_ev.time.size == 0:
        return {'sums': None, 'count': None, 'lat': None, 'lon': None}

    ds_der = compute_derived(ds_ev)
    sums = {}
    for v in ds_der.data_vars:
        da = ds_der[v]
        sums[v] = da.sum(dim='time', skipna=True).compute()
    count = ds_der['jet_speed_250'].notnull().sum(dim='time').compute()
    lat = ds_der.latitude.values
    lon = ds_der.longitude.values
    return {'sums': sums, 'count': count, 'lat': lat, 'lon': lon}


def combine(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    combined = {'sums': {}, 'count': None, 'lat': None, 'lon': None}
    for res in results:
        if res['count'] is None:
            continue
        if combined['count'] is None:
            combined['count'] = res['count']
            combined['sums']  = res['sums']
            combined['lat'], combined['lon'] = res['lat'], res['lon']
        else:
            combined['count'] += res['count']
            for v, arr in res['sums'].items():
                combined['sums'][v] += arr
    return combined


def save_composites(
    out: Path,
    results: Dict[int, Dict[int, Dict[int, Any]]],
    wts: List[int], months: List[int], offs: List[int],
    lat: np.ndarray, lon: np.ndarray,
    period: Dict[str, Any]
):
    derived_vars = [
        'jet_speed_250','div_250','pv_500',
        'shear_250_850','mfc_850','conv_850'
    ]
    shape = (len(wts), len(months), len(offs), len(lat), len(lon))
    data = {v: np.full(shape, np.nan, np.float32) for v in derived_vars}
    cnt_arr = np.zeros((len(wts), len(months), len(offs)), np.int32)

    for i, wt in enumerate(wts):
        for j, m in enumerate(months):
            for k, o in enumerate(offs):
                comp = results[wt][o][m]
                if comp['count'] is None:
                    continue
                cnt = comp['count'].values if hasattr(comp['count'], 'values') else comp['count']
                cnt_arr[i, j, k] = cnt
                for v in derived_vars:
                    arr = comp['sums'][v].values if hasattr(comp['sums'][v], 'values') else comp['sums'][v]
                    data[v][i, j, k] = arr / cnt

    ds_out = xr.Dataset(
        {f"{v}_mean": (('wt','month','off','lat','lon'), data[v]) for v in derived_vars}
        | {"event_count": (('wt','month','off'), cnt_arr)},
        coords={'wt': wts, 'month': months, 'off': offs, 'lat': lat, 'lon': lon}
    )
    ds_out.attrs.update({
        'description': (
            "JJA composites of derived single-level variables: "
            "jet_speed_250 (250 hPa), div_250 (250 hPa), pv_500 (500 hPa), "
            "shear_250_850 (250–850 hPa), mfc_850 (850 hPa moisture-flux convergence), "
            "conv_850 (850 hPa wind convergence)."
        ),
        'history': f"Created on {pd.Timestamp.now(tz='UTC')}"
    })
    enc = {v:{'zlib':True,'complevel':4,'_FillValue':1e20} for v in ds_out.data_vars}
    out.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(out, encoding=enc)
    logging.info(f"Wrote composites to {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute JJA single-level derived composites for MCS events."
    )
    parser.add_argument("--data_dir", type=Path,
                        default="/data/reloclim/normal/INTERACT/ERA5/pressure_levels",
                        help="Directory containing ERA5 monthly files (YYYY-MM_NA.nc)")
    parser.add_argument("--period", choices=PERIODS, default="evaluation")
    parser.add_argument("--wt_csv_base", default="./csv/composite_")
    parser.add_argument("--region", required=True)
    parser.add_argument("--output_dir", type=Path,
                        default="./composites/ERA5/derived/", help="Output directory")
    parser.add_argument("--ncores", type=int, default=32)
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--time_offsets", default="-12,0,12")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--noMCS", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    warnings.filterwarnings("ignore")
    dask.config.set({"array.slicing.split_large_chunks": True})

    period = PERIODS[args.period]

    if args.noMCS:
        csv_path = Path(f"{args.wt_csv_base}{args.region}_nomcs.csv")
    else:
        csv_path = Path(f"{args.wt_csv_base}{args.region}_mcs.csv")
    df = pd.read_csv(csv_path, parse_dates=[0])
    df.columns = ['datetime' if i==0 else c for i,c in enumerate(df.columns)]
    df['datetime'] = df['datetime'].dt.round('H')
    df['year'], df['month'] = df['datetime'].dt.year, df['datetime'].dt.month
    df = df[df['year'].between(period['start'],period['end']) & df['month'].isin(TARGET_MONTHS)]

    offset_cols = create_offset_cols(df)
    offs = sorted(int(x) for x in args.time_offsets.split(','))
    wts = sorted(df['wt'].unique())
    if 0 not in wts: wts.insert(0,0)

    results = {wt:{off:{m:None for m in TARGET_MONTHS} for off in offs} for wt in wts}
    for wt in wts:
        subset = df if wt==0 else df[df['wt']==wt]
        logging.info(f"Processing WT={wt}, events={len(subset)}")
        for off in offs:
            col = offset_cols.get(off,'datetime')
            grp = subset.dropna(subset=[col])
            tasks = [(yr,m,pd.DatetimeIndex(g[col].tolist()), args.data_dir)
                     for (yr,m),g in grp.groupby([grp[col].dt.year, grp['month']])]
            logging.info(f"  Offset {off}h → {len(tasks)} tasks")
            if args.serial or args.ncores<=1:
                out_res = [process_group(t) for t in tasks]
            else:
                with Pool(args.ncores) as p: out_res = p.map(process_group, tasks)
            by_month = {m:[] for m in TARGET_MONTHS}
            for (_yr,m,_times,_dir),r in zip(tasks,out_res): by_month[m].append(r)
            for m in TARGET_MONTHS: results[wt][off][m] = combine(by_month[m])

    sample = next(r for wt_d in results.values() for off_d in wt_d.values() for r in off_d.values() if r['count'] is not None)
    lat, lon = sample['lat'], sample['lon']

    suffix = "_nomcs" if args.noMCS else ""
    out_file = args.output_dir / f"composite_derived_{args.region}_{period['name']}{suffix}.nc"
    save_composites(out_file, results, wts, TARGET_MONTHS, offs, lat, lon, period)

if __name__ == "__main__":
    main()

