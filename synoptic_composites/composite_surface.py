#!/usr/bin/env python3
"""
Compute JJA ERA5 composites for MCS events for surface variables,
including calculation of monthly-hourly climatology means, stratified by weather type.
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

# Domain boundaries and target months
DOMAIN_LAT = (20, 55)
DOMAIN_LON = (-20, 40)
TARGET_MONTHS = [6, 7, 8]
PERIODS = {
    "historical": {"start": 1996, "end": 2005, "name": "historical"},
    "evaluation": {"start": 2000, "end": 2009, "name": "evaluation"}
}


def get_data_file(data_dir: Path, year: int, pattern: str) -> Path:
    """Return file path for a given year."""
    return data_dir / pattern.format(year=year)


def standardize_ds(ds: xr.Dataset) -> xr.Dataset:
    """Rename coords to latitude/longitude, sort latitude, subset domain."""
    rename = {}
    if 'lat' in ds.coords: rename['lat'] = 'latitude'
    if 'lon' in ds.coords: rename['lon'] = 'longitude'
    ds = ds.rename(rename)
    lat = ds['latitude']
    if lat[0] > lat[-1]: ds = ds.reindex(latitude=np.sort(lat))
    return ds.sel(latitude=slice(*DOMAIN_LAT), longitude=slice(*DOMAIN_LON))


def create_offset_cols(df: pd.DataFrame) -> Dict[int, str]:
    """Automatically extract offset column names from the DataFrame."""
    offset_cols = {}
    found_base = False
    time_cols_in_csv = [c for c in df.columns if 'time' in c.lower()]

    for col in time_cols_in_csv:
        if col.startswith("time_minus"):
            try:
                offset = -int(col.replace("time_minus", "").replace("h", ""))
                offset_cols[offset] = col
            except ValueError: logging.warning(f"Could not parse offset: {col}")
        elif col.startswith("time_plus"):
            try:
                offset = int(col.replace("time_plus", "").replace("h", ""))
                offset_cols[offset] = col
            except ValueError: logging.warning(f"Could not parse offset: {col}")
        elif col == "time_0h":
            offset_cols[0] = col
            found_base = True

    if not found_base and 0 not in offset_cols:
        logging.warning("Base time column 'time_0h' not found.")
        if time_cols_in_csv:
            base_col_guess = time_cols_in_csv[0]
            logging.info(f"Using '{base_col_guess}' as base time (offset 0).")
            offset_cols[0] = base_col_guess
        else: raise ValueError("No suitable time column found in CSV.")
    return offset_cols



def process_group(task: Tuple) -> Dict[str, Any]:
    """Load data and climatology for one year-month group, return sums and counts."""
    year, month, times, data_dir, pattern, var, clim_ds = task
    result = {'sum': None, 'clim_sum': None, 'count': None, 'lat': None, 'lon': None}
    file = get_data_file(data_dir, year, pattern)
    if not file.exists(): return result

    ds = standardize_ds(xr.open_dataset(file, chunks={'time':'auto'}))
    if var not in ds: return result
    raw = ds[var].reindex(time=times, method='nearest', tolerance=pd.Timedelta('1H')).dropna('time', how='all')
    if raw.time.size == 0: return result

    result['count'] = raw.notnull().sum(dim='time').compute()
    result['sum']   = raw.sum(dim='time', skipna=True).compute()
    result['lat']   = ds['latitude'].values
    result['lon']   = ds['longitude'].values

    clim = clim_ds[var].sel(month=month)
    hours = pd.to_datetime(raw['time'].values).hour
    result['clim_sum'] = clim.sel(hour=hours).sum(dim='hour')
    return result


def combine(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Sum across groups to get total sum, clim_sum, and count."""
    combined = {'sum': None, 'clim_sum': None, 'count': None, 'lat': None, 'lon': None}
    for res in results:
        if res['count'] is None: continue
        cnt = res['count'] if isinstance(res['count'], np.ndarray) else res['count'].values
        if not np.any(cnt): continue
        if combined['count'] is None:
            combined.update(res)
        else:
            combined['count']   += cnt
            combined['sum']     += res['sum']
            combined['clim_sum']+= res['clim_sum']
    return combined


def save_composites(out: Path, data_var: str,
                    results: Dict[int, Dict[int, Dict[int, Any]]],
                    wts: List[int], months: List[int], offs: List[int],
                    lat: np.ndarray, lon: np.ndarray, period: Dict[str,Any]):
    """Assemble DataArrays and write to NetCDF."""
    shape = (len(wts), len(months), len(offs), len(lat), len(lon))
    mean_arr = np.full(shape, np.nan, np.float32)
    clim_arr = np.full(shape, np.nan, np.float32)
    cnt_arr  = np.zeros(shape, np.int32)

    for i, wt in enumerate(wts):
        for j, m in enumerate(months):
            for k, o in enumerate(offs):
                comp = results[wt][o][m]
                # extract arrays safely
                raw_cnt = comp['count']
                if raw_cnt is None:
                    cnt = np.array(0)
                else:
                    cnt = raw_cnt.values if hasattr(raw_cnt, 'values') else raw_cnt
                raw_sum = comp['sum']
                s = raw_sum.values if hasattr(raw_sum, 'values') else (raw_sum or 0)
                raw_cs = comp['clim_sum']
                cs = raw_cs.values if hasattr(raw_cs, 'values') else (raw_cs or 0)
                cnt_arr[i, j, k] = cnt
                with np.errstate(divide='ignore', invalid='ignore'):
                    mean = s / cnt
                    clim_mean = cs / cnt
                mean_arr[i, j, k] = np.where(cnt > 0, mean, np.nan)
                clim_arr[i, j, k] = np.where(cnt > 0, clim_mean, np.nan)

    ds_out = xr.Dataset({
        f"{data_var}_mean": (('wt','month','off','lat','lon'), mean_arr),
        f"{data_var}_clim_mean": (('wt','month','off','lat','lon'), clim_arr),
        "event_count":    (('wt','month','off','lat','lon'), cnt_arr)},
        coords={'wt':wts,'month':months,'off':offs,'lat':lat,'lon':lon})
    ds_out.attrs.update({
        'description': f"JJA composites for {data_var} ({period['name']} {period['start']}-{period['end']})",
        'history':     f"Created on {pd.Timestamp.now(tz='UTC')}"})
    enc = {}
    for v in ds_out.data_vars:
        if np.issubdtype(ds_out[v].dtype, np.floating):
            enc[v] = {'zlib': True, 'complevel': 4, '_FillValue': np.float32(1e20)}
        else:
            enc[v] = {'zlib': True, 'complevel': 4}
    out.parent.mkdir(exist_ok=True, parents=True)
    ds_out.to_netcdf(out, encoding=enc)
    logging.info(f"Saved to {out}")

def main():
    parser = argparse.ArgumentParser(
        description="Compute JJA monthly composites for surface variables for MCS events.")
    parser.add_argument("--data_dir",    type=Path,
                        default="/data/reloclim/normal/INTERACT/ERA5/surface/", help="ERA5 input directory")
    parser.add_argument("--clim_base_dir", type=Path,
                        default="/home/dkn/climatology/ERA5/",          help="Climatology directory")
    parser.add_argument("--period",      choices=PERIODS, default="evaluation")
    parser.add_argument("--data_var",    default="msl")
    parser.add_argument("--file_pattern",default="slp_{year}_NA.nc")
    parser.add_argument("--wt_csv_base", default="/nas/home/dkn/Desktop/MoCCA/composites/scripts/synoptic_composites/csv/composite_")
    parser.add_argument("--region",      required=True)
    parser.add_argument("--output_dir",  type=Path,
                        default="/home/dkn/composites/ERA5/",        help="Output directory")
    parser.add_argument("--ncores",      type=int, default=32)
    parser.add_argument("--serial",      action="store_true")
    parser.add_argument("--time_offsets", default="-12,0,12")
    parser.add_argument("--debug",       action="store_true")
    parser.add_argument("--noMCS",       action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")
    warnings.filterwarnings("ignore")
    dask.config.set(array={'slicing':{'split_large_chunks':True}})

    period = PERIODS[args.period]
    clim_file = args.clim_base_dir / f"era5_surf_{args.data_var}_clim_may_sep_{period['name']}_{period['start']}-{period['end']}.nc"
    if not clim_file.exists():
        sys.exit(f"Missing climatology file: {clim_file}")
    clim_ds = standardize_ds(xr.open_dataset(clim_file))

    # Load event CSV and determine offsets
    if args.noMCS:
        csv_path = Path(f"{args.wt_csv_base}{args.region}_nomcs.csv")
        base_col      = 'datetime'
        offsets_list  = ['0']
    else:
        csv_path = Path(f"{args.wt_csv_base}{args.region}_mcs.csv")
        base_col      = 'time_0h'
        offsets_list  = args.time_offsets.split(',')

    df = pd.read_csv(csv_path, parse_dates=[base_col])
    df[base_col] = df[base_col].dt.round('H')
    df['year'], df['month'] = df[base_col].dt.year, df[base_col].dt.month
    df = df[df['year'].between(period['start'], period['end']) & df['month'].isin(TARGET_MONTHS)]
    offset_cols = create_offset_cols(df)
    offs = sorted(int(o) for o in offsets_list)
    wts = sorted(df['wt'].unique())
    if 0 not in wts: wts.insert(0,0)

    # Prepare tasks and process
    results = {wt:{o:{} for o in offs} for wt in wts}
    for wt in wts:
        subset = df if wt==0 else df[df['wt']==wt]
        logging.info(f"Processing WT {wt}, events: {len(subset)}")
        for o in offs:
            col = offset_cols[o]
            grp = subset.dropna(subset=[col])
            tasks = []
            for (yr, m), g in grp.groupby([pd.to_datetime(grp[col]).dt.year, grp['month']]):
                times = pd.DatetimeIndex(g[col].tolist())
                if times.empty: continue
                tasks.append((yr, m, times, args.data_dir, args.file_pattern, args.data_var, clim_ds))
            logging.info(f"  Offset {o}h: {len(tasks)} tasks")
            if args.serial or args.ncores==1:
                res = [process_group(t) for t in tasks]
            else:
                with Pool(args.ncores) as p:
                    res = p.map(process_group, tasks)
            by_month = {m: [] for m in TARGET_MONTHS}
            for (_,m,_,_,_,_,_), r in zip(tasks, res): by_month[m].append(r)
            for m in TARGET_MONTHS:
                results[wt][o][m] = combine(by_month[m])

    # Determine lat/lon
    first_wt = next(iter(results.values()))
    first_off = next(iter(first_wt.values()))
    sample = next(iter(first_off.values()))
    lat, lon = sample['lat'], sample['lon']

    suffix = '_nomcs' if args.noMCS else ''
    out_file = args.output_dir / f"composite_surface_{args.region}_{args.data_var}_wt_clim_{args.period}{suffix}.nc"
    save_composites(out_file, args.data_var, results, wts, TARGET_MONTHS, offs, lat, lon, period)

if __name__ == '__main__':
    main()
