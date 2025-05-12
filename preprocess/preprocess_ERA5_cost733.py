#!/usr/bin/env python3
"""
preprocess_era5_for_cost733.py
──────────────────────────────
Create daily-mean ERA5 NetCDF files whose dimensions are called
**lon / lat / time** – the naming that cost733class expects.
"""

from __future__ import annotations
import argparse
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import xarray as xr
import dask
from dask.diagnostics import ProgressBar

# ─────────────── user paths ──────────────────────────────────────────────
HOURLY_SLP_DIR = Path("/data/reloclim/normal/INTERACT/ERA5/surface")
HOURLY_PL_DIR  = Path("/data/reloclim/normal/INTERACT/ERA5/pressure_levels")
OUT_DIR        = Path("/home/dkn/ERA5")

# ─────────────── constants ───────────────────────────────────────────────
DOMAIN = dict(longitude=slice(-5, 35), latitude=slice(55, 30))   # lon 0..360, lat ↓
# Try britain domain
#DOMAIN = dict(longitude=slice(-21, 11), latitude=slice(66, 44))   # lon 0..360, lat ↓
CHUNKS = {"time": 365, "lat": 100, "lon": 100}
ENC    = dict(zlib=True, complevel=4, dtype="float32",
              _FillValue=np.float32(1e20))

# ─────────────── helpers ─────────────────────────────────────────────────
def _rename_and_transpose(da: xr.DataArray) -> xr.DataArray:
    """longitude→lon, latitude→lat, and put time first."""
    return (
        da.rename({"longitude": "lon", "latitude": "lat"})
          .transpose("time", "lat", "lon")
    )

def daily_mean_slp(year: int, slp_dir: Path) -> xr.DataArray:
    fp = slp_dir / f"slp_{year}_NA.nc"
    da = xr.open_dataset(fp, chunks={"time": 744})["msl"]
    #da = da.sel(**DOMAIN).resample(time="1D").mean()
    da = da.sel(time=da["time"].dt.hour == 12)
    
    return _rename_and_transpose(da)

def daily_mean_pl(year: int, var: str, pl_dir: Path) -> xr.DataArray:
    files = sorted(pl_dir.glob(f"{year}-??_NA.nc"))
    if not files:
        raise FileNotFoundError(f"no pressure-level files for {year}")
    da = (
        xr.open_mfdataset(files, chunks={"time": 744})[var]
          .sel(level=850)
          .drop_vars("level", errors="ignore")
          .sel(**DOMAIN)
          .resample(time="1D").mean()
    )
    return _rename_and_transpose(da)

def _nc_enc(da: xr.DataArray, var: str) -> dict[str, dict]:
        """Return a valid encoding dict whose chunks never exceed dim sizes."""
        sz = da.sizes                      # dict: dim → length
        cs = [min(sz[d], CHUNKS[d]) for d in ("time", "lat", "lon")]
        return {var: dict(ENC, chunksizes=cs)}

# ─────────────── main ────────────────────────────────────────────────────
def main(start: int, end: int, ncores: int,
         slp_dir: Path, pl_dir: Path, out_dir: Path) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)

    with Pool(processes=ncores) as pool:
        print("⏳  Loading daily MSLP …")
        slp = xr.concat(pool.map(partial(daily_mean_slp, slp_dir=slp_dir),
                                 range(start, end + 1)),
                        dim="time").chunk(CHUNKS)

        print("⏳  Loading z850, U500, V500 …")
        z850 = xr.concat(pool.map(partial(daily_mean_pl, var="z", pl_dir=pl_dir),
                                  range(start, end + 1)),
                         dim="time").rename("z850").chunk(CHUNKS)
        u500 = xr.concat(pool.map(partial(daily_mean_pl, var="u", pl_dir=pl_dir),
                                  range(start, end + 1)),
                         dim="time").rename("u500").chunk(CHUNKS)
        v500 = xr.concat(pool.map(partial(daily_mean_pl, var="v", pl_dir=pl_dir),
                                  range(start, end + 1)),
                         dim="time").rename("v500").chunk(CHUNKS)

    print("⚙️  Computing wind magnitude …")
    wspd500 = np.hypot(u500, v500).rename("wspd500")
    wspd500.attrs.update(long_name="500 hPa wind-speed magnitude", units="m s-1")

    print("💾  Writing NetCDFs …")
    with ProgressBar():
        #slp    .to_netcdf(out_dir / f"slp_gb_daily_{start}_{end}.nc",
        #                encoding=_nc_enc(slp,    "msl"))
        z850   .to_netcdf(out_dir / f"z500_gb_daily_{start}_{end}.nc",
                        encoding=_nc_enc(z850,   "z850"))


    print("✅  Done – files saved in", out_dir)


# ─────────────── CLI ─────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pre-process ERA5 for cost733class")
    p.add_argument("--start", type=int, default=2001)
    p.add_argument("--end",   type=int, default=2020)
    p.add_argument("--ncores", type=int, default=32)
    p.add_argument("--slp-dir", type=Path, default=HOURLY_SLP_DIR)
    p.add_argument("--pl-dir",  type=Path, default=HOURLY_PL_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = p.parse_args()

    dask.config.set(scheduler="threads",
                    num_workers=max(2, a.ncores // 4))

    main(a.start, a.end, a.ncores, a.slp_dir, a.pl_dir, a.out_dir)
