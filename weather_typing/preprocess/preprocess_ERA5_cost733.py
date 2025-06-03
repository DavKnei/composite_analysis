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
HOURLY_PL_DIR = Path("/data/reloclim/normal/INTERACT/ERA5/pressure_levels")
OUT_DIR = Path("/home/dkn/ERA5")

# ─────────────── constants ───────────────────────────────────────────────
DOMAIN = dict(longitude=slice(-20, 43), latitude=slice(65, 25))  # lon 0..360, lat ↓

CHUNKS = {"time": 365, "lat": 100, "lon": 100}
ENC = dict(zlib=True, complevel=4, dtype="float32", _FillValue=np.float32(1e20))

# ─────────────── helpers ─────────────────────────────────────────────────
def _rename_and_transpose(da: xr.DataArray) -> xr.DataArray:
    """longitude→lon, latitude→lat, and put time first."""
    return da.rename({"longitude": "lon", "latitude": "lat"}).transpose(
        "time", "lat", "lon"
    )


def daily_mean_pl(year: int, var: str, pl_dir: Path) -> xr.DataArray:
    files = sorted(pl_dir.glob(f"{year}-??_NA.nc"))
    if not files:
        raise FileNotFoundError(f"no pressure-level files for {year}")
    da = (
        xr.open_mfdataset(files, chunks={"time": 744})[var]
        .sel(level=500)
        .drop_vars("level", errors="ignore")
        .sel(**DOMAIN)
        .resample(time="1D")
        .mean()
    )
    return _rename_and_transpose(da)


def _nc_enc(da: xr.DataArray, var: str) -> dict[str, dict]:
    """Return a valid encoding dict whose chunks never exceed dim sizes."""
    sz = da.sizes  # dict: dim → length
    cs = [min(sz[d], CHUNKS[d]) for d in ("time", "lat", "lon")]
    return {var: dict(ENC, chunksizes=cs)}


# ─────────────── main ────────────────────────────────────────────────────
def main(start: int, end: int, ncores: int, pl_dir: Path, out_dir: Path) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)

    with Pool(processes=ncores) as pool:
        print("⏳  Loading z500 …")
        z500 = (
            xr.concat(
                pool.map(
                    partial(daily_mean_pl, var="z", pl_dir=pl_dir),
                    range(start, end + 1),
                ),
                dim="time",
            )
            .rename("z500")
            .chunk(CHUNKS)
        )

    print("💾  Writing NetCDFs …")
    with ProgressBar():
        z500.to_netcdf(
            out_dir / f"z500_EUR_daily_{start}_{end}.nc", encoding=_nc_enc(z500, "z500")
        )

    print("✅  Done – files saved in", out_dir)


# ─────────────── CLI ─────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pre-process ERA5 for cost733class")
    p.add_argument("--start", type=int, default=2001)
    p.add_argument("--end", type=int, default=2020)
    p.add_argument("--ncores", type=int, default=32)
    p.add_argument("--pl-dir", type=Path, default=HOURLY_PL_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = p.parse_args()

    dask.config.set(scheduler="threads", num_workers=max(2, a.ncores // 4))

    main(a.start, a.end, a.ncores, a.pl_dir, a.out_dir)
