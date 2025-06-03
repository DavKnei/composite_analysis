#!/usr/bin/env python3
"""
run_cost733_ascii.py  –  CAP on MSLP *and* Z500 in one go
──────────────────────────────────────────────────────────
Usage examples
--------------
# 27-class CAP built from MSLP+Z500 over Eastern Alps (all months)
python run_cost733_ascii.py --method CAP_MSLZ500 --ncl 27 --region eastern_alps

# 9-class CAP for JJA only
python run_cost733_ascii.py --method CAP_MSLZ500 --ncl 9  \
                            --region eastern_alps --months 6,7,8
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr

# ──────────────────────────────────────────────────────────
DATA_DIR = Path("/home/dkn/ERA5")                  # ASCII matrices
OUT_DIR  = Path("./csv"); OUT_DIR.mkdir(exist_ok=True)

METHOD_SPECS = {
    # ---------- single-level methods exactly as before ----------
    "GWT_MSL" : ["slp_daily_2001_2020_filtered_?.dat"],
    "GWT_Z500": ["z500_daily_2001_2020_filtered_?.dat"],
    "GWT_Z850": ["z850_daily_2001_2020_filtered_?.dat"],
    "LIT"     : ["slp_daily_2001_2020_?.dat"],
    "JCT"     : ["slp_daily_2001_2020_?.dat"],
    "CAP"     : ["z850_daily_2001_2020_?.dat"],
    "GWTWS"   : ["slp_daily_2001_2020_filtered_?.dat",
                "z500_daily_2001_2020_filtered_?.dat"],
    # ---------- NEW ► CAP on msl & z500 together ---------------
    "CAP_MSLZ500": ["slp_daily_2001_2020_filtered_?.dat",
                    "z500_daily_2001_2020_filtered_?.dat"],
    "CAP_Z850Z500": ["z850_daily_2001_2020_filtered_?.dat",
                    "z500_daily_2001_2020_filtered_?.dat"],                
}

def cost_keyword(method: str) -> str:
    return method.split("_", 1)[0]          # GWT , CAP , …

# ────────────────────────── helpers ─────────────────────────
def build_dat_flag(path: Path, start: str) -> str:
    if not path.exists():
        sys.exit(f"Missing input file {path}")
    nrows  = sum(1 for _ in open(path))
    first  = pd.to_datetime(start)
    last   = first + pd.Timedelta(days=nrows-1)
    nc     = xr.open_dataset(path.with_suffix(".nc"))
    lonmin, lonmax = float(nc.lon.min()), float(nc.lon.max())
    latmin, latmax = float(nc.lat.min()), float(nc.lat.max())

    if "z500" in str(path): 
        return (f"pth:{path}@fmt:ascii@fdt:{first:%Y:%m:%d:%H}"
                f"@ldt:{last:%Y:%m:%d:%H}@ddt:1d"
                f"@lon:{lonmin}:{lonmax}:1@lat:{latmin}:{latmax}:1")
                #"@fil:-31@nrm:1")
    else:
        return (f"pth:{path}@fmt:ascii@fdt:{first:%Y:%m:%d:%H}"
            f"@ldt:{last:%Y:%m:%d:%H}@ddt:1d"
            f"@lon:{lonmin}:{lonmax}:1@lat:{latmin}:{latmax}:1")
            #"@fil:-31@nrm:1")


def cla_to_csv(cla: Path, offset_min: int = 30) -> Path:
    df = pd.read_csv(cla, delim_whitespace=True, header=None,
                     names=["Y","M","D","H","wt"])
    base = (pd.to_datetime(dict(year=df.Y, month=df.M, day=df.D)) +
            pd.Timedelta(minutes=offset_min))
    times = base.repeat(24) + pd.to_timedelta(
            np.tile(np.arange(24), base.size), unit="h")
    out = pd.DataFrame({"datetime": times, "wt": df.wt.repeat(24)})
    csv = cla.with_suffix(".csv"); out.to_csv(csv, index=False)
    return csv
# ───────────────────────── main runner ──────────────────────
def run(method: str, ncl: int, start: str, region: str, mon_str: str):
    files = [Path(DATA_DIR, f).with_name(f.replace("?", region))
             for f in METHOD_SPECS[method]]
    dat_flags = sum([["-dat", build_dat_flag(p, start)] for p in files], [])
    ascii0    = files[0]
    days      = sum(1 for _ in open(ascii0))
    first_dt  = pd.to_datetime(start)
    last_dt   = first_dt + pd.Timedelta(days=days-1)
    per_flag  = ["-per", f"{first_dt:%Y:%m:%d:%H},{last_dt:%Y:%m:%d:%H},1d",
                 "-mon", mon_str]

    # ---------- CAP (all flavours) ----------------------------------
    if cost_keyword(method) == "CAP":
        hcl = OUT_DIR / f"{method}_{region}_ncl{ncl}_HCL.cla"
        final = OUT_DIR / f"{method}_{region}_ncl{ncl}.cla"

        base_flags = dat_flags + per_flag + ["-pcw", "0.5"]

        # 1️⃣  start-partition with HCL
        subprocess.run(["cost733class", *base_flags,
                        "-met", "HCL", "-ncl", str(ncl),
                        "-cla", str(hcl)], check=True)
        # 2️⃣  k-means refinement
        subprocess.run(["cost733class", *base_flags,
                        "-met", "KMN",
                        "-clain", f"pth:{hcl}", "dtc:4",
                        "-ncl",  str(ncl), "-cla", str(final)], check=True)
        cla_to_csv(final)
        print("✔", final)
        return

    # ---------- other methods --------------------------------------
    cla = OUT_DIR / f"{method}_{region}_ncl{ncl}.cla"

    if method.startswith("JCT"):
        #dat_flag = dat_flags[1].replace("@nrm:1", "").replace("@fil:-31", "")
        dat_flags[1] = dat_flag
    cmd = ["cost733class", *dat_flags, *per_flag,
                    "-met", cost_keyword(method),
                    "-ncl", str(ncl), "-cla", str(cla)]
    print("run...", cmd)
    subprocess.run(cmd, check=True)
    cla_to_csv(cla)
    print("✔", cla)

# ───────────────────────── command line ─────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="GWT_Z500", choices=METHOD_SPECS)
    ap.add_argument("--ncl",    type=int, default=10)
    ap.add_argument("--region", required=True)
    ap.add_argument("--start",  default="2001-01-01")
    ap.add_argument("--months", default="1,2,3,4,5,6,7,8,9,10,11,12",
                    help="CSV list (no spaces) – e.g. 6,7,8")
    args = ap.parse_args()

    mon_str = ",".join(f"{int(m):02d}" for m in args.months.split(","))
    run(args.method, args.ncl, args.start, args.region, mon_str)
