#!/usr/bin/env python3
"""
convert_netcdf_to_ascii.py  ────────────────────────────────────────────────
FAST conversion of ERA5 daily NetCDF → COST733 ASCII **plus** a small, tidy
NetCDF copy for visual inspection.

Includes manual preprocessing steps to mimic cost733class flags:
- Row-wise centering (`@nrm:1`)
- 31-day Gaussian high-pass filtering (`@fil:-31`)

Improvements vs. last patch
---------------------------
* **Manual Preprocessing**: Added functions `normalize_data` and
  `high_pass_filter` to apply preprocessing before saving.
* **Bug‑fix**: Reworked Gaussian filter using rolling().dot() for better
  dask/xarray compatibility, resolving KeyError.
* **Speed**: keeps the NumPy‑based ASCII write.
* **NetCDF output**: compression, 32‑bit floats, sensible `encoding`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
import xarray as xr
import dask
from dask.diagnostics import ProgressBar

FILL_VALUE = np.float32(1e20)
ENC_NC = dict(zlib=True, complevel=4, dtype="float32", _FillValue=FILL_VALUE)

# ───────────────────────── helpers ──────────────────────────


def calc_center_lat_lon(region: str, f: Path) -> tuple[float, float]:
    """Calculate center lat/lon for predefined regions."""
    with open("../../regions.yaml", "r") as f:
        SUBREGIONS = yaml.safe_load(f)
    if region not in SUBREGIONS:
        (f"{region} not in regions.yaml. Available: {list(SUBREGIONS.keys())}")

    b = SUBREGIONS[region]
    return (b["lat_min"] + b["lat_max"]) / 2, (b["lon_min"] + b["lon_max"]) / 2


def regrid_bilinear(
    da: xr.DataArray, lon_step: float = 1.0, lat_step: float = 1.0
) -> xr.DataArray:
    """Bilinear regridding (e.g., 0.25° → 1°); Dask‑safe."""
    print(f"Regridding {da.name} to {lat_step}x{lon_step} degree...")
    lon_new = np.arange(np.floor(da.lon.min()), np.ceil(da.lon.max()) + 1e-6, lon_step)
    lat_new = np.arange(np.floor(da.lat.min()), np.ceil(da.lat.max()) + 1e-6, lat_step)
    # Ensure longitude wraps correctly if needed (though less critical for regional domains)
    if lon_new.max() > 180 and da.lon.min() >= 0:
        # Crude check if we might need wrapping logic for global data
        pass  # Add wrapping logic if necessary for global data conversion

    # Preserve attributes
    original_attrs = da.attrs
    regridded_da = da.interp(lon=lon_new, lat=lat_new, method="linear")
    # Ensure dtype doesn't accidentally change (e.g., to float64)
    regridded_da = regridded_da.astype(da.dtype)
    regridded_da.attrs = original_attrs
    return regridded_da


def gaussian_lowpass_31day(da: xr.DataArray, sigma: float = 7.3) -> xr.DataArray:
    """
    Centre‑weighted 31‑day Gaussian low-pass filter (no edge loss, NaN aware).
    Sigma chosen to approximate a 31-day window effect.
    Uses xarray's rolling().dot() method for robustness with dask.
    """
    window_size = 31
    print(f"Applying Gaussian low-pass filter (sigma={sigma}, window={window_size})...")

    # Define the Gaussian weights as an xarray DataArray
    window_dim = "window"
    weights = xr.DataArray(
        np.exp(-0.5 * ((np.arange(window_size) - window_size // 2) / sigma) ** 2),
        dims=[window_dim],
    )
    weights /= weights.sum()  # Normalize weights

    # Pad the array to handle boundaries; mode='reflect' is often suitable
    # Use pad_width that matches half the window size
    pad_width = window_size // 2
    padded_da = da.pad(time=(pad_width, pad_width), mode="reflect")

    # Apply the rolling operation and dot product with weights
    # Ensure rolling happens along the 'time' dimension
    # Need to handle NaNs: convolve data and valid mask separately
    # 1. Convolve padded data (NaNs treated as 0 for convolution)
    data_vals = padded_da.fillna(0)
    rolled_data = data_vals.rolling(time=window_size, center=True).construct(window_dim)
    smoothed_data = rolled_data.dot(weights, dims=window_dim)

    # 2. Convolve a mask of valid (non-NaN) data points
    mask = xr.where(padded_da.notnull(), 1.0, 0.0)  # 1 where valid, 0 where NaN
    rolled_mask = mask.rolling(time=window_size, center=True).construct(window_dim)
    smoothed_mask = rolled_mask.dot(weights, dims=window_dim)

    # 3. Combine: Divide smoothed data by smoothed mask, avoid division by zero
    epsilon = 1e-6  # To prevent division by very small numbers
    filtered_padded = xr.where(
        smoothed_mask > epsilon, smoothed_data / smoothed_mask, np.nan
    )

    # Remove the padding by selecting the original time range
    # Use .isel to select by index, robust to time coordinate details
    original_time_indices = slice(pad_width, -pad_width)
    filtered_da = filtered_padded.isel(time=original_time_indices)

    # Restore original variable name and attributes
    filtered_da = filtered_da.rename(da.name)
    filtered_da.attrs = da.attrs
    # Ensure dtype is float32 if original was, as processing might change it
    if da.dtype == np.float32 or da.dtype == np.float64:
        filtered_da = filtered_da.astype(np.float32)

    return filtered_da


# --- Preprocessing Functions ---
def normalize_data(da: xr.DataArray) -> xr.DataArray:
    """
    Mimics cost733class @nrm:1 (row-wise/object centering).
    Subtracts the spatial mean of each time step from that time step's field.
    """
    print("Applying row-wise centering (@nrm:1)...")
    # Keep attributes to reapply later
    original_attrs = da.attrs
    original_name = da.name
    # Calculate spatial mean, ensuring it keeps the 'time' dimension
    spatial_mean_per_timestep = da.mean(
        dim=["lat", "lon"], skipna=True, keep_attrs=True
    )
    # Subtract the spatial mean (broadcasts along time)
    normalized_da = da - spatial_mean_per_timestep
    # Restore name and attributes
    normalized_da = normalized_da.rename(original_name)
    normalized_da.attrs = original_attrs
    # Ensure dtype is float32 if original was
    if da.dtype == np.float32 or da.dtype == np.float64:
        normalized_da = normalized_da.astype(np.float32)
    return normalized_da


def high_pass_filter(da: xr.DataArray, sigma: float = 7.3) -> xr.DataArray:
    """
    Mimics cost733class @fil:-31 (Gaussian high-pass filter).
    Calculated as: Original Data - Low-Pass Filtered Data.
    Uses a 31-day Gaussian low-pass filter.
    """
    print("Applying 31-day high-pass filter (@fil:-31)...")
    # Keep attributes to reapply later
    original_attrs = da.attrs
    original_name = da.name
    # Calculate the low-pass filtered version
    low_pass_da = gaussian_lowpass_31day(da, sigma=sigma)
    # Subtract the low-pass from the original to get the high-pass
    high_pass_da = da - low_pass_da
    # Restore name and attributes
    high_pass_da = high_pass_da.rename(original_name)
    high_pass_da.attrs = original_attrs
    # Ensure dtype is float32 if original was
    if da.dtype == np.float32 or da.dtype == np.float64:
        high_pass_da = high_pass_da.astype(np.float32)
    return high_pass_da


# --- End Preprocessing ---


def save_ascii(da: xr.DataArray, out_path: Path, fmt: str = "%.3f") -> None:
    """
    Write the DataArray to a COST733-compatible ASCII file.
    Transposes to (time, lat, lon), sorts lat/lon, flattens spatial dims,
    and saves the matrix. Uses a slightly higher precision format.
    """
    print(f"Preparing ASCII output for {out_path}...")
    # Ensure correct order: time varies slowest, lon varies fastest within lat
    mat = (
        da.transpose("time", "lat", "lon")
        .sortby("lat", ascending=True)  # COST733 often expects South-to-North
        .sortby("lon", ascending=True)  # West-to-East
        .fillna(FILL_VALUE)  # Use fill value for NaNs
        .astype(np.float32)  # Ensure float32
        .values.reshape(da.sizes["time"], -1)
    )  # Reshape time x (lat*lon)

    # Save the matrix
    np.savetxt(out_path, mat, fmt=fmt)
    print(f"💾  Wrote {out_path}  ({mat.shape[0]} rows × {mat.shape[1]} cols)")


# ───────────────────────── main ─────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert daily NetCDF to COST733 ASCII + NC (fast) with manual preprocessing"
    )
    p.add_argument(
        "--inpath",
        type=Path,
        default="/home/dkn/ERA5/z500_EUR_daily_2001_2020.nc",
        help="input NetCDF file",
    )
    p.add_argument(
        "--regions_file",
        type=Path,
        default="../../regions.yaml",
        help=".yaml file including the SUBREGIONS",
    )
    p.add_argument(
        "--outpath",
        type=Path,
        default="/home/dkn/ERA5/z500_daily_2001_2020_filtered",  # Changed default name
        help="output root (without extension), '_filtered' suffix recommended",
    )
    p.add_argument(
        "--data_var",
        default="z500",
        help="variable name in NetCDF (e.g. z500, msl, z850)",
    )
    p.add_argument(
        "--chunks", default="time:365", help="dask chunk spec, e.g. time:365"
    )
    p.add_argument(
        "--skip_filter",
        action="store_true",
        help="Skip normalization and filtering steps",
    )
    args = p.parse_args()

    if not args.inpath.exists():
        sys.exit(f"Input file {args.inpath} not found")

    # Setup Dask client (optional, but good practice for managing resources)
    # Adjust workers/threads based on your system
    # from dask.distributed import Client, LocalCluster
    # cluster = LocalCluster(n_workers=4, threads_per_worker=8) # Example config
    # client = Client(cluster)
    # print(f"Dask dashboard link: {client.dashboard_link}")
    dask.config.set(
        scheduler="threads"
    )  # Use threads scheduler if not using distributed

    chunk_dict = {k: int(v) for k, v in (c.split(":") for c in args.chunks.split(","))}
    print(f"Opening dataset {args.inpath} with chunks: {chunk_dict}")
    # Using engine='h5netcdf' might be necessary for some compressed files
    try:
        ds = xr.open_dataset(args.inpath, chunks=chunk_dict)
    except ValueError as e:
        print(f"Warning: Error opening with default engine: {e}. Trying h5netcdf...")
        try:
            ds = xr.open_dataset(args.inpath, chunks=chunk_dict, engine="h5netcdf")
        except Exception as e2:
            sys.exit(f"Failed to open dataset {args.inpath} with both engines: {e2}")

    # Ensure ascending latitude (South to North)
    if ds.lat.values[0] > ds.lat.values[-1]:
        print("Reversing latitude dimension to be ascending (South to North).")
        ds = ds.reindex(lat=np.sort(ds.lat.values))

    # Select the data variable
    if args.data_var not in ds:
        sys.exit(
            f"Error: Variable '{args.data_var}' not found in {args.inpath}. Available variables: {list(ds.data_vars)}"
        )
    da_full = ds[args.data_var]
    # Ensure data is float32 for processing
    if da_full.dtype != np.float32:
        print(f"Converting data type from {da_full.dtype} to float32.")
        da_full = da_full.astype(np.float32)

    # Define regions to process
    regions_to_process = ["Alps", "Balcan", "Eastern_Europe", "France"]

    for region in regions_to_process:
        print(f"\n--- Processing region: {region} ---")
        try:
            lat_c, lon_c = calc_center_lat_lon(region, args.regions_file)
        except ValueError as e:
            print(f"Skipping region {region}: {e}")
            continue  # Skip to the next region if definition is missing

        # Regrid to 1 degree
        da_reg = regrid_bilinear(da_full, 1.0, 1.0)

        # Select spatial domain
        lat_min, lat_max = lat_c - 10, lat_c + 10
        lon_min, lon_max = lon_c - 15, lon_c + 15
        print(f"Selecting domain: Lat {lat_min}-{lat_max}, Lon {lon_min}-{lon_max}")
        # Use .sel for selection
        da_reg = da_reg.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        # Check if the selection resulted in valid dimensions
        if da_reg.sizes["lat"] == 0 or da_reg.sizes["lon"] == 0:
            print(
                f"Warning: Spatial selection for region '{region}' resulted in zero size. Skipping."
            )
            continue

        # --- Apply Manual Preprocessing ---
        if not args.skip_filter:
            # 1. High-pass filter (@fil:-31)
            da_processed = high_pass_filter(da_reg)
            # 2. Normalize (@nrm:1)
            da_processed = normalize_data(da_processed)

        else:
            print("Skipping manual normalization and filtering steps.")
            da_processed = da_reg  # Use original regional data if skipping

        # Ensure the variable has a name (can be lost in processing)
        if da_processed.name is None:
            da_processed = da_processed.rename(args.data_var)

        # Define output paths
        root = Path(str(args.outpath))  # Use the specified output path root
        # Add region to the filename stem
        ascii_path = root.with_stem(root.stem + f"_{region}").with_suffix(".dat")
        nc_path = ascii_path.with_suffix(".nc")
        ascii_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure output dir exists

        # Transpose for saving
        da_save = da_processed.transpose("time", "lat", "lon")

        # --- Compute and Save ---
        print("Computing processed data (if using Dask)...")
        with ProgressBar():
            # Compute the Dask array before saving to avoid potential issues with lazy saving
            # Especially important after complex operations like filtering
            da_save_computed = da_save.compute()

            # Save ASCII
            save_ascii(da_save_computed, ascii_path)  # Pass the computed array

            # Save NetCDF
            print(f"Saving NetCDF to {nc_path}...")
            # Define encoding for the specific variable
            encoding_var = {da_save_computed.name: ENC_NC}
            da_save_computed.astype("float32").to_netcdf(nc_path, encoding=encoding_var)
            print(f"💾  Wrote {nc_path}")

    print("\n--- Processing complete ---")


if __name__ == "__main__":
    main()
