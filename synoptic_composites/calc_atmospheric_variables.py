#!/usr/bin/env python3
"""
Module for calculating derived atmospheric variables from xarray Datasets.
"""
import xarray as xr
import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def calculate_theta_e_on_levels(ds_input_tq: xr.Dataset) -> xr.Dataset:
    """
    Calculates equivalent potential temperature (theta_e) from t and q
    on multiple pressure levels within an xarray.Dataset. Input t and q are
    expected to be instantaneous values.

    Args:
        ds_input_tq: An xarray.Dataset expected to contain 't' (temperature in K)
                     and 'q' (specific humidity in kg/kg) as data variables,
                     and 'level' as a coordinate (pressure in hPa).
                     Data for 't' and 'q' should ideally be loaded into memory
                     (e.g., via .load()) before calling this function if they are Dask arrays.
    Returns:
        An xarray.Dataset containing the calculated 'theta_e' DataArray, preserving
        the original coordinates (time, level, lat, lon).
    """
    if not ({'t', 'q'}.issubset(ds_input_tq.data_vars)):
        raise ValueError("Input Dataset for theta_e calculation must contain 't' and 'q' data variables.")
    if 'level' not in ds_input_tq.coords:
        raise ValueError("Input Dataset for theta_e calculation must have 'level' coordinate (in hPa).")

    logger.debug("Calculating theta_e on levels using iterative approach...")

    t_da = ds_input_tq['t']
    q_da = ds_input_tq['q']
    pressure_levels_hpa = ds_input_tq['level'].values

    # List to store theta_e DataArrays for each level
    theta_e_levels_list = []

    for p_hpa in pressure_levels_hpa:
        logger.debug(f"  Processing level: {p_hpa} hPa")
        t_at_level = t_da.sel(level=p_hpa).data # Get NumPy array
        q_at_level = q_da.sel(level=p_hpa).data # Get NumPy array
        
        p_units = p_hpa * units.hPa
        theta_e_result_np = np.full_like(t_at_level, np.nan, dtype=np.float32)

        # Ensure inputs to MetPy are NumPy arrays with units
        t_units_np = t_at_level * units.kelvin
        q_units_np = q_at_level * units('kg/kg')
        
        # Create a mask for valid calculation points
        valid_mask = np.isfinite(t_at_level) & np.isfinite(q_at_level) & (t_at_level > 0) & (q_at_level >= -1e-5)
        
        # Clip q to be non-negative for calculations, only for valid points
        q_units_for_calc = q_units_np.copy() # Operate on a copy
        q_units_for_calc[~valid_mask] = np.nan # Set invalid points to NaN before potential clipping
        q_units_for_calc[q_units_for_calc < 0 * units('kg/kg')] = 0 * units('kg/kg')


        if np.any(valid_mask):
            # Select only valid data for MetPy functions to avoid issues with NaNs/Infs
            t_v = t_units_np[valid_mask]
            q_v_candidate = q_units_for_calc[valid_mask] 
            
            # Cap specific humidity at 1.2 * saturation mixing ratio
            sat_mr = mpcalc.saturation_mixing_ratio(p_units, t_v)
            q_v = np.minimum(q_v_candidate, 1.2 * sat_mr)
            
            dewpoint = mpcalc.dewpoint_from_specific_humidity(p_units, t_v, q_v)
            theta_e_values = mpcalc.equivalent_potential_temperature(p_units, t_v, dewpoint)
            # Place calculated values back into the full-shaped array using the mask
            theta_e_result_np[valid_mask] = theta_e_values.magnitude
        
        # Create a DataArray for this level's theta_e
        # Preserve original non-level coordinates from t_da (or q_da)
        coords_for_level = {
            dim: t_da.coords[dim] for dim in t_da.dims if dim != 'level'
        }
        theta_e_level_da = xr.DataArray(
            theta_e_result_np,
            coords=coords_for_level,
            dims=[dim for dim in t_da.dims if dim != 'level'],
            name='theta_e_temp_level' # Temporary name
        )
        theta_e_levels_list.append(theta_e_level_da)

    # Concatenate the list of DataArrays along the 'level' dimension
    if not theta_e_levels_list:
        logger.warning("No theta_e data was calculated for any level.")
        # Return an empty or NaN-filled dataset matching expected structure
        # This part might need refinement based on how you want to handle fully empty results
        return xr.Dataset({'theta_e': xr.DataArray(
            np.nan, 
            coords=ds_input_tq.coords, # Use original coords
            dims=ds_input_tq['t'].dims, # Use original dims
            name='theta_e'
        )})


    final_theta_e_da = xr.concat(theta_e_levels_list, dim=pd.Index(pressure_levels_hpa, name='level'))
    final_theta_e_da = final_theta_e_da.rename('theta_e')
    final_theta_e_da.attrs = {
        'long_name': 'Equivalent potential temperature', 
        'units': 'K',
        'calculation_details': 'Calculated iteratively per level using MetPy from t and q'
    }
    
    logger.debug("Theta_e calculation complete (iterative approach).")
    return final_theta_e_da.to_dataset()
