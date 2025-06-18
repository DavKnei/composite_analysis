# Weather Typing Workflow

This dir of the repository contains the scripts and workflow used for weather typing based on ERA5 reanalysis data. The process involves a two-step preprocessing phase followed by a classification step using the `cost733class` tool.

## Workflow Overview

The weather typing procedure is executed as follows:

1.  **Initial ERA5 Preprocessing**: Hourly ERA5 pressure level data is processed to create daily-mean NetCDF files. This step ensures the data has the correct temporal resolution and dimensional structure (`time`, `lat`, `lon`) for subsequent analysis. This is performed by the `preprocess_ERA5_cost733.py` script.
2.  **Conversion to ASCII and Further Preprocessing**: The daily-mean NetCDF files are then converted to the ASCII format required by `cost733class`. During this step, the data is also regridded, spatially subsetted for specific regions, and undergoes filtering. This is handled by the `convert_netcdf_to_ascii.py` script. Both preprocessing scripts are located in the `preprocess/` subdirectory.
3.  **Weather Typing**: The final weather typing (classification) is performed using the `cost733class` command-line tool, orchestrated by the `run_cost733_ascii.py` script located in the root of this specific experiment's subdirectory. This script takes the preprocessed ASCII files as input.

## Step 1: Initial ERA5 Preprocessing

The script `preprocess/preprocess_ERA5_cost733.py` is responsible for the initial preparation of ERA5 data.

* **Input**: Hourly ERA5 pressure level data (NetCDF format). For example, from a path like `/data/reloclim/normal/INTERACT/ERA5/pressure_levels`.
* **Key Operations**:
    * Takes a start year, end year, number of cores, input directory, and output directory as arguments.
    * Selects data for a predefined spatial domain (longitude: -20°E to 43°E, latitude: 65°N to 25°N).
    * For a specified variable (e.g., geopotential height 'z'):
        * Reads and concatenates monthly data files for each year.
        * Selects a specific pressure level (e.g., 500 hPa).
        * Calculates daily means from hourly data.
        * Renames coordinate dimensions to `lon` (longitude), `lat` (latitude), and `time`.
        * Transposes dimensions to the order (`time`, `lat`, `lon`).
    * Concatenates the processed daily data across all specified years.
    * The selected variable is renamed (e.g., 'z' becomes 'z500').
    * Uses Dask for efficient, chunked processing.
* **Output**: A single NetCDF file (e.g., `z500_Alps_daily_{start}_{end}.nc`) containing the daily-mean data for the specified variable, period, and domain, ready for the next stage. The output NetCDF is compressed (zlib, level 4) and uses 32-bit floats.

## Step 2: Conversion to ASCII and Further Preprocessing

The script `preprocess/convert_netcdf_to_ascii.py` takes the output from Step 1 and prepares it for the `cost733class` tool.

* **Input**:
    * Daily-mean NetCDF file generated in Step 1 (e.g., `z500_daily_2001_2020.nc`).
    * A `regions.yaml` file defining the geographical boundaries of analysis subregions.
* **Key Operations**:
    * Uses Dask for parallel processing.
    * Ensures latitude coordinates are in ascending order (South to North).
    * Converts data to `float32` precision if not already.
    * Iterates over a list of predefined regions (e.g., "southern_alps", "Alps"):
        * Calculates the center latitude and longitude for the current region based on `regions.yaml`.
        * **Regridding**: Performs bilinear regridding of the data to a 1.0° x 1.0° resolution using the `regrid_bilinear` function.
        * **Domain Selection**: Selects a spatial subset of the data centered on the region (typically ±10° latitude and ±15° longitude from the center).
        * **Data Filtering (applied by default, can be skipped with `--skip_filter`):**
            * **High-pass Filter (`@fil:-31`)**: A 31-day Gaussian high-pass filter is applied. This is achieved by subtracting a 31-day Gaussian low-pass filtered version of the data from the original data. The low-pass filter (`gaussian_lowpass_31day`) is NaN-aware.
        * The processed data is transposed to (`time`, `lat`, `lon`) order.
        * Dask computations are triggered to process the data.
* **Output (for each region)**:
    * **ASCII File (`.dat`)**: A COST733-compatible ASCII file (e.g., `z500_daily_2001_2020_filtered_eastern_alps.dat`). Data is sorted by latitude (ascending) and longitude (ascending). Missing values are replaced with a `FILL_VALUE` (1e20). The spatial dimensions are flattened, resulting in a matrix of `time` rows and `(lat*lon)` columns.
    * **NetCDF File (`.nc`)**: A compressed (zlib, level 4), 32-bit float NetCDF version of the processed data (e.g., `z500_daily_2001_2020_filtered_eastern_alps.nc`) for verification and inspection.

## Step 3: Weather Typing with `run_cost733_ascii.py`

The script `run_cost733_ascii.py` (located in this directory) executes the `cost733class` tool for weather typing.

* **Input**:
    * The ASCII `.dat` files generated in Step 2. These are expected to be in a specified data directory (default: `/home/dkn/ERA5`).
    * **Method**: The classification method (e.g., `GWT_Z500`, `CAP_MSLZ500`).
        * **Current Default (to be potentially changed in script)**: `GWT_Z500`.
    * **Number of Classes (`--ncl`)**: The desired number of weather types.
        * **Current Default (to be potentially changed in script)**: `10`.
    * **Region**: The specific region (e.g., `Alps`) corresponding to the input file.
    * **Start Date**: The start date of the analysis period (e.g., `2001-01-01`).
    * **Months**: A comma-separated list of months to include in the classification (e.g., `6,7,8` for JJA). Default is all months.
* **Key Operations**:
    * The script maps the chosen `--method` to the required input ASCII file patterns.
    * It dynamically constructs the `-dat` flag for `cost733class` by extracting metadata (time range, lat/lon bounds) from the corresponding `.nc` file associated with each `.dat` input file.
    * **Classification Process**:
        * For `CAP` (Cluster Analysis of Principal components) methods:
            1.  An initial partition is created using Hierarchical Clustering (`HCL`).
            2.  This partition is then refined using k-means (`KMN`).
        * For other methods (like `GWT` - Grosswettertypen): `cost733class` is run directly with the specified method.
    * The output classification file (`.cla`) from `cost733class` is converted to a more user-friendly CSV format.
* **Output**:
    * A classification file in `.cla` format (e.g., `GWT_Z500_Alps_ncl10.cla`).
    * A CSV version of the classification (e.g., `GWT_Z500_Alps_ncl10.csv`), with 'datetime' and 'wt' (weather type) columns.
    * Output files are saved in a specified output directory (default: `./csv`).

## How to Run

1.  **Ensure ERA5 data is available** for `preprocess/preprocess_ERA5_cost733.py`. Modify paths within the script if necessary.
    ```bash
    # Example execution (adjust parameters as needed)
    python preprocess/preprocess_ERA5_cost733.py --start 2001 --end 2020 --ncores 8 --pl-dir /path/to/hourly/era5 --out-dir /home/dkn/ERA5
    ```
2.  **Run the NetCDF to ASCII conversion**. Make sure your `regions.yaml` is correctly configured.
    ```bash
    # Example execution (ensure input file from step 1 exists and regions.yaml is present)
    python preprocess/convert_netcdf_to_ascii.py --inpath /home/dkn/ERA5/z500_gb_daily_2001_2020.nc --regions_file preprocess/regions.yaml --outpath /home/dkn/ERA5/z500_daily_2001_2020_filtered --data_var z500
    ```
    *Note: The default input path in `convert_netcdf_to_ascii.py` is `/home/dkn/ERA5/z500_daily_2001_2020.nc`. The output path root is `/home/dkn/ERA5/z500_daily_2001_2020_filtered`, so output files will be named like `z500_daily_2001_2020_filtered_eastern_alps.dat`.*
3.  **Perform the weather typing**. Adjust the method, number of classes, region, and other parameters as needed.
    ```bash
    # Example for GWT_Z500 with 10 classes for Alps
    python run_cost733_ascii.py --method GWT_Z500 --ncl 10 --region Alps --start 2001-01-01
    ```
    *Ensure the `DATA_DIR` in `run_cost733_ascii.py` points to where the `.dat` and `.nc` files from step 2 are stored (default `/home/dkn/ERA5`). The output CSVs will be in `./csv/`.*

## Dependencies

* Python 3
* xarray
* numpy
* pandas
* dask
* PyYAML (for `convert_netcdf_to_ascii.py` to read `regions.yaml`)
* `cost733class` command-line tool (must be in system PATH or its location specified).

