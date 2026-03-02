# SHYFEM Antsiranana Bay Analysis

## Overview

This notebook analyzes SHYFEM ocean model output for Antsiranana Bay, Madagascar (March 31 - April 5, 2021).

It uses **xugrid** and **hvplot** (from the OpenScienceComputing fork) for handling unstructured grid data from ocean models.

## Installation

### Option 1: Conda
```bash
conda create -n shyfem python=3.11 -y
conda activate shyfem
conda install -c conda-forge xugrid hvplot holoviews panel bokeh xarray netcdf4 pandas matplotlib
```

### Option 2: Pip
```bash
python -m venv shyfem-env
source shyfem-env/bin/activate
pip install -r requirements.txt
```

## Requirements

See `requirements.txt` for the full list. Key packages:
- **xugrid** - For handling unstructured UGRID data
- **hvplot** - Interactive plotting (with xugrid support)
- **holoviews** - Visualization library
- **panel** - For interactive dashboards
- **xarray** - For N-dimensional data arrays
- **netcdf4** - For reading NetCDF files

## Launch Notebook

```bash
jupyter notebook SHYFEM_Analysis.ipynb
```

## Data

The notebook loads data from:
- `surf.nos.nc` - Raw SHYFEM output
- `surf.ous.nc` - Processed SHYFEM output

These are loaded directly from the public URL using xugrid.

## Features

- Load unstructured grid data with xugrid
- Interactive temperature, salinity, and velocity plots
- Vector field visualization
- Time series analysis
- Summary statistics
- Export to HTML

## Credits

Based on the FVCOM notebook by Rich Signell:
https://gist.github.com/rsignell/2ee54d379b15be4b8a101c5dec68bd6d

Modified to use xugrid for SHYFEM data.
