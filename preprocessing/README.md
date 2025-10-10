# GPCRactDB Preprocessing Pipeline

This directory contains the scripts to build the GPCRactDB from raw public data.
The pipeline is divided into four sequential steps. Please run the scripts in numerical order.

## Prerequisites

- Ensure all dependencies from the main `environment.yml` are installed.

## Step-by-Step Protocol

### Step 1: Parse and Integrate Raw Data

This script collects data from various public sources located in `data/raw/` and integrates them into a unified format.

```bash
python preprocessing/01_parse_raw_data.py
