# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
DEBUG VERSION - CBottle Video Conditional Animation
"""

import torch
import numpy as np
import os
import sys
import traceback
from datetime import datetime, timedelta

print("=== IMPORT PHASE ===")
print("Importing earth2studio modules...")

try:
    from earth2studio.data import WB2ERA5
    print("✓ Imported WB2ERA5")
except Exception as e:
    print(f"✗ Failed to import WB2ERA5: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from earth2studio.lexicon import WB2Lexicon, CBottleLexicon
    print("✓ Imported Lexicons")
except Exception as e:
    print(f"✗ Failed to import Lexicons: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from earth2studio.data.utils import fetch_data
    print("✓ Imported fetch_data")
except Exception as e:
    print(f"✗ Failed to import fetch_data: {e}")
    traceback.print_exc()
    sys.exit(1)

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n=== SETUP ===")
print(f"Using device: {device}")

# Generate times
def generate_datetime_array(year, month, day, hour=0, minute=0, second=0):
    start_time = datetime(year, month, day, hour, minute, second)
    times = [start_time + timedelta(hours=6*i) for i in range(12)]
    return np.array(times, dtype="datetime64[ns]")

date_dict = {'year': 2022, 'month': 6, 'day': 1}
times = generate_datetime_array(date_dict['year'], date_dict['month'], date_dict['day'])
print(f"Generated {len(times)} timestamps")

# Determine variables
print("\n=== VARIABLE DETERMINATION ===")
wb2_vars = set(WB2Lexicon.VOCAB.keys())
cbottle_vars = list(CBottleLexicon.VOCAB.keys())
available_in_era5 = sorted([v for v in cbottle_vars if v in wb2_vars])
print(f"Using {len(available_in_era5)} ERA5 variables")

# Create data source
print("\n=== DATA SOURCE CREATION ===")
try:
    era5_ds = WB2ERA5()
    print("✓ Created WB2ERA5 data source")
except Exception as e:
    print(f"✗ Failed to create WB2ERA5: {e}")
    traceback.print_exc()
    sys.exit(1)

# Fetch data with detailed error tracking
print("\n=== DATA FETCH ===")
print("About to call fetch_data...")
print(f"  times shape: {times.shape}")
print(f"  num variables: {len(available_in_era5)}")
print(f"  device: {device}")

try:
    # Call __call__ directly to avoid fetch_data wrapper
    print("\nTrying direct data source call...")
    da = era5_ds(times, available_in_era5)
    print(f"✓ Data source returned xarray with shape: {da.shape}")
    print(f"  Data dims: {da.dims}")
    print(f"  Data coords: {list(da.coords.keys())}")

    print("\nConverting to torch tensor on CPU first...")
    data_cpu = torch.Tensor(da.values)
    print(f"✓ Created CPU tensor with shape: {data_cpu.shape}")

    print(f"\nMoving tensor to {device}...")
    data_gpu = data_cpu.to(device)
    print(f"✓ Moved tensor to {device}")

    print("\n✓✓✓ SUCCESS - No HEALPix error!")

except Exception as e:
    print(f"\n✗✗✗ ERROR OCCURRED:")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)

print("\n=== TEST COMPLETE ===")
