# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CBottle Video Conditional Animation
====================================

Create conditional weather forecast animations using CBottleVideo with ERA5 data.
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors
import numpy as np
import pandas as pd
import os
import gc
from datetime import datetime
import cartopy.crs as ccrs

from earth2studio.data import WB2ERA5
from earth2studio.lexicon import WB2Lexicon, CBottleLexicon
from earth2studio.models.dx import CBottleInfill
from earth2studio.models.px import CBottleVideo
from earth2studio.data.utils import fetch_data

# Configuration
os.makedirs("outputs", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters - CONFIGURE THESE
VIDEO_VARIABLE = "t850"
INITIALIZATION_TIME = datetime(2022, 6, 1)
N_FRAMES = 12  # 0-66 hours in 6-hour steps
SEED = 42

# ============================================================================
# Step 1: Determine Available ERA5 Variables
# ============================================================================
print("\nDetermining available ERA5 variables...")
wb2_vars = set(WB2Lexicon.VOCAB.keys())
cbottle_vars = list(CBottleLexicon.VOCAB.keys())
available_in_era5 = sorted([v for v in cbottle_vars if v in wb2_vars])

print(f"Using {len(available_in_era5)} ERA5 variables for conditioning")

# ============================================================================
# Step 2: Fetch and Infill ERA5 Data
# ============================================================================
print("\nFetching ERA5 data...")
era5_ds = WB2ERA5()
times = np.array([INITIALIZATION_TIME], dtype="datetime64[ns]")

# Fetch ERA5 data ONCE
era5_x, era5_coords = fetch_data(era5_ds, times, available_in_era5, device=device)
print(f"ERA5 data shape: {era5_x.shape}")

# Load and run CBottleInfill
print("\nRunning CBottleInfill...")
package_infill = CBottleInfill.load_default_package()
cbottle_infill = CBottleInfill.load_model(
    package_infill,
    input_variables=available_in_era5,
    sampler_steps=18
)
cbottle_infill = cbottle_infill.to(device)
cbottle_infill.set_seed(SEED)

# Run infilling
infilled_x, infilled_coords = cbottle_infill(era5_x, era5_coords)
print(f"Infilled data shape: {infilled_x.shape}")

# CRITICAL: Free CBottleInfill and ERA5 data from GPU
del cbottle_infill, era5_x, era5_coords
torch.cuda.empty_cache()
gc.collect()
print("✓ Freed CBottleInfill from GPU memory")

# ============================================================================
# Step 3: Prepare Conditional Input
# ============================================================================
print("\nPreparing conditional input...")

# Add batch dimension if needed
if len(infilled_x.shape) == 5:  # [time, lead_time, variable, lat, lon]
    x_cond = infilled_x.unsqueeze(0)  # [batch, time, lead_time, variable, lat, lon]
else:
    x_cond = infilled_x

print(f"Conditional input shape: {x_cond.shape}")

# Optional: Apply masking (condition only on first/last timesteps)
# x_cond_masked = x_cond.clone()
# x_cond_masked[:, 1:-1, :, :, :, :] = float('nan')  # Mask middle timesteps
# x_cond = x_cond_masked

# Setup coordinates
coords_cond = {
    "time": infilled_coords["time"] if "time" in infilled_coords else times,
    "batch": np.array([0]),
    "variable": infilled_coords["variable"],
    "lead_time": infilled_coords.get("lead_time", np.array([np.timedelta64(0, "h")])),
    "lat": infilled_coords["lat"],
    "lon": infilled_coords["lon"],
}

# ============================================================================
# Step 4: Run CBottleVideo Inference
# ============================================================================
print("\nLoading CBottleVideo...")
package_video = CBottleVideo.load_default_package()
cbottle_video = CBottleVideo.load_model(package_video, seed=SEED)
cbottle_video = cbottle_video.to(device)

print("Running conditional video generation...")
iterator = cbottle_video.create_iterator(x_cond, coords_cond)

# CRITICAL: Move x_cond off GPU after starting iterator
# (Keep a CPU copy if you need it later)
x_cond_cpu = x_cond.cpu()
del x_cond, infilled_x, infilled_coords
torch.cuda.empty_cache()
print("✓ Moved input data to CPU")

# Collect outputs (moved to CPU immediately)
outputs = []
coords_list = []
for step, (output, output_coords) in enumerate(iterator):
    lead_time = output_coords["lead_time"][0]
    hours = int(lead_time / np.timedelta64(1, "h"))
    print(f"  Step {step}: +{hours}h")

    # CRITICAL: Move to CPU immediately
    outputs.append(output.cpu())
    coords_list.append(output_coords)

    # Free GPU memory after each step
    del output
    if step % 3 == 0:  # Periodic cleanup
        torch.cuda.empty_cache()

    if step >= N_FRAMES - 1:
        break

# CRITICAL: Free CBottleVideo from GPU
del cbottle_video
torch.cuda.empty_cache()
gc.collect()
print("✓ Freed CBottleVideo from GPU memory")

# ============================================================================
# Step 5: Create Animation
# ============================================================================
print(f"\nCreating animation for variable: {VIDEO_VARIABLE}")

# Find variable index
var_names = coords_list[0]["variable"]
try:
    var_idx = np.where(var_names == VIDEO_VARIABLE)[0][0]
except IndexError:
    print(f"Error: Variable '{VIDEO_VARIABLE}' not found!")
    print(f"Available variables: {list(var_names)}")
    exit(1)

# Setup plot
plt.style.use("dark_background")
projection = ccrs.Orthographic(central_longitude=0.0, central_latitude=45.0)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection=projection)

# Get data range for colormap
data_min = min(outputs[i][0, 0, 0, var_idx].min() for i in range(len(outputs)))
data_max = max(outputs[i][0, 0, 0, var_idx].max() for i in range(len(outputs)))
norm = matplotlib.colors.Normalize(vmin=data_min, vmax=data_max)

# First frame
data_first = outputs[0][0, 0, 0, var_idx].numpy()
img = ax.pcolormesh(
    coords_list[0]["lon"],
    coords_list[0]["lat"],
    data_first,
    transform=ccrs.PlateCarree(),
    cmap="viridis",
    norm=norm,
)
ax.coastlines()
ax.gridlines()
plt.colorbar(img, ax=ax, orientation="horizontal", shrink=0.5, pad=0.05,
             label=f"{VIDEO_VARIABLE}")

# Initial title
lead_time = coords_list[0]["lead_time"][0]
hours = int(lead_time / np.timedelta64(1, "h"))
time_str = pd.Timestamp(coords_list[0]["time"][0] + lead_time).strftime("%Y-%m-%d %H:%M")
title = ax.set_title(f"Conditional: {VIDEO_VARIABLE} +{hours:03d}h ({time_str})")
fig.tight_layout()

def update(frame):
    """Update animation frame"""
    data = outputs[frame][0, 0, 0, var_idx].numpy()
    img.set_array(data.ravel())

    lead_time = coords_list[frame]["lead_time"][0]
    hours = int(lead_time / np.timedelta64(1, "h"))
    time_str = pd.Timestamp(coords_list[frame]["time"][0] + lead_time).strftime("%Y-%m-%d %H:%M")
    title.set_text(f"Conditional: {VIDEO_VARIABLE} +{hours:03d}h ({time_str})")
    return [img, title]

# Create and save animation
print("Rendering animation...")
anim = animation.FuncAnimation(fig, update, frames=len(outputs), interval=500, blit=True)

output_file = f"outputs/21_cbottle_conditional_{VIDEO_VARIABLE}.mp4"
writer = animation.FFMpegWriter(fps=2)
anim.save(output_file, writer=writer, dpi=100)
plt.close()

print(f"✓ Animation saved to {output_file}")
print("\nMemory-efficient execution complete!")
