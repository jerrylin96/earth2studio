
import cbottle.inference
import torch
from cbottle.visualizations import visualize
import cbottle.netcdf_writer
from cbottle.datasets.dataset_3d import get_dataset
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors
from dataclasses import dataclass
from typing import Annotated
import numpy as np
import pandas as pd
import os
import sys
import gc
import xarray as xr

from datetime import datetime, timedelta
import earth2grid
from earth2grid import healpix
import cartopy.crs as ccrs

from earth2studio.data import WB2ERA5, CBottle3D
from earth2studio.lexicon import WB2Lexicon, CBottleLexicon
from earth2studio.models.dx import CBottleInfill
from earth2studio.data.utils import fetch_data
from earth2studio.lexicon.wb2 import WB2Lexicon
from earth2studio.lexicon.cbottle import CBottleLexicon
from earth2studio.models.px import CBottleVideo
from earth2studio.io import ZarrBackend

from cbottle.datasets.merged_dataset import TimeMergedMapStyle
from cbottle.dataclass_parser import parse_args, healpix_order

import tempfile
from tqdm import tqdm

# Get the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the CBottleVideo model
package = CBottleVideo.load_default_package()
cbottle_video = CBottleVideo.load_model(package, seed=42)
cbottle_video = cbottle_video.to(device)

projection = ccrs.Orthographic(central_longitude=0.0, central_latitude=45.0)

video_variable = "t850"
video_var_idx = np.where(uncond_coords_list[0]["variable"] == video_variable)[0][0]

wb2_vars = list(WB2Lexicon.VOCAB.keys())
cbottle_vars = list(CBottleLexicon.VOCAB.keys())

available_in_era5 = []
missing_from_era5 = []

for cbottle_var in cbottle_vars:
    if cbottle_var in wb2_vars:
        available_in_era5.append(cbottle_var)
    else:
        missing_from_era5.append(cbottle_var)

available_in_era5 = sorted(available_in_era5)
missing_from_era5 = sorted(missing_from_era5)

# Load ERA5 data source
era5_ds = WB2ERA5()

# Load CBottleInfill model to generate all required variables
input_variables = available_in_era5

# Fetch ERA5 data
print("Fetching ERA5 data and running infill...")
era5_x, era5_coords = fetch_data(era5_ds, times, input_variables, device=device)

package_infill = CBottleInfill.load_default_package()
cbottle_infill = CBottleInfill.load_model(
    package_infill, input_variables=input_variables, sampler_steps=18
)
cbottle_infill = cbottle_infill.to(device)
cbottle_infill.set_seed(42)

# Fetch ERA5 data
print("Fetching ERA5 data and running infill...")
era5_x, era5_coords = fetch_data(era5_ds, times, input_variables, device=device)

# Infill to get all 45 CBottleVideo variables
infilled_x, infilled_coords = cbottle_infill(era5_x, era5_coords)

# Reshape for CBottleVideo input: [batch, time, lead_time, variable, lat, lon]
x_cond = infilled_x.unsqueeze(0)  # Add batch dimension
x_cond_masked = x_cond.clone()
del x_cond
gc.collect()
x_cond_masked[:,1:-1,:,:,:,:] = float('nan')

print(f"Conditioned input shape: {x_cond_masked.shape}")

# Update coordinates for CBottleVideo
coords_cond = cbottle_video.input_coords()
coords_cond["time"] = times
coords_cond["batch"] = np.array([0])
coords_cond["variable"] = infilled_coords["variable"]

# Run conditional inference
print("Running conditional generation...")
iterator_cond = cbottle_video.create_iterator(x_cond_masked, coords_cond)
cond_outputs = []
cond_coords_list = []
for step, (output, output_coords) in enumerate(iterator_cond):
    lead_time = output_coords["lead_time"][0]
    hours = int(lead_time / np.timedelta64(1, "h"))
    print(f"Step {step}: lead_time = +{hours}h")
    cond_outputs.append(output.cpu())
    cond_coords_list.append(output_coords)
    if step >= 11:  # Get first 12 frames (0-66 hours)
        break

fig_cond = plt.figure(figsize=(12, 8))
ax_cond = fig_cond.add_subplot(111, projection=projection)

# Set up the first frame
data_cond = cond_outputs[0][0, 0, 0, video_var_idx].numpy()

img_cond = ax_cond.pcolormesh(
    cond_coords_list[0]["lon"],
    cond_coords_list[0]["lat"],
    data_cond,
    transform=ccrs.PlateCarree(),
    cmap="viridis",
    norm=norm,
)
ax_cond.coastlines()
ax_cond.gridlines()
plt.colorbar(
    img_cond, ax=ax_cond, orientation="horizontal", shrink=0.5, pad=0.05, label="T850 (K)"
)

lead_time = cond_coords_list[0]["lead_time"][0]
hours = int(lead_time / np.timedelta64(1, "h"))
time_str = pd.Timestamp(
    cond_coords_list[0]["time"][0] + cond_coords_list[0]["lead_time"][0]
).strftime("%Y-%m-%d %H:%M")
title_cond = ax_cond.set_title(
    f"Conditional Generation (ERA5): {video_variable} +{hours:03d}h ({time_str})"
)

fig_cond.tight_layout()


def update_cond(frame):
    """Update conditional animation frame"""
    data = cond_outputs[frame][0, 0, 0, video_var_idx].numpy()
    img_cond.set_array(data.ravel())

    lead_time = cond_coords_list[frame]["lead_time"][0]
    hours = int(lead_time / np.timedelta64(1, "h"))
    time_str = pd.Timestamp(
        cond_coords_list[frame]["time"][0] + cond_coords_list[frame]["lead_time"][0]
    ).strftime("%Y-%m-%d %H:%M")
    title_cond.set_text(
        f"Conditional Generation (ERA5): {video_variable} +{hours:03d}h ({time_str})"
    )
    return [img_cond, title_cond]


# Create animation
print("Creating conditional video...")
anim_cond = animation.FuncAnimation(
    fig_cond, update_cond, frames=len(cond_outputs), interval=500, blit=True
)

# Save video
anim_cond.save("outputs/19_cbottle_video_conditional.mp4", writer=writer, dpi=100)
plt.close()