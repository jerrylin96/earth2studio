import argparse
import calendar
import gc
import shutil
import os
from pathlib import Path
import numpy as np
import torch
from datetime import datetime, timedelta

# Earth2Studio imports
from earth2studio.data import WB2ERA5
from earth2studio.lexicon import WB2Lexicon, CBottleLexicon
from earth2studio.data.utils import fetch_data
from earth2studio.models.dx import CBottleInfill

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True, help='Year to process (e.g., 2020)')
    parser.add_argument('--output_dir', type=str, required=True)
    return parser.parse_args()

def create_datetime_array(year, month):
    _, num_days = calendar.monthrange(year, month)
    # 4 samples per day (6-hour intervals)
    start_time = datetime(year=year, month=month, day=1)
    return np.array([start_time + timedelta(hours=6*i) for i in range(num_days * 4)], dtype="datetime64[ns]")

def main(args):
    year = args.year
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Starting Processing for Year {year} ---")

    # 1. Setup Model (Load ONCE for the whole year)
    # ---------------------------------------------------------
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a GPU for efficient infilling.")

    device_gpu = torch.device("cuda")
    device_cpu = torch.device("cpu")
    
    # Identify variables
    wb2_vars = set(WB2Lexicon.VOCAB.keys())
    cbottle_vars = set(CBottleLexicon.VOCAB.keys())
    input_variables = sorted(list(cbottle_vars.intersection(wb2_vars)))
    
    print(f"Loading CBottleInfill model to {torch.cuda.get_device_name(0)}...")
    
    # Load model context
    package = CBottleInfill.load_default_package()
    model = CBottleInfill.load_model(package, input_variables=input_variables, sampler_steps=18)
    model = model.to(device_gpu)
    model.set_seed(42)
    
    # 2. Loop Through Months
    # ---------------------------------------------------------
    for month in range(1, 13):
        filename = f"era5_infilled_{year}_{month:02d}.pt"
        save_path = output_path / filename
        
        # A. Resume Capability: Skip if already done
        if save_path.exists():
            # Optional: Check file size to ensure it's not a corrupted fragment
            if save_path.stat().st_size > 1024: 
                print(f"[SKIP] Month {month:02d} already exists.")
                continue

        print(f"\nProcessing {year}-{month:02d}...")

        # B. Fetch Data (CPU)
        times = create_datetime_array(year, month)
        ds = WB2ERA5()
        
        try:
            # Fetch directly to CPU
            data, coords = fetch_data(ds, times, input_variables, device=device_cpu)
        except Exception as e:
            print(f"Error fetching {year}-{month:02d}: {e}")
            # We exit to avoid saving bad data; Scheduler will retry or user investigates
            exit(1)

        # C. Infill (GPU)
        # Use inference_mode to save memory and speed up
        with torch.inference_mode():
             # Move data to GPU -> Infill -> Move Result to CPU
            infilled_data, infilled_coords = model(data.to(device_gpu), coords)
            infilled_data = infilled_data.cpu()

        # D. Save (Atomic)
        print(f"Saving to {filename}...")
        torch.save({'data': infilled_data, 'coords': infilled_coords}, save_path)
        
        # E. Cleanup Memory & Cache
        del data, infilled_data
        

    print(f"--- Year {year} Completed Successfully ---")

if __name__ == "__main__":
    main(setup_args())
    