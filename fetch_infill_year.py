import argparse
import calendar
import gc
import os
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from datetime import datetime, timedelta
import logging
from loguru import logger

# Earth2Studio imports
from earth2studio.data import WB2ERA5
from earth2studio.lexicon import WB2Lexicon, CBottleLexicon
from earth2studio.data.utils import fetch_data
from earth2studio.models.dx import CBottleInfill

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True, help='Year to process')
    parser.add_argument('--output_dir', type=str, required=True)
    return parser.parse_args()

def create_datetime_array(year, month):
    _, num_days = calendar.monthrange(year, month)
    start_time = datetime(year=year, month=month, day=1)
    return np.array([start_time + timedelta(hours=6*i) for i in range(num_days * 4)], dtype="datetime64[ns]")

def diagnose_nans(data, var_names):
    """
    Scans the tensor for NaNs and prints a detailed report of which variables/times are affected.
    Assumes data shape is (Time, Variable, Lat, Lon).
    """
    if not torch.isnan(data).any():
        return # Clean data
    
    print("\n" + "!"*50)
    print(f"CRITICAL FAILURE: NaNs detected in input data!")
    print("!"*50)
    print(f"Diagnostic Report:")
    
    # Iterate over variables (Dimension 1)
    for i, var_name in enumerate(var_names):
        # Slice: All times, specific variable, all lat/lon
        # Dimensions: time, lead_time (size 1), var, lat, lon
        var_slice = data[:, 0, i, :, :]
        nan_count = torch.isnan(var_slice).sum().item()
        
        if nan_count > 0:
            print(f"  - Variable '{var_name}' (Index {i}): {nan_count} NaNs found.")
            
            # Find which time steps are affected
            # Collapse lat/lon (dim 1, 2 of the slice) to see if time step has any nan
            affected_times = torch.where(torch.isnan(var_slice).any(dim=(1, 2)))[0]

            
            # Print the first few affected time indices as a hint
            limit = 5
            time_list = affected_times.tolist()
            shown = time_list[:limit]
            print(f"    -> Affected Time Steps (Indices): {shown}{'...' if len(time_list)>limit else ''}")
            
    print("!"*50 + "\n")
    raise ValueError("Input data contains NaNs (see report above).")

def main(args):
    # 1. Silence Logs
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    logging.getLogger('earth2studio').setLevel(logging.WARNING)

    year = args.year
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Starting Processing for Year {year} ---")

    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a GPU for efficient infilling.")

    device_gpu = torch.device("cuda")
    device_cpu = torch.device("cpu")
    
    wb2_vars = set(WB2Lexicon.VOCAB.keys())
    cbottle_vars = set(CBottleLexicon.VOCAB.keys())
    input_variables = sorted(list(cbottle_vars.intersection(wb2_vars)))
    
    print(f"Loading CBottleInfill model to {torch.cuda.get_device_name(0)}...")
    
    package = CBottleInfill.load_default_package()
    model = CBottleInfill.load_model(package, input_variables=input_variables, sampler_steps=18)
    model = model.to(device_gpu)
    model.set_seed(42)
    
    for month in range(1, 13):
        filename = f"era5_infilled_{year}_{month:02d}.pt"
        save_path = output_path / filename
        
        # A. Resume Capability
        if save_path.exists():
            try:
                if save_path.stat().st_size < 1024 * 1024: 
                    raise ValueError("File too small")
                test_load = torch.load(save_path, map_location='cpu', weights_only=False)
                if 'data' not in test_load or 'coords' not in test_load:
                    raise ValueError("Missing keys in checkpoint")
                del test_load
                print(f"[SKIP] Month {month:02d} already complete and verified.")
                continue
            except Exception as e:
                print(f"[WARNING] Corrupt file found ({filename}). Error: {e}")
                save_path.unlink()

        print(f"\nProcessing {year}-{month:02d}...")

        # B. Fetch Data (CPU)
        times = create_datetime_array(year, month)
        ds = WB2ERA5()
        
        max_retries = 5
        fetch_success = False
        data = None
        coords = None

        for attempt in range(max_retries):
            try:
                data, coords = fetch_data(ds, times, input_variables, device=device_cpu)
                fetch_success = True
                break
            except Exception as e:
                wait_time = 15 * (attempt + 1)
                print(f"[WARNING] Fetch attempt {attempt+1}/{max_retries} failed. Retrying in {wait_time}s...")
                if attempt == max_retries - 1:
                    print(f"CRITICAL FAILURE for {year}-{month:02d}:")
                    traceback.print_exc()
                time.sleep(wait_time)

        if not fetch_success:
            exit(1)

        # C. Infill (GPU)
        try:
            # --- DIAGNOSTIC STEP ---
            # Check for NaNs before we even touch the GPU
            # We assume 'coords' has a key 'variable' which is the list of names
            var_names = coords.get('variable', input_variables)
            diagnose_nans(data, var_names)
            # -----------------------

            with torch.inference_mode():
                infilled_data, infilled_coords = model(data.to(device_gpu), coords)
                infilled_data = infilled_data.cpu()

            print(f"Saving to {filename}...")
            torch.save({'data': infilled_data, 'coords': infilled_coords}, save_path)
            
            del data, infilled_data
            torch.cuda.empty_cache()
            gc.collect()
            print(f"Infilled and saved {year}-{month:02d}:")
        except Exception as e:
            # We catch the ValueError from diagnose_nans here and exit cleanly
            print(f"Error during Infilling/Saving {year}-{month:02d}:")
            # If it's our specific NaN error, print the message, otherwise traceback
            if "Input data contains NaNs" in str(e):
                print(str(e))
            else:
                traceback.print_exc()
            exit(1)

    print(f"--- Year {year} Completed Successfully ---")

if __name__ == "__main__":
    main(setup_args())