import argparse
import calendar
import gc
import os
import time
import traceback
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
    parser.add_argument('--year', type=int, required=True, help='Year to process')
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
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a GPU for efficient infilling.")

    device_gpu = torch.device("cuda")
    device_cpu = torch.device("cpu")
    
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
    for month in range(1, 13):
        filename = f"era5_infilled_{year}_{month:02d}.pt"
        save_path = output_path / filename
        
# A. Resume Capability with Integrity Check
        if save_path.exists():
            try:
                # 1. Quick Size Check (Don't waste time loading empty files)
                if save_path.stat().st_size < 1024 * 1024: # < 1MB
                    raise ValueError("File too small")

                # 2. Deep Integrity Check
                # Critical: map_location='cpu' prevents GPU OOM
                # Critical: weights_only=False fixes the numpy security error
                test_load = torch.load(save_path, map_location='cpu', weights_only=False)
                
                # Check structure
                if 'data' not in test_load or 'coords' not in test_load:
                    raise ValueError("Missing keys in checkpoint")
                
                # Success - Clean up memory immediately
                del test_load
                print(f"[SKIP] Month {month:02d} already complete and verified.")
                continue

            except Exception as e:
                print(f"[WARNING] Corrupt file found ({filename}). Error: {e}")
                print("Deleting and re-processing...")
                save_path.unlink() # Delete the bad file

        print(f"\nProcessing {year}-{month:02d}...")

        # B. Fetch Data (CPU) -- WITH RETRY LOGIC
        times = create_datetime_array(year, month)
        ds = WB2ERA5()
        
        max_retries = 5
        fetch_success = False
        data = None
        coords = None

        for attempt in range(max_retries):
            try:
                # Fetch directly to CPU
                data, coords = fetch_data(ds, times, input_variables, device=device_cpu)
                fetch_success = True
                break # Success! Exit loop
            except Exception as e:
                wait_time = 15 * (attempt + 1)
                print(f"[WARNING] Fetch attempt {attempt+1}/{max_retries} failed. Retrying in {wait_time}s...")
                # Only print traceback on the final failure to keep logs clean
                if attempt == max_retries - 1:
                    print(f"CRITICAL FAILURE for {year}-{month:02d}:")
                    traceback.print_exc()
                time.sleep(wait_time)

        if not fetch_success:
            print(f"Skipping {year}-{month:02d} due to repeated fetch failures.")
            # We exit with error code so the Scheduler knows to mark this job as failed
            exit(1)

        # C. Infill (GPU)
        try:
            with torch.inference_mode():
                infilled_data, infilled_coords = model(data.to(device_gpu), coords)
                infilled_data = infilled_data.cpu()

            # D. Save (Atomic)
            print(f"Saving to {filename}...")
            torch.save({'data': infilled_data, 'coords': infilled_coords}, save_path)
            
            # E. Cleanup Memory
            del data, infilled_data
            torch.cuda.empty_cache()
            gc.collect()
            print(f"Infilled and saved {year}-{month:02d}:")
        except Exception as e:
            print(f"Error during Infilling/Saving {year}-{month:02d}")
            traceback.print_exc()
            exit(1)

    print(f"--- Year {year} Completed Successfully ---")

if __name__ == "__main__":
    main(setup_args())