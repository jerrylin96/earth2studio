import argparse
import calendar
import gc
import logging
from pathlib import Path
import numpy as np
import torch
from datetime import datetime, timedelta

# Earth2Studio imports
from earth2studio.data import WB2ERA5
from earth2studio.lexicon import WB2Lexicon, CBottleLexicon
from earth2studio.data.utils import fetch_data
from earth2studio.models.dx import CBottleInfill

# Suppress verbose logging
logging.getLogger('earth2studio').setLevel(logging.WARNING)


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True, help='Year to process (e.g., 2020)')
    parser.add_argument('--output_dir', type=str, required=True)
    return parser.parse_args()


def create_datetime_array(year, month):
    _, num_days = calendar.monthrange(year, month)
    # 4 samples per day (6-hour intervals)
    start_time = datetime(year=year, month=month, day=1)
    return np.array(
        [start_time + timedelta(hours=6*i) for i in range(num_days * 4)],
        dtype="datetime64[ns]"
    )


def main(args):
    year = args.year
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting Year {year} ===")

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
    model = CBottleInfill.load_model(
        package,
        input_variables=input_variables,
        sampler_steps=18
    )
    model = model.to(device_gpu)
    model.set_seed(42)

    try:
        # 2. Loop Through Months
        # ---------------------------------------------------------
        for month in range(1, 13):
            filename = f"era5_infilled_{year}_{month:02d}.pt"
            save_path = output_path / filename

            # A. Resume Capability: Skip if already done
            if save_path.exists():
                try:
                    # Verify file integrity by attempting to load
                    test_load = torch.load(save_path)
                    if 'data' in test_load and 'coords' in test_load:
                        print(f"[SKIP] {month:02d}/12 - Already complete and verified")
                        continue
                except Exception:
                    print(f"[WARN] Corrupted file detected, re-processing {month:02d}")
                    save_path.unlink()

            print(f"\nProcessing {year}-{month:02d} ({month}/12)...")

            # B. Fetch Data (CPU)
            times = create_datetime_array(year, month)
            ds = WB2ERA5()

            try:
                # Fetch directly to CPU
                data, coords = fetch_data(ds, times, input_variables, device=device_cpu)
            except Exception as e:
                print(f"ERROR fetching {year}-{month:02d}: {e}")
                raise

            # C. Infill (GPU)
            # Use inference_mode to save memory and speed up
            with torch.inference_mode():
                # Move data to GPU -> Infill -> Move Result to CPU
                infilled_data, infilled_coords = model(data.to(device_gpu), coords)
                infilled_data = infilled_data.cpu()

            # D. Save (Atomic)
            print(f"  Saving {filename}...")
            torch.save({'data': infilled_data, 'coords': infilled_coords}, save_path)

            # E. Cleanup Memory & Cache
            del data, infilled_data
            torch.cuda.empty_cache()
            gc.collect()

        print(f"\n=== Year {year} Complete ===")

    except Exception as e:
        print(f"\n!!! Year {year} FAILED: {e}")
        # Cleanup before exit
        del model
        torch.cuda.empty_cache()
        raise


if __name__ == "__main__":
    main(setup_args())
