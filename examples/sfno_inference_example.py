"""
SFNO (Spherical Fourier Neural Operator) Inference Example

This script demonstrates how to run inference using the SFNO weather forecasting model
in Earth2Studio. SFNO is a 73-variable global prognostic model that operates on a
0.25 degree lat-lon grid with 6-hour time steps.

For more information:
- Paper: https://arxiv.org/abs/2306.03838
- Code: https://github.com/NVIDIA/modulus-makani
- Model: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/models/sfno_73ch_small
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

from earth2studio.models.px import SFNO
from earth2studio.data import WB2ERA5
from earth2studio.data.utils import fetch_data
from earth2studio.io import ZarrBackend


def single_step_forecast():
    """
    Example 1: Single 6-hour forecast step

    This demonstrates the basic usage of SFNO:
    - Load the model
    - Fetch initial conditions from ERA5
    - Run one forecast step (6 hours)
    """
    print("="*70)
    print("Example 1: Single 6-Hour Forecast")
    print("="*70)

    # 1. Load SFNO model
    print("\n1. Loading SFNO model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")

    package = SFNO.load_default_package()
    model = SFNO.load_model(package, device=str(device))
    model = model.to(device)

    print(f"   Model loaded: {model}")
    print(f"   Variables: {len(model.variables)} ({', '.join(model.variables[:5])}...)")

    # 2. Fetch initial conditions from ERA5
    print("\n2. Fetching initial conditions from ERA5...")
    initial_time = np.array([np.datetime64("2022-01-01T00:00")])

    era5_ds = WB2ERA5()

    # Get the input coordinates from the model
    lead_time = model.input_coords()["lead_time"]  # [0h]
    variables = model.input_coords()["variable"]   # 73 variables

    # Fetch data (this downloads from WeatherBench2)
    x, coords = fetch_data(
        era5_ds,
        initial_time,
        variables,
        lead_time,
        device=device
    )

    print(f"   Input shape: {x.shape}")
    print(f"   Input coords: time={coords['time'][0]}, lead_time={coords['lead_time'][0]}")

    # 3. Run forecast
    print("\n3. Running 6-hour forecast...")
    with torch.inference_mode():
        forecast, forecast_coords = model(x, coords)

    print(f"   Forecast shape: {forecast.shape}")
    print(f"   Forecast time: {forecast_coords['time'][0]} + {forecast_coords['lead_time'][0]}")
    print(f"   = {forecast_coords['time'][0] + forecast_coords['lead_time'][0]}")

    return forecast, forecast_coords


def multi_step_forecast(num_steps=8):
    """
    Example 2: Multi-step forecast (time integration)

    This demonstrates autoregressive forecasting:
    - Use output from step N as input to step N+1
    - Forecast multiple days ahead
    - Track forecast trajectory

    Parameters
    ----------
    num_steps : int
        Number of 6-hour steps (e.g., 8 steps = 48 hours = 2 days)
    """
    print("\n"+"="*70)
    print(f"Example 2: Multi-Step Forecast ({num_steps * 6} hours)")
    print("="*70)

    # Load model
    print("\n1. Loading SFNO model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    package = SFNO.load_default_package()
    model = SFNO.load_model(package, device=str(device))
    model = model.to(device)

    # Fetch initial conditions
    print("\n2. Fetching initial conditions...")
    initial_time = np.array([np.datetime64("2022-06-15T00:00")])

    era5_ds = WB2ERA5()
    x, coords = fetch_data(
        era5_ds,
        initial_time,
        model.input_coords()["variable"],
        model.input_coords()["lead_time"],
        device=device
    )

    # Run multi-step forecast
    print(f"\n3. Running {num_steps}-step forecast...")

    forecasts = []
    forecast_times = []

    current_x = x
    current_coords = coords

    for step in range(num_steps):
        with torch.inference_mode():
            forecast, forecast_coords = model(current_x, current_coords)

        # Store results
        forecasts.append(forecast.cpu())
        forecast_time = forecast_coords['time'][0] + forecast_coords['lead_time'][0]
        forecast_times.append(forecast_time)

        # Use output as next input (autoregressive)
        current_x = forecast
        current_coords = forecast_coords

        # Update coords for next step (lead_time resets to 0h)
        current_coords['lead_time'] = np.array([np.timedelta64(0, 'h')])

        print(f"   Step {step+1}/{num_steps}: Forecast to {forecast_time}")

    # Stack all forecasts
    all_forecasts = torch.cat(forecasts, dim=1)  # Concatenate along time dimension

    print(f"\n4. Forecast complete!")
    print(f"   Total forecast shape: {all_forecasts.shape}")
    print(f"   Forecast range: {initial_time[0]} to {forecast_times[-1]}")
    print(f"   Total hours: {num_steps * 6}")

    return all_forecasts, forecast_times, current_coords


def forecast_with_iterator():
    """
    Example 3: Using the built-in iterator for forecasting

    SFNO provides a convenient iterator interface for multi-step forecasting
    that automatically handles the autoregressive loop.
    """
    print("\n"+"="*70)
    print("Example 3: Using Model Iterator")
    print("="*70)

    # Load model
    print("\n1. Loading SFNO model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    package = SFNO.load_default_package()
    model = SFNO.load_model(package, device=str(device))
    model = model.to(device)

    # Fetch initial conditions
    print("\n2. Fetching initial conditions...")
    initial_time = np.array([np.datetime64("2022-03-01T00:00")])

    era5_ds = WB2ERA5()
    x, coords = fetch_data(
        era5_ds,
        initial_time,
        model.input_coords()["variable"],
        model.input_coords()["lead_time"],
        device=device
    )

    # Create iterator
    print("\n3. Creating forecast iterator...")
    iterator = model.create_iterator(x, coords)

    # Generate forecasts
    print("\n4. Generating 5-day forecast (20 steps)...")
    num_steps = 20

    forecasts = []
    for i, (forecast, forecast_coords) in enumerate(iterator):
        if i >= num_steps:
            break

        forecasts.append(forecast.cpu())
        forecast_time = forecast_coords['time'][0] + forecast_coords['lead_time'][0]

        if i % 4 == 0:  # Print every 24 hours
            print(f"   Day {i//4}: {forecast_time}")

    print(f"\n5. Generated {len(forecasts)} forecast steps")

    return forecasts


def save_forecast_to_zarr():
    """
    Example 4: Save forecast to Zarr format

    Demonstrates how to save SFNO forecasts in a standard format
    for later analysis or comparison.
    """
    print("\n"+"="*70)
    print("Example 4: Saving Forecast to Zarr")
    print("="*70)

    # Run a short forecast
    print("\n1. Running 24-hour forecast...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    package = SFNO.load_default_package()
    model = SFNO.load_model(package, device=str(device))
    model = model.to(device)

    initial_time = np.array([np.datetime64("2022-01-15T00:00")])
    era5_ds = WB2ERA5()

    x, coords = fetch_data(
        era5_ds,
        initial_time,
        model.input_coords()["variable"],
        model.input_coords()["lead_time"],
        device=device
    )

    # Generate 4 steps (24 hours)
    forecasts = []
    all_coords = []

    current_x = x
    current_coords = coords

    for step in range(4):
        with torch.inference_mode():
            forecast, forecast_coords = model(current_x, current_coords)

        forecasts.append(forecast.cpu())
        all_coords.append(forecast_coords)

        current_x = forecast
        current_coords = forecast_coords
        current_coords['lead_time'] = np.array([np.timedelta64(0, 'h')])

    # Stack forecasts
    all_forecasts = torch.cat(forecasts, dim=1)

    # Save to Zarr
    print("\n2. Saving to Zarr...")
    output_path = "outputs/sfno_forecast.zarr"

    io = ZarrBackend(output_path)
    io.add_array(
        all_coords[-1],
        "forecast",
        all_forecasts.numpy()
    )

    print(f"   Saved to: {output_path}")
    print(f"   Array shape: {all_forecasts.shape}")

    return output_path


def visualize_forecast():
    """
    Example 5: Simple visualization of SFNO forecast

    Creates a simple plot showing temperature at 850 hPa from a forecast.
    """
    print("\n"+"="*70)
    print("Example 5: Visualizing Forecast")
    print("="*70)

    # Run forecast
    print("\n1. Running forecast...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    package = SFNO.load_default_package()
    model = SFNO.load_model(package, device=str(device))
    model = model.to(device)

    initial_time = np.array([np.datetime64("2022-07-04T00:00")])
    era5_ds = WB2ERA5()

    x, coords = fetch_data(
        era5_ds,
        initial_time,
        model.input_coords()["variable"],
        model.input_coords()["lead_time"],
        device=device
    )

    with torch.inference_mode():
        forecast, forecast_coords = model(x, coords)

    # Extract t850 (temperature at 850 hPa)
    print("\n2. Extracting t850...")
    var_idx = np.where(forecast_coords['variable'] == 't850')[0][0]
    t850 = forecast.cpu()[0, 0, 0, var_idx, :, :].numpy()

    # Plot
    print("\n3. Creating plot...")
    fig, ax = plt.subplots(figsize=(12, 6))

    lats = forecast_coords['lat']
    lons = forecast_coords['lon']

    im = ax.contourf(lons, lats, t850, levels=20, cmap='RdBu_r')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'SFNO Forecast: Temperature at 850 hPa\n{initial_time[0]} + 6h')
    plt.colorbar(im, ax=ax, label='Temperature (K)')

    plt.savefig('outputs/sfno_t850_forecast.png', dpi=150, bbox_inches='tight')
    print(f"   Saved plot to: outputs/sfno_t850_forecast.png")

    plt.close()


def main():
    """Run all examples"""
    import os
    os.makedirs('outputs', exist_ok=True)

    print("\n" + "="*70)
    print("SFNO Inference Examples")
    print("="*70)

    # Example 1: Single step
    forecast, coords = single_step_forecast()

    # Example 2: Multi-step
    all_forecasts, times, coords = multi_step_forecast(num_steps=8)

    # Example 3: Iterator
    forecasts = forecast_with_iterator()

    # Example 4: Save to Zarr
    zarr_path = save_forecast_to_zarr()

    # Example 5: Visualization
    visualize_forecast()

    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)


if __name__ == "__main__":
    main()
