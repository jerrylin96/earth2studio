#!/usr/bin/env python3
"""
Using ALL available ERA5 variables for CBottleInfill
====================================================

This script shows that you can use all 34 ERA5-available CBottle variables
as input to CBottleInfill, not just the 13 from the example.
"""

# All CBottle variables that ARE available in ERA5 (from our analysis)
AVAILABLE_IN_ERA5 = [
    # Surface variables (6)
    'msl', 'sst', 't2m', 'tcwv', 'u10m', 'v10m',

    # Pressure level variables at all 7 available levels (28 variables)
    # u, v, t, z at: 50, 200, 300, 500, 700, 850, 1000 hPa
    't1000', 't200', 't300', 't50', 't500', 't700', 't850',
    'u1000', 'u200', 'u300', 'u50', 'u500', 'u700', 'u850',
    'v1000', 'v200', 'v300', 'v50', 'v500', 'v700', 'v850',
    'z1000', 'z200', 'z300', 'z50', 'z500', 'z700', 'z850',
]

# What the example used (subset for demonstration)
EXAMPLE_SUBSET = [
    "u10m", "v10m", "t2m", "msl",
    "z50", "u50", "v50",
    "z500", "u500", "v500",
    "z1000", "u1000", "v1000",
]

# What CBottle needs but ERA5 doesn't have
MISSING_FROM_ERA5 = [
    # Surface (7)
    'rlut', 'rsut', 'rsds',  # Radiation
    'tclw', 'tciw',          # Cloud water/ice
    'tpf',                    # Precipitation flux
    'sic',                    # Sea ice

    # 10 hPa level (4)
    't10', 'u10', 'v10', 'z10',
]

print("=" * 80)
print("CBottleInfill Input Options")
print("=" * 80)

print("\n1. MINIMAL (Example uses this):")
print(f"   {len(EXAMPLE_SUBSET)} variables")
print(f"   {EXAMPLE_SUBSET}")

print("\n2. MAXIMAL (Use all available ERA5):")
print(f"   {len(AVAILABLE_IN_ERA5)} variables")
print(f"   {sorted(AVAILABLE_IN_ERA5)}")

print("\n3. Variables CBottleInfill MUST generate (not in ERA5):")
print(f"   {len(MISSING_FROM_ERA5)} variables")
print(f"   {sorted(MISSING_FROM_ERA5)}")

print("\n" + "=" * 80)
print("Comparison")
print("=" * 80)

print(f"""
Using MINIMAL input ({len(EXAMPLE_SUBSET)} vars):
  - Pros: Faster to fetch from ERA5, less data transfer
  - Cons: CBottleInfill must generate {45 - len(EXAMPLE_SUBSET)} variables ({100*(45-len(EXAMPLE_SUBSET))/45:.1f}%)

Using MAXIMAL input ({len(AVAILABLE_IN_ERA5)} vars):
  - Pros: More information → potentially better infilled results
  - Pros: CBottleInfill only generates {45 - len(AVAILABLE_IN_ERA5)} variables ({100*(45-len(AVAILABLE_IN_ERA5))/45:.1f}%)
  - Cons: Slightly slower to fetch from ERA5

Recommendation: Use ALL {len(AVAILABLE_IN_ERA5)} available variables for best results!
""")

print("=" * 80)
print("Usage Example")
print("=" * 80)

print("""
from earth2studio.data import WB2ERA5
from earth2studio.models.dx import CBottleInfill
from earth2studio.data.utils import fetch_data

# Option 1: Use minimal subset (faster, but less accurate)
input_variables_minimal = [
    "u10m", "v10m", "t2m", "msl",
    "z50", "u50", "v50",
    "z500", "u500", "v500",
    "z1000", "u1000", "v1000",
]

# Option 2: Use ALL available ERA5 variables (recommended!)
input_variables_maximal = [
    'msl', 'sst', 't2m', 'tcwv', 'u10m', 'v10m',
    't1000', 't200', 't300', 't50', 't500', 't700', 't850',
    'u1000', 'u200', 'u300', 'u50', 'u500', 'u700', 'u850',
    'v1000', 'v200', 'v300', 'v50', 'v500', 'v700', 'v850',
    'z1000', 'z200', 'z300', 'z50', 'z500', 'z700', 'z850',
]

# Fetch your chosen variables
era5_ds = WB2ERA5()
era5_x, era5_coords = fetch_data(
    era5_ds, times, input_variables_maximal, device=device
)

# CBottleInfill will generate the missing variables
package = CBottleInfill.load_default_package()
cbottle_infill = CBottleInfill.load_model(
    package,
    input_variables=input_variables_maximal  # Use maximal for best results
)
infilled_x, infilled_coords = cbottle_infill(era5_x, era5_coords)

# Result: 34 → 45 variables (only 11 generated, vs 32 with minimal)
""")

print("\n" + "=" * 80)
print("Key Insight")
print("=" * 80)
print("""
The 13-variable list in the example was chosen for DEMONSTRATION purposes
to show how CBottleInfill fills in missing variables.

For PRODUCTION use, you should use ALL 34 available ERA5 variables to give
CBottleInfill the most information possible, resulting in better physics
for the 11 variables it must still generate (radiation, clouds, etc.).
""")
