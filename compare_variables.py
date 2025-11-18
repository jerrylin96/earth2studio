#!/usr/bin/env python3
"""
Compare available variables between WB2ERA5 and CBottle
========================================================

This script definitively shows:
1. What variables are available in WB2ERA5 (ERA5 reanalysis)
2. What variables CBottle models require
3. Which CBottle variables are available/missing in ERA5
"""

import numpy as np
from earth2studio.lexicon import WB2Lexicon, CBottleLexicon

# Get all available WB2ERA5 variables
print("=" * 80)
print("WB2ERA5 (ERA5 Reanalysis) Available Variables")
print("=" * 80)

wb2_vars = list(WB2Lexicon.VOCAB.keys())
print(f"Total WB2ERA5 variables: {len(wb2_vars)}")
print("\nSurface variables:")
surface_vars = [v for v in wb2_vars if not any(char.isdigit() for char in v)]
print(f"  {surface_vars}")

print("\nPressure level variables (by parameter):")
pressure_params = ["u", "v", "w", "z", "t", "q", "r"]
for param in pressure_params:
    param_vars = [v for v in wb2_vars if v.startswith(param) and v[len(param):].isdigit()]
    if param_vars:
        levels = sorted([int(v[len(param):]) for v in param_vars])
        print(f"  {param}: {levels}")

# Get all CBottle required variables
print("\n" + "=" * 80)
print("CBottle Required Variables")
print("=" * 80)

cbottle_vars = list(CBottleLexicon.VOCAB.keys())
print(f"Total CBottle variables: {len(cbottle_vars)}")

cbottle_surface = [v for v in cbottle_vars if not any(char.isdigit() for char in v)]
print(f"\nSurface variables ({len(cbottle_surface)}):")
print(f"  {cbottle_surface}")

print("\nPressure level variables:")
cbottle_params = ["u", "v", "t", "z"]
for param in cbottle_params:
    param_vars = [v for v in cbottle_vars if v.startswith(param) and v[len(param):].isdigit()]
    if param_vars:
        levels = sorted([int(v[len(param):]) for v in param_vars])
        print(f"  {param}: {levels}")

# Compare: What's available vs missing
print("\n" + "=" * 80)
print("Comparison: ERA5 Coverage of CBottle Variables")
print("=" * 80)

available_in_era5 = []
missing_from_era5 = []

for cbottle_var in cbottle_vars:
    if cbottle_var in wb2_vars:
        available_in_era5.append(cbottle_var)
    else:
        missing_from_era5.append(cbottle_var)

print(f"\nCBottle variables AVAILABLE in ERA5 ({len(available_in_era5)}/{len(cbottle_vars)}):")
print(f"  {sorted(available_in_era5)}")

print(f"\nCBottle variables MISSING from ERA5 ({len(missing_from_era5)}/{len(cbottle_vars)}):")
print(f"  {sorted(missing_from_era5)}")

# Show what the missing variables are
print("\n" + "=" * 80)
print("Missing Variable Details")
print("=" * 80)

missing_surface = [v for v in missing_from_era5 if not any(char.isdigit() for char in v)]
print(f"\nMissing surface variables ({len(missing_surface)}):")
for var in sorted(missing_surface):
    print(f"  - {var}: {CBottleLexicon.VOCAB[var]}")

missing_pressure = [v for v in missing_from_era5 if any(char.isdigit() for char in v)]
if missing_pressure:
    print(f"\nMissing pressure level variables ({len(missing_pressure)}):")
    # Group by parameter
    from collections import defaultdict
    by_param = defaultdict(list)
    for var in missing_pressure:
        # Extract parameter (non-digit prefix)
        param = ''.join([c for c in var if not c.isdigit()])
        level = ''.join([c for c in var if c.isdigit()])
        by_param[param].append(int(level))

    for param, levels in sorted(by_param.items()):
        print(f"  {param}: missing levels {sorted(levels)}")

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print(f"ERA5 has {len(available_in_era5)}/{len(cbottle_vars)} CBottle variables")
print(f"ERA5 is missing {len(missing_from_era5)}/{len(cbottle_vars)} CBottle variables")
print(f"\nConclusion: {'ERA5 has all CBottle variables!' if len(missing_from_era5) == 0 else 'ERA5 is missing key variables - CBottleInfill is needed'}")
