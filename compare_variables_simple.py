#!/usr/bin/env python3
"""
Compare available variables between WB2ERA5 and CBottle
========================================================
"""

# WB2 Lexicon variables (from wb2.py)
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

wb2_surface = {
    "u10m", "v10m", "t2m", "sp", "lsm", "z", "msl", "sst", "tcwv",
    "tp06", "tp12", "tp24"
}

wb2_pressure_params = ["u", "v", "w", "z", "t", "q", "r"]
wb2_vars = set(wb2_surface)
for param in wb2_pressure_params:
    for level in LEVELS:
        wb2_vars.add(f"{param}{level}")

# CBottle Lexicon variables (from cbottle.py)
cbottle_surface = {
    "tcwv", "tclw", "tciw", "t2m", "u10m", "v10m", "rlut", "rsut",
    "msl", "tpf", "rsds", "sst", "sic"
}

cbottle_pressure_params = ["u", "v", "t", "z"]
cbottle_pressure_levels = [1000, 850, 700, 500, 300, 200, 50, 10]
cbottle_vars = set(cbottle_surface)
for param in cbottle_pressure_params:
    for level in cbottle_pressure_levels:
        cbottle_vars.add(f"{param}{level}")

# Print results
print("=" * 80)
print("WB2ERA5 (ERA5 Reanalysis) Available Variables")
print("=" * 80)
print(f"Total WB2ERA5 variables: {len(wb2_vars)}")
print(f"\nSurface variables ({len(wb2_surface)}):")
print(f"  {sorted(wb2_surface)}")
print(f"\nPressure levels available: {LEVELS}")
print(f"Pressure parameters: {wb2_pressure_params}")
print(f"Total pressure level variables: {len(wb2_pressure_params) * len(LEVELS)}")

print("\n" + "=" * 80)
print("CBottle Required Variables")
print("=" * 80)
print(f"Total CBottle variables: {len(cbottle_vars)}")
print(f"\nSurface variables ({len(cbottle_surface)}):")
print(f"  {sorted(cbottle_surface)}")
print(f"\nPressure levels required: {cbottle_pressure_levels}")
print(f"Pressure parameters: {cbottle_pressure_params}")
print(f"Total pressure level variables: {len(cbottle_pressure_params) * len(cbottle_pressure_levels)}")

print("\n" + "=" * 80)
print("Comparison: ERA5 Coverage of CBottle Variables")
print("=" * 80)

available_in_era5 = cbottle_vars & wb2_vars
missing_from_era5 = cbottle_vars - wb2_vars

print(f"\nCBottle variables AVAILABLE in ERA5: {len(available_in_era5)}/{len(cbottle_vars)}")
print(f"  {sorted(available_in_era5)}")

print(f"\nCBottle variables MISSING from ERA5: {len(missing_from_era5)}/{len(cbottle_vars)}")
print(f"  {sorted(missing_from_era5)}")

# Detailed missing analysis
missing_surface = {v for v in missing_from_era5 if not any(c.isdigit() for c in v)}
missing_pressure = {v for v in missing_from_era5 if any(c.isdigit() for c in v)}

print("\n" + "=" * 80)
print("Missing Variable Details")
print("=" * 80)

print(f"\nMissing SURFACE variables ({len(missing_surface)}):")
for var in sorted(missing_surface):
    descriptions = {
        "tclw": "Total cloud liquid water",
        "tciw": "Total cloud ice water",
        "rlut": "Outgoing longwave radiation (TOA)",
        "rsut": "Outgoing shortwave radiation (TOA)",
        "rsds": "Downwelling shortwave radiation (surface)",
        "tpf": "Total precipitation flux",
        "sic": "Sea ice concentration"
    }
    print(f"  - {var}: {descriptions.get(var, 'Unknown')}")

if missing_pressure:
    print(f"\nMissing PRESSURE level variables ({len(missing_pressure)}):")
    from collections import defaultdict
    by_param = defaultdict(list)
    for var in missing_pressure:
        param = ''.join([c for c in var if not c.isdigit()])
        level = int(''.join([c for c in var if c.isdigit()]))
        by_param[param].append(level)

    for param, levels in sorted(by_param.items()):
        print(f"  {param}: missing levels {sorted(levels)}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"✓ ERA5 provides {len(available_in_era5)}/{len(cbottle_vars)} CBottle variables ({100*len(available_in_era5)/len(cbottle_vars):.1f}%)")
print(f"✗ ERA5 is missing {len(missing_from_era5)}/{len(cbottle_vars)} CBottle variables ({100*len(missing_from_era5)/len(cbottle_vars):.1f}%)")

if len(missing_from_era5) > 0:
    print(f"\n{'='*80}")
    print("CONCLUSION: CBottleInfill IS NECESSARY")
    print("="*80)
    print("ERA5 is missing critical variables (radiation, clouds, some pressure levels).")
    print("CBottleInfill uses the diffusion model to infer these missing variables")
    print("from the available ERA5 data, leveraging physics learned from ICON training.")
else:
    print(f"\nCONCLUSION: ERA5 has all CBottle variables - CBottleInfill not strictly needed")
