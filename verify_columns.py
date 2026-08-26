"""Verify the COLUMN_NAMES fix: check alignment and data ranges."""

import numpy as np
import pandas as pd

from test_metal.features import COLUMN_NAMES

EXCEL_PATH = r"C:\1001110001000111101(1)\Python\Test_metal\source_data.xls"

print(f"Total COLUMN_NAMES: {len(COLUMN_NAMES)}")
print()

# Show indices of key columns
key_cols = ["steel_S_before", "steel_S_after", "steel_Si_before", "steel_Si_after"]
for col in key_cols:
    idx = COLUMN_NAMES.index(col)
    print(f"  {col}: index {idx}")

print()
print("=" * 80)
print("Verify: raw headers at the new key column positions")
print("=" * 80)
book = pd.ExcelFile(EXCEL_PATH, engine="xlrd")
df_raw = pd.read_excel(book, header=3, usecols="B:CN", engine="xlrd")

# Apply the pipeline's column renaming
df = df_raw.iloc[:, : len(COLUMN_NAMES)].copy()
df.columns = COLUMN_NAMES[: len(df.columns)]

print(f"Renamed shape: {df.shape}")
print()

# Check the key columns
expected_ranges = {
    "steel_S_before": (0.004, 0.076),
    "steel_S_after": (0.001, 0.036),
    "steel_Si_before": (0.01, 0.20),
    "steel_Si_after": (0.012, 0.30),
}

for col, (exp_min, exp_max) in expected_ranges.items():
    if col in df.columns:
        actual_min = df[col].min()
        actual_max = df[col].max()
        match = (actual_min >= exp_min - 1e-9) and (actual_max <= exp_max + 1e-9)
        print(
            f"  {col}: range [{actual_min:.6f}, {actual_max:.6f}] vs expected [{exp_min}, {exp_max}] -> {'PASS' if match else 'FAIL'}"
        )

print()
print("=" * 80)
print("Independent verification: S decreases, Si increases")
print("=" * 80)

# S verification
mask_s = df["steel_S_before"].notna() & df["steel_S_after"].notna()
x_s = df.loc[mask_s, "steel_S_before"].values
y_s = df.loc[mask_s, "steel_S_after"].values
if len(x_s) > 1:
    slope_s = np.polyfit(x_s, y_s, 1)
    y_hat = slope_s[0] * x_s + slope_s[1]
    ss_res = np.sum((y_s - y_hat) ** 2)
    ss_tot = np.sum((y_s - y_s.mean()) ** 2)
    r2_s = 1 - ss_res / ss_tot
    pct_s_decrease = (y_s < x_s).mean() * 100
    pct_s_increase = (y_s > x_s).mean() * 100
    print(f"  S: R2 = {r2_s:.4f} (threshold 0.3 -> {'PASS' if r2_s > 0.3 else 'FAIL'})")
    print(f"  S: slope = {slope_s[0]:.6f}, intercept = {slope_s[1]:.6f}")
    print(f"  S: before mean = {x_s.mean():.6f}, after mean = {y_s.mean():.6f}")
    print(f"  S: after < before in {pct_s_decrease:.1f}% of rows")
    print(f"  S: after > before in {pct_s_increase:.1f}% of rows")

# Si verification
mask_si = df["steel_Si_before"].notna() & df["steel_Si_after"].notna()
x_si = df.loc[mask_si, "steel_Si_before"].values
y_si = df.loc[mask_si, "steel_Si_after"].values
if len(x_si) > 1:
    slope_si = np.polyfit(x_si, y_si, 1)
    y_hat = slope_si[0] * x_si + slope_si[1]
    ss_res = np.sum((y_si - y_hat) ** 2)
    ss_tot = np.sum((y_si - y_si.mean()) ** 2)
    r2_si = 1 - ss_res / ss_tot
    pct_si_increase = (y_si > x_si).mean() * 100
    pct_si_decrease = (y_si < x_si).mean() * 100
    print(f"\n  Si: R2 = {r2_si:.4f} (threshold 0.3 -> {'PASS' if r2_si > 0.3 else 'FAIL'})")
    print(f"  Si: slope = {slope_si[0]:.6f}, intercept = {slope_si[1]:.6f}")
    print(f"  Si: before mean = {x_si.mean():.6f}, after mean = {y_si.mean():.6f}")
    print(f"  Si: after > before in {pct_si_increase:.1f}% of rows")
    print(f"  Si: after < before in {pct_si_decrease:.1f}% of rows")

# Verify sulfur_reduction_ratio correlation
print()
print("=" * 80)
print("Verify sulfur_reduction_ratio = (S_before - S_after) / S_before")
print("=" * 80)
mask_r = (
    df["steel_S_before"].notna()
    & df["steel_S_after"].notna()
    & df["sulfur_reduction_ratio"].notna()
)
computed_ratio = (df.loc[mask_r, "steel_S_before"] - df.loc[mask_r, "steel_S_after"]) / df.loc[
    mask_r, "steel_S_before"
]
stored_ratio = df.loc[mask_r, "sulfur_reduction_ratio"]
correlation = np.corrcoef(computed_ratio.values, stored_ratio.values)[0, 1]
print(f"  Correlation: {correlation:.7f}")
print(
    f"  Computed ratio: min={computed_ratio.min():.6f}, max={computed_ratio.max():.6f}, mean={computed_ratio.mean():.6f}"
)
print(
    f"  Stored ratio:   min={stored_ratio.min():.6f}, max={stored_ratio.max():.6f}, mean={stored_ratio.mean():.6f}"
)

# Show the full column alignment after fix
print()
print("=" * 80)
print("Full COLUMN_NAMES alignment after fix (positions 43-50):")
print("=" * 80)
for i in range(43, 51):
    raw_header = str(df_raw.columns[i])
    cn = COLUMN_NAMES[i] if i < len(COLUMN_NAMES) else "<NONE>"
    print(f"  [{i}] raw='{raw_header[:50]}' -> COLUMN_NAMES='{cn}'")
    vals = df_raw.iloc[:3, i].tolist()
    print(f"       values: {vals}")

print()
print("=" * 80)
print("Full COLUMN_NAMES alignment after fix (tech-params, positions 71-91):")
print("=" * 80)
for i in range(71, min(91, len(COLUMN_NAMES))):
    raw_header = str(df_raw.columns[i])
    cn = COLUMN_NAMES[i] if i < len(COLUMN_NAMES) else "<NONE>"
    vals = df_raw.iloc[:3, i].tolist()
    print(f"  [{i}] raw='{raw_header[:50]}' -> COLUMN_NAMES='{cn}' | vals={vals}")
