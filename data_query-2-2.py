import pandas as pd
import requests

# ── Zooplankton API skipped (too slow, running out of time) ──────────────────
# Will add back if time permits after submission deadline.

# ── Fetch Hydrographic Bottle Data only ──────────────────────────────────────
def fetch_erddap_data(base_url, dataset_id, variables):
    url = f"{base_url}{dataset_id}.csv?{','.join(variables)}"
    response = requests.get(url)
    response.raise_for_status()
    return pd.read_csv(url, low_memory=False, skiprows=[1])  # row 1 = units header

bottle_variables = [
    'time', 'latitude', 'longitude', 'depthm', 't_degc', 'salinity', 'oxy',
    'po4um', 'sio3um', 'no2um', 'no3um', 'nh3um', 'chlora', 'dic1', 'ta1', 'ph1'
]

bottle_base = "https://coastwatch.pfeg.noaa.gov/erddap/tabledap/"
print("Fetching hydrographic bottle data...")
bottle_df = fetch_erddap_data(bottle_base, "siocalcofiHydroBottle", bottle_variables)

print("Hydrographic Bottle Data Columns:")
print(", ".join(bottle_df.columns.tolist()))
print(bottle_df.head())

# ── Save to CSV so model_pipeline.py can load it directly ────────────────────
bottle_df['data_source'] = 'hydrographic_bottle'
usable_data_df = bottle_df

usable_data_df.to_csv("combined_data.csv", index=False)
print(f"\n✅ Saved combined_data.csv — shape: {usable_data_df.shape}")
print("Ready for model_pipeline.py")