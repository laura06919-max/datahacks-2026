import pandas as pd
import requests

# Function to fetch data from ERDDAP
def fetch_erddap_data(base_url, dataset_id, variables):
    url = f"{base_url}{dataset_id}.csv?{','.join(variables)}"
    response = requests.get(url)
    response.raise_for_status()
    return pd.read_csv(url)

# Define usable columns for Zooplankton Data (biological/ecosystem features)
zoo_variables = [
    'time', 'latitude', 'longitude', 'line', 'station',
    'volume_sampled', 'small_plankton', 'total_plankton'
]

# Fetch recent zooplankton data (e.g., last 5 years for manageable size)
zoo_base = "https://oceanview.pfeg.noaa.gov/erddap/tabledap/"
zoo_df = fetch_erddap_data(zoo_base, "erdCalCOFIzoovol", zoo_variables)
print("Zooplankton Data Columns:")
print(zoo_df.columns.tolist())
print(zoo_df.head())

# Define usable chemistry features for Hydrographic Bottle Data
bottle_variables = [
    'time', 'latitude', 'longitude', 'depthm', 't_degc', 'salinity', 'oxy',
    'po4um', 'sio3um', 'no2um', 'no3um', 'nh3um', 'chlora', 'dic1', 'ta1', 'ph1'
]

# Fetch recent bottle data (e.g., last 5 years)
bottle_base = "https://coastwatch.pfeg.noaa.gov/erddap/tabledap/"
bottle_df = fetch_erddap_data(bottle_base, "siocalcofiHydroBottle", bottle_variables)
print("\nHydrographic Bottle Data Columns (Chemistry Features):")
print(bottle_df.columns.tolist())
print(bottle_df.head())

# Optional: Merge on common keys (e.g., time, lat, lon, line, station) for combined dataset
# Note: Exact matching may require preprocessing (e.g., rounding depths, aligning times)
# For demo, just show the selected data

# Add a source column to distinguish datasets
zoo_df['data_source'] = 'zooplankton'
bottle_df['data_source'] = 'hydrographic_bottle'

# Combine into one dataframe (note: different columns, so NaN for missing)
combined_df = pd.concat([zoo_df, bottle_df], ignore_index=True, sort=False)
print(f"\nCombined DataFrame shape: {combined_df.shape}")
print("Combined DataFrame columns:")
print(combined_df.columns.tolist())
print(combined_df.head())

# Store in the environment (as a global variable for this script)
# In a real application, you might save to CSV or database
usable_data_df = combined_df
print("\nUsable data stored in 'usable_data_df' dataframe.")