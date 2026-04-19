import pandas as pd
import numpy as np
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD ───────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv('194903-202105_Bottle.csv', low_memory=False, encoding='latin1')
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

# ── 2. EXTRACT YEAR FROM DEPTH_ID ────────────────────────────
# Format: 19-4903CR-... where '49' = 1949, '03' = 2003
df['Year'] = df['Depth_ID'].str.split('-').str[1].str[:2].astype(float)
df['Year'] = df['Year'].apply(lambda x: 1900 + x if x >= 49 else 2000 + x)

# ── 3. SELECT KEY FEATURES ───────────────────────────────────
features = ['T_degC', 'Salnty', 'O2ml_L', 'O2Sat', 'ChlorA',
            'PO4uM', 'NO3uM', 'NO2uM', 'Depthm', 'Year']

df_clean = df[features].copy()

# ── 4. DROP ROWS WHERE CORE COLUMNS ARE NULL ─────────────────
df_clean = df_clean.dropna(subset=['O2ml_L', 'NO3uM', 'T_degC', 'Salnty', 'Year'])
print(f"After dropping nulls: {df_clean.shape[0]:,} rows")
print(f"Year range: {int(df_clean['Year'].min())} to {int(df_clean['Year'].max())}")

# Fill remaining nulls with median
df_clean = df_clean.fillna(df_clean.median(numeric_only=True))

# ── 5. HEALTH LABELING ───────────────────────────────────────
# O2ml_L: hypoxia < 1.4, stressed 1.4-4.0, healthy > 4.0
# NO3uM:  high nitrate > 20 = nutrient stress
# O2Sat:  < 40% = severely depleted

def label_health(row):
    o2  = row['O2ml_L']
    no3 = row['NO3uM']
    o2s = row['O2Sat']

    if o2 < 1.4 or o2s < 40:
        return 'Critical'
    elif o2 < 4.0 or no3 > 20:
        return 'Stressed'
    else:
        return 'Healthy'

print("Labeling ecosystem health...")
df_clean['health_label'] = df_clean.apply(label_health, axis=1)

print("\n=== HEALTH LABEL DISTRIBUTION ===")
print(df_clean['health_label'].value_counts())
print(df_clean['health_label'].value_counts(normalize=True).round(3))

# ── 6. ML TRAINING ───────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

print("\nPreparing ML pipeline...")

# Exclude O2ml_L and O2Sat — used to define labels (would cause leakage)
# Predict health purely from nutrients, temperature, salinity, depth
ml_features = ['T_degC', 'Salnty', 'PO4uM', 'NO3uM', 'NO2uM', 'Depthm', 'ChlorA']

X = df_clean[ml_features]
y = df_clean['health_label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training on {X_train.shape[0]:,} samples...")
print(f"Testing on  {X_test.shape[0]:,} samples...")

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
print("Model trained!")

# ── 7. EVALUATE ──────────────────────────────────────────────
y_pred = rf.predict(X_test)

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

importances = pd.Series(rf.feature_importances_, index=ml_features)
importances = importances.sort_values(ascending=False)
print("\n=== FEATURE IMPORTANCE ===")
print(importances.round(3))

# ── 8. SAVE MODEL ────────────────────────────────────────────
with open('calcofi_model.pkl', 'wb') as f:
    pickle.dump(rf, f)
print("\nModel saved to calcofi_model.pkl")

# ── 9. CHART 1 — Oxygen distribution by health class ─────────
fig1 = px.histogram(df_clean, x='O2ml_L', color='health_label',
                    title='Oxygen Distribution by Health Class',
                    color_discrete_map={
                        'Healthy': '#2ecc71',
                        'Stressed': '#3498db',
                        'Critical': '#e74c3c'
                    },
                    nbins=80, barmode='overlay', opacity=0.7)
fig1.show()

# ── 10. CHART 2 — Health trend over time ─────────────────────
trend = df_clean.groupby(['Year', 'health_label']).size().reset_index(name='Count')
trend['Year'] = trend['Year'].astype(int)

print("\n=== YEAR RANGE CHECK ===")
print(f"Years: {trend['Year'].min()} to {trend['Year'].max()}")
print(trend.head(9).to_string())

fig2 = px.area(trend, x='Year', y='Count', color='health_label',
               title='California Ocean Health 1949-2021',
               color_discrete_map={
                   'Healthy': '#2ecc71',
                   'Stressed': '#3498db',
                   'Critical': '#e74c3c'
               })
fig2.update_layout(xaxis_title='Year', yaxis_title='Sample Count')
fig2.show()

# ── 11. CHART 3 — Feature importance ─────────────────────────
imp_df = importances.reset_index()
imp_df.columns = ['Feature', 'Importance']

fig3 = px.bar(imp_df, x='Feature', y='Importance',
              title='What Predicts Ocean Health? (Feature Importance)',
              color='Importance',
              color_continuous_scale='RdYlGn')
fig3.update_layout(showlegend=False)
fig3.show()

# ── 12. CHART 4 — Health by depth zone ───────────────────────
df_clean['Depth_Zone'] = pd.cut(
    df_clean['Depthm'],
    bins=[0, 50, 150, 300, 600, 5400],
    labels=['0-50m', '50-150m', '150-300m', '300-600m', '600m+']
)

depth_health = df_clean.groupby(
    ['Depth_Zone', 'health_label'], observed=True
).size().reset_index(name='Count')

fig4 = px.bar(depth_health, x='Depth_Zone', y='Count',
              color='health_label',
              title='Ocean Health by Depth Zone',
              barmode='group',
              color_discrete_map={
                  'Healthy': '#2ecc71',
                  'Stressed': '#3498db',
                  'Critical': '#e74c3c'
              })
fig4.update_layout(xaxis_title='Depth Zone', yaxis_title='Sample Count')
fig4.show()

print("\nAll done! 4 charts generated.")
print("Charts: oxygen dist, health over time, feature importance, health by depth")
# ── 13. CHART 5 — Geographic health map ──────────────────────
print("Loading cast file for lat/lon...")
cast = pd.read_csv('194903-202105_Cast.csv', low_memory=False, encoding='latin1')
cast = cast[['Sta_ID', 'Lat_Dec', 'Lon_Dec']].dropna()

# Merge lat/lon into df_clean using Sta_ID
df_clean['Sta_ID'] = df['Sta_ID'].loc[df_clean.index]
df_map = df_clean.merge(cast.drop_duplicates('Sta_ID'), on='Sta_ID', how='left')
df_map = df_map.dropna(subset=['Lat_Dec', 'Lon_Dec'])

print(f"Map dataframe: {df_map.shape[0]:,} rows with coordinates")

# Sample 10k points so the map loads fast
map_df = df_map.sample(10000, random_state=42)

fig5 = px.scatter_mapbox(
    map_df,
    lat='Lat_Dec',
    lon='Lon_Dec',
    color='health_label',
    color_discrete_map={
        'Healthy': '#2ecc71',
        'Stressed': '#3498db',
        'Critical': '#e74c3c'
    },
    zoom=4,
    height=600,
    title='California Current Ecosystem Health Map',
    hover_data=['T_degC', 'O2ml_L', 'NO3uM']
)
fig5.update_layout(mapbox_style='carto-positron')
fig5.show()
# ── SAVE PROCESSED DATA ──────────────────────────────────────
df_map.to_csv('calcofi_processed.csv', index=False)
print("Saved calcofi_processed.csv")