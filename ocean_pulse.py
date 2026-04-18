import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np

    mo.md("""
    # 🌊 Marine Ecosystem Health Classifier

    ## Team: Ocean Pulse
    ### Members: Laura · Maggie · Chau · David
    ##### DataHacks 2026 | CalCOFI Dataset
    """)
    return mo, pd


@app.cell
def _(mo):
    import os
    from dotenv import load_dotenv
    from groq import Groq

    load_dotenv("datahacks.env")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    groq_client = Groq(api_key=GROQ_API_KEY)

    mo.md("✅ API Key loaded securely!")
    return (groq_client,)


@app.cell
def _(mo):
    mo.md("""
    ## 📋 Project Overview

    The California Current is one of the most productive marine ecosystems on Earth —
    but decades of climate change, ocean acidification, and warming waters are pushing
    it toward a tipping point. **Can we predict when it's too late?**

    Using **70+ years of ocean data** from the Scripps Institution of Oceanography,
    we built a machine learning pipeline that classifies the health state of the
    California Current ecosystem — and forecasts where it's heading.

    ---

    ## 🎯 Goal
    > Classify the health state of the California Current ecosystem based on
    > decades of chemical and biological ocean measurements.

    ---

    ## 📦 Dataset
    **CalCOFI — California Cooperative Oceanic Fisheries Investigations**

    | Dataset | Contents |
    |---------|----------|
    | Hydrographic Bottle Data | Temperature, oxygen, salinity, nutrients, depth |
    | Zooplankton Volume Data | Plankton abundance, biovolume, species diversity |

    - 📅 Time range: **1951 – 2023** (70+ years)
    - 📍 Coverage: **California coastline**
    - 🔬 Collected via CTD casts, net tows, and underway systems

    ---

    ## 🏷️ Health Classification Labels

    | Label | Criteria | Meaning |
    |-------|----------|---------|
    | ✅ **Healthy** | O₂ > 4.5 ml/L, pH > 8.0, abundant plankton | Ecosystem thriving |
    | ⚠️ **Stressed** | O₂ 2–4.5 ml/L, pH 7.8–8.0, declining plankton | Early warning signs |
    | 🚨 **Critical** | O₂ < 2 ml/L, pH < 7.8, plankton collapse | Hypoxic dead zone risk |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 🌍 Why This Matters

    The California Current feeds millions of people.
    It supports **$180M+** in annual fisheries revenue.
    It is changing — slowly, measurably, and preventably.

    - 🌡️ Ocean temperatures have risen **+1.5°C** since 1950
    - 💨 Dissolved oxygen has dropped in key zones
    - 🧪 pH is falling — oceans are **30% more acidic** than pre-industrial levels
    - 🦐 Plankton populations are shifting and collapsing

    > *"The California Current is the canary in the coal mine
    > for climate change impacts on marine ecosystems."*

    **We built an early warning system.**
    Our model reads 70 years of ocean data and classifies:
    ✅ Healthy · ⚠️ Stressed · 🚨 Critical
    """)
    return


@app.cell
def _(mo, pd):
    mo.md("## 📦 Data Loading")

    df = pd.read_csv('zooplankton.csv', skiprows=1, header=1)

    df.columns = ['cruise', 'ship', 'ship_code', 'order_occupied', 
                  'tow_type', 'tow_number', 'net_location', 
                  'time', 'latitude', 'longitude', 
                  'line', 'station', 'volume_sampled', 
                  'small_plankton', 'total_plankton']

    mo.md(f"✅ Loaded **{len(df):,} rows** and **{len(df.columns)} columns**")
    df.head()
    return


@app.cell
def _(mo, pd):
    import plotly.express as px
    import pickle

    mo.md("## 🤖 ML Model + Results")

    # 加载数据
    df_bottle = pd.read_csv('data/194903-202105_Bottle.csv', 
                             low_memory=False, encoding='latin1')

    # 健康标签
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

    df_bottle['health_label'] = df_bottle.apply(label_health, axis=1)

    # 图表 — 健康趋势
    df_bottle['Year'] = df_bottle['Depth_ID'].str.split('-').str[1].str[:2].astype(float)
    df_bottle['Year'] = df_bottle['Year'].apply(lambda x: 1900 + x if x >= 49 else 2000 + x)

    trend = df_bottle.groupby(['Year', 'health_label']).size().reset_index(name='Count')

    fig2 = px.area(trend, x='Year', y='Count', color='health_label',
                   title='California Ocean Health 1949-2023',
                   color_discrete_map={
                       'Healthy': '#2ecc71',
                       'Stressed': '#f39c12',
                       'Critical': '#e74c3c'
                   })
    fig2
    return


@app.cell
def _(groq_client, mo):
    def explain_health(prediction, feature_values):
        prompt = f"""
        You are a marine biologist. An ML model classified a California ocean 
        water sample as '{prediction}' ecosystem health.

        Measurements: {feature_values}

        In 2-3 simple sentences explain:
        1. What this means for marine life
        2. What human activities likely caused this
        Keep it accessible to a general audience.
        """
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    # Test
    sample = {
        'T_degC': 12.5,
        'Salnty': 33.8,
        'PO4uM': 2.1,
        'NO3uM': 25.0,
        'Depthm': 100
    }

    result = explain_health('Stressed', sample)
    mo.md(f"### 🤖 AI Explanation:\n{result}")
    return


@app.cell
def _(mo, pd):
    import plotly.express as px
    import pickle
    import warnings
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    warnings.filterwarnings('ignore')

    mo.md("## 🤖 ML Model + Results")

    # 加载数据
    df_bottle = pd.read_csv('194903-202105_Bottle.csv', low_memory=False, encoding='latin1')
    mo.md(f"✅ Loaded **{len(df_bottle):,} rows**")

    # 年份
    df_bottle['Year'] = df_bottle['Depth_ID'].str.split('-').str[1].str[:2].astype(float)
    df_bottle['Year'] = df_bottle['Year'].apply(lambda x: 1900 + x if x >= 49 else 2000 + x)

    # 选择特征
    features = ['T_degC', 'Salnty', 'O2ml_L', 'O2Sat', 'ChlorA', 'PO4uM', 'NO3uM', 'NO2uM', 'Depthm', 'Year']
    df_clean = df_bottle[features].copy()
    df_clean = df_clean.dropna(subset=['O2ml_L', 'NO3uM', 'T_degC', 'Salnty', 'Year'])
    df_clean = df_clean.fillna(df_clean.median(numeric_only=True))

    # 健康标签
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

    df_clean['health_label'] = df_clean.apply(label_health, axis=1)

    # ML 模型
    ml_features = ['T_degC', 'Salnty', 'PO4uM', 'NO3uM', 'NO2uM', 'Depthm', 'ChlorA']
    X = df_clean[ml_features]
    y = df_clean['health_label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # 保存模型
    with open('calcofi_model.pkl', 'wb') as f:
        pickle.dump(rf, f)

    # Chart 1 — 健康趋势
    trend = df_clean.groupby(['Year', 'health_label']).size().reset_index(name='Count')
    trend['Year'] = trend['Year'].astype(int)

    fig2 = px.area(trend, x='Year', y='Count', color='health_label',
                   title='California Ocean Health 1949-2021',
                   color_discrete_map={
                       'Healthy': '#2ecc71',
                       'Stressed': '#f39c12',
                       'Critical': '#e74c3c'
                   })

    # Chart 2 — Feature importance
    importances = pd.Series(rf.feature_importances_, index=ml_features).sort_values(ascending=False)
    imp_df = importances.reset_index()
    imp_df.columns = ['Feature', 'Importance']

    fig3 = px.bar(imp_df, x='Feature', y='Importance',
                  title='What Predicts Ocean Health?',
                  color='Importance',
                  color_continuous_scale='RdYlGn')

    # Chart 3 — 地图
    cast = pd.read_csv('194903-202105_Cast.csv', low_memory=False, encoding='latin1')
    cast = cast[['Sta_ID', 'Lat_Dec', 'Lon_Dec']].dropna()
    df_clean['Sta_ID'] = df_bottle['Sta_ID'].loc[df_clean.index]
    df_map = df_clean.merge(cast.drop_duplicates('Sta_ID'), on='Sta_ID', how='left')
    df_map = df_map.dropna(subset=['Lat_Dec', 'Lon_Dec'])
    map_df = df_map.sample(10000, random_state=42)

    fig5 = px.scatter_mapbox(map_df, lat='Lat_Dec', lon='Lon_Dec',
                              color='health_label',
                              color_discrete_map={
                                  'Healthy': '#2ecc71',
                                  'Stressed': '#f39c12',
                                  'Critical': '#e74c3c'
                              },
                              zoom=4, height=600,
                              title='California Current Ecosystem Health Map',
                              hover_data=['T_degC', 'O2ml_L', 'NO3uM'])
    fig5.update_layout(mapbox_style='carto-positron')

    fig2
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
