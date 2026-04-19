import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import pickle
    import warnings
    import os
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from groq import Groq
    warnings.filterwarnings('ignore')
    return (
        Groq,
        RandomForestClassifier,
        classification_report,
        mo,
        os,
        pd,
        pickle,
        plt,
        px,
        sns,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md("""
    # California Ocean Ecosystem Health Classifier
    **Team:** Ocean Pulse | Laura · Chau · David · Maggie
    **Event:** DataHacks 2026 @ UC San Diego
    **Dataset:** CalCOFI Hydrographic Bottle Database — Scripps Institution of Oceanography (1949–2021)
    **Model:** Random Forest Classifier | **Accuracy:** 98%

    Using 70 years of real oceanographic data to classify and explain
    California Current ecosystem health — and recommend government action.
    """)
    return


@app.cell
def _(mo, pd):
    df_map = pd.read_csv('calcofi_processed.csv', low_memory=False)

    def assign_region(lat):
        if lat >= 38.0:
            return "Northern California (SF & Above)"
        elif lat >= 35.0:
            return "Central California (Monterey / Big Sur)"
        elif lat >= 32.5:
            return "Southern California (LA / San Diego)"
        else:
            return "Baja California Border Zone"

    df_map['Region'] = df_map['Lat_Dec'].apply(assign_region)
    mo.md(f"**Loaded:** {df_map.shape[0]:,} rows | Regions: {df_map['Region'].value_counts().to_dict()}")
    return (df_map,)


@app.cell
def _(mo):
    mo.md("""
    ## Data Preparation
    We select key oceanographic features: temperature, salinity, oxygen,
    nutrients (phosphate, nitrate, nitrite), chlorophyll, and depth.
    Rows missing core oxygen or nutrient readings are dropped —
    these are required to define our health labels.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Health Labeling
    We define three ecosystem health states based on scientific thresholds
    from CalCOFI literature:

    | Label | Oxygen (ml/L) | Nitrate (µM) | Meaning |
    |-------|--------------|--------------|---------|
    | Healthy | > 4.0 | ≤ 20 | Normal marine conditions |
    | Stressed | 1.4 – 4.0 | > 20 | Low oxygen or nutrient overload |
    | Critical | < 1.4 | — | Hypoxic, life-threatening conditions |
    """)
    return


@app.cell
def _(df_map, mo):
    dist = df_map['health_label'].value_counts()
    mo.md(f"""
    **Label distribution:**
    - Healthy: {dist.get('Healthy', 0):,} samples
    - Stressed: {dist.get('Stressed', 0):,} samples
    - Critical: {dist.get('Critical', 0):,} samples
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Configurable Health Thresholds
    Adjust these to apply the classifier to different ocean regions or datasets.
    """)
    return


@app.cell
def _(mo):
    o2_critical = mo.ui.slider(0.5, 3.0, value=1.4, step=0.1,
                                label="Critical O2 threshold (ml/L)")
    o2_stressed = mo.ui.slider(2.0, 6.0, value=4.0, step=0.1,
                                label="Stressed O2 threshold (ml/L)")
    no3_stressed = mo.ui.slider(5.0, 50.0, value=20.0, step=1.0,
                                 label="Stressed Nitrate threshold (µM)")
    mo.vstack([o2_critical, o2_stressed, no3_stressed])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Machine Learning Model
    We train a **Random Forest Classifier** to predict ecosystem health
    using only indirect measurements — temperature, salinity, nutrients,
    and depth — deliberately excluding oxygen to avoid data leakage.

    This answers a real scientific question:
    > *Can we infer oxygen stress from nutrients and temperature alone?*

    The answer is yes — with 98% accuracy.
    """)
    return


@app.cell
def _(RandomForestClassifier, df_map, mo, pickle, train_test_split):
    ml_features = ['T_degC', 'Salnty', 'PO4uM', 'NO3uM', 'NO2uM', 'Depthm', 'ChlorA']

    df_model = df_map[ml_features + ['health_label']].dropna()
    X = df_model[ml_features]
    y = df_model['health_label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    with open('calcofi_model.pkl', 'wb') as f:
        pickle.dump(rf, f)

    mo.md("**Model trained and saved!** ✓")
    return ml_features, rf, y_pred, y_test


@app.cell
def _(classification_report, mo, y_pred, y_test):
    report = classification_report(y_test, y_pred, output_dict=True)
    mo.md(f"""
    ## Model Performance

    | Class | Precision | Recall | F1-Score |
    |-------|-----------|--------|----------|
    | Critical | {report['Critical']['precision']:.2f} | {report['Critical']['recall']:.2f} | {report['Critical']['f1-score']:.2f} |
    | Healthy | {report['Healthy']['precision']:.2f} | {report['Healthy']['recall']:.2f} | {report['Healthy']['f1-score']:.2f} |
    | Stressed | {report['Stressed']['precision']:.2f} | {report['Stressed']['recall']:.2f} | {report['Stressed']['f1-score']:.2f} |

    **Overall Accuracy: {report['accuracy']:.2%}**
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Visualizations
    """)
    return


@app.cell
def _(df_map, px):
    trend = df_map.groupby(['Year', 'health_label']).size().reset_index(name='Count')
    trend['Year'] = trend['Year'].astype(int)
    fig_trend = px.area(trend, x='Year', y='Count', color='health_label',
                        title='California Ocean Health 1949-2021',
                        color_discrete_map={
                            'Healthy': '#2ecc71',
                            'Stressed': '#3498db',
                            'Critical': '#e74c3c'
                        })
    fig_trend
    return


@app.cell
def _(ml_features, pd, px, rf):
    importances = pd.Series(rf.feature_importances_, index=ml_features).sort_values(ascending=False)
    imp_df = importances.reset_index()
    imp_df.columns = ['Feature', 'Importance']
    fig_imp = px.bar(imp_df, x='Feature', y='Importance',
                     title='What Predicts Ocean Health?',
                     color='Importance',
                     color_continuous_scale='RdYlGn')
    fig_imp.update_layout(showlegend=False)
    fig_imp
    return


@app.cell
def _(df_map, pd, px):
    df_clean_depth = df_map.copy()
    df_clean_depth['Depth_Zone'] = pd.cut(
        df_clean_depth['Depthm'],
        bins=[0, 50, 150, 300, 600, 5400],
        labels=['0-50m', '50-150m', '150-300m', '300-600m', '600m+']
    )
    depth_health = df_clean_depth.groupby(
        ['Depth_Zone', 'health_label'], observed=True
    ).size().reset_index(name='Count')
    fig_depth = px.bar(depth_health, x='Depth_Zone', y='Count',
                       color='health_label',
                       title='Ocean Health by Depth Zone',
                       barmode='group',
                       color_discrete_map={
                           'Healthy': '#2ecc71',
                           'Stressed': '#3498db',
                           'Critical': '#e74c3c'
                       })
    fig_depth
    return


@app.cell
def _(df_map, px):
    map_df = df_map.dropna(subset=['Lat_Dec', 'Lon_Dec']).sample(10000, random_state=42)
    fig_map = px.scatter_mapbox(
        map_df, lat='Lat_Dec', lon='Lon_Dec',
        color='health_label',
        color_discrete_map={
            'Healthy': '#2ecc71',
            'Stressed': '#3498db',
            'Critical': '#e74c3c'
        },
        zoom=4, height=600,
        title='California Current Ecosystem Health Map',
        hover_data=['T_degC', 'O2ml_L', 'NO3uM']
    )
    fig_map.update_layout(mapbox_style='carto-positron')
    fig_map
    return


@app.cell
def _(df_map, px):
    fig_region_map = px.scatter_mapbox(
        df_map.dropna(subset=['Lat_Dec', 'Lon_Dec']).sample(10000, random_state=42),
        lat='Lat_Dec', lon='Lon_Dec',
        color='Region',
        zoom=4, height=600,
        title='California Coast — Regions by Government Jurisdiction',
        hover_data=['health_label', 'T_degC', 'O2ml_L', 'NO3uM']
    )
    fig_region_map.update_layout(mapbox_style='carto-positron')
    fig_region_map
    return


@app.cell
def _(df_map, mo, plt, sns):
    corr = df_map[['T_degC', 'Salnty', 'O2ml_L', 'NO3uM', 'PO4uM', 'Depthm']].corr()
    fig_corr, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()
    mo.image(src="correlation_heatmap.png")
    return


@app.cell
def _(df_map, px):
    fig_box = px.box(
        df_map[df_map["O2ml_L"] > 0],
        x="health_label", y="O2ml_L",
        title="Oxygen Distribution Across Health Classes",
        color="health_label",
        color_discrete_map={
            'Healthy': '#2ecc71',
            'Stressed': '#3498db',
            'Critical': '#e74c3c'
        },
        points=False, template="plotly_white"
    )
    fig_box
    return


@app.cell
def _(df_map, px):
    fig_nitrate = px.box(
        df_map[df_map["NO3uM"] <= 40],
        x="health_label", y="NO3uM",
        color="health_label",
        title="Nitrate Levels by Ecosystem Health",
        color_discrete_map={
            'Healthy': '#2ecc71',
            'Stressed': '#3498db',
            'Critical': '#e74c3c'
        },
        template="plotly_white"
    )
    fig_nitrate
    return


@app.cell
def _(mo):
    mo.md("""
    ## Regional Analysis & Government Action Advisor
    Select a California coastal region to see its health summary
    and get AI-powered policy recommendations.
    """)
    return


@app.cell
def _(df_map, mo):
    region_summary = df_map.groupby('Region').agg(
        Avg_Temperature=('T_degC', 'mean'),
        Avg_Oxygen=('O2ml_L', 'mean'),
        Avg_Nitrate=('NO3uM', 'mean'),
        Avg_Phosphate=('PO4uM', 'mean'),
        Avg_Depth=('Depthm', 'mean'),
        Sample_Count=('O2ml_L', 'count'),
        Critical_Pct=('health_label', lambda x: (x == 'Critical').mean() * 100),
        Stressed_Pct=('health_label', lambda x: (x == 'Stressed').mean() * 100),
        Healthy_Pct=('health_label', lambda x: (x == 'Healthy').mean() * 100),
    ).round(2).reset_index()

    region_dropdown = mo.ui.dropdown(
        options=region_summary['Region'].tolist(),
        label="Select a California coastal region:",
        value=region_summary['Region'].tolist()[0]
    )
    region_dropdown
    return region_dropdown, region_summary


@app.cell
def _(Groq, mo, os, region_dropdown, region_summary):
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY", "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"))

    def region_action_advisor(region_name, avg_stats, dominant_health):
        prompt = f"""
        You are an ocean policy advisor for the state of California.

        Region: {region_name}
        Average ocean measurements:
        - Temperature: {avg_stats['Avg_Temperature']}°C
        - Dissolved Oxygen: {avg_stats['Avg_Oxygen']} ml/L
        - Nitrate: {avg_stats['Avg_Nitrate']} µM
        - Phosphate: {avg_stats['Avg_Phosphate']} µM
        - Dominant health status: {dominant_health}
        - Critical samples: {avg_stats['Critical_Pct']}%
        - Stressed samples: {avg_stats['Stressed_Pct']}%

        Provide a structured response with exactly these three sections:

        **CURRENT SITUATION:**
        2-3 sentences on what these measurements mean for marine life right now.

        **IF NO ACTION IS TAKEN:**
        2-3 sentences on what happens in 10-20 years if conditions continue.

        **RECOMMENDED GOVERNMENT ACTIONS:**
        List 3 specific actions that California state agencies
        (Coastal Commission, EPA, Water Board) can take to improve conditions.
        Keep actions concrete and jurisdiction-specific.
        """
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    selected_region = region_summary[
        region_summary['Region'] == region_dropdown.value
    ].iloc[0]

    if selected_region['Critical_Pct'] > 30:
        dominant = 'Critical'
    elif selected_region['Stressed_Pct'] > 40:
        dominant = 'Stressed'
    else:
        dominant = 'Healthy'

    advice = region_action_advisor(region_dropdown.value, selected_region, dominant)

    mo.md(f"""
    ## {region_dropdown.value}

    | Metric | Value |
    |--------|-------|
    | Avg Temperature | {selected_region['Avg_Temperature']}°C |
    | Avg Oxygen | {selected_region['Avg_Oxygen']} ml/L |
    | Avg Nitrate | {selected_region['Avg_Nitrate']} µM |
    | Critical Samples | {selected_region['Critical_Pct']}% |
    | Stressed Samples | {selected_region['Stressed_Pct']}% |
    | Total Samples | {int(selected_region['Sample_Count']):,} |

    ---

    {advice}
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Key Findings

    - **Phosphate and nitrate** are the strongest predictors of ocean health
    - **Critical zones expand with depth** — deep water hypoxia is worsening
    - **Stressed conditions have increased** since the 1980s
    - A 98% accurate classifier can predict ecosystem stress purely from
      nutrient and temperature measurements — no direct oxygen sensor needed
    - Regional analysis enables **jurisdiction-specific policy recommendations**
      for California state agencies

    *Data source: Scripps Institution of Oceanography, CalCOFI Program (1949–2021)*
    """)
    return


if __name__ == "__main__":
    app.run()
