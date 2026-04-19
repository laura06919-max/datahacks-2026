"""
Ocean Pulse — Live Demo
Team: Laura · Maggie · Chau · David
DataHacks 2026 | CalCOFI Dataset

Run:
    pip install streamlit plotly pandas scikit-learn
    streamlit run streamlit_app.py

Place in same folder:
    194903-202105_Bottle.csv
    zooplankton.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle, os, warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Ocean Pulse · DataHacks 2026",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #040d17;
    color: #cfe8ff;
}
h1,h2,h3,h4 { font-family: 'Space Mono', monospace; }
.hero {
    background: linear-gradient(135deg, #071e36 0%, #0a3060 60%, #040d17 100%);
    border: 1px solid #1a3d6b;
    border-radius: 16px;
    padding: 2.5rem 3rem 2rem;
    margin-bottom: 1.5rem;
}
.hero h1 { font-size:2.4rem; margin:0; color:#fff; letter-spacing:-1px; }
.hero p  { color:#7ab8d8; margin:.4rem 0 0; font-size:.95rem; line-height:1.6; }
.predict-result {
    border-radius: 14px;
    padding: 1.8rem 2rem;
    text-align: center;
    margin-top: .5rem;
}
.stButton>button {
    background: linear-gradient(135deg,#0077b6,#00b4d8) !important;
    color: #fff !important; border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono',monospace !important;
    font-weight: 700 !important; width: 100% !important;
    padding: .65rem 1rem !important; font-size: 1rem !important;
}
</style>
""", unsafe_allow_html=True)

HEALTH_COLORS = {"Healthy": "#2ecc71", "Stressed": "#f39c12", "Critical": "#e74c3c"}
HEALTH_EMOJI  = {"Healthy": "✅", "Stressed": "⚠️", "Critical": "🚨"}
DARK = dict(
    paper_bgcolor="#040d17", plot_bgcolor="#071828",
    font=dict(color="#cfe8ff", family="DM Sans"),
    xaxis=dict(gridcolor="#1a3050", linecolor="#1a3050"),
    yaxis=dict(gridcolor="#1a3050", linecolor="#1a3050"),
)

ML_FEATURES = ['T_degC','Salnty','PO4uM','NO3uM','NO2uM','Depthm','ChlorA']


# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="⏳ Loading 895k CalCOFI bottle samples…")
def load_bottle():
    df = pd.read_csv('194903-202105_Bottle.csv',
                     low_memory=False, encoding='latin1')

    # Year from Depth_ID  e.g. "19-4903CR-..."  → 1949
    df['Year'] = (df['Depth_ID'].str.split('-').str[1].str[:2]
                    .astype(float, errors='ignore'))
    df['Year'] = df['Year'].apply(
        lambda x: (1900+x) if pd.notna(x) and x >= 49 else (2000+x if pd.notna(x) else np.nan)
    )

    # Lat/Lon from Sta_ID  e.g. "054.0 056.0"  → line=54, station=56
    _p = df['Sta_ID'].str.split(' ', expand=True)
    df['_line'] = pd.to_numeric(_p[0], errors='coerce')
    df['_sta']  = pd.to_numeric(_p[1], errors='coerce')
    df['Lat_Dec'] = 34.05 - (df['_line'] - 80.0) * 0.083
    df['Lon_Dec'] = -(122.50 - (df['_sta']  - 60.0) * 0.093)
    # Keep only valid CA Current bounding box
    valid_geo = df['Lat_Dec'].between(28,39) & df['Lon_Dec'].between(-128,-115)
    df.loc[~valid_geo, ['Lat_Dec','Lon_Dec']] = np.nan
    df.drop(columns=['_line','_sta'], inplace=True)

    # Clean O2 — remove physically impossible values (5 outliers in real data)
    df['O2ml_L'] = pd.to_numeric(df['O2ml_L'], errors='coerce')
    df.loc[~df['O2ml_L'].between(0, 10), 'O2ml_L'] = np.nan
    df['O2Sat']  = pd.to_numeric(df['O2Sat'],  errors='coerce')
    df.loc[~df['O2Sat'].between(0, 150), 'O2Sat'] = np.nan

    for c in ML_FEATURES:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing the fields we need
    df_c = df.dropna(subset=['O2ml_L','T_degC','Salnty','Year']).copy()
    # Fill remaining NaNs with column medians
    fill_cols = ML_FEATURES + ['O2Sat','O2ml_L']
    df_c[fill_cols] = df_c[fill_cols].fillna(df_c[fill_cols].median(numeric_only=True))

    # Health labels
    def label(row):
        o2  = row['O2ml_L']
        no3 = row['NO3uM'] if pd.notna(row['NO3uM']) else 0
        o2s = row['O2Sat'] if pd.notna(row['O2Sat']) else 100
        if o2 < 1.4 or o2s < 40:   return 'Critical'
        elif o2 < 4.0 or no3 > 20: return 'Stressed'
        return 'Healthy'

    df_c['health_label'] = df_c.apply(label, axis=1)

    df_c['Depth_Zone'] = pd.cut(
        df_c['Depthm'],
        bins=[0,50,150,300,600,6000],
        labels=['0–50m','50–150m','150–300m','300–600m','600m+']
    )
    return df_c


@st.cache_data(show_spinner="⏳ Loading zooplankton data…")
def load_zoo():
    zoo = pd.read_csv('zooplankton.csv', skiprows=[1])  # row 1 = units
    zoo['plankton_density'] = (
        zoo['total_plankton'] / zoo['volume_sampled'].replace(0, np.nan)
    )
    zoo['small_fraction'] = (
        zoo['small_plankton'] / zoo['total_plankton'].replace(0, np.nan)
    )
    return zoo


@st.cache_resource(show_spinner="⏳ Training Random Forest on real data…")
def get_model(n_rows):
    if os.path.exists('calcofi_model.pkl'):
        with open('calcofi_model.pkl','rb') as f:
            return pickle.load(f)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    _df = load_bottle()
    X   = _df[ML_FEATURES].fillna(_df[ML_FEATURES].median())
    y   = _df['health_label']
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2,
                                         random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=100, max_depth=15,
                                random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    with open('calcofi_model.pkl','wb') as f:
        pickle.dump(rf, f)
    return rf


try:
    df  = load_bottle()
    zoo = load_zoo()
except FileNotFoundError as e:
    st.error(f"❌ CSV not found: `{e}`\n\n"
             "Put `194903-202105_Bottle.csv` and `zooplankton.csv` "
             "in the same folder as `streamlit_app.py`, then reload.")
    st.stop()

model = get_model(len(df))


# ── Hero ──────────────────────────────────────────────────────────────────────
counts = df['health_label'].value_counts()
total  = len(df)

st.markdown(f"""
<div class="hero">
  <h1>🌊 Ocean Pulse</h1>
  <p>
    Marine Ecosystem Health Classifier &nbsp;·&nbsp;
    California Current &nbsp;·&nbsp; 1949–2021<br>
    <span style="font-size:.8rem;color:#4a8aaa;">
      {total:,} bottle samples &nbsp;·&nbsp;
      {len(zoo):,} zooplankton tows &nbsp;·&nbsp;
      Team: Laura · Maggie · Chau · David &nbsp;·&nbsp;
      DataHacks 2026 · Scripps Institution of Oceanography / NOAA CalCOFI
    </span>
  </p>
</div>
""", unsafe_allow_html=True)

k1,k2,k3,k4 = st.columns(4)
k1.metric("🔬 Total Samples", f"{total:,}")
k2.metric("✅ Healthy",  f"{counts.get('Healthy',0)/total:.1%}",
          f"{counts.get('Healthy',0):,} samples")
k3.metric("⚠️ Stressed", f"{counts.get('Stressed',0)/total:.1%}",
          f"{counts.get('Stressed',0):,} samples")
k4.metric("🚨 Critical", f"{counts.get('Critical',0)/total:.1%}",
          f"{counts.get('Critical',0):,} samples")

st.markdown("---")

tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
    "🔍 Live Predictor",
    "📈 Health Over Time",
    "🗺️ Geographic Map",
    "🌊 Depth Analysis",
    "🦐 Zooplankton",
    "🤖 Model Insights",
])


# ════════════════════════════════════════════════════════════════
# TAB 1  ·  LIVE PREDICTOR
# ════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 🔍 Classify Any Water Sample in Real Time")
    st.caption("Sliders use real value ranges from 895k CalCOFI bottle samples.")

    sl, sr = st.columns([1,1], gap="large")

    with sl:
        t_degc = st.slider("🌡️ Temperature (°C)",   float(df['T_degC'].quantile(.01)),
                           float(df['T_degC'].quantile(.99)), 12.5, 0.1)
        salnty = st.slider("🧂 Salinity (psu)",      float(df['Salnty'].quantile(.01)),
                           float(df['Salnty'].quantile(.99)), 33.8, 0.05)
        po4    = st.slider("🔵 Phosphate PO₄ (µM)",  0.0,
                           float(df['PO4uM'].quantile(.99)), 2.1, 0.05)
        no3    = st.slider("🟢 Nitrate NO₃ (µM)",    0.0,
                           float(df['NO3uM'].quantile(.99)), 15.0, 0.5)
        no2    = st.slider("🟡 Nitrite NO₂ (µM)",    0.0,
                           float(df['NO2uM'].quantile(.99)), 0.05, 0.01)
        depth  = st.slider("🌊 Depth (m)",
                           int(df['Depthm'].min()), int(df['Depthm'].quantile(.99)),
                           80, 5)
        chlora = st.slider("🌿 Chlorophyll-A (µg/L)", 0.0,
                           float(df['ChlorA'].quantile(.99)), 0.4, 0.05)

    with sr:
        sample = np.array([[t_degc, salnty, po4, no3, no2, depth, chlora]])
        pred   = model.predict(sample)[0]
        proba  = model.predict_proba(sample)[0]
        colour = HEALTH_COLORS[pred]
        emoji  = HEALTH_EMOJI[pred]

        st.markdown(f"""
        <div class="predict-result"
             style="background:linear-gradient(145deg,{colour}18,{colour}06);
                    border:1px solid {colour}50;">
          <div style="font-size:4rem;line-height:1">{emoji}</div>
          <div style="font-size:2.2rem;font-weight:700;color:{colour};
                      font-family:'Space Mono',monospace;margin:.3rem 0">{pred}</div>
          <div style="font-size:.8rem;color:#7ab3d4;letter-spacing:1px;
                      text-transform:uppercase">Ecosystem Status</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Confidence**")
        for cls, prob in sorted(zip(model.classes_, proba), key=lambda x: -x[1]):
            c = HEALTH_COLORS[cls]
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px">'
                f'<span style="width:72px;font-size:.85rem">{cls}</span>'
                f'<div style="flex:1;background:#1a3050;border-radius:4px;height:13px">'
                f'<div style="width:{prob*100:.1f}%;background:{c};border-radius:4px;height:13px"></div>'
                f'</div><span style="width:40px;text-align:right;font-family:monospace;'
                f'font-size:.85rem">{prob:.1%}</span></div>',
                unsafe_allow_html=True,
            )

        msgs = {
            "Healthy":  "✅ Oxygen and nutrient levels are within healthy ranges. "
                        "Marine life is well-supported at this depth and temperature.",
            "Stressed": "⚠️ Early warning. Elevated nitrate or lower oxygen suggests "
                        "biological oxygen demand is rising — likely upwelling or runoff.",
            "Critical": "🚨 Hypoxic. O₂ < 1.4 ml/L is lethal to most fish and "
                        "invertebrates. Dead-zone conditions are present.",
        }
        st.info(msgs[pred])

        with st.expander("📖 Label definitions"):
            st.markdown("""
| Label | O₂ (ml/L) | O₂ Sat | NO₃ (µM) |
|-------|-----------|--------|---------|
| ✅ Healthy  | ≥ 4.0  | ≥ 40% | ≤ 20 |
| ⚠️ Stressed | 1.4–4.0 | ≥ 40% | > 20 |
| 🚨 Critical | < 1.4  | < 40% | any  |

O₂ is excluded from model inputs (it defines the label) — prediction
uses temperature, salinity, nutrients, depth, and chlorophyll only.
            """)


# ════════════════════════════════════════════════════════════════
# TAB 2  ·  HEALTH OVER TIME
# ════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📈 California Current Health 1949–2021")
    st.caption(f"{total:,} samples across {int(df['Year'].max()-df['Year'].min())} years")

    trend = (df.groupby(['Year','health_label'])
               .size().reset_index(name='Count'))
    trend['Year'] = trend['Year'].astype(int)

    fig_area = px.area(trend, x='Year', y='Count', color='health_label',
                       color_discrete_map=HEALTH_COLORS,
                       title="Sample Health Distribution Over Time",
                       labels={"Count":"Bottle Samples"})
    fig_area.update_layout(**DARK, legend_title="Status")
    st.plotly_chart(fig_area, use_container_width=True)

    pct = (df.groupby('Year')
             .apply(lambda x: (x['health_label']=='Healthy').mean())
             .reset_index(name='pct'))
    pct['Year']    = pct['Year'].astype(int)
    pct['rolling'] = pct['pct'].rolling(5, center=True, min_periods=1).mean()

    fig_pct = go.Figure()
    fig_pct.add_trace(go.Scatter(
        x=pct['Year'], y=pct['pct'], mode='markers', name='Annual',
        marker=dict(color='#00b4d8', size=4, opacity=0.4),
    ))
    fig_pct.add_trace(go.Scatter(
        x=pct['Year'], y=pct['rolling'], mode='lines', name='5-yr avg',
        line=dict(color='#2ecc71', width=3),
    ))
    fig_pct.update_layout(**DARK,
                          title="% Healthy Samples (5-year rolling average)",
                          yaxis=dict(tickformat=".0%", gridcolor="#1a3050"),
                          xaxis=dict(gridcolor="#1a3050"))
    st.plotly_chart(fig_pct, use_container_width=True)

    st.markdown("**By decade**")
    dec = df.copy()
    dec['Decade'] = (dec['Year']//10*10).astype(int).astype(str) + 's'
    tbl = (dec.groupby(['Decade','health_label'])
              .size().unstack(fill_value=0))
    tbl['Total'] = tbl.sum(axis=1)
    for cls in ['Healthy','Stressed','Critical']:
        if cls in tbl.columns:
            tbl[f'{cls} %'] = (tbl[cls]/tbl['Total']*100).round(1)
    st.dataframe(tbl, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB 3  ·  GEOGRAPHIC MAP
# ════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🗺️ Where Is the Ocean Under Stress?")
    st.caption("Lat/lon approximated from CalCOFI line·station codes (Sta_ID)")

    yr_min, yr_max = int(df['Year'].min()), int(df['Year'].max())
    yr_sel = st.slider("Filter by year range", yr_min, yr_max, (1990, 2021))

    map_df = (df.dropna(subset=['Lat_Dec','Lon_Dec'])
                .query("@yr_sel[0] <= Year <= @yr_sel[1]")
                .sample(min(15000, len(df)), random_state=42))

    st.caption(f"Showing {len(map_df):,} samples")

    fig_map = px.scatter_mapbox(
        map_df, lat='Lat_Dec', lon='Lon_Dec',
        color='health_label', color_discrete_map=HEALTH_COLORS,
        zoom=5, height=580,
        title='California Current Ecosystem Health Map',
        hover_data={c:True for c in ['T_degC','O2ml_L','NO3uM','Year']
                    if c in map_df.columns},
        opacity=0.65,
    )
    fig_map.update_layout(
        mapbox_style='carto-darkmatter',
        paper_bgcolor="#040d17",
        font=dict(color="#cfe8ff"),
        margin=dict(l=0,r=0,t=40,b=0),
    )
    st.plotly_chart(fig_map, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB 4  ·  DEPTH ANALYSIS
# ════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🌊 How Does Health Change With Depth?")

    c1, c2 = st.columns(2)

    with c1:
        dh = (df.groupby(['Depth_Zone','health_label'], observed=True)
                .size().reset_index(name='Count'))
        fig_dh = px.bar(dh, x='Depth_Zone', y='Count', color='health_label',
                        barmode='group', color_discrete_map=HEALTH_COLORS,
                        title='Health Labels by Depth Zone')
        fig_dh.update_layout(**DARK)
        st.plotly_chart(fig_dh, use_container_width=True)

    with c2:
        o2d = df.groupby('Depth_Zone', observed=True)['O2ml_L'].mean().reset_index()
        fig_o2d = px.bar(o2d, x='Depth_Zone', y='O2ml_L',
                         title='Mean Dissolved O₂ by Depth Zone',
                         color='O2ml_L', color_continuous_scale='RdYlGn')
        fig_o2d.add_hline(y=1.4, line_dash="dash", line_color="#e74c3c",
                          annotation_text="Hypoxia 1.4 ml/L")
        fig_o2d.add_hline(y=4.0, line_dash="dash", line_color="#f39c12",
                          annotation_text="Stressed 4.0 ml/L")
        fig_o2d.update_layout(**DARK)
        st.plotly_chart(fig_o2d, use_container_width=True)

    fig_hist = px.histogram(
        df, x='O2ml_L', color='health_label',
        color_discrete_map=HEALTH_COLORS,
        nbins=80, barmode='overlay', opacity=0.75,
        title='Dissolved Oxygen Distribution by Health Class',
        labels={'O2ml_L':'Dissolved Oxygen (ml/L)'},
    )
    fig_hist.add_vline(x=1.4, line_dash="dash", line_color="#e74c3c",
                       annotation_text="1.4")
    fig_hist.add_vline(x=4.0, line_dash="dash", line_color="#f39c12",
                       annotation_text="4.0")
    fig_hist.update_layout(**DARK)
    st.plotly_chart(fig_hist, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB 5  ·  ZOOPLANKTON
# ════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 🦐 Zooplankton Biological Signal")
    st.caption(f"{len(zoo)} tow samples · Jan 2023 · California coast · "
               "Source: NOAA ERDDAP `erdCalCOFIzoovol`")

    z1, z2 = st.columns(2)

    with z1:
        fig_zmap = px.scatter_mapbox(
            zoo.dropna(subset=['latitude','longitude']),
            lat='latitude', lon='longitude',
            color='plankton_density', size='total_plankton',
            color_continuous_scale='Blues',
            zoom=6, height=400,
            title='Zooplankton Density (Jan 2023)',
            hover_data=['volume_sampled','small_plankton','total_plankton'],
            size_max=20,
        )
        fig_zmap.update_layout(
            mapbox_style='carto-darkmatter',
            paper_bgcolor="#040d17",
            font=dict(color="#cfe8ff"),
            margin=dict(l=0,r=0,t=40,b=0),
        )
        st.plotly_chart(fig_zmap, use_container_width=True)

    with z2:
        fig_zbar = px.bar(
            zoo.sort_values('total_plankton', ascending=False),
            x='station', y='total_plankton',
            color='plankton_density', color_continuous_scale='Blues',
            title='Total Plankton by Station',
            labels={'total_plankton':'Total Plankton (ml/1000m³)',
                    'station':'Station'},
        )
        fig_zbar.update_layout(**DARK)
        st.plotly_chart(fig_zbar, use_container_width=True)

    fig_zsc = px.scatter(
        zoo, x='total_plankton', y='small_plankton',
        color='plankton_density', color_continuous_scale='Blues',
        title='Small vs Total Plankton',
        labels={'total_plankton':'Total (ml/1000m³)',
                'small_plankton':'Small (ml/1000m³)'},
        trendline='ols',
    )
    fig_zsc.update_layout(**DARK)
    st.plotly_chart(fig_zsc, use_container_width=True)

    st.info("💡 These 29 tows are from Jan 2023. Once Chau's full zooplankton "
            "fetch is complete, `plankton_density` and `small_fraction` "
            "will be added as model features for extra biological signal.")


# ════════════════════════════════════════════════════════════════
# TAB 6  ·  MODEL INSIGHTS
# ════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("### 🤖 What Did the Model Learn?")

    m1, m2 = st.columns([1,1], gap="large")

    with m1:
        imp_df = pd.DataFrame({
            'Feature':    ML_FEATURES,
            'Importance': model.feature_importances_,
        }).sort_values('Importance', ascending=True)

        fig_imp = px.bar(imp_df, x='Importance', y='Feature',
                         orientation='h', title='Feature Importance (Random Forest)',
                         color='Importance', color_continuous_scale='Blues')
        fig_imp.update_layout(**DARK, showlegend=False)
        st.plotly_chart(fig_imp, use_container_width=True)

    with m2:
        st.markdown("**What each feature means scientifically**")
        sci = {
            'T_degC': ("🌡️ Temperature",
                "Warmer water holds less O₂ and speeds up microbial respiration — "
                "the primary physical driver of hypoxia in the California Current."),
            'NO3uM':  ("🟢 Nitrate",
                "Upwelling-driven nitrate fuels phytoplankton blooms. Decomposing "
                "blooms consume O₂, creating hypoxic dead zones downstream."),
            'Depthm': ("🌊 Depth",
                "Below the pycnocline (~100m), O₂ replenishment slows dramatically. "
                "Depth is a strong structural proxy for oxygen minimum zone exposure."),
            'PO4uM':  ("🔵 Phosphate",
                "Excess PO₄ relative to nitrogen (breaking Redfield N:P ≈ 16:1) "
                "signals active denitrification — a direct hypoxia marker."),
            'ChlorA': ("🌿 Chlorophyll-A",
                "High bloom biomass (high Chl-A) precedes O₂ crashes as organic "
                "matter sinks and decomposes, consuming dissolved oxygen."),
            'Salnty': ("🧂 Salinity",
                "Low-salinity surface lenses stratify the water column, cutting off "
                "O₂ replenishment to deeper layers — a precursor to dead zones."),
            'NO2uM':  ("🟡 Nitrite",
                "Elevated subsurface nitrite marks active denitrification — a "
                "direct biochemical fingerprint of oxygen-depleted conditions."),
        }
        top_order = imp_df.sort_values('Importance', ascending=False)['Feature'].tolist()
        for feat in top_order:
            if feat in sci:
                title, body = sci[feat]
                st.markdown(f"**{title}**  \n{body}")
                st.divider()

    # Classification report
    st.markdown("### 📊 Classification Report — 20% held-out test set")
    from sklearn.metrics import classification_report as _cr
    from sklearn.model_selection import train_test_split as _tts
    X  = df[ML_FEATURES].fillna(df[ML_FEATURES].median())
    y  = df['health_label']
    _, X_te, _, y_te = _tts(X, y, test_size=0.2, random_state=42, stratify=y)
    y_pred_te = model.predict(X_te)
    report_df = pd.DataFrame(_cr(y_te, y_pred_te, output_dict=True)).T.round(3)
    st.dataframe(
        report_df.style.background_gradient(
            subset=['f1-score','precision','recall'], cmap='RdYlGn'),
        use_container_width=True,
    )

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    import plotly.figure_factory as ff
    labs = ['Healthy','Stressed','Critical']
    cm   = confusion_matrix(y_te, y_pred_te, labels=labs)
    fig_cm = ff.create_annotated_heatmap(
        cm, x=labs, y=labs, colorscale='Blues', showscale=True)
    fig_cm.update_layout(
        **DARK, title="Confusion Matrix — Test Set",
        xaxis_title="Predicted", yaxis_title="Actual",
    )
    st.plotly_chart(fig_cm, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#2a5a7a;font-size:.78rem;padding:.5rem'>"
    "🌊 Ocean Pulse &nbsp;·&nbsp; DataHacks 2026 &nbsp;·&nbsp; UC San Diego &nbsp;|&nbsp;"
    " Data: CalCOFI / Scripps Institution of Oceanography / NOAA ERDDAP &nbsp;|&nbsp;"
    " Team: Laura · Maggie · Chau · David"
    "</div>",
    unsafe_allow_html=True,
)
