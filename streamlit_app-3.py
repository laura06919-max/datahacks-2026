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

tab7,tab1,tab8,tab2,tab3,tab4,tab6 = st.tabs([
    "📡 Data Sources & Forecast",
    "🔍 Live Predictor",
    "🌎 Regional Policy Advisor",
    "📈 Health Over Time",
    "🗺️ Geographic Map",
    "🌊 Depth Analysis",
    "🤖 Model Insights",
])


# ════════════════════════════════════════════════════════════════
# TAB 1  ·  LIVE PREDICTOR
# ════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 🔍 Classify Any Water Sample in Real Time")
    st.markdown("Sliders use real value ranges from 895k CalCOFI bottle samples.")

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
    st.markdown(f"{total:,} samples across {int(df['Year'].max()-df['Year'].min())} years")

    trend = (df.groupby(['Year','health_label'])
               .size().reset_index(name='Count'))
    trend['Year'] = trend['Year'].astype(int)

    fig_area = px.area(trend, x='Year', y='Count', color='health_label',
                   color_discrete_map=HEALTH_COLORS,
                   title="Sample Health Distribution Over Time",
                   labels={"Count":"Bottle Samples"})
    fig_area.update_layout(**DARK, legend_title="Status")
    st.plotly_chart(fig_area, use_container_width=True)

    st.markdown(
        "Track how ocean health has evolved from 1949 to today. "
        "Each line shows how many samples fall into healthy, stressed, "
        "or critical conditions each year."
    )

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
                      title="% Healthy Samples (5-year rolling average)")
    fig_pct.update_xaxes(tickformat="", gridcolor="#1a3050")
    fig_pct.update_yaxes(tickformat=".0%", gridcolor="#1a3050")
    st.plotly_chart(fig_pct, use_container_width=True)
    
    st.markdown(
    "Track the share of healthy ocean samples over time. "
    "Dots show yearly values, while the line highlights the overall trend."
    )

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
    st.markdown("Lat/lon approximated from CalCOFI line·station codes (Sta_ID)")

    yr_min, yr_max = int(df['Year'].min()), int(df['Year'].max())
    yr_sel = st.slider("Filter by year range", yr_min, yr_max, (1990, 2021))

    map_df = (df.dropna(subset=['Lat_Dec','Lon_Dec'])
                .query("@yr_sel[0] <= Year <= @yr_sel[1]")
                .sample(min(15000, len(df)), random_state=42))

    st.markdown(f"Showing {len(map_df):,} samples")

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

        st.markdown(
            "See how ecosystem health varies across ocean depths. "
            "Surface waters are mostly healthy, but as you go deeper, "
            "stressed and critical conditions become more common."
        )

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

        st.markdown(
            "Explore how oxygen levels change with depth. "
            "Lower oxygen at deeper levels can lead to stressed or even hypoxic "
            "(low-oxygen) conditions for marine life."
        )

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
    st.markdown(
        "Explore how dissolved oxygen levels relate to ecosystem health. "
        "Lower oxygen levels are strongly linked to stressed and critical "
        "conditions for marine life."
    )


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

# ════════════════════════════════════════════════════════════════
# TAB 7  ·  DATA SOURCES & FORECAST
# ════════════════════════════════════════════════════════════════
with tab7:
    st.markdown("### 📡 Get Ocean Data For Your Location")
    st.markdown("""
    Enter your location and we'll fetch the nearest real ocean measurements 
    and run them through our model automatically.
    """)

    # ── Location Input ────────────────────────────────────────
    col1, col2 = st.columns([2, 1])
    with col1:
        location_input = st.text_input(
            "Enter a city, or coastal location:",
            placeholder="e.g. San Diego, CA  or  La Jolla"
        )
    with col2:
        radius_km = st.slider("Search radius (km)", 50, 500, 200)

    # Location → lat/lon mapping for California coastalities
    LOCATION_COORDS = {
        # San Diego area
        'san diego': (32.72, -117.15),
        'la jolla': (32.85, -117.27),
        '92037': (32.85, -117.27),
        '92101': (32.72, -117.17),
        '92109': (32.77, -117.23),
        # LA area
        'los angeles': (33.95, -118.40),
        'santa monica': (34.02, -118.50),
        'long beach': (33.77, -118.19),
        'malibu': (34.03, -118.78),
        # Central CA
        'santa barbara': (34.42, -119.70),
        'monterey': (36.60, -121.89),
        'big sur': (36.27, -121.81),
        'san luis obispo': (35.28, -120.66),
        # Northern CA
        'san francisco': (37.75, -122.45),
        'half moon bay': (37.46, -122.43),
        'bodega bay': (38.33, -123.05),
        'eureka': (40.80, -124.16),
    }

    if location_input:
        # Try to match location
        loc_key = location_input.lower().strip()
        matched_coords = None

        for key, coords in LOCATION_COORDS.items():
            if key in loc_key or loc_key in key:
                matched_coords = coords
                matched_name = key.title()
                break

        if matched_coords is None:
            st.warning(f"Location '{location_input}' not recognized. Try a California coastal city like 'San Diego', 'Monterey', or 'San Francisco'.")
        else:
            lat, lon = matched_coords
            st.success(f"📍 Found: **{matched_name}** ({lat:.2f}°N, {abs(lon):.2f}°W)")

            # Show nearby historical data from our dataset
            st.markdown("#### Nearby Ocean Samples From Our Dataset")

            # Filter by approximate location
            lat_range = radius_km / 111
            lon_range = radius_km / 85

            nearby = df[
                (df['Lat_Dec'].between(lat - lat_range, lat + lat_range)) &
                (df['Lon_Dec'].between(lon - lon_range, lon + lon_range))
            ].copy()

            if len(nearby) == 0:
                st.warning("No samples found in this area. Try increasing the search radius.")
            else:
                st.info(f"Found **{len(nearby):,}** historical samples within {radius_km}km of {matched_name}")

                # Show recent samples
                recent_nearby = nearby[nearby['Year'] >= 2015].copy()
                if len(recent_nearby) == 0:
                    recent_nearby = nearby.tail(100)

                # Average measurements
                avg_measurements = recent_nearby[ML_FEATURES].mean()

                st.markdown("#### Average Recent Measurements For This Area")
                metric_cols = st.columns(4)
                metric_cols[0].metric("🌡️ Temperature", f"{avg_measurements['T_degC']:.1f}°C")
                metric_cols[1].metric("🧂 Salinity", f"{avg_measurements['Salnty']:.2f} PSU")
                metric_cols[2].metric("🔵 Phosphate", f"{avg_measurements['PO4uM']:.2f} µM")
                metric_cols[3].metric("🟢 Nitrate", f"{avg_measurements['NO3uM']:.1f} µM")

                # Auto-predict using average measurements
                pred_input = pd.DataFrame([avg_measurements[ML_FEATURES]])
                pred = model.predict(pred_input)[0]
                proba = model.predict_proba(pred_input)[0]
                color = HEALTH_COLORS[pred]
                emoji = HEALTH_EMOJI[pred]

                st.markdown("#### Model Prediction For This Location")
                st.markdown(f"""
                <div class="predict-result" style="background:{color}22;
                     border:2px solid {color};margin:.5rem 0">
                  <div style="font-size:3rem">{emoji}</div>
                  <div style="font-size:1.8rem;font-weight:700;color:{color};
                              font-family:'Space Mono',monospace">{pred}</div>
                  <div style="color:#cfe8ff;margin-top:.5rem">
                    Based on {len(recent_nearby):,} nearby samples
                    (2015–2021)
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Probability bars
                for cls, prob in sorted(zip(model.classes_, proba), key=lambda x: -x[1]):
                    c = HEALTH_COLORS[cls]
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px">'
                        f'<span style="width:80px">{cls}</span>'
                        f'<div style="flex:1;background:#1a3050;border-radius:4px;height:13px">'
                        f'<div style="width:{prob*100:.1f}%;background:{c};border-radius:4px;height:13px"></div>'
                        f'</div><span style="width:45px;text-align:right;font-family:monospace">{prob:.1%}</span></div>',
                        unsafe_allow_html=True
                    )

                # Map of nearby samples
                st.markdown("#### Nearby Sample Locations")
                fig_nearby = px.scatter_mapbox(
                    recent_nearby.dropna(subset=['Lat_Dec','Lon_Dec']).sample(
                        min(2000, len(recent_nearby)), random_state=42),
                    lat='Lat_Dec', lon='Lon_Dec',
                    color='health_label',
                    color_discrete_map=HEALTH_COLORS,
                    zoom=7, height=400,
                    title=f'Ocean Health Near {matched_name}',
                    hover_data=['T_degC','O2ml_L','NO3uM','Year']
                )
                fig_nearby.update_layout(
                    mapbox_style='carto-darkmatter',
                    paper_bgcolor="#040d17",
                    font=dict(color="#cfe8ff"),
                    margin=dict(l=0,r=0,t=40,b=0)
                )
                st.plotly_chart(fig_nearby, use_container_width=True)

                # Health trend for this location
                st.markdown("#### Health Trend For This Area Over Time")
                loc_trend = nearby.groupby(['Year','health_label']).size().reset_index(name='Count')
                loc_trend['Year'] = loc_trend['Year'].astype(int)
                fig_loc = px.area(loc_trend, x='Year', y='Count', color='health_label',
                                  color_discrete_map=HEALTH_COLORS,
                                  title=f'Ocean Health Trend Near {matched_name}')
                fig_loc.update_layout(**DARK)
                st.plotly_chart(fig_loc, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔮 Future Projection")
    target_year = st.slider("Project ocean conditions to year:", 2022, 2040, 2026)

    if st.button("🔮 Generate Projection"):
        with st.spinner("Computing historical trends..."):
            recent = df[df['Year'] >= 2000].copy()
            trend_rows = []

            for feat in ML_FEATURES:
                yearly = recent.groupby('Year')[feat].mean().reset_index().dropna()
                if len(yearly) > 3:
                    slope, intercept = np.polyfit(yearly['Year'], yearly[feat], 1)
                    projected = slope * target_year + intercept
                    last_known = yearly[feat].iloc[-1]
                    change_pct = ((projected - last_known) / abs(last_known) * 100) if last_known != 0 else 0
                    trend_rows.append({
                        'Feature': feat,
                        'Last Known (2021)': round(last_known, 3),
                        'Annual Change': round(slope, 4),
                        f'Projected ({target_year})': round(projected, 3),
                        'Change %': round(change_pct, 1)
                    })

            trend_df = pd.DataFrame(trend_rows)
            st.dataframe(trend_df.set_index('Feature'), use_container_width=True)

            synthetic = {r['Feature']: r[f'Projected ({target_year})']
                        for _, r in trend_df.iterrows()}
            synthetic_df = pd.DataFrame([synthetic])
            pred  = model.predict(synthetic_df[ML_FEATURES])[0]
            proba = model.predict_proba(synthetic_df[ML_FEATURES])[0]
            color = HEALTH_COLORS[pred]
            emoji = HEALTH_EMOJI[pred]

            st.markdown(f"""
            <div class="predict-result" style="background:{color}22;
                 border:2px solid {color};margin-top:1rem">
              <div style="font-size:3rem">{emoji}</div>
              <div style="font-size:1.8rem;font-weight:700;color:{color};
                          font-family:'Space Mono',monospace">{pred}</div>
              <div style="color:#cfe8ff;margin-top:.5rem">
                Projected for <strong>{target_year}</strong> | 
                Confidence: {max(proba):.1%}
              </div>
            </div>
            """, unsafe_allow_html=True)
# ════════════════════════════════════════════════════════════════
# TAB 8  ·  REGIONAL POLICY ADVISOR
# ════════════════════════════════════════════════════════════════
with tab8:
    st.markdown("### 🌎 Regional Ocean Health & Government Action Advisor")
    st.markdown("""
    Select a California coastal region to see its health summary
    and get AI-powered policy recommendations from our LLM.
    """)

    def assign_region(lat):
        if lat >= 38.0:
            return "Northern California (SF & Above)"
        elif lat >= 35.0:
            return "Central California (Monterey / Big Sur)"
        elif lat >= 32.5:
            return "Southern California (LA / San Diego)"
        else:
            return "Baja California Border Zone"

    df['Region'] = df['Lat_Dec'].apply(assign_region)

    region_summary = df.groupby('Region').agg(
        Avg_Temperature=('T_degC', 'mean'),
        Avg_Oxygen=('O2ml_L', 'mean'),
        Avg_Nitrate=('NO3uM', 'mean'),
        Avg_Phosphate=('PO4uM', 'mean'),
        Sample_Count=('O2ml_L', 'count'),
        Critical_Pct=('health_label', lambda x: (x == 'Critical').mean() * 100),
        Stressed_Pct=('health_label', lambda x: (x == 'Stressed').mean() * 100),
        Healthy_Pct=('health_label', lambda x: (x == 'Healthy').mean() * 100),
    ).round(2).reset_index()

    selected = st.selectbox(
        "Select a California coastal region:",
        region_summary['Region'].tolist()
    )

    row = region_summary[region_summary['Region'] == selected].iloc[0]

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Avg Temperature", f"{row['Avg_Temperature']:.1f}°C")
    r2.metric("Avg Oxygen", f"{row['Avg_Oxygen']:.2f} ml/L")
    r3.metric("Critical %", f"{row['Critical_Pct']:.1f}%")
    r4.metric("Stressed %", f"{row['Stressed_Pct']:.1f}%")

    # Region health trend chart
    region_df = df[df['Region'] == selected]
    reg_trend = region_df.groupby(['Year','health_label']).size().reset_index(name='Count')
    reg_trend['Year'] = reg_trend['Year'].astype(int)
    fig_reg = px.area(reg_trend, x='Year', y='Count', color='health_label',
                      color_discrete_map=HEALTH_COLORS,
                      title=f'Ocean Health Trend — {selected}')
    fig_reg.update_layout(**DARK)
    st.plotly_chart(fig_reg, use_container_width=True)

    # Groq policy recommendation
    st.markdown("#### AI Policy Recommendations")
    if st.button("🤖 Get Policy Recommendations"):
        try:
            from groq import Groq
            groq_client = Groq(api_key="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")

            dominant = 'Critical' if row['Critical_Pct'] > 30 else \
                       'Stressed' if row['Stressed_Pct'] > 40 else 'Healthy'

            prompt = f"""
You are an ocean policy advisor for the state of California.
You are specifically analyzing the {selected} coastal region.

Average measurements:
- Temperature: {row['Avg_Temperature']}°C
- Dissolved Oxygen: {row['Avg_Oxygen']} ml/L
- Nitrate: {row['Avg_Nitrate']} µM
- Phosphate: {row['Avg_Phosphate']} µM
- Critical samples: {row['Critical_Pct']}%
- Stressed samples: {row['Stressed_Pct']}%

Provide exactly three sections:

**CURRENT SITUATION IN {selected.upper()}:**
2-3 sentences on what these measurements mean for marine life.

**IF NO ACTION IS TAKEN:**
2-3 sentences on consequences in 10-20 years.

**RECOMMENDED GOVERNMENT ACTIONS:**
3 specific actions for California agencies (Coastal Commission, EPA, 
Water Board) specific to this region's geography and industries.
"""
            with st.spinner("Consulting AI policy advisor..."):
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}]
                )
                advice = response.choices[0].message.content
                st.markdown(advice)

        except Exception as e:
            st.error(f"API error: {e}")
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
