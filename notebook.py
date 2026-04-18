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
def _(mo):
    from groq import Groq

    GROQ_API_KEY = "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"

    groq_client = Groq(api_key=GROQ_API_KEY)

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
def _():
    return


if __name__ == "__main__":
    app.run()
