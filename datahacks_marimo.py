import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
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
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    ## 📋 Project Overview

    **Goal:** Classify California Current ecosystem health using decades of ocean data

    **Dataset:** CalCOFI (Scripps Institution of Oceanography) -  Hydrographic Bottle Data & Bottle + Cast Tables

    **Health Labels:**
    - ✅ Healthy — High oxygen, normal pH, abundant plankton
    - ⚠️ Stressed — Declining oxygen, acidifying waters
    - 🚨 Critical — Hypoxic zones, ecosystem collapse risk

    **ML Model:** XGBoost Multi-class Classifier
    """)
    return


@app.cell
def _(mo):
    from groq import Groq

    GROQ_API_KEY = "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"

    client = Groq(api_key=GROQ_API_KEY)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Say: API is working!"}]
    )

    mo.md(f"### API Test Result: {response.choices[0].message.content}")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 🎤 Pitch Outline

    1. **Problem** — California's ocean ecosystem is under threat from climate change
    2. **Data** — 70+ years of CalCOFI ocean chemistry & biological data (Scripps)
    3. **Solution** — XGBoost ML model classifies ecosystem as Healthy / Stressed / Critical
    4. **Results** — Declining oxygen and rising acidity signal growing ecosystem stress
    5. **Impact** — An early warning system to help scientists protect the California Current
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
