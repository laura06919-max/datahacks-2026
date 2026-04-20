# 🌊 Ocean Pulse — DataHacks 2026

> 🏆 **3rd Place — Machine Learning / AI Track** | DataHacks 2026 @ UC San Diego

**Team:** Ocean Pulse | Laura · Chau · David · Maggie  
**Track:** Machine Learning / AI  
**Event:** DataHacks 2026 @ UC San Diego  

---

## Project Overview

The California Current is one of the most productive marine ecosystems 
on Earth. However, after decades of climate change, ocean acidification, and 
warming waters are pushing it toward a tipping point.

Using **70 years of real oceanographic data** from the Scripps Institution 
of Oceanography (CalCOFI program, 1949–2021), we built a machine learning 
pipeline that classifies the health state of the California Current 
ecosystem into three states:

| Label | Meaning |
|-------|---------|
| 🟢 Healthy | Normal marine conditions, oxygen > 4.0 ml/L |
| 🟡 Stressed | Low oxygen or nutrient overload, early warning signs |
| 🔴 Critical | Hypoxic conditions, life-threatening for marine life |

---

## Results

- **Model:** Random Forest Classifier
- **Accuracy:** 98%
- **Training samples:** 286,819
- **Key finding:** Phosphate and nitrate are the strongest predictors of ocean health. Therefore, oxygen stress can be inferred from nutrients alone

---

## Dataset

**CalCOFI — California Cooperative Oceanic Fisheries Investigations**  
Scripps Institution of Oceanography, UC San Diego  
- Hydrographic Bottle Data (1949–2021)
- 895,371 raw samples across the California coastline

---

## Features Used

| Feature | Description |
|---------|-------------|
| T_degC | Water temperature (°C) |
| Salnty | Salinity (PSU) |
| PO4uM | Phosphate concentration (µM) |
| NO3uM | Nitrate concentration (µM) |
| NO2uM | Nitrite concentration (µM) |
| ChlorA | Chlorophyll-A (µg/L) |
| Depthm | Sample depth (meters) |

---

## Project Structure

```
datahacks-2026/
├── calcofi_analysis.py      # ML pipeline — data loading, labeling, training
├── calcofi_marimo.py        # Marimo interactive notebook
├── streamlit_app-3.py       # Interactive web dashboard
├── docs/                    # Sphinx documentation
│   └── build/html/          # Generated HTML docs
└── README.md
```
---

## How To Run

**ML Pipeline:**
```
python3 calcofi_analysis.py
```

**Interactive Marimo Notebook:**
```
MARIMO_OUTPUT_MAX_BYTES=10000000 marimo edit calcofi_marimo.py
```

**Streamlit Dashboard:**
```
streamlit run streamlit_app-3.py
```
**View Sphinx Documentation:**
Open `docs/build/html/index.html` in your browser, or run:
```
open docs/build/html/index.html
```
**Interactive website:**
```
https://laura06919-max.github.io/datahacks-2026/
```
---
 
## Data Source

California Cooperative Oceanic Fisheries Investigations (CalCOFI)  
https://calcofi.org  
Scripps Institution of Oceanography © 2021
