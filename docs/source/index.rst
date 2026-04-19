DataHacks 2026 — Ocean Pulse
=============================

.. toctree::
   :maxdepth: 2
   :caption: Contents

   overview
   methodology
   results
   modules

Overview
--------
Ocean Pulse is a machine learning project built at DataHacks 2026 using 
70 years of real oceanographic data from the Scripps Institution of 
Oceanography (CalCOFI program, 1949–2021).

We classify California Current ecosystem health into three states 
(Healthy, Stressed, and Critical), using a Random Forest classifier 
trained on water chemistry measurements.

**Team:** Ocean Pulse (Laura, Chau, David, Maggie)  
**Track:** Machine Learning / AI  
**Dataset:** CalCOFI Hydrographic Bottle Database (Scripps)  
**Model Accuracy:** 98%

Key Findings
------------
- Phosphate and nitrate are the strongest predictors of ocean health
- Critical hypoxic zones worsen significantly below 300m depth
- Stressed conditions have increased since the 1980s
- Ocean health can be predicted from nutrients alone and no oxygen sensor are needed

Data Source
-----------
California Cooperative Oceanic Fisheries Investigations (CalCOFI)  
Scripps Institution of Oceanography, UC San Diego  
https://calcofi.org

Usage
-----
To run the full pipeline::

   python3 calcofi_analysis.py

To launch the interactive Marimo notebook::

   MARIMO_OUTPUT_MAX_BYTES=10000000 marimo edit calcofi_marimo.py

Requirements
------------
Install dependencies::

   pip install pandas numpy scikit-learn plotly marimo sphinx groq seaborn matplotlib xgboost