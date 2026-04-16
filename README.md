# Graphene THz Metamaterial Optimizer

ML-based optimization of graphene THz metamaterial biosensors using COMSOL simulation data with an interactive Streamlit dashboard.

## Project Overview

This project applies machine learning to optimize graphene-based terahertz (THz) metamaterial devices for biosensing applications. It automates the analysis of COMSOL electromagnetic simulation data, trains regression models to predict S-parameter behavior, and provides an interactive web dashboard for exploration and design prediction.

### Key Features

- **Automated Data Pipeline** — Parses COMSOL `.txt` exports, extracts geometric parameters from filenames, identifies S12/S22 dip frequencies
- **ON/OFF State Analysis** — Pairs simulations at different conductivity states (sigma=0.3 vs 1.2) to compute frequency shifts
- **ML Prediction Models** — Random Forest, Gradient Boosting, and Gaussian Process regressors with Leave-One-Out cross-validation
- **Interactive Dashboard** — 5-page Streamlit app with Plotly charts for data exploration, model evaluation, and new design prediction
- **Parameter Sensitivity** — Identifies which geometric parameters (dx, g_w, c_w, h_graph) most influence device performance

## Project Structure

```
├── streamlit_app.py                 # Streamlit web dashboard (5 pages)
├── config.py              # Centralized path configuration
├── data_loader.py         # COMSOL file parser and data pipeline
├── ml_model.py            # ML model training and evaluation
├── analyze.py             # Text report generation
├── visualize.py           # Matplotlib plot generation
├── generate_report.py     # Word document report generation
├── requirements.txt       # Python dependencies
├── data/                  # COMSOL simulation exports (19 files)
└── output/                # Generated reports, plots, ML results
```

## Simulation Parameters

| Parameter | Description | Range |
|-----------|------------|-------|
| `dx` | Unit cell size (um) | 23 - 35 |
| `g_w` | Gap width (um) | 3 - 5 |
| `c_w` | Capacitor width (um) | 12 - 28 |
| `h_graph` | Graphene height (um) | -6 to 6 |
| `w_graph` | Graphene width (um) | 1 (fixed) |
| `w_au` | Gold width (um) | 4 (fixed) |
| `sigma` | Conductivity state | 0.3 (ON), 1.2 (OFF) |

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:** pandas, numpy, matplotlib, scikit-learn, python-docx, streamlit, plotly

## Usage

### Run the Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

The dashboard has 5 pages:
1. **Project Overview** — Key metrics and parameter summary
2. **Data Explorer** — Browse all simulations, ON/OFF pairs, and best configurations
3. **S-Parameter Curves** — Interactive Plotly charts of S12 transmission data
4. **ML Model Results** — Model comparison, actual vs predicted plots, feature importance
5. **Predict New Design** — Input parameters via sliders and get ML predictions

### Run Individual Scripts

```bash
python data_loader.py       # Verify data loading (prints summary)
python analyze.py           # Generate text analysis report
python ml_model.py          # Train and evaluate ML models
python visualize.py         # Generate matplotlib plots
python generate_report.py   # Generate Word document report
```

## ML Model Performance

Evaluated using Leave-One-Out cross-validation on 19 samples:

| Model | S12 Dip (R²) | Dip Freq (R²) |
|-------|--------------|----------------|
| Random Forest | 0.34 | 0.63 |
| Gradient Boosting | 0.58 | 0.49 |
| Gaussian Process | 0.44 | 0.75 |

**Key finding:** `h_graph` (graphene height) dominates S12 dip prediction with 69.6% feature importance.

## Current Results

- **Best S12 dip:** -16.64 dB (target: -20 dB)
- **Best frequency shift:** 60 GHz (ON/OFF switching)
- **Gap to target:** 3.36 dB improvement needed
- **19 simulations** with **7 complete ON/OFF pairs**

## License

This project is part of a Final Year Project (FYP) for academic purposes.
