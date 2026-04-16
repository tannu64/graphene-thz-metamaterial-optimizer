"""
app.py - Streamlit web interface for THz Metamaterial Optimizer.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import DATA_DIR
from data_loader import load_all_data, pair_on_off, get_full_curves
from ml_model import (
    prepare_dataset, prepare_pair_dataset, build_models,
    evaluate_models, train_final_models, feature_importance,
)

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="THz Metamaterial Optimizer",
    page_icon="📡",
    layout="wide",
)

# ============================================================
# Cached Loading
# ============================================================

@st.cache_data
def load_data():
    df = load_all_data(DATA_DIR)
    pairs = pair_on_off(df)
    return df, pairs


@st.cache_data
def load_curves():
    return get_full_curves(DATA_DIR)


@st.cache_resource
def load_ml(_df, _pairs):
    """Train and evaluate all ML models. Cached so it runs once."""
    import warnings
    warnings.filterwarnings("ignore")

    X_all, Y_all, feat_cols, target_cols = prepare_dataset(_df)
    models = build_models()

    results_all, _ = evaluate_models(X_all, Y_all, models)
    trained_all, scaler_all = train_final_models(X_all, Y_all, build_models())
    imp_all = feature_importance(X_all, Y_all, feat_cols)

    results_pairs, imp_pairs = None, None
    if len(_pairs) >= 5:
        X_p, Y_p, _, _ = prepare_pair_dataset(_pairs)
        results_pairs, _ = evaluate_models(X_p, Y_p, build_models())
        imp_pairs = feature_importance(X_p, Y_p, feat_cols)

    return (results_all, results_pairs, trained_all, scaler_all,
            imp_all, imp_pairs, feat_cols, target_cols)


# Load everything
df, pairs = load_data()
curves = load_curves()
(results_all, results_pairs, trained_all, scaler_all,
 imp_all, imp_pairs, feat_cols, target_cols) = load_ml(df, pairs)


# ============================================================
# Sidebar
# ============================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Project Overview",
    "Data Explorer",
    "S-Parameter Curves",
    "ML Model Results",
    "Predict New Design",
    "Phase 3: Optimisation",
])

st.sidebar.markdown("---")
st.sidebar.caption(f"Dataset: {len(df)} simulations")
st.sidebar.caption(f"ON/OFF pairs: {len(pairs)}")


# ============================================================
# Page 1: Project Overview
# ============================================================
def render_overview():
    st.title("THz Metamaterial Optimizer")
    st.markdown("**AI-Powered Parameter Optimization for Graphene Metamaterial THz Devices**")
    st.markdown("---")

    # Key metrics
    best_dip = df['s12_min'].min()
    best_dip_freq = df.loc[df['s12_min'].idxmin(), 's12_min_freq']
    best_shift = pairs['freq_shift_ghz'].max() if len(pairs) > 0 else 0
    gap = -20 - best_dip

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Simulations", len(df))
    col2.metric("ON/OFF Pairs", len(pairs))
    col3.metric("Best S12 Dip", f"{best_dip:.2f} dB")

    col4, col5, col6 = st.columns(3)
    col4.metric("Best Freq Shift", f"{best_shift:.0f} GHz")
    col5.metric("Gap to -20 dB Target", f"{gap:.2f} dB")
    col6.metric("Best Dip Frequency", f"{best_dip_freq:.0f} GHz")

    st.markdown("---")

    # Objectives
    st.subheader("Optimization Objectives")
    obj1, obj2 = st.columns(2)
    with obj1:
        st.info(
            "**Objective 1: Maximize S12 Dip Depth**\n\n"
            "Target: -20 dB or deeper. Deeper dip = stronger resonance = better modulation."
        )
    with obj2:
        st.info(
            "**Objective 2: Maximize Frequency Shift**\n\n"
            "Larger ON/OFF frequency shift = better tunability of the device."
        )

    # Parameter table
    st.subheader("Design Parameters")
    param_data = {
        'Parameter': ['dx', 'g_w', 'c_w', 'h_graph', 'w_graph', 'w_au'],
        'Description': [
            'Unit cell size', 'Capacitor gap width', 'Capacitor width',
            'Graphene extra height', 'Graphene extra width', 'Gold line width',
        ],
        'Range': [
            f"{df['dx'].min()}-{df['dx'].max()} um",
            f"{df['g_w'].min()}-{df['g_w'].max()} um",
            f"{df['c_w'].min()}-{df['c_w'].max()} um",
            f"{df['h_graph'].min()} to {df['h_graph'].max()} um",
            f"{df['w_graph'].unique()[0]} (fixed)",
            f"{df['w_au'].unique()[0]} (fixed)",
        ],
        'Status': [
            'Varied', 'Varied', 'Varied', 'Varied', 'Fixed', 'Fixed',
        ],
    }
    st.dataframe(pd.DataFrame(param_data), use_container_width=True)


# ============================================================
# Page 2: Data Explorer
# ============================================================
def render_data_explorer():
    st.title("Data Explorer")

    tab1, tab2, tab3 = st.tabs(["All Simulations", "ON/OFF Pairs", "Best Configurations"])

    with tab1:
        st.subheader(f"All {len(df)} Simulations")
        display = df[['dx', 'g_w', 'c_w', 'h_graph', 'sigma', 's12_min', 's12_min_freq']].copy()
        display.columns = ['dx (um)', 'g_w (um)', 'c_w (um)', 'h_graph (um)', 'sigma (mS)',
                           'S12 Dip (dB)', 'Dip Freq (GHz)']
        display['S12 Dip (dB)'] = display['S12 Dip (dB)'].round(2)
        display['Dip Freq (GHz)'] = display['Dip Freq (GHz)'].round(0)
        st.dataframe(display, use_container_width=True)

    with tab2:
        st.subheader(f"{len(pairs)} ON/OFF Pairs")
        if len(pairs) > 0:
            pair_display = pairs[['dx', 'g_w', 'c_w', 'h_graph',
                                  's12_min_on', 's12_min_freq_on',
                                  's12_min_off', 's12_min_freq_off',
                                  'freq_shift_ghz', 'avg_dip']].copy()
            pair_display.columns = ['dx', 'g_w', 'c_w', 'h_graph',
                                    'ON Dip (dB)', 'ON Freq (GHz)',
                                    'OFF Dip (dB)', 'OFF Freq (GHz)',
                                    'Freq Shift (GHz)', 'Avg Dip (dB)']
            for c in ['ON Dip (dB)', 'OFF Dip (dB)', 'Avg Dip (dB)']:
                pair_display[c] = pair_display[c].round(2)
            st.dataframe(pair_display, use_container_width=True)
        else:
            st.warning("No ON/OFF pairs found.")

    with tab3:
        st.subheader("Best Configurations")

        best_dip_row = df.loc[df['s12_min'].idxmin()]
        st.success(
            f"**Deepest S12 Dip:** {best_dip_row['s12_min']:.2f} dB at {best_dip_row['s12_min_freq']:.0f} GHz\n\n"
            f"Config: dx={best_dip_row['dx']:.0f}, g_w={best_dip_row['g_w']:.0f}, "
            f"c_w={best_dip_row['c_w']:.0f}, h_graph={best_dip_row['h_graph']:.0f}, "
            f"sigma={best_dip_row['sigma']}"
        )

        if len(pairs) > 0:
            best_shift_row = pairs.loc[pairs['freq_shift_ghz'].idxmax()]
            st.success(
                f"**Largest Frequency Shift:** {best_shift_row['freq_shift_ghz']:.0f} GHz\n\n"
                f"Config: dx={best_shift_row['dx']:.0f}, g_w={best_shift_row['g_w']:.0f}, "
                f"c_w={best_shift_row['c_w']:.0f}, h_graph={best_shift_row['h_graph']:.0f}"
            )

            best_avg_row = pairs.loc[pairs['avg_dip'].idxmin()]
            st.success(
                f"**Deepest Average Dip (ON+OFF):** {best_avg_row['avg_dip']:.2f} dB\n\n"
                f"Config: dx={best_avg_row['dx']:.0f}, g_w={best_avg_row['g_w']:.0f}, "
                f"c_w={best_avg_row['c_w']:.0f}, h_graph={best_avg_row['h_graph']:.0f}"
            )


# ============================================================
# Page 3: S-Parameter Curves (Plotly)
# ============================================================
def render_curves():
    st.title("S-Parameter Curves")

    chart = st.selectbox("Select Chart", [
        "All S12 Transmission Curves",
        "S12 Dip vs Capacitor Width (c_w)",
        "S12 Dip vs Unit Cell Size (dx)",
        "ON/OFF State Comparison",
        "Frequency Shift Summary",
    ])

    if chart == "All S12 Transmission Curves":
        fig = go.Figure()
        for fname in sorted(curves.keys(), key=lambda f: curves[f]['params']['sigma']):
            p = curves[fname]['params']
            c = curves[fname]['curve']
            color = 'royalblue' if p['sigma'] == 0.3 else 'crimson'
            label = f"dx={p['dx']} gw={p['g_w']} cw={p['c_w']} hg={p['h_graph']} σ={p['sigma']}"
            fig.add_trace(go.Scatter(
                x=c['freq_ghz'], y=c['s12'], mode='lines',
                name=label, line=dict(color=color, width=1.5),
            ))
        fig.update_layout(
            title="All S12 Transmission Curves (Blue=ON, Red=OFF)",
            xaxis_title="Frequency (GHz)", yaxis_title="S12 (dB)",
            height=550, legend=dict(font=dict(size=9)),
        )
        st.plotly_chart(fig, use_container_width=True)

    elif chart == "S12 Dip vs Capacitor Width (c_w)":
        filtered = df[(df['sigma'] == 0.3) & (df['dx'] == 35) & (df['h_graph'] == 2)]
        fig = go.Figure()
        for gw, group in filtered.groupby('g_w'):
            group = group.sort_values('c_w')
            fig.add_trace(go.Scatter(
                x=group['c_w'], y=group['s12_min'], mode='lines+markers',
                name=f'g_w={gw} um', marker=dict(size=10),
            ))
        fig.update_layout(
            title="S12 Dip Depth vs Capacitor Width (dx=35, h_graph=2, ON state)",
            xaxis_title="c_w - Capacitor Width (um)",
            yaxis_title="S12 Dip Depth (dB)",
            yaxis=dict(autorange='reversed'), height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

    elif chart == "S12 Dip vs Unit Cell Size (dx)":
        dx_sweep = df[(df['g_w'] == 3) & (df['c_w'] == 28) &
                       (df['sigma'] == 0.3) & (df['h_graph'] == 2)].sort_values('dx')
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dx_sweep['dx'], y=dx_sweep['s12_min'], mode='lines+markers',
            marker=dict(size=10, color='royalblue'),
            text=[f"{f:.0f} GHz" for f in dx_sweep['s12_min_freq']],
            textposition='top right', textfont=dict(size=10),
            name='S12 Dip',
        ))
        fig.update_layout(
            title="S12 Dip Depth vs Unit Cell Size (g_w=3, c_w=28, h_graph=2, ON state)",
            xaxis_title="dx - Unit Cell Size (um)",
            yaxis_title="S12 Dip Depth (dB)",
            yaxis=dict(autorange='reversed'), height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

    elif chart == "ON/OFF State Comparison":
        if len(pairs) == 0:
            st.warning("No ON/OFF pairs found.")
            return

        n = len(pairs)
        fig = make_subplots(rows=1, cols=n,
                            subplot_titles=[
                                f"dx={r['dx']:.0f} gw={r['g_w']:.0f} cw={r['c_w']:.0f} hg={r['h_graph']:.0f}"
                                for _, r in pairs.iterrows()
                            ])

        for i, (_, pair) in enumerate(pairs.iterrows(), 1):
            for fname, data in curves.items():
                p = data['params']
                if (p['dx'] == pair['dx'] and p['g_w'] == pair['g_w'] and
                        p['c_w'] == pair['c_w'] and p['h_graph'] == pair['h_graph']):
                    c = data['curve']
                    color = 'royalblue' if p['sigma'] == 0.3 else 'crimson'
                    name = 'ON' if p['sigma'] == 0.3 else 'OFF'
                    fig.add_trace(go.Scatter(
                        x=c['freq_ghz'], y=c['s12'], mode='lines',
                        line=dict(color=color, width=2),
                        name=name, showlegend=(i == 1),
                    ), row=1, col=i)

            # Annotate frequency shift
            fig.add_annotation(
                x=(pair['s12_min_freq_on'] + pair['s12_min_freq_off']) / 2,
                y=pair['avg_dip'],
                text=f"Δf={pair['freq_shift_ghz']:.0f} GHz",
                showarrow=False, font=dict(color='green', size=12, family='Arial Black'),
                row=1, col=i,
            )

        fig.update_layout(height=450, title_text="ON vs OFF State Comparison")
        fig.update_xaxes(title_text="Freq (GHz)")
        fig.update_yaxes(title_text="S12 (dB)")
        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Frequency Shift Summary":
        if len(pairs) == 0:
            st.warning("No ON/OFF pairs found.")
            return

        labels = [f"dx={r['dx']:.0f} gw={r['g_w']:.0f}\ncw={r['c_w']:.0f} hg={r['h_graph']:.0f}"
                  for _, r in pairs.iterrows()]

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=labels, y=pairs['freq_shift_ghz'], name='Freq Shift (GHz)',
            marker_color='mediumseagreen', text=[f"{v:.0f}" for v in pairs['freq_shift_ghz']],
            textposition='outside',
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=labels, y=pairs['avg_dip'], name='Avg Dip (dB)',
            mode='markers', marker=dict(size=14, color='crimson', symbol='diamond'),
        ), secondary_y=True)

        fig.update_layout(
            title="Frequency Shift & Dip Depth per Configuration",
            height=500, barmode='group',
        )
        fig.update_yaxes(title_text="Frequency Shift (GHz)", secondary_y=False)
        fig.update_yaxes(title_text="Avg S12 Dip (dB)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Page 4: ML Model Results
# ============================================================
def render_ml_results():
    st.title("ML Model Results")
    st.caption("Evaluated using Leave-One-Out cross-validation")

    dataset = st.selectbox("Dataset", ["All Simulations (19 samples)", "ON/OFF Pairs (7 pairs)"])

    if "All" in dataset:
        results = results_all
        imp = imp_all
    else:
        if results_pairs is None:
            st.warning("Not enough ON/OFF pairs for evaluation.")
            return
        results = results_pairs
        imp = imp_pairs

    # Metrics table
    st.subheader("Model Performance")
    rows = []
    for model_name, targets in results.items():
        for target_name, m in targets.items():
            rows.append({
                'Model': model_name,
                'Target': target_name,
                'R²': round(m['R2'], 4),
                'RMSE': round(m['RMSE'], 4),
                'MAE': round(m['MAE'], 4),
            })
    metrics_df = pd.DataFrame(rows)
    st.dataframe(metrics_df, use_container_width=True)

    # Best model per target
    for target_name in list(results.values())[0].keys():
        best_model = max(results.keys(), key=lambda m: results[m][target_name]['R2'])
        best_r2 = results[best_model][target_name]['R2']
        st.success(f"**Best for {target_name}:** {best_model} (R² = {best_r2:.4f})")

    # Actual vs Predicted
    st.subheader("Actual vs Predicted")
    target_names = list(list(results.values())[0].keys())
    cols = st.columns(len(target_names))

    for idx, target_name in enumerate(target_names):
        with cols[idx]:
            fig = go.Figure()
            for model_name in results:
                m = results[model_name][target_name]
                fig.add_trace(go.Scatter(
                    x=m['y_true'], y=m['y_pred'], mode='markers',
                    name=model_name, marker=dict(size=8),
                ))
            # Diagonal reference
            all_vals = np.concatenate([results[mn][target_name]['y_true'] for mn in results])
            vmin, vmax = all_vals.min(), all_vals.max()
            fig.add_trace(go.Scatter(
                x=[vmin, vmax], y=[vmin, vmax], mode='lines',
                line=dict(dash='dash', color='gray'), name='Perfect',
                showlegend=False,
            ))
            fig.update_layout(
                title=target_name, xaxis_title="Actual", yaxis_title="Predicted",
                height=400, legend=dict(font=dict(size=9)),
            )
            st.plotly_chart(fig, use_container_width=True)

    # Feature Importance
    st.subheader("Feature Importance (Random Forest)")
    fig = go.Figure()
    for col_name in imp.columns:
        fig.add_trace(go.Bar(
            y=imp.index, x=imp[col_name], name=col_name, orientation='h',
        ))
    fig.update_layout(
        barmode='group', height=350,
        xaxis_title="Importance", yaxis_title="Parameter",
        legend=dict(font=dict(size=10)),
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Page 5: Predict New Design
# ============================================================
def render_predict():
    st.title("Predict New Design")
    st.markdown("Enter parameter values to get instant S12 dip and frequency predictions.")

    # Training data ranges for warnings
    train_ranges = {
        'dx': (df['dx'].min(), df['dx'].max()),
        'g_w': (df['g_w'].min(), df['g_w'].max()),
        'c_w': (df['c_w'].min(), df['c_w'].max()),
        'h_graph': (df['h_graph'].min(), df['h_graph'].max()),
    }

    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.subheader("Input Parameters")

        dx = st.slider("dx (unit cell size, um)", 20, 40, 35, step=1)
        g_w = st.slider("g_w (capacitor gap, um)", 2, 6, 3, step=1)
        c_w = st.slider("c_w (capacitor width, um)", 10, 35, 28, step=1)
        h_graph = st.slider("h_graph (graphene height, um)", -8, 10, 2, step=1)

        st.caption("Fixed parameters: w_graph=1, w_au=4")

        model_choice = st.selectbox("Model", list(trained_all.keys()))

        # Check if outside training range
        outside = []
        for name, val in [('dx', dx), ('g_w', g_w), ('c_w', c_w), ('h_graph', h_graph)]:
            lo, hi = train_ranges[name]
            if val < lo or val > hi:
                outside.append(f"{name}={val} (training range: {lo}-{hi})")

        if outside:
            st.warning(
                "**Extrapolation warning:** These values are outside the training data range:\n\n"
                + "\n".join(f"- {o}" for o in outside)
                + "\n\nPredictions may be unreliable."
            )

    with col_result:
        st.subheader("Predictions")

        # Build input
        X_input = np.array([[dx, g_w, c_w, h_graph, 1, 4]])  # w_graph=1, w_au=4
        X_scaled = scaler_all.transform(X_input)

        for target_name in target_cols:
            model = trained_all[model_choice][target_name]
            pred = model.predict(X_scaled)[0]

            # GP uncertainty
            if model_choice == 'Gaussian Process':
                pred_val, pred_std = model.predict(X_scaled, return_std=True)
                pred = pred_val[0]
                std = pred_std[0]
                st.metric(target_name, f"{pred:.2f}", help=f"±{std:.2f} (1 std)")
                st.caption(f"95% confidence: [{pred - 1.96*std:.2f}, {pred + 1.96*std:.2f}]")
            else:
                st.metric(target_name, f"{pred:.2f}")

        # Compare to target
        s12_pred = trained_all[model_choice][target_cols[0]].predict(X_scaled)[0]
        gap = -20 - s12_pred
        if s12_pred <= -20:
            st.success(f"This design meets the -20 dB target! (predicted: {s12_pred:.2f} dB)")
        elif s12_pred <= -16:
            st.info(f"Close to target. Gap: {gap:.2f} dB to reach -20 dB")
        else:
            st.warning(f"Gap: {gap:.2f} dB to reach -20 dB target")


# ============================================================
# Page 6: Phase 3 Optimisation (Bayesian + NSGA-II)
# ============================================================
@st.cache_data
def load_phase3_bo():
    import os
    path = os.path.join('output', 'phase3_bo_top10.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def load_phase3_pareto():
    import os
    path = os.path.join('output', 'phase3_pareto_front.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def render_phase3():
    st.title("Phase 3: Multi-Objective Optimisation")
    st.markdown(
        "Use the trained ML surrogates to search the parameter space for "
        "designs that maximise device performance, without running any new "
        "COMSOL simulations."
    )
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs([
        "Bayesian Optimisation (Single Objective)",
        "NSGA-II (Multi Objective Pareto)",
        "Validation Package",
    ])

    with tab1:
        st.subheader("Bayesian Optimisation — Maximise S12 Dip Depth")
        st.markdown(
            "Uses **scikit-optimize**'s `gp_minimize` with Expected Improvement "
            "acquisition function. The GP surrogate is consulted ~60 times to "
            "find the parameter set with the deepest predicted dip."
        )

        bo_df = load_phase3_bo()
        if bo_df is None:
            st.warning(
                "Run `python phase3_bayesopt.py` first to generate results.")
        else:
            col_pred = [c for c in bo_df.columns if c.startswith('predicted')][0]
            st.markdown(f"**Best predicted dip:** {bo_df[col_pred].min():.3f} dB")
            st.markdown(
                "**Top 10 candidates** (ranked by predicted S12 dip depth):")
            st.dataframe(bo_df, use_container_width=True)

            import os
            conv_path = os.path.join('output', 'plots', 'phase3_bo_convergence.png')
            if os.path.exists(conv_path):
                st.image(conv_path, caption="BO convergence trace")

    with tab2:
        st.subheader("NSGA-II — Pareto Trade-off: Dip Depth vs Frequency Shift")
        st.markdown(
            "Uses **pymoo**'s NSGA-II with population=40, generations=50. "
            "Objectives: (1) maximise dip depth, (2) maximise |ON/OFF frequency "
            "shift|. The Pareto front shows the best-achievable trade-offs."
        )

        pareto_df = load_phase3_pareto()
        if pareto_df is None:
            st.warning(
                "Run `python phase3_nsga2.py` first to generate results.")
        else:
            st.markdown(
                f"**Pareto front size:** {len(pareto_df)} non-dominated solutions")

            # Interactive Plotly scatter
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pareto_df['predicted_dip_dB'],
                y=pareto_df['predicted_freq_shift_abs_GHz'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=pareto_df['h_graph'],
                    colorscale='Viridis',
                    colorbar=dict(title='h_graph (μm)'),
                    line=dict(color='black', width=0.5),
                ),
                text=[f"dx={r.dx}, g_w={r.g_w}, c_w={r.c_w}, "
                      f"h_graph={r.h_graph}, w_au={r.w_au}"
                      for r in pareto_df.itertuples()],
                hovertemplate="%{text}<br>Dip: %{x:.2f} dB<br>"
                              "Shift: %{y:.1f} GHz<extra></extra>",
            ))
            fig.update_layout(
                xaxis_title="Predicted S12 Dip (dB, more negative = deeper)",
                yaxis_title="Predicted |Frequency Shift| (GHz, larger = better)",
                title="Pareto Front — hover for parameter details",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Pareto-optimal parameter sets:**")
            st.dataframe(pareto_df, use_container_width=True)

            # Download CSV button
            csv = pareto_df.to_csv(index=False).encode()
            st.download_button(
                "Download Pareto front as CSV",
                csv, "phase3_pareto_front.csv", "text/csv"
            )

    with tab3:
        st.subheader("Top 5 Candidates for COMSOL Validation")
        st.markdown(
            "These 5 candidates are selected from the Pareto front to be "
            "validated by running new COMSOL simulations. Comparing predicted "
            "vs. actual performance will quantify the surrogate's real-world "
            "accuracy."
        )
        pareto_df = load_phase3_pareto()
        if pareto_df is not None:
            top5 = pareto_df.head(5).copy()
            top5.insert(0, 'candidate', range(1, len(top5) + 1))
            st.dataframe(top5, use_container_width=True)
            st.markdown(
                "**Acceptance criteria for surrogate validation:**\n"
                "- MAE (dip) ≤ 1.5 dB\n"
                "- MAE (frequency) ≤ 20 GHz\n"
                "- R² ≥ 0.6 on the 5 held-out points"
            )
        else:
            st.info("Run NSGA-II first to populate this tab.")


# ============================================================
# Routing
# ============================================================
if page == "Project Overview":
    render_overview()
elif page == "Data Explorer":
    render_data_explorer()
elif page == "S-Parameter Curves":
    render_curves()
elif page == "ML Model Results":
    render_ml_results()
elif page == "Predict New Design":
    render_predict()
elif page == "Phase 3: Optimisation":
    render_phase3()
