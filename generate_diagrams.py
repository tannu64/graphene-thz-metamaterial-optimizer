"""
generate_diagrams.py - Generates all diagrams and plots needed for the Phase 2 report.
Run: python generate_diagrams.py
Output: All images saved to output/plots/
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import graphviz

from config import DATA_DIR, OUTPUT_DIR, PLOTS_DIR
from data_loader import load_all_data, pair_on_off
from ml_model import (
    prepare_dataset, prepare_pair_dataset, build_models,
    evaluate_models, train_final_models, feature_importance,
)

os.makedirs(PLOTS_DIR, exist_ok=True)


# ============================================================
# 1. ML WORKFLOW DIAGRAM (Graphviz)
# ============================================================
def generate_ml_workflow():
    """Pipeline: COMSOL -> Data Extraction -> Dataset -> ML Models -> Predictions -> Dashboard"""
    dot = graphviz.Digraph('ML_Pipeline', format='png')
    dot.attr(rankdir='LR', bgcolor='white', dpi='200')
    dot.attr('node', shape='box', style='filled,rounded', fontname='Arial', fontsize='11')
    dot.attr('edge', fontname='Arial', fontsize='9', color='#333333')

    # Nodes with colors
    dot.node('comsol', 'COMSOL\nMultiphysics\nSimulation', fillcolor='#E3F2FD', color='#1565C0')
    dot.node('extract', 'Data Extraction\n& Parsing\n(data_loader.py)', fillcolor='#E8F5E9', color='#2E7D32')
    dot.node('dataset', 'Dataset\n19 simulations\n7 ON/OFF pairs', fillcolor='#FFF3E0', color='#E65100')
    dot.node('preprocess', 'Feature Scaling\n& LOO CV Split', fillcolor='#F3E5F5', color='#6A1B9A')
    dot.node('models', 'ML Models\nRF | GB | GP', fillcolor='#FCE4EC', color='#C62828')
    dot.node('eval', 'Evaluation\nR² | RMSE | MAE', fillcolor='#E0F7FA', color='#00695C')
    dot.node('predict', 'Predictions\n& Uncertainty', fillcolor='#FFF9C4', color='#F57F17')
    dot.node('dashboard', 'Streamlit\nDashboard', fillcolor='#E8EAF6', color='#283593')

    # Edges
    dot.edge('comsol', 'extract', label=' .txt files\n (S12, S22)')
    dot.edge('extract', 'dataset', label=' Parse filenames\n Extract params')
    dot.edge('dataset', 'preprocess', label=' X: 6 features\n Y: dip, freq')
    dot.edge('preprocess', 'models', label=' Train/Test\n (LOO CV)')
    dot.edge('models', 'eval', label=' Cross-validated\n predictions')
    dot.edge('eval', 'predict', label=' Best model\n selection')
    dot.edge('predict', 'dashboard', label=' Interactive\n exploration')

    output_path = os.path.join(PLOTS_DIR, 'ml_workflow_diagram')
    dot.render(output_path, cleanup=True)
    print(f"  Saved: {output_path}.png")


# ============================================================
# 2. RANDOM FOREST CONCEPT DIAGRAM (Graphviz)
# ============================================================
def generate_rf_diagram():
    """Shows how Random Forest works: multiple trees -> average prediction."""
    dot = graphviz.Digraph('Random_Forest', format='png')
    dot.attr(rankdir='TB', bgcolor='white', dpi='200')
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', color='#555555')

    # Input
    dot.node('input', 'Input Parameters\n(dx, g_w, c_w, h_graph, w_graph, w_au)',
             shape='box', style='filled,rounded', fillcolor='#E3F2FD', color='#1565C0')

    # Bootstrap samples
    dot.node('b1', 'Bootstrap\nSample 1', shape='box', style='filled', fillcolor='#FFF3E0', color='#E65100')
    dot.node('b2', 'Bootstrap\nSample 2', shape='box', style='filled', fillcolor='#FFF3E0', color='#E65100')
    dot.node('b3', 'Bootstrap\nSample ...', shape='box', style='filled', fillcolor='#FFF3E0', color='#E65100')
    dot.node('b4', 'Bootstrap\nSample 100', shape='box', style='filled', fillcolor='#FFF3E0', color='#E65100')

    # Trees
    dot.node('t1', 'Decision\nTree 1', shape='triangle', style='filled', fillcolor='#E8F5E9', color='#2E7D32')
    dot.node('t2', 'Decision\nTree 2', shape='triangle', style='filled', fillcolor='#E8F5E9', color='#2E7D32')
    dot.node('t3', 'Decision\nTree ...', shape='triangle', style='filled', fillcolor='#E8F5E9', color='#2E7D32')
    dot.node('t4', 'Decision\nTree 100', shape='triangle', style='filled', fillcolor='#E8F5E9', color='#2E7D32')

    # Predictions
    dot.node('p1', 'Pred 1', shape='box', style='filled,rounded', fillcolor='#FCE4EC', color='#C62828')
    dot.node('p2', 'Pred 2', shape='box', style='filled,rounded', fillcolor='#FCE4EC', color='#C62828')
    dot.node('p3', 'Pred ...', shape='box', style='filled,rounded', fillcolor='#FCE4EC', color='#C62828')
    dot.node('p4', 'Pred 100', shape='box', style='filled,rounded', fillcolor='#FCE4EC', color='#C62828')

    # Average
    dot.node('avg', 'Average All Predictions\nŷ = (1/T) Σ tree_t(x)',
             shape='box', style='filled,rounded,bold', fillcolor='#E8EAF6', color='#283593')

    # Edges
    for b in ['b1', 'b2', 'b3', 'b4']:
        dot.edge('input', b)
    for b, t in [('b1','t1'), ('b2','t2'), ('b3','t3'), ('b4','t4')]:
        dot.edge(b, t)
    for t, p in [('t1','p1'), ('t2','p2'), ('t3','p3'), ('t4','p4')]:
        dot.edge(t, p)
    for p in ['p1', 'p2', 'p3', 'p4']:
        dot.edge(p, 'avg')

    output_path = os.path.join(PLOTS_DIR, 'rf_concept_diagram')
    dot.render(output_path, cleanup=True)
    print(f"  Saved: {output_path}.png")


# ============================================================
# 3. MODEL COMPARISON CONCEPT DIAGRAM (Graphviz)
# ============================================================
def generate_model_comparison_diagram():
    """Side-by-side comparison of RF vs GB vs GP approaches."""
    dot = graphviz.Digraph('Model_Comparison', format='png')
    dot.attr(rankdir='TB', bgcolor='white', dpi='200')
    dot.attr('node', fontname='Arial', fontsize='10')

    # Title
    dot.node('title', 'Three ML Approaches Compared', shape='plaintext',
             fontsize='14', fontname='Arial Bold')

    # RF cluster
    with dot.subgraph(name='cluster_rf') as c:
        c.attr(label='Random Forest', style='filled', color='#1565C0',
               fillcolor='#E3F2FD', fontname='Arial Bold', fontsize='12')
        c.node('rf1', 'Parallel trees\n(bagging)', shape='box', style='filled,rounded', fillcolor='white')
        c.node('rf2', 'Each tree sees\nrandom subset', shape='box', style='filled,rounded', fillcolor='white')
        c.node('rf3', 'Average predictions\nfor final output', shape='box', style='filled,rounded', fillcolor='white')
        c.node('rf4', 'Strength: Robust\n& feature importance', shape='box', style='filled,rounded',
               fillcolor='#C8E6C9', color='#2E7D32')
        c.edge('rf1', 'rf2')
        c.edge('rf2', 'rf3')
        c.edge('rf3', 'rf4')

    # GB cluster
    with dot.subgraph(name='cluster_gb') as c:
        c.attr(label='Gradient Boosting', style='filled', color='#C62828',
               fillcolor='#FCE4EC', fontname='Arial Bold', fontsize='12')
        c.node('gb1', 'Sequential trees\n(boosting)', shape='box', style='filled,rounded', fillcolor='white')
        c.node('gb2', 'Each tree corrects\nprevious errors', shape='box', style='filled,rounded', fillcolor='white')
        c.node('gb3', 'Weighted sum\nfor final output', shape='box', style='filled,rounded', fillcolor='white')
        c.node('gb4', 'Strength: Highest\naccuracy on S12 dip', shape='box', style='filled,rounded',
               fillcolor='#C8E6C9', color='#2E7D32')
        c.edge('gb1', 'gb2')
        c.edge('gb2', 'gb3')
        c.edge('gb3', 'gb4')

    # GP cluster
    with dot.subgraph(name='cluster_gp') as c:
        c.attr(label='Gaussian Process', style='filled', color='#6A1B9A',
               fillcolor='#F3E5F5', fontname='Arial Bold', fontsize='12')
        c.node('gp1', 'Bayesian approach\n(probabilistic)', shape='box', style='filled,rounded', fillcolor='white')
        c.node('gp2', 'Models data as\nGaussian distribution', shape='box', style='filled,rounded', fillcolor='white')
        c.node('gp3', 'Prediction +\nuncertainty (±σ)', shape='box', style='filled,rounded', fillcolor='white')
        c.node('gp4', 'Strength: Best for\nfrequency prediction', shape='box', style='filled,rounded',
               fillcolor='#C8E6C9', color='#2E7D32')
        c.edge('gp1', 'gp2')
        c.edge('gp2', 'gp3')
        c.edge('gp3', 'gp4')

    # Invisible edges for alignment
    dot.edge('title', 'rf1', style='invis')
    dot.edge('title', 'gb1', style='invis')
    dot.edge('title', 'gp1', style='invis')

    output_path = os.path.join(PLOTS_DIR, 'model_comparison_diagram')
    dot.render(output_path, cleanup=True)
    print(f"  Saved: {output_path}.png")


# ============================================================
# 4. R² COMPARISON BAR CHART (Matplotlib)
# ============================================================
def generate_r2_comparison():
    """Bar chart: R² for RF vs GB vs GP across all targets."""
    df = load_all_data(DATA_DIR)
    pairs = pair_on_off(df)

    X_all, Y_all, feat_cols, target_cols = prepare_dataset(df)
    models = build_models()
    results_all, _ = evaluate_models(X_all, Y_all, models)

    model_names = list(results_all.keys())
    targets = list(results_all[model_names[0]].keys())

    fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 5))
    if len(targets) == 1:
        axes = [axes]

    colors = ['#1565C0', '#C62828', '#6A1B9A']

    for idx, target in enumerate(targets):
        r2_values = [results_all[m][target]['R2'] for m in model_names]
        bars = axes[idx].bar(model_names, r2_values, color=colors, edgecolor='black', linewidth=0.5)

        # Add value labels on bars
        for bar, val in zip(bars, r2_values):
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                          f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

        axes[idx].set_title(target, fontsize=13, fontweight='bold')
        axes[idx].set_ylabel('R² Score', fontsize=11)
        axes[idx].set_ylim(0, 1.0)
        axes[idx].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='R²=0.5 baseline')
        axes[idx].legend(fontsize=9)
        axes[idx].grid(axis='y', alpha=0.3)

    plt.suptitle('Model Performance Comparison (R² Score, LOO CV)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, 'ml_r2_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# 5. ACTUAL VS PREDICTED SCATTER (Matplotlib)
# ============================================================
def generate_actual_vs_predicted():
    """Scatter: actual vs predicted for all 3 models, both targets."""
    df = load_all_data(DATA_DIR)
    X_all, Y_all, feat_cols, target_cols = prepare_dataset(df)
    models = build_models()
    results_all, _ = evaluate_models(X_all, Y_all, models)

    model_names = list(results_all.keys())
    targets = list(results_all[model_names[0]].keys())
    colors = {'Random Forest': '#1565C0', 'Gradient Boosting': '#C62828', 'Gaussian Process': '#6A1B9A'}
    markers = {'Random Forest': 'o', 'Gradient Boosting': 's', 'Gaussian Process': '^'}

    fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 5.5))
    if len(targets) == 1:
        axes = [axes]

    for idx, target in enumerate(targets):
        ax = axes[idx]
        all_vals = []

        for model_name in model_names:
            m = results_all[model_name][target]
            y_true = m['y_true']
            y_pred = m['y_pred']
            all_vals.extend(y_true)
            all_vals.extend(y_pred)

            ax.scatter(y_true, y_pred, c=colors[model_name], marker=markers[model_name],
                      s=60, alpha=0.7, label=f"{model_name} (R²={m['R2']:.3f})", edgecolors='black', linewidth=0.3)

        # Perfect prediction line
        vmin, vmax = min(all_vals), max(all_vals)
        margin = (vmax - vmin) * 0.05
        ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin],
                'k--', alpha=0.4, label='Perfect prediction')

        ax.set_xlabel('Actual', fontsize=11)
        ax.set_ylabel('Predicted', fontsize=11)
        ax.set_title(target, fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    plt.suptitle('Actual vs Predicted (Leave-One-Out Cross-Validation)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, 'ml_actual_vs_predicted.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# 6. FEATURE IMPORTANCE BAR CHART (Matplotlib)
# ============================================================
def generate_feature_importance():
    """Horizontal bar chart of RF feature importance."""
    df = load_all_data(DATA_DIR)
    X_all, Y_all, feat_cols, target_cols = prepare_dataset(df)
    imp = feature_importance(X_all, Y_all, feat_cols)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    x = np.arange(len(imp.index))
    width = 0.35
    colors = ['#1565C0', '#C62828']

    for i, col in enumerate(imp.columns):
        bars = ax.barh(x + i * width, imp[col], width, label=col, color=colors[i],
                       edgecolor='black', linewidth=0.3)
        # Add value labels
        for bar, val in zip(bars, imp[col]):
            if val > 0.02:
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{val:.1%}', va='center', fontsize=9)

    ax.set_yticks(x + width / 2)
    ax.set_yticklabels(imp.index, fontsize=11)
    ax.set_xlabel('Importance', fontsize=11)
    ax.set_title('Feature Importance (Random Forest)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'ml_feature_importance.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# 7. DEVICE GEOMETRY SCHEMATIC (Matplotlib)
# ============================================================
def generate_geometry_schematic():
    """Annotated schematic of the metamaterial unit cell with labeled parameters."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-55, 55)
    ax.set_ylim(-55, 55)
    ax.set_aspect('equal')
    ax.set_facecolor('#F5F5F5')

    # Unit cell boundary (dx x dx)
    unit_cell = plt.Rectangle((-35, -35), 70, 70, fill=False,
                               edgecolor='#333333', linewidth=2, linestyle='--')
    ax.add_patch(unit_cell)

    # Substrate (light gray fill)
    substrate = plt.Rectangle((-35, -35), 70, 70, fill=True,
                               facecolor='#E8E8E8', edgecolor='none', alpha=0.5)
    ax.add_patch(substrate)

    # Gold cross structure (w_au wide lines)
    w_au = 4
    # Vertical gold line
    ax.add_patch(plt.Rectangle((-w_au/2, -35), w_au, 70,
                                facecolor='#FFD700', edgecolor='#B8860B', linewidth=1))
    # Horizontal gold line
    ax.add_patch(plt.Rectangle((-35, -w_au/2), 70, w_au,
                                facecolor='#FFD700', edgecolor='#B8860B', linewidth=1))

    # Capacitor plates (c_w tall, on either side of center gap)
    c_w = 20  # visual representation
    g_w = 3
    plate_width = 6
    # Left plate
    ax.add_patch(plt.Rectangle((-g_w/2 - plate_width, -c_w/2), plate_width, c_w,
                                facecolor='#FFD700', edgecolor='#B8860B', linewidth=1.5))
    # Right plate
    ax.add_patch(plt.Rectangle((g_w/2, -c_w/2), plate_width, c_w,
                                facecolor='#FFD700', edgecolor='#B8860B', linewidth=1.5))

    # Graphene layer (blue, semi-transparent, over capacitor area)
    h_graph = 6
    w_graph = 1
    graphene_h = c_w + 2 * h_graph
    graphene_w = g_w + 2 * plate_width + 2 * w_graph
    ax.add_patch(plt.Rectangle((-graphene_w/2, -graphene_h/2), graphene_w, graphene_h,
                                facecolor='#2196F3', edgecolor='#0D47A1', linewidth=1,
                                alpha=0.3))

    # ---- DIMENSION ANNOTATIONS ----

    # dx (unit cell size) - top
    ax.annotate('', xy=(35, 40), xytext=(-35, 40),
                arrowprops=dict(arrowstyle='<->', color='#C62828', lw=2))
    ax.text(0, 43, 'dx', ha='center', fontsize=14, fontweight='bold', color='#C62828')

    # g_w (capacitor gap) - center
    ax.annotate('', xy=(g_w/2, -c_w/2 - 5), xytext=(-g_w/2, -c_w/2 - 5),
                arrowprops=dict(arrowstyle='<->', color='#2E7D32', lw=2))
    ax.text(0, -c_w/2 - 8, 'g_w', ha='center', fontsize=12, fontweight='bold', color='#2E7D32')

    # c_w (capacitor width/height) - right side
    ax.annotate('', xy=(g_w/2 + plate_width + 4, c_w/2), xytext=(g_w/2 + plate_width + 4, -c_w/2),
                arrowprops=dict(arrowstyle='<->', color='#E65100', lw=2))
    ax.text(g_w/2 + plate_width + 7, 0, 'c_w', ha='left', fontsize=12, fontweight='bold',
            color='#E65100', rotation=90, va='center')

    # h_graph (graphene extra height) - left side
    ax.annotate('', xy=(-graphene_w/2 - 4, c_w/2), xytext=(-graphene_w/2 - 4, graphene_h/2),
                arrowprops=dict(arrowstyle='<->', color='#0D47A1', lw=2))
    ax.text(-graphene_w/2 - 7, c_w/2 + h_graph/2, 'h_graph', ha='right', fontsize=11,
            fontweight='bold', color='#0D47A1', rotation=90, va='center')

    # w_au label
    ax.annotate('w_au', xy=(w_au/2 + 1, 25), fontsize=10, fontweight='bold', color='#B8860B',
                arrowprops=dict(arrowstyle='->', color='#B8860B'),
                xytext=(15, 30))

    # Legend
    gold_patch = mpatches.Patch(facecolor='#FFD700', edgecolor='#B8860B', label='Gold (Au)')
    graphene_patch = mpatches.Patch(facecolor='#2196F3', alpha=0.3, edgecolor='#0D47A1', label='Graphene')
    substrate_patch = mpatches.Patch(facecolor='#E8E8E8', edgecolor='#333333', label='Substrate')
    ax.legend(handles=[gold_patch, graphene_patch, substrate_patch], loc='lower right', fontsize=10)

    ax.set_title('Unit Cell Geometry — Graphene Metamaterial THz Modulator',
                 fontsize=13, fontweight='bold', pad=20)
    ax.set_xlabel('Position (μm)', fontsize=11)
    ax.set_ylabel('Position (μm)', fontsize=11)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'device_geometry_schematic.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# 8. LOO CROSS-VALIDATION DIAGRAM (Graphviz)
# ============================================================
def generate_loo_diagram():
    """Visualise Leave-One-Out cross-validation process."""
    dot = graphviz.Digraph('LOO_CV', format='png')
    dot.attr(rankdir='TB', bgcolor='white', dpi='200')
    dot.attr('node', fontname='Arial', fontsize='10', shape='box', style='filled,rounded')

    dot.node('dataset', 'Full Dataset\n(n = 19 simulations)', fillcolor='#E3F2FD', color='#1565C0')

    # Folds
    dot.node('fold1', 'Fold 1: Train on 18\nTest on sample 1', fillcolor='#FFF3E0', color='#E65100')
    dot.node('fold2', 'Fold 2: Train on 18\nTest on sample 2', fillcolor='#FFF3E0', color='#E65100')
    dot.node('fold3', '...', fillcolor='#FFF3E0', color='#E65100', shape='plaintext')
    dot.node('fold19', 'Fold 19: Train on 18\nTest on sample 19', fillcolor='#FFF3E0', color='#E65100')

    # Results
    dot.node('pred1', 'Prediction 1', fillcolor='#E8F5E9', color='#2E7D32')
    dot.node('pred2', 'Prediction 2', fillcolor='#E8F5E9', color='#2E7D32')
    dot.node('pred3', '...', fillcolor='#E8F5E9', color='#2E7D32', shape='plaintext')
    dot.node('pred19', 'Prediction 19', fillcolor='#E8F5E9', color='#2E7D32')

    # Final metrics
    dot.node('metrics', 'Final Metrics\nR² = 1 - (SS_res / SS_tot)\nRMSE = √(Σ(y - ŷ)² / n)\nMAE = Σ|y - ŷ| / n',
             fillcolor='#FCE4EC', color='#C62828')

    # Edges
    for f in ['fold1', 'fold2', 'fold3', 'fold19']:
        dot.edge('dataset', f)
    for f, p in [('fold1','pred1'), ('fold2','pred2'), ('fold3','pred3'), ('fold19','pred19')]:
        dot.edge(f, p)
    for p in ['pred1', 'pred2', 'pred3', 'pred19']:
        dot.edge(p, 'metrics')

    output_path = os.path.join(PLOTS_DIR, 'loo_cv_diagram')
    dot.render(output_path, cleanup=True)
    print(f"  Saved: {output_path}.png")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    print("Generating diagrams for Phase 2 report...\n")

    print("[1/8] ML Workflow Diagram...")
    generate_ml_workflow()

    print("[2/8] Random Forest Concept Diagram...")
    generate_rf_diagram()

    print("[3/8] Model Comparison Diagram...")
    generate_model_comparison_diagram()

    print("[4/8] R² Comparison Bar Chart...")
    generate_r2_comparison()

    print("[5/8] Actual vs Predicted Scatter...")
    generate_actual_vs_predicted()

    print("[6/8] Feature Importance Chart...")
    generate_feature_importance()

    print("[7/8] Device Geometry Schematic...")
    generate_geometry_schematic()

    print("[8/8] LOO Cross-Validation Diagram...")
    generate_loo_diagram()

    print(f"\nAll diagrams saved to: {PLOTS_DIR}")
