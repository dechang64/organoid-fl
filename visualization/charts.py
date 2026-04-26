# ── visualization/charts.py ──
"""
Visualization Module
====================
Plotly-based interactive charts for Organoid-FL platform.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional

from utils.constants import COLORS, CLASS_INFO


def fl_convergence(history: list[dict], title: str = "Federated Learning Convergence") -> go.Figure:
    """Multi-panel FL training convergence chart.

    Shows: validation accuracy, validation loss, per-client accuracy, training time.
    """
    if not history:
        return go.Figure().update_layout(
            title=dict(text="No training data", font=dict(size=14)),
            template="plotly_white", height=400,
        )

    rounds = [h["round"] for h in history]
    val_acc = [h["val_acc"] * 100 for h in history]
    val_loss = [h["val_loss"] for h in history]
    elapsed = [h["elapsed"] for h in history]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Validation Accuracy (%)", "Validation Loss", "Per-Client Accuracy (%)", "Round Time (s)"),
        vertical_spacing=0.12,
        horizontal_spacing=0.12,
    )

    # Val accuracy
    fig.add_trace(go.Scatter(
        x=rounds, y=val_acc, mode="lines+markers+text",
        text=[f"{v:.1f}%" for v in val_acc], textposition="top center",
        line=dict(color=COLORS["primary"], width=2),
        marker=dict(size=6),
        name="Val Accuracy",
    ), row=1, col=1)

    # Val loss
    fig.add_trace(go.Scatter(
        x=rounds, y=val_loss, mode="lines+markers",
        line=dict(color=COLORS["secondary"], width=2),
        marker=dict(size=6),
        name="Val Loss",
    ), row=1, col=2)

    # Per-client accuracy
    if history and "client_metrics" in history[0]:
        n_clients = len(history[0]["client_metrics"])
        client_colors = [COLORS.get(cls, COLORS["primary"]) for cls in list(COLORS.keys())[:n_clients]]
        for cid in range(n_clients):
            client_accs = [h["client_metrics"][cid]["train_acc"] * 100 for h in history]
            fig.add_trace(go.Scatter(
                x=rounds, y=client_accs, mode="lines+markers",
                line=dict(color=client_colors[cid], width=1.5),
                marker=dict(size=4),
                name=f"Client {cid + 1}",
            ), row=2, col=1)

    # Elapsed time
    fig.add_trace(go.Bar(
        x=rounds, y=elapsed,
        marker_color=COLORS["accent"],
        name="Time (s)",
    ), row=2, col=2)

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        template="plotly_white",
        height=600,
        showlegend=True,
    )
    return fig


def data_distribution(df: pd.DataFrame, title: str = "Data Distribution Across Clients") -> go.Figure:
    """Stacked bar chart of class distribution per client."""
    if df.empty:
        return go.Figure().update_layout(
            title=dict(text="No data", font=dict(size=14)),
            template="plotly_white", height=300,
        )

    fig = go.Figure()
    classes = df["Class"].unique()
    for cls in classes:
        cls_df = df[df["Class"] == cls]
        color = CLASS_INFO.get(cls, {}).get("color", COLORS["primary"])
        fig.add_trace(go.Bar(
            x=cls_df["Client"],
            y=cls_df["Count"],
            name=CLASS_INFO.get(cls, {}).get("label", cls),
            marker_color=color,
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        barmode="stack",
        template="plotly_white",
        height=400,
        xaxis_title="Client",
        yaxis_title="Sample Count",
    )
    return fig


def confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    title: str = "Confusion Matrix",
) -> go.Figure:
    """Interactive heatmap confusion matrix."""
    from sklearn.metrics import confusion_matrix as sk_cm

    cm = sk_cm(y_true, y_pred)
    n = len(class_names)

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[CLASS_INFO.get(c, {}).get("label", c) for c in class_names],
        y=[CLASS_INFO.get(c, {}).get("label", c) for c in class_names],
        text=cm,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=True,
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        template="plotly_white",
        height=450,
        xaxis_title="Predicted",
        yaxis_title="Actual",
    )
    return fig


def tsne_visualization(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    title: str = "Feature Space (t-SNE)",
) -> go.Figure:
    """t-SNE 2D scatter plot of feature vectors."""
    from sklearn.manifold import TSNE

    # Subsample if too many points
    max_points = 500
    if len(features) > max_points:
        idx = np.random.choice(len(features), max_points, replace=False)
        features = features[idx]
        labels = labels[idx]

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    coords = tsne.fit_transform(features)

    fig = go.Figure()
    for cls_id, cls_name in enumerate(class_names):
        mask = labels == cls_id
        color = CLASS_INFO.get(cls_name, {}).get("color", COLORS["primary"])
        label = CLASS_INFO.get(cls_name, {}).get("label", cls_name)
        fig.add_trace(go.Scatter(
            x=coords[mask, 0],
            y=coords[mask, 1],
            mode="markers",
            name=label,
            marker=dict(color=color, size=5, opacity=0.7),
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        template="plotly_white",
        height=500,
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
    )
    return fig


def knn_results(
    query_id: str,
    results: list[tuple[str, float]],
    class_names: Optional[list[str]] = None,
    title: str = "k-Nearest Neighbors",
) -> go.Figure:
    """Bar chart of kNN similarity scores."""
    if not results:
        return go.Figure().update_layout(
            title=dict(text="No results", font=dict(size=14)),
            template="plotly_white", height=300,
        )

    ids = [r[0] for r in results]
    scores = [r[1] for r in results]

    colors = []
    for score in scores:
        if score > 0.9:
            colors.append(COLORS["healthy"])
        elif score > 0.7:
            colors.append(COLORS["early_stage"])
        else:
            colors.append(COLORS["late_stage"])

    fig = go.Figure(go.Bar(
        x=ids,
        y=scores,
        marker_color=colors,
        text=[f"{s:.3f}" for s in scores],
        textposition="auto",
    ))

    fig.update_layout(
        title=dict(text=f"{title} (query: {query_id})", font=dict(size=16)),
        template="plotly_white",
        height=400,
        yaxis_title="Cosine Similarity",
        yaxis_range=[0, 1.05],
    )
    return fig


def audit_timeline(chain_df: pd.DataFrame, title: str = "Audit Chain Timeline") -> go.Figure:
    """Timeline visualization of audit chain blocks."""
    if chain_df.empty:
        return go.Figure().update_layout(
            title=dict(text="No audit data", font=dict(size=14)),
            template="plotly_white", height=300,
        )

    # Color by operation type
    op_colors = {
        "genesis": "#6b7280",
        "fl_round": COLORS["primary"],
        "fl_aggregate": COLORS["secondary"],
        "vector_insert": COLORS["accent"],
        "vector_search": COLORS["healthy"],
        "vector_delete": COLORS["late_stage"],
    }

    colors = [op_colors.get(op, "#6b7280") for op in chain_df["Operation"]]

    fig = go.Figure(go.Scatter(
        x=chain_df["Block"],
        y=[1] * len(chain_df),
        mode="markers+text",
        marker=dict(color=colors, size=12),
        text=chain_df["Operation"],
        textposition="top center",
        hovertext=chain_df["Details"],
        hoverinfo="text",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        template="plotly_white",
        height=300,
        showlegend=False,
        xaxis_title="Block Index",
        yaxis_visible=False,
    )
    return fig


def model_comparison(
    results: dict[str, dict],
    title: str = "Algorithm Comparison",
) -> go.Figure:
    """Compare FL algorithms (FedAvg vs FedProx etc.)."""
    if not results:
        return go.Figure().update_layout(
            title=dict(text="No comparison data", font=dict(size=14)),
            template="plotly_white", height=300,
        )

    algos = list(results.keys())
    final_accs = [results[a].get("final_accuracy", 0) * 100 for a in algos]
    total_times = [results[a].get("total_time", 0) for a in algos]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Final Accuracy (%)", "Total Training Time (s)"),
    )

    fig.add_trace(go.Bar(
        x=algos, y=final_accs,
        marker_color=COLORS["primary"],
        text=[f"{v:.1f}%" for v in final_accs],
        textposition="auto",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=algos, y=total_times,
        marker_color=COLORS["accent"],
        text=[f"{v:.1f}s" for v in total_times],
        textposition="auto",
    ), row=1, col=2)

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        template="plotly_white",
        height=400,
    )
    return fig
