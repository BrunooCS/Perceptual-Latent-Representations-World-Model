import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

import plotly.graph_objects as go

def plot_latent_space(
    latent_space: np.ndarray, 
    save_path: str, 
    name: str = '', 
    save: bool = True
) -> None:
    """
    Scatter plot of the latent space.

    Args:
        latent_space (np.ndarray): The latent vectors to visualize.
        save_path (str): File path to save the plot.
        name (str, optional): Additional name or label for the plot. Defaults to ''.
        save (bool, optional): Whether to save or show the plot. Defaults to True.
    """
    sns.set_style("whitegrid")  # Adds a subtle grid for readability
    sns.set_context("paper", font_scale=1.5)  # Adjust for paper-ready appearance
    plt.figure(figsize=(10, 8))  # Increased size for print clarity

    plt.scatter(latent_space[:, 0], latent_space[:, 1], s=6, color='black', alpha=0.7)
    plt.title(f'Latent Space Visualization {name}', fontsize=18, pad=20)
    plt.xlabel('Dimension 1', fontsize=16)
    plt.ylabel('Dimension 2', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.gca().set_facecolor('white')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    if save:
        plt.tight_layout()
        plt.savefig(save_path, dpi=400)  # High DPI for publications
        plt.close()
    else:
        plt.show()

def plot_latent_space_density(
    latent_space: np.ndarray, 
    save_path: str, 
    name: str = '', 
    save: bool = True
) -> None:
    """
    Density plot of the latent space.

    Args:
        latent_space (np.ndarray): The latent vectors to visualize.
        save_path (str): File path to save the plot.
        name (str, optional): Additional name or label for the plot. Defaults to ''.
        save (bool, optional): Whether to save or show the plot. Defaults to True.
    """
    sns.set_style("whitegrid")  # Adds a subtle grid for readability
    sns.set_context("paper", font_scale=1.5)  # Adjust for paper-ready appearance
    plt.figure(figsize=(10, 8))  # Increased size for print clarity

    sns.kdeplot(
        x=latent_space[:, 0], 
        y=latent_space[:, 1], 
        fill=True, 
        cmap="viridis",  # Colorblind-friendly colormap
        thresh=0, 
        alpha=0.8
    )
    sns.kdeplot(
        x=latent_space[:, 0], 
        y=latent_space[:, 1], 
        color="black", 
        levels=15, 
        linewidths=0.5
    )

    plt.title(f'Latent Space Density Visualization {name}', fontsize=18, pad=20)
    plt.xlabel('Dimension 1', fontsize=16)
    plt.ylabel('Dimension 2', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.gca().set_facecolor('white')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    if save:
        plt.tight_layout()
        plt.savefig(save_path, dpi=400)  # High DPI for publications
        plt.close()
    else:
        plt.show()


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """
    Convert a hex color string to RGBA format.

    Args:
        hex_color (str): Hex color code (e.g., '#86D293').
        alpha (float, optional): Alpha value for the color. Defaults to 1.0.

    Returns:
        str: The RGBA string.
    """
    hex_color = hex_color.lstrip('#')
    return f"rgba({int(hex_color[0:2], 16)}, {int(hex_color[2:4], 16)}, {int(hex_color[4:6], 16)}, {alpha})"


def plot_cma_es_results(
    metrics: pd.DataFrame,
    colors: list = None,
    metric_columns: list = None,
) -> None:
    """
    Plot CMA-ES training results using Plotly.

    Args:
        metrics (pd.DataFrame): DataFrame containing 'generation' and reward metrics.
        colors (list, optional): List of colors for each metric. Defaults to predefined list.
        metric_columns (list, optional): Columns in 'metrics' to plot. Defaults to a typical set.
        path (str, optional): Directory to save the output plot. Defaults to 'plots'.
    """
    if colors is None:
        colors = ['#86D293', '#FFCF96', '#FF8080']
    if metric_columns is None:
        metric_columns = ['best_reward', 'mean_reward', 'worst_reward']

    fig = go.Figure()

    for metric, color in zip(metric_columns, colors):
        fig.add_trace(go.Scatter(
            x=metrics['generation'],
            y=metrics[metric],
            mode='lines',
            name=metric.replace('_', ' ').title(),
            line=dict(color=color, width=2.5),
            fill='tozeroy',
            fillcolor=hex_to_rgba(color, 0.2)
        ))

    fig.update_layout(
        height=600,
        width=900,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Helvetica, Arial, sans-serif", size=14, color="#333333"),
        title=dict(
            text='CMA-ES Training Results', 
            x=0.5, 
            y=0.95,
            xanchor='center', 
            yanchor='top',
            font=dict(size=20, color="#333333")
        ),
        legend=dict(
            title='', 
            title_font_size=16, 
            font_size=14,
            bgcolor='rgba(255,255,255,0)',
            bordercolor='rgba(0,0,0,0)',
            orientation='h', 
            yanchor='bottom', 
            y=1.02,
            xanchor='center', 
            x=0.5
        ),
        xaxis=dict(
            showgrid=True, 
            gridcolor='rgba(200,200,200,0.2)',
            linecolor='rgba(200,200,200,0.5)', 
            linewidth=1,
            mirror=True, 
            title='Generation'
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor='rgba(200,200,200,0.2)',
            linecolor='rgba(200,200,200,0.5)', 
            linewidth=1,
            mirror=True, 
            title='Reward'
        )
    )

    fig.update_traces(hovertemplate='%{y} in Generation %{x}<extra></extra>')
    
    fig.show()

