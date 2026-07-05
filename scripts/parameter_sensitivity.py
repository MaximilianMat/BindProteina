#!/usr/bin/env python3
"""
Parameter Sensitivity Analysis for Seed Search

Analyzes the sensitivity of mean_success_rate to parameters like distance,
rotation, fragment, and orientation. Outputs plots and statistics.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def calculate_relative_gradients(df, param, other_params, rate_col='mean_success_rate', is_periodic=False, period=None):
    """
    Calculate the local gradient |Δ success_rate| / |Δ param| for a continuous parameter.
    Returns relative changes (divided by the global mean success rate).
    """
    gradients = []
    global_mean = df[rate_col].mean()
    eps = 1e-6 # To avoid division by zero if global mean is 0
    
    for name, group in df.groupby(other_params):
        group = group.sort_values(by=param)
        vals = group[param].values
        rates = group[rate_col].values
        
        if len(vals) < 2:
            continue
            
        for i in range(len(vals) - 1):
            d_rate = abs(rates[i+1] - rates[i])
            # We just want the relative change between adjacent steps
            # "Relative to global mean" gives a stable percentage.
            rel_change = d_rate / (global_mean + eps)
            gradients.append({
                **dict(zip(other_params, name)),
                param: vals[i],
                'rel_change': rel_change
            })
            
        if is_periodic and period is not None:
            # Wrap around from last to first
            d_rate = abs(rates[0] - rates[-1])
            rel_change = d_rate / (global_mean + eps)
            gradients.append({
                **dict(zip(other_params, name)),
                param: vals[-1],
                'rel_change': rel_change
            })
            
    return pd.DataFrame(gradients)

def coarsening_cost(df, param, fine_values, coarse_values, other_params, rate_col='mean_success_rate'):
    """
    Calculate the loss in max success rate if we only search the coarse_values.
    """
    losses = []
    for name, group in df.groupby(other_params):
        fine_max = group[rate_col].max()
        
        # Filter to coarse grid
        coarse_group = group[group[param].isin(coarse_values)]
        if len(coarse_group) == 0:
            continue
        coarse_max = coarse_group[rate_col].max()
        
        loss = fine_max - coarse_max
        # Relative loss
        global_mean = df[rate_col].mean()
        rel_loss = loss / (global_mean + 1e-6)
        
        losses.append({
            **dict(zip(other_params, name)),
            'fine_max': fine_max,
            'coarse_max': coarse_max,
            'absolute_loss': loss,
            'relative_loss': rel_loss
        })
    return pd.DataFrame(losses)

def main():
    parser = argparse.ArgumentParser(description="Parameter Sensitivity Analysis")
    parser.add_argument("--input", required=True, type=Path, help="Path to summary_stats.csv")
    parser.add_argument("--output", default=Path("sensitivity_analysis_out"), type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    
    # Ensure types
    rot_col = 'angle' if 'angle' in df.columns else 'rotation'
    rate_col = 'success_rate' if 'success_rate' in df.columns else 'mean_success_rate'
    
    df['distance'] = pd.to_numeric(df['distance'])
    df[rot_col] = pd.to_numeric(df[rot_col])
    
    global_mean = df[rate_col].mean()
    print(f"Global Mean Success Rate: {global_mean:.4f}")
    
    # ---------------------------------------------------------
    # 1. Main Effects (Global Norm)
    # ---------------------------------------------------------
    sns.set_theme(style="whitegrid")
    params = ['orientation', 'distance', 'fragment', rot_col]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, p in enumerate(params):
        sns.pointplot(data=df, x=p, y=rate_col, ax=axes[i], capsize=.1, errorbar='se')
        axes[i].set_title(f"Main Effect: {p}")
        axes[i].set_ylabel(rate_col.replace('_', ' ').title())
        # Rotate x labels if it's fragment
        if p == 'fragment':
            axes[i].tick_params(axis='x', rotation=45)
            
    plt.tight_layout()
    plt.savefig(args.output / "main_effects.png")
    plt.close()

    # ---------------------------------------------------------
    # 2. Local Sensitivity (Relative Gradients)
    # ---------------------------------------------------------
    # Distance
    grad_dist = calculate_relative_gradients(
        df, param='distance', 
        other_params=['orientation', 'fragment', rot_col], 
        rate_col=rate_col,
        is_periodic=False
    )
    
    # Rotation (Periodic 0=360, distance=360-288=72)
    grad_rot = calculate_relative_gradients(
        df, param=rot_col, 
        other_params=['orientation', 'fragment', 'distance'], 
        rate_col=rate_col,
        is_periodic=True, period=360
    )
    
    print("\n--- Average Relative Step Sensitivity ---")
    print(f"Distance (step=2A): {grad_dist['rel_change'].mean():.2f}x global mean")
    print(f"{rot_col.capitalize()} (step=72d): {grad_rot['rel_change'].mean():.2f}x global mean")
    
    # Plot Local Sensitivities
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(data=grad_dist, y='rel_change', ax=axes[0])
    axes[0].set_title("Distance Sensitivity (Relative Change per Step)")
    axes[0].set_ylabel("Relative Change (Δ / Global Mean)")
    
    sns.boxplot(data=grad_rot, y='rel_change', ax=axes[1])
    axes[1].set_title(f"{rot_col.capitalize()} Sensitivity (Relative Change per Step)")
    axes[1].set_ylabel("Relative Change (Δ / Global Mean)")
    plt.tight_layout()
    plt.savefig(args.output / "local_sensitivity_boxplots.png")
    plt.close()

    # ---------------------------------------------------------
    # 3. Fragment-level Interactions
    # ---------------------------------------------------------
    # Is sensitivity different among fragments?
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.barplot(data=grad_dist, x='fragment', y='rel_change', ax=axes[0], errorbar='se')
    axes[0].set_title("Distance Sensitivity by Fragment")
    axes[0].tick_params(axis='x', rotation=45)
    
    sns.barplot(data=grad_rot, x='fragment', y='rel_change', ax=axes[1], errorbar='se')
    axes[1].set_title(f"{rot_col.capitalize()} Sensitivity by Fragment")
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(args.output / "fragment_interactions.png")
    plt.close()

    # ---------------------------------------------------------
    # 4. Coarsening Cost ("Granularity Needed")
    # ---------------------------------------------------------
    print("\n--- Cost of Coarsening (Granularity Analysis) ---")
    
    # Rotation: What if we only sample [0, 144, 288]? (step=144)
    rot_coarse = [0, 144, 288]
    cost_rot = coarsening_cost(df, rot_col, df[rot_col].unique(), rot_coarse, ['orientation', 'fragment', 'distance'], rate_col=rate_col)
    mean_rel_loss_rot = cost_rot['relative_loss'].mean()
    print(f"Reducing {rot_col} sampling to {rot_coarse} loses on average {mean_rel_loss_rot*100:.1f}% of success rate (relative to global mean).")
    
    # Distance: What if we only sample [10, 14]? (step=4)
    dist_coarse = [10, 14]
    cost_dist = coarsening_cost(df, 'distance', df['distance'].unique(), dist_coarse, ['orientation', 'fragment', rot_col], rate_col=rate_col)
    mean_rel_loss_dist = cost_dist['relative_loss'].mean()
    print(f"Reducing distance sampling to {dist_coarse} loses on average {mean_rel_loss_dist*100:.1f}% of success rate (relative to global mean).")
    
    # Save the coarsening data
    cost_rot.to_csv(args.output / f"coarsening_cost_{rot_col}.csv", index=False)
    cost_dist.to_csv(args.output / "coarsening_cost_distance.csv", index=False)

    print(f"\nAnalysis complete. Results and plots saved to {args.output}")

if __name__ == "__main__":
    main()
