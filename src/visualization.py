#!/usr/bin/env python3
"""
Complete Calibration Visualization Suite
Creates all 6 visualization types as per user requirements
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300

def create_all_visualizations(all_results, save_dir):
    """
    Create all 6 visualization types
    
    Args:
        all_results: Dict[scenario_name -> Dict[method -> metrics]]
        save_dir: Directory to save plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("  [1/6] Creating 3-panel heatmap...", flush=True)
    create_triple_heatmap(all_results, save_dir / "1_heatmaps_comparison.png")
    
    print("  [2/6] Creating reliability diagrams grid...", flush=True)
    create_reliability_grid(all_results, save_dir / "2_reliability_diagrams_all.png")
    
    print("  [3/6] Creating overall performance bar chart...", flush=True)
    create_overall_bar_chart(all_results, save_dir / "3_overall_performance.png")
    
    print("  [4/6] Creating improvement matrix...", flush=True)
    create_improvement_matrix(all_results, save_dir / "4_improvement_matrix.png")
    
    print("  [5/6] Creating method ranking...", flush=True)
    create_method_ranking(all_results, save_dir / "5_method_ranking.png")
    
    print("  [6/6] Creating Brier Score comparison...", flush=True)
    create_brier_comparison(all_results, save_dir / "6_brier_comparison.png")
    
    print(f"\n  ✅ All 6 visualizations saved to {save_dir}/", flush=True)

# ============================================================================
# VISUALIZATION 1: Three Side-by-Side He atmaps
# ============================================================================

def create_triple_heatmap(all_results, save_path):
    """3 heatmaps: Accuracy | F1 | ECE"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    scenarios = list(all_results.keys())
    methods = ['uncalibrated', 'platt', 'isotonic', 'temperature', 'beta']
    method_labels = ['Uncal', 'Platt', 'Isotonic', 'Temperature', 'Beta']
    
    # Prepare data
    accuracy_data = []
    f1_data = []
    ece_data = []
    
    for scenario in scenarios:
        acc_row, f1_row, ece_row = [], [], []
        for method in methods:
            if method in all_results[scenario]:
                acc_row.append(all_results[scenario][method]['accuracy'])
                f1_row.append(all_results[scenario][method]['f1'])
                ece_row.append(all_results[scenario][method]['ece'])
            else:
                acc_row.append(0)
                f1_row.append(0)
                ece_row.append(0)
        accuracy_data.append(acc_row)
        f1_data.append(f1_row)
        ece_data.append(ece_row)
    
    # Heatmap A: Accuracy
    df_acc = pd.DataFrame(accuracy_data, index=scenarios, columns=method_labels)
    sns.heatmap(df_acc, annot=True, fmt='.3f', cmap='Greens', ax=axes[0],
                cbar_kws={'label': 'Accuracy'}, vmin=0.90, vmax=1.0,
                linewidths=0.5, linecolor='white')
    axes[0].set_title('Accuracy (Higher is Better)', fontsize=14, fontweight='bold', pad=10)
    axes[0].set_xlabel('Calibration Method', fontsize=12)
    axes[0].set_ylabel('Scenario', fontsize=12)
    
    # Heatmap B: F1 Score
    df_f1 = pd.DataFrame(f1_data, index=scenarios, columns=method_labels)
    sns.heatmap(df_f1, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1],
                cbar_kws={'label': 'F1 Score'}, vmin=0.90, vmax=1.0,
                linewidths=0.5, linecolor='white')
    axes[1].set_title('F1 Score (Higher is Better)', fontsize=14, fontweight='bold', pad=10)
    axes[1].set_xlabel('Calibration Method', fontsize=12)
    axes[1].set_ylabel('')
    
    # Heatmap C: ECE
    df_ece = pd.DataFrame(ece_data, index=scenarios, columns=method_labels)
    sns.heatmap(df_ece, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=axes[2],
                cbar_kws={'label': 'ECE (lower is better)'},
                linewidths=0.5, linecolor='white')
    axes[2].set_title('ECE (Lower is Better)', fontsize=14, fontweight='bold', pad=10)
    axes[2].set_xlabel('Calibration Method', fontsize=12)
    axes[2].set_ylabel('')
    
    plt.suptitle('Calibration Methods Comparison - All Models', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# VISUALIZATION 2: Reliability Diagrams Grid (5 scenarios)
# ============================================================================

def create_reliability_grid(all_results, save_path):
    """5 reliability diagrams showing before/after calibration"""
    from calibration import get_reliability_diagram_data
    
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    scenarios = list(all_results.keys())
    
    for idx, scenario in enumerate(scenarios):
        if idx >= 5:  # Max 5 scenarios (leave one spot for legend/summary)
            break
        
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        scenario_data = all_results[scenario]
        
        # Find best calibration method (lowest ECE)
        best_method = 'temperature'  # default
        best_ece = 1.0
        for method in ['platt', 'isotonic', 'temperature', 'beta']:
            if method in scenario_data and scenario_data[method]['ece'] < best_ece:
                best_ece = scenario_data[method]['ece']
                best_method = method
        
        # Plot uncalibrated
        if 'uncalibrated' in scenario_data:
            probs = scenario_data['uncalibrated']['probs']
            labels = scenario_data['uncalibrated']['labels']
            bin_centers, bin_accs, _, _ = get_reliability_diagram_data(probs, labels, n_bins=10)
            ax.plot(bin_centers, bin_accs, 'o-', color='gray', alpha=0.6, 
                   label=f'Uncal (ECE={scenario_data["uncalibrated"]["ece"]:.3f})', linewidth=2, markersize=6)
        
        # Plot best calibrated
        if best_method in scenario_data:
            probs = scenario_data[best_method]['probs']
            labels = scenario_data[best_method]['labels']
            bin_centers, bin_accs, _, _ = get_reliability_diagram_data(probs, labels, n_bins=10)
            ax.plot(bin_centers, bin_accs, 'o-', color='green', alpha=0.8,
                   label=f'Best Calib (ECE={scenario_data[best_method]["ece"]:.3f})', linewidth=2, markersize=6)
        
        # Perfect calibration
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1.5, label='Perfect')
        
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('Mean Predicted Probability', fontsize=10)
        ax.set_ylabel('Fraction of Positives', fontsize=10)
        ax.set_title(f'{scenario}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Summary panel
    ax_summary = fig.add_subplot(gs[1, 2])
    ax_summary.axis('off')
    
    summary_text = "Calibration Summary:\n\n"
    for scenario in scenarios:
        uncal_ece = all_results[scenario]['uncalibrated']['ece']
        best_ece = min([all_results[scenario][m]['ece'] for m in ['platt', 'isotonic', 'temperature', 'beta'] 
                       if m in all_results[scenario]])
        improvement = (uncal_ece - best_ece) / uncal_ece * 100
        summary_text += f"{scenario}:\n  {improvement:.1f}% ECE reduction\n\n"
    
    ax_summary.text(0.1, 0.9, summary_text, transform=ax_summary.transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Reliability Diagrams - Before vs After Calibration', fontsize=16, fontweight='bold')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# VISUALIZATION 3: Overall Performance Bar Chart
# ============================================================================

def create_overall_bar_chart(all_results, save_path):
    """Bar chart comparing all scenarios with best calibration"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    scenarios = list(all_results.keys())
    x = np.arange(len(scenarios))
    width = 0.25
    
    # Get best calibrated metrics for each scenario
    accuracy_vals = []
    f1_vals = []
    ece_vals = []
    
    for scenario in scenarios:
        # Find best method (lowest ECE)
        best_method = min(
            [(m, all_results[scenario][m]['ece']) for m in ['platt', 'isotonic', 'temperature', 'beta']
             if m in all_results[scenario]],
            key=lambda x: x[1]
        )[0]
        
        accuracy_vals.append(all_results[scenario][best_method]['accuracy'])
        f1_vals.append(all_results[scenario][best_method]['f1'])
        ece_vals.append(all_results[scenario][best_method]['ece'])
    
    bars1 = ax.bar(x - width, accuracy_vals, width, label='Accuracy', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, f1_vals, width, label='F1 Score', color='#e74c3c', alpha=0.8)
    bars3 = ax.bar(x + width, ece_vals, width, label='ECE', color='#2ecc71', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Scenario', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Final Model Comparison (Calibrated)', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=0)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# VISUALIZATION 4: Improvement Matrix
# ============================================================================

def create_improvement_matrix(all_results, save_path):
    """Table showing ECE improvement for each scenario"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    table_data = []
    headers = ['Scenario', 'Before\n(Uncalibrated)', 'After\n(Best)', 'Δ ECE', '% Improvement', 'Best Method']
    
    for scenario in all_results.keys():
        uncal_ece = all_results[scenario]['uncalibrated']['ece']
        
        # Find best calibration
        best_method, best_ece = min(
            [(m, all_results[scenario][m]['ece']) for m in ['platt', 'isotonic', 'temperature', 'beta']
             if m in all_results[scenario]],
            key=lambda x: x[1]
        )
        
        delta = uncal_ece - best_ece
        improvement = (delta / uncal_ece * 100)
        
        table_data.append([
            scenario,
            f'{uncal_ece:.4f}',
            f'{best_ece:.4f}',
            f'↓ {delta:.4f}',
            f'{improvement:.1f}% ⬇️',
            best_method.capitalize()
        ])
    
    table = ax.table(cellText=table_data, colLabels=headers, loc='center',
                    cellLoc='center', colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style cells
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if j == 4:  # Improvement column
                table[(i, j)].set_facecolor('#2ecc71' if float(table_data[i-1][4].split('%')[0]) > 90 else '#f39c12')
            else:
                table[(i, j)].set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
    
    plt.title('Calibration Improvement Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# VISUALIZATION 5: Method Ranking
# ============================================================================

def create_method_ranking(all_results, save_path):
    """Horizontal bar chart showing average ECE per calibration method"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['platt', 'isotonic', 'temperature', 'beta']
    method_labels = ['Platt Scaling', 'Isotonic Regression', 'Temperature Scaling', 'Beta Calibration']
    
    # Calculate average ECE across all scenarios
    avg_ece = []
    for method in methods:
        eces = [all_results[s][method]['ece'] for s in all_results.keys() if method in all_results[s]]
        avg_ece.append(np.mean(eces) if eces else 0)
    
    # Sort by performance
    sorted_indices = np.argsort(avg_ece)
    sorted_labels = [method_labels[i] for i in sorted_indices]
    sorted_ece = [avg_ece[i] for i in sorted_indices]
    
    colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(sorted_ece))]
    bars = ax.barh(sorted_labels, sorted_ece, color=colors, alpha=0.8)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f'{width:.4f}', ha='left', va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Average ECE (across all scenarios)', fontsize=12, fontweight='bold')
    ax.set_title('Calibration Method Ranking (Lower is Better)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# VISUALIZATION 6: Brier Score Comparison
# ============================================================================

def create_brier_comparison(all_results, save_path):
    """Before/after Brier score comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    scenarios = list(all_results.keys())
    
    # Before calibration
    brier_before = [all_results[s]['uncalibrated']['brier'] for s in scenarios]
    
    # After calibration (best method)
    brier_after = []
    for s in scenarios:
        best_method = min(
            [(m, all_results[s][m]['ece']) for m in ['platt', 'isotonic', 'temperature', 'beta']
             if m in all_results[s]],
            key=lambda x: x[1]
        )[0]
        brier_after.append(all_results[s][best_method]['brier'])
    
    x = np.arange(len(scenarios))
    
    # Before
    bars1 = ax1.bar(x, brier_before, color='#e74c3c', alpha=0.7)
    ax1.set_title('Before Calibration', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Brier Score (↓ lower is better)', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # After
    bars2 = ax2.bar(x, brier_after, color='#2ecc71', alpha=0.7)
    ax2.set_title('After Calibration (Best Method)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Brier Score (↓ lower is better)', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Add improvement arrows
    for i, (before, after) in enumerate(zip(brier_before, brier_after)):
        improvement = (before - after) / before * 100
        ax2.text(i, after * 0.5, f'↓{improvement:.0f}%',
                ha='center', fontsize=10, fontweight='bold', color='darkgreen')
    
    plt.suptitle('Brier Score Comparison: Before vs After Calibration', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
