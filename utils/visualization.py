# -*- coding: utf-8 -*-
"""
Hyper-CAD-BEV v6.5-Sparse: Figure Generation
=============================================
Generates all figures from the manuscript:
  - Fig 4(a): Pareto frontier accuracy vs compute efficiency
  - Fig 4(b): Module ablation bar chart
  - Fig 4(c): Robustness under extreme conditions
  - Fig 4(d): Cross-platform cost-performance comparison
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os
from typing import Dict, List, Optional

plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans',
    'axes.titlesize': 12, 'axes.labelsize': 11})


class FigureGenerator:
    """Generate publication-quality figures for the manuscript."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.colors = {'ours': '#d62728', 'dense': '#1f77b4',
            'mono': '#ff7f0e', 'neuro': '#2ca02c', 'baseline': '#7f7f7f'}
    
    def fig4a_pareto(self, sota_params: Dict, save: bool = True):
        """Fig 4(a): Pareto frontier - accuracy vs compute efficiency."""
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        for name, p in sota_params.items():
            if 'v6.5' in name:
                ax.scatter(p['compute'], p['miou'], c=self.colors['ours'], s=180, zorder=5, edgecolors='black', linewidth=1.5)
                ax.annotate(name, (p['compute'], p['miou']), textcoords='offset points',
                    xytext=(10, 10), fontsize=9, fontweight='bold')
            elif 'Dense' in name or 'BEVFormer' in name or 'BEVDet' in name:
                ax.scatter(p['compute'], p['miou'], c=self.colors['dense'], s=80, zorder=3)
            elif 'Mono' in name or 'Single' in name or 'v5.2' in name:
                ax.scatter(p['compute'], p['miou'], c=self.colors['mono'], s=80, zorder=3)
            else:
                ax.scatter(p['compute'], p['miou'], c=self.colors['neuro'], s=80, zorder=3)
        ax.set_xlabel('Effective Compute (TOPS)')
        ax.set_ylabel('mIoU (%)')
        ax.set_title('(a) Accuracy-Efficiency Pareto Frontier')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0.05, color='red', linestyle='--', alpha=0.5, label='Edge threshold')
        ax.legend(loc='lower right', fontsize=8)
        fig.tight_layout()
        if save:
            fig.savefig(os.path.join(self.output_dir, 'fig4a_pareto.pdf'), dpi=300, bbox_inches='tight')
            fig.savefig(os.path.join(self.output_dir, 'fig4a_pareto.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        return fig
    
    def fig4b_ablation(self, ablation_results: Dict, save: bool = True):
        """Fig 4(b): Module ablation contributions."""
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        modules = list(ablation_results.keys())
        values = list(ablation_results.values())
        colors_bar = [self.colors['ours'], '#ff9999', '#ffcccc', '#ffcccc', '#ffcccc', '#ffcccc']
        bars = ax.barh(modules[::-1], values[::-1], color=colors_bar[::-1])
        ax.set_xlabel('mIoU (%)')
        ax.set_title('(b) Core Module Ablation Study')
        for bar, val in zip(bars, values[::-1]):
            ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}', va='center', fontsize=9)
        fig.tight_layout()
        if save:
            fig.savefig(os.path.join(self.output_dir, 'fig4b_ablation.pdf'), dpi=300, bbox_inches='tight')
            fig.savefig(os.path.join(self.output_dir, 'fig4b_ablation.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        return fig
    
    def fig4c_robustness(self, slope_data: Dict, weather_data: Dict, save: bool = True):
        """Fig 4(c): Robustness under extreme conditions."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        # Slope robustness
        slopes = list(slope_data.keys())
        for method, color in [('MonoBEV v2', 'blue'), ('v6.0-Neuro', 'green'), ('v6.5-Sparse', 'red')]:
            vals = [slope_data[s][method] for s in slopes]
            ax1.plot(range(len(slopes)), vals, 'o-', label=method, color=color, linewidth=2)
        ax1.set_xticks(range(len(slopes)))
        ax1.set_xticklabels([s.replace('deg','') for s in slopes])
        ax1.set_xlabel('Slope Angle')
        ax1.set_ylabel('mIoU (%)')
        ax1.set_title('(c) Terrain Slope Robustness')
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.3)
        # Weather robustness
        conditions = list(weather_data.keys())
        for method, color in [('MonoBEV v2', 'blue'), ('v6.0-Neuro', 'green'), ('v6.5-Sparse', 'red')]:
            vals = [weather_data[c][method] for c in conditions]
            ax2.plot(range(len(conditions)), vals, 'o-', label=method, color=color, linewidth=2)
        ax2.set_xticks(range(len(conditions)))
        ax2.set_xticklabels([c.replace('_',' ').title() for c in conditions], rotation=45, ha='right')
        ax2.set_title('(c) Weather Robustness')
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        if save:
            fig.savefig(os.path.join(self.output_dir, 'fig4c_robustness.pdf'), dpi=300, bbox_inches='tight')
            fig.savefig(os.path.join(self.output_dir, 'fig4c_robustness.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        return fig
    
    def fig4d_cost_performance(self, cross_platform: Dict, save: bool = True):
        """Fig 4(d): Cross-platform cost-performance."""
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        platforms = list(cross_platform.keys())
        tops = [cross_platform[p]['tops'] for p in platforms]
        energy = [cross_platform[p]['energy_pJ'] for p in platforms]
        x = range(len(platforms))
        ax.bar(x, tops, width=0.35, label='TOPS', color='steelblue')
        ax.bar([i+0.35 for i in x], energy, width=0.35, label='Energy (pJ)', color='salmon')
        ax.set_xticks([i+0.175 for i in x])
        ax.set_xticklabels(platforms)
        ax.set_ylabel('Value')
        ax.set_title('(d) Cross-Platform Cost-Performance')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()
        if save:
            fig.savefig(os.path.join(self.output_dir, 'fig4d_platform.pdf'), dpi=300, bbox_inches='tight')
            fig.savefig(os.path.join(self.output_dir, 'fig4d_platform.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        return fig


print('FigureGenerator loaded')
