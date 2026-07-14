# -*- coding: utf-8 -*-
"""
Implements all metrics from the manuscript:
  - mIoU (mean Intersection over Union)
  - Geometric Error (cm)
  - Edge Smoothness (Gradient Loss)
  - Reconstruction Error (MSE)
  - Energy per frame (mJ)
  - Energy Efficiency (mIoU/J)
"""

import numpy as np
import json
import csv
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class BEVMetrics:
    """Compute all BEV perception evaluation metrics."""
    
    @staticmethod
    def compute_miou(pred: np.ndarray, gt: np.ndarray, n_classes: int = 20) -> float:
        """Mean Intersection over Union.
        pred: (Nx, Ny, n_classes) or (Nx, Ny) with class indices
        gt: same shape
        """
        if pred.ndim == 3:
            pred = np.argmax(pred, axis=-1)
        if gt.ndim == 3:
            gt = np.argmax(gt, axis=-1)
        ious = []
        for c in range(n_classes):
            pred_c = (pred == c)
            gt_c = (gt == c)
            intersection = (pred_c & gt_c).sum()
            union = (pred_c | gt_c).sum()
            if union > 0:
                ious.append(intersection / union)
        return float(np.mean(ious)) * 100
    
    @staticmethod
    def compute_geometric_error(pred_elevation: np.ndarray, gt_elevation: np.ndarray) -> float:
        """Geometric error in centimeters.
        pred_elevation: predicted terrain height (m)
        gt_elevation: ground truth terrain height (m)
        """
        return float(np.sqrt(np.mean((pred_elevation - gt_elevation)**2))) * 100
    
    @staticmethod
    def compute_edge_smoothness(field: np.ndarray) -> float:
        """Edge smoothness via gradient loss."""
        gx = np.gradient(field, axis=0)
        gy = np.gradient(field, axis=1)
        return float(np.mean(np.sqrt(gx**2 + gy**2)))
    
    @staticmethod
    def compute_mse(pred: np.ndarray, gt: np.ndarray) -> float:
        """Mean Squared Error."""
        return float(np.mean((pred - gt)**2))
    
    @staticmethod
    def compute_energy_efficiency(miou: float, energy_mJ: float) -> float:
        """Energy efficiency in mIoU/J."""
        return miou / (energy_mJ / 1000.0) if energy_mJ > 0 else 0.0
    
    @staticmethod
    def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two distributions."""
        a_f = a.flatten()
        b_f = b.flatten()
        return float(np.dot(a_f, b_f) / (np.linalg.norm(a_f) * np.linalg.norm(b_f) + 1e-12))


class ExperimentTracker:
    """Track and export experiment results."""
    
    def __init__(self, exp_name: str, output_dir: str):
        self.exp_name = exp_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results: List[Dict] = []
        self.start_time = datetime.now()
    
    def log(self, **kwargs):
        """Log a result entry."""
        self.results.append(kwargs)
    
    def to_csv(self, filename: str = None):
        """Export results to CSV (handles heterogeneous dicts)."""
        if filename is None:
            filename = f'{self.exp_name}_results.csv'
        fpath = os.path.join(self.output_dir, filename)
        if self.results:
            # Collect all unique fieldnames across all entries
            all_keys = []
            for r in self.results:
                for k in r.keys():
                    if k not in all_keys:
                        all_keys.append(k)
            with open(fpath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(self.results)
        return fpath
    
    def to_json(self, filename: str = None):
        """Export results to JSON."""
        if filename is None:
            filename = f'{self.exp_name}_results.json'
        fpath = os.path.join(self.output_dir, filename)
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
        return fpath
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return dict(experiment=self.exp_name, n_entries=len(self.results),
            elapsed_sec=elapsed, output_dir=self.output_dir)


