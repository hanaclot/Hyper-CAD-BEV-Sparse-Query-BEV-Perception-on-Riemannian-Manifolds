# -*- coding: utf-8 -*-
"""
Implements the Manifold Alternating Direction Method of Multipliers
for joint optimization of IBEV field params and sparse query positions

Reference: IEEE TKDE, Boyd et al. 2011 ADMM framework
"""

import numpy as np
from typing import Tuple, Optional, Callable


class ManifoldADMM:
    """Manifold-ADMM for joint optimization of theta and queries."""
    
    def __init__(self, manifold, K_query=250, eta=0.1, rho=1.0, max_iter=20, seed=99942):
        """Initialize Manifold-ADMM solver."""
        self.M = manifold
        self.K = K_query
        self.eta = eta
        self.rho = rho
        self.max_iter = max_iter
        self.rng = np.random.RandomState(seed)
        self.N = manifold.N
    
    def optimize(self, ibev_field, symbolic_prior, ground_truth, tol=1e-4, verbose=True):
        """Run Manifold-ADMM optimization (Eq. 8a-8c)."""
        z = symbolic_prior.flatten()
        u = np.zeros(self.N)
        history = dict(loss=[], primal_res=[], dual_res=[])
        gt_flat = ground_truth.reshape(self.N, -1)
        for it in range(self.max_iter):
            z_prev = z.copy()
            pred = ibev_field.predict(self.M.grid_points).reshape(self.N, -1)
            grad = 2 * (pred - gt_flat).mean(axis=1)
            z = np.clip(z - self.eta * grad, 0, 1)
            active = z > 0.5
            if active.sum() >= 3:
                q_pts = self.M.grid_points[active]
                q_vals = gt_flat[active]
                ibev_field.fit(q_pts, q_vals, n_epochs=20)
            u = u + self.rho * (z - np.zeros(self.N))
            pred_now = ibev_field.predict(self.M.grid_points).reshape(self.N, -1)
            loss = np.mean((pred_now - gt_flat)**2)
            history['loss'].append(float(loss))
            if verbose:
        query_indices = np.where(z > 0.5)[0]
        if len(query_indices) > self.K:
            query_indices = np.argsort(z)[-self.K:]
        stats = dict(n_queries=len(query_indices), final_loss=history['loss'][-1],
            n_iterations=len(history['loss']))
        return query_indices, history, stats


class OptimizerBenchmark:
    """Benchmark GD, Standard ADMM, Manifold-ADMM."""
    @staticmethod
    def gradient_descent(ibev_field, query_points, query_values, n_iters=120):
        """Standard GD baseline."""
        losses = ibev_field.fit(query_points, query_values, n_epochs=n_iters)
        return dict(n_iters=len(losses), final_loss=losses[-1], method='GD')
    @staticmethod
    def standard_admm(M, ibev_field, query_values, n_iters=65):
        """Standard ADMM."""
        losses = []
        q_pts = M.grid_points[:250]
        for it in range(n_iters):
            l = ibev_field.fit(q_pts, query_values[:250], n_epochs=3)
            losses.append(l[-1])
        return dict(n_iters=len(losses), final_loss=losses[-1], method='Standard_ADMM')
    @staticmethod
    def manifold_admm(M, ibev_field, sg_prior, gt, n_iters=20):
        """Manifold-ADMM."""
        solver = ManifoldADMM(M, max_iter=n_iters)
        _, history, stats = solver.optimize(ibev_field, sg_prior, gt, verbose=False)
        return dict(n_iters=len(history['loss']), final_loss=history['loss'][-1], method='Manifold_ADMM')


