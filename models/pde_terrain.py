# -*- coding: utf-8 -*-
"""
Implements OffroadTerrainGenerator, ReactionDiffusionPDE,
and ImplicitBEVField as described in Sections II.A, II.C.

Components:
  1. OffroadTerrainGenerator - generates realistic rural terrain
  2. ReactionDiffusionPDE - solves reaction-diffusion PDE on manifold
  3. ImplicitBEVField - implicit neural BEV field from sparse queries
  4. AnisotropicDiffusionField - D(p) coefficient field

Reference: IEEE TKDE Submission ef6c319b
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy.ndimage import zoom


class OffroadTerrainGenerator:
    """Generate realistic rural unstructured terrain."""
    
    def __init__(self, manifold, seed=12345):
        self.M = manifold
        self.rng = np.random.RandomState(seed)
    
    def generate_rural_terrain(self, slope_deg=0.0, roughness=0.3,
                                include_obstacles=True, include_ridges=True,
                                road_width=3.5, ridge_width=0.5):
        """Generate rural terrain with roads, ridges, obstacles."""
        Nx, Ny = self.M.Nx, self.M.Ny
        dx, dy = self.M.dx, self.M.dy
        X = self.M.X
        
        slope_rad = np.deg2rad(slope_deg)
        h = np.tan(slope_rad) * X
        
        for scale in [8.0, 4.0, 2.0]:
            noise = self.rng.randn(Nx//4, Ny//4)
            noise_up = zoom(noise, (Nx/noise.shape[0], Ny/noise.shape[1]), order=1)
            h += roughness * noise_up / scale
        
        road_center = Ny // 2
        road_half = int(road_width / dy / 2)
        h[:, road_center-road_half:road_center+road_half] *= 0.05
        
        if include_ridges:
            ridge_half = int(ridge_width / dy / 2)
            for rx in [Nx//4, Nx//2, 3*Nx//4]:
                h[rx-ridge_half:rx+ridge_half, :] += 0.3 * roughness
        
        n_obs = 0
        if include_obstacles:
            n_obs = self.rng.randint(3, 8)
            for _ in range(n_obs):
                cx = self.rng.randint(20, Nx-20)
                cy = self.rng.randint(20, Ny-20)
                r = self.rng.randint(2, 5)
                Yg, Xg = np.ogrid[:Nx, :Ny]
                mask = (Xg-cx)**2 + (Yg-cy)**2 < r**2
                h[mask] += self.rng.uniform(0.3, 1.0)
        
        semantic = np.zeros((Nx, Ny), dtype=np.int32)
        semantic[:, :Ny//2] = 0
        semantic[:, Ny//2:] = 1
        semantic[np.abs(h) > 1.0] = 2
        
        meta = dict(slope_deg=slope_deg, roughness=roughness, n_obstacles=n_obs)
        return h, semantic, meta


class AnisotropicDiffusionField:
    """Compute D(p) - anisotropic diffusion coefficient field."""
    
    def __init__(self, D_drivable=0.8, D_boundary=0.01):
        self.D_drivable = D_drivable
        self.D_boundary = D_boundary
    
    def compute(self, manifold, semantic=None):
        """D(p)=0.8 drivable, D(p)=0.01 at boundaries."""
        D = np.full((manifold.Nx, manifold.Ny), self.D_drivable)
        if semantic is not None:
            D[semantic == 2] = self.D_boundary
        edges = manifold.compute_edge_mask()
        D[edges] = self.D_boundary
        return D


class ReactionDiffusionPDE:
    """Solve reaction-diffusion PDE on Riemannian manifold (Eq. 1)."""
    
    def __init__(self, manifold, gamma=0.5, dt=0.01, max_iter=500):
        self.M = manifold
        self.gamma = gamma
        self.dt = dt
        self.max_iter = max_iter
    
    def solve(self, u0, D_field=None, reaction_func=None, source_func=None):
        """Solve Eq.1: du/dt = div(D grad u) + R(u) + S."""
        u = u0.copy()
        history = [u.copy()]
        for it in range(self.max_iter):
            diffusion = self.M.covariant_diffusion(u, D_field)
            reaction = np.zeros_like(u)
            if reaction_func is not None:
                reaction = reaction_func(u, self.M.X, self.M.Y)
            source = np.zeros_like(u)
            if source_func is not None:
                source = source_func(self.M.X, self.M.Y)
            u = u + self.dt * (diffusion + self.gamma * reaction + source)
            u = np.clip(u, 0, 1)
            if it % 50 == 0:
                history.append(u.copy())
        if len(history) == 1:
            history.append(u.copy())
        return u, history
    
    def solve_euclidean(self, u0, D_field=None, reaction_func=None, source_func=None):
        """Solve Eq.1 on Euclidean plane (no manifold curvature)."""
        u = u0.copy()
        history = [u.copy()]
        for it in range(self.max_iter):
            diffusion = self.M.euclidean_laplacian(u)
            if D_field is not None:
                diffusion = diffusion * D_field
            reaction = np.zeros_like(u)
            if reaction_func is not None:
                reaction = reaction_func(u, self.M.X, self.M.Y)
            source = np.zeros_like(u)
            if source_func is not None:
                source = source_func(self.M.X, self.M.Y)
            u = u + self.dt * (diffusion + self.gamma * reaction + source)
            u = np.clip(u, 0, 1)
            if it % 50 == 0:
                history.append(u.copy())
        return u, history


class ImplicitBEVField:
    """Implicit neural field for BEV semantic reconstruction."""
    
    def __init__(self, manifold, hidden_dim=64, n_classes=20, lr=0.01, seed=42):
        self.M = manifold
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.lr = lr
        self.rng = np.random.RandomState(seed)
        self._init_params()
    
    def _init_params(self):
        """Xavier initialization for 2-layer MLP."""
        scale1 = np.sqrt(6.0 / (2 + self.hidden_dim))
        scale2 = np.sqrt(6.0 / (self.hidden_dim + self.n_classes))
        self.W1 = self.rng.uniform(-scale1, scale1, (2, self.hidden_dim))
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = self.rng.uniform(-scale2, scale2, (self.hidden_dim, self.n_classes))
        self.b2 = np.zeros(self.n_classes)
    
    def _siren_forward(self, p):
        """Sine-activated MLP forward pass."""
        h = np.sin(p @ self.W1 + self.b1)
        return h @ self.W2 + self.b2
    
    def predict(self, grid_points=None):
        """Predict BEV field on grid."""
        if grid_points is None:
            grid_points = self.M.grid_points
        logits = self._siren_forward(grid_points)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return probs.reshape(self.M.Nx, self.M.Ny, self.n_classes)
    
    def fit(self, query_points, query_values, n_epochs=200):
        """Fit IBEV field to sparse query observations."""
        losses = []
        for epoch in range(n_epochs):
            pred = self._siren_forward(query_points)
            loss = np.mean((pred - query_values)**2)
            losses.append(float(loss))
            grad_pred = 2 * (pred - query_values) / len(query_points)
            h = np.sin(query_points @ self.W1 + self.b1)
            dh = np.cos(query_points @ self.W1 + self.b1)
            grad_W2 = h.T @ grad_pred
            grad_b2 = grad_pred.sum(axis=0)
            grad_h = grad_pred @ self.W2.T
            grad_W1 = query_points.T @ (grad_h * dh)
            grad_b1 = (grad_h * dh).sum(axis=0)
            self.W1 -= self.lr * grad_W1
            self.b1 -= self.lr * grad_b1
            self.W2 -= self.lr * grad_W2
            self.b2 -= self.lr * grad_b2
        return losses
    
    def compute_hessian_norm_field(self):
        """Compute Hessian norm of predicted BEV field."""
        probs = self.predict()
        return np.stack([self.M.hessian_norm(probs[:,:,c]) for c in range(self.n_classes)], axis=-1)


