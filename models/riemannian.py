# -*- coding: utf-8 -*-
"""
Hyper-CAD-BEV v6.5-Sparse: Riemannian Manifold Module
======================================================
Implements the Riemannian geometry framework for BEV perception
as described in Section II.A-II.B of the manuscript.

Key mathematical components:
  - Metric tensor g_ij from terrain elevation h(x,y)
  - Christoffel symbols Gamma^k_ij
  - Gaussian curvature K
  - Covariant Laplacian nabla_g . (D nabla_g u)
  - Hessian norm ||nabla^2_g u||_F (Theorem 2: optimal query criterion)

Reference: ef6c319b-af69-4df4-a606-021de639c471
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Callable, Optional, Tuple, Dict
import warnings


class RiemannianManifold:
    """
    Bounded 2D Riemannian manifold (M, g_ij) induced by terrain elevation.
    
    The manifold is parameterized as a Monge patch:
        z = h(x, y)  for (x, y) in [0, Lx] x [0, Ly]
    
    The metric tensor is:
        g_11 = 1 + (dh/dx)^2
        g_12 = g_21 = (dh/dx)(dh/dy)
        g_22 = 1 + (dh/dy)^2
    """
    
    def __init__(self, Nx: int = 200, Ny: int = 200, Lx: float = 50.0, Ly: float = 50.0):
        """
        Args:
            Nx, Ny: Grid resolution
            Lx, Ly: Physical domain size in meters
        """
        self.Nx, self.Ny = Nx, Ny
        self.Lx, self.Ly = Lx, Ly
        self.dx = Lx / (Nx - 1)
        self.dy = Ly / (Ny - 1)
        self._build_grid()
        
    def _build_grid(self):
        """Build coordinate grid."""
        x = np.linspace(0, self.Lx, self.Nx)
        y = np.linspace(0, self.Ly, self.Ny)
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        self.grid_points = np.stack([self.X.flatten(), self.Y.flatten()], axis=-1)
        self.N = self.Nx * self.Ny
    
    def set_elevation(self, h_array: np.ndarray):
        """Set elevation from numpy array."""
        self.h = h_array.reshape(self.Nx, self.Ny)
        self._compute_metric()
        self._compute_christoffel()
        self._compute_curvature()
    
    def set_elevation_func(self, h_func: Callable):
        """Set elevation from a function h(x,y)."""
        self.h = h_func(self.X, self.Y)
        self._compute_metric()
        self._compute_christoffel()
        self._compute_curvature()
    
    def _compute_metric(self):
        """Compute metric tensor g_ij and its inverse."""
        hx = np.gradient(self.h, self.dx, axis=0)
        hy = np.gradient(self.h, self.dy, axis=1)
        
        self.g11 = 1.0 + hx**2
        self.g12 = hx * hy
        self.g22 = 1.0 + hy**2
        
        self.det_g = self.g11 * self.g22 - self.g12**2
        self.sqrt_det_g = np.sqrt(np.maximum(self.det_g, 1e-12))
        
        det_inv = 1.0 / np.maximum(self.det_g, 1e-12)
        self.g11_inv = self.g22 * det_inv
        self.g12_inv = -self.g12 * det_inv
        self.g22_inv = self.g11 * det_inv
    
    def _compute_christoffel(self):
        """Compute Christoffel symbols Gamma^k_ij for the induced metric."""
        hx = np.gradient(self.h, self.dx, axis=0)
        hy = np.gradient(self.h, self.dy, axis=1)
        hxx = np.gradient(hx, self.dx, axis=0)
        hxy = np.gradient(hx, self.dy, axis=1)
        hyy = np.gradient(hy, self.dy, axis=1)
        
        # Gamma^1_ij (first coordinate)
        self.Gamma1_11 = self.g11_inv * hx * hxx + self.g12_inv * hx * hxy
        self.Gamma1_12 = self.g11_inv * hx * hxy + self.g12_inv * hx * hyy
        self.Gamma1_22 = self.g11_inv * hy * hxy + self.g12_inv * hy * hyy
        
        # Gamma^2_ij (second coordinate)
        self.Gamma2_11 = self.g12_inv * hx * hxx + self.g22_inv * hx * hxy
        self.Gamma2_12 = self.g12_inv * hx * hxy + self.g22_inv * hx * hyy
        self.Gamma2_22 = self.g12_inv * hy * hxy + self.g22_inv * hy * hyy
    
    def _compute_curvature(self):
        """Compute Gaussian curvature K of the induced metric."""
        hx = np.gradient(self.h, self.dx, axis=0)
        hy = np.gradient(self.h, self.dy, axis=1)
        hxx = np.gradient(hx, self.dx, axis=0)
        hxy = np.gradient(hx, self.dy, axis=1)
        hyy = np.gradient(hy, self.dy, axis=1)
        
        numerator = hxx * hyy - hxy**2
        denominator = (1.0 + hx**2 + hy**2)**2
        self.K = numerator / np.maximum(denominator, 1e-12)
    
    def covariant_laplacian(self, u: np.ndarray) -> np.ndarray:
        """
        Compute the covariant (Riemannian) Laplacian:
            Delta_g u = (1/sqrt(det(g))) partial_i (sqrt(det(g)) g^{ij} partial_j u)
        """
        u = np.asarray(u).reshape(self.Nx, self.Ny)
        ux = np.gradient(u, self.dx, axis=0)
        uy = np.gradient(u, self.dy, axis=1)
        
        Fx = self.sqrt_det_g * (self.g11_inv * ux + self.g12_inv * uy)
        Fy = self.sqrt_det_g * (self.g12_inv * ux + self.g22_inv * uy)
        
        div_F = np.gradient(Fx, self.dx, axis=0) + np.gradient(Fy, self.dy, axis=1)
        return div_F / np.maximum(self.sqrt_det_g, 1e-12)
    
    def euclidean_laplacian(self, u: np.ndarray) -> np.ndarray:
        """Standard Euclidean Laplacian for comparison."""
        u = np.asarray(u).reshape(self.Nx, self.Ny)
        uxx = np.gradient(np.gradient(u, self.dx, axis=0), self.dx, axis=0)
        uyy = np.gradient(np.gradient(u, self.dy, axis=1), self.dy, axis=1)
        return uxx + uyy
    
    def covariant_diffusion(self, u: np.ndarray, D_field: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute nabla_g . (D(p) nabla_g u) -- the anisotropic diffusion term
        from Equation (1) with position-dependent diffusion coefficient.
        """
        u = np.asarray(u).reshape(self.Nx, self.Ny)
        ux = np.gradient(u, self.dx, axis=0)
        uy = np.gradient(u, self.dy, axis=1)
        
        if D_field is None:
            D_field = np.ones_like(u)
        D_field = np.asarray(D_field).reshape(self.Nx, self.Ny)
        
        Fx = D_field * self.sqrt_det_g * (self.g11_inv * ux + self.g12_inv * uy)
        Fy = D_field * self.sqrt_det_g * (self.g12_inv * ux + self.g22_inv * uy)
        
        div_F = np.gradient(Fx, self.dx, axis=0) + np.gradient(Fy, self.dy, axis=1)
        return div_F / np.maximum(self.sqrt_det_g, 1e-12)
    
    def hessian_norm(self, u: np.ndarray) -> np.ndarray:
        """
        Compute the Frobenius norm of the covariant Hessian tensor:
            ||nabla^2_g u||_F = sqrt(sum_{i,j} |H_ij|^2)
        
        The covariant Hessian is:
            H_ij = partial_i partial_j u - Gamma^k_ij partial_k u
        
        This is the KEY quantity from Theorem 2: the optimal query density
        rho*(p) is proportional to ||nabla^2_g u_true(p)||_F.
        """
        u = np.asarray(u).reshape(self.Nx, self.Ny)
        ux = np.gradient(u, self.dx, axis=0)
        uy = np.gradient(u, self.dy, axis=1)
        uxx = np.gradient(ux, self.dx, axis=0)
        uxy = np.gradient(ux, self.dy, axis=1)
        uyy = np.gradient(uy, self.dy, axis=1)
        
        # Covariant Hessian components
        H11 = uxx - self.Gamma1_11 * ux - self.Gamma2_11 * uy
        H12 = uxy - self.Gamma1_12 * ux - self.Gamma2_12 * uy
        H22 = uyy - self.Gamma1_22 * ux - self.Gamma2_22 * uy
        
        return np.sqrt(H11**2 + 2.0 * H12**2 + H22**2)
    
    def volume_element(self) -> np.ndarray:
        """Return dmu(p) = sqrt(det(g)) dx dy."""
        return self.sqrt_det_g
    
    def manifold_integral(self, f: np.ndarray) -> float:
        """Compute integral_M f(p) dmu(p)."""
        f = np.asarray(f).reshape(self.Nx, self.Ny)
        return float(np.sum(f * self.sqrt_det_g) * self.dx * self.dy)
    
    def manifold_norm_L2(self, u: np.ndarray) -> float:
        """Compute ||u||_L2(M) = sqrt(integral_M |u|^2 dmu)."""
        return float(np.sqrt(self.manifold_integral(np.asarray(u)**2)))
    
    def total_variation(self, u: np.ndarray) -> float:
        """Total variation on manifold: V_g = integral_M ||nabla_g u|| dmu."""
        u = np.asarray(u).reshape(self.Nx, self.Ny)
        ux = np.gradient(u, self.dx, axis=0)
        uy = np.gradient(u, self.dy, axis=1)
        grad_norm = np.sqrt(ux**2 + uy**2)
        return float(self.manifold_integral(grad_norm))
    
    def get_statistics(self) -> Dict:
        """Return diagnostic statistics about the manifold geometry."""
        return {
            'gaussian_curvature_mean': float(np.mean(self.K)),
            'gaussian_curvature_max': float(np.max(np.abs(self.K))),
            'gaussian_curvature_std': float(np.std(self.K)),
            'metric_det_mean': float(np.mean(self.det_g)),
            'metric_det_range': (float(np.min(self.det_g)), float(np.max(self.det_g))),
            'elevation_range': (float(np.min(self.h)), float(np.max(self.h))),
            'elevation_std': float(np.std(self.h)),
        }
    
    def compute_edge_mask(self, percentile: float = 85.0) -> np.ndarray:
        """Identify edges based on Gaussian curvature threshold."""
        return np.abs(self.K) > np.percentile(np.abs(self.K), percentile)


print('RiemannianManifold v2.0 loaded -- 9 geometric operators ready')
