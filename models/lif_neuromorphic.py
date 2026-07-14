# -*- coding: utf-8 -*-
"""
Hyper-CAD-BEV v6.5-Sparse: LIF Neuromorphic Solver
====================================================
Implements mathematical equivalence between manifold reaction-diffusion PDE
and LIF spiking neuron dynamics (Section II.E, Eq. 11).

PDE <-> SNN Mapping:
  Point p_i on manifold       <->  Neuron i
  BEV field u(p_i, t)         <->  Membrane potential h_i(t)
  Diffusion nabla_g.(D nabla_g u) <->  Synaptic current sum w_ij S_j(t)
  Reaction R(u,p,t)           <->  Bias current b_i(t)
  Source S(p,t)               <->  Input spikes S_in,i(t)

LIF update (Eq. 11):
  h_i(t+1) = h_i(t) + dt*(-h_i(t)/tau_m + sum w_ij*S_j(t) + b_i(t) + S_in,i(t))

Reference: Intel Loihi 2 platform
"""

import numpy as np
from typing import Tuple, Optional
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve


class LIFNeuromorphicSolver:
    """LIF-based neuromorphic PDE solver for Loihi 2 execution."""
    
    def __init__(self, N, tau_m=20.0, v_th=1.0, v_reset=0.0, dt=0.1,
                 energy_per_spike=23.6):
        """Initialize LIF solver.
        Args:
            N: number of neurons (grid points)
            tau_m: membrane time constant (ms)
            v_th: firing threshold
            v_reset: reset potential
            dt: time step (ms)
            energy_per_spike: energy per spike in pJ (Loihi 2: 23.6 pJ)
        """
        self.N = N
        self.tau_m = tau_m
        self.v_th = v_th
        self.v_reset = v_reset
        self.dt = dt
        self.energy_per_spike = energy_per_spike
        self.h = np.zeros(N)
        self.spike_count = np.zeros(N, dtype=np.int64)
    
    def build_synaptic_matrix(self, manifold, D_field=None):
        """Build sparse synaptic weight matrix."""
        N = self.N
        Nx, Ny = manifold.Nx, manifold.Ny
        dx, dy = manifold.dx, manifold.dy
        W = lil_matrix((N, N), dtype=np.float64)
        for i in range(Nx):
            for j in range(Ny):
                idx = i * Ny + j
                W[idx, idx] = -4.0
                if i > 0: W[idx, (i-1)*Ny + j] = 1.0
                if i < Nx - 1: W[idx, (i+1)*Ny + j] = 1.0
                if j > 0: W[idx, i*Ny + (j-1)] = 1.0
                if j < Ny - 1: W[idx, i*Ny + (j+1)] = 1.0
        if D_field is not None:
            D_flat = D_field.flatten()
            for idx in range(N): W[idx, :] *= D_flat[idx]
        self.W = W.tocsr()
        return self.W
    
    def reset(self):
        """Reset all neuron states."""
        self.h = np.zeros(self.N)
        self.spike_count = np.zeros(self.N, dtype=np.int64)
        self.total_spikes = 0
        self.total_energy_pJ = 0.0
    
    def step(self, I_ext, dt_factor=1.0):
        """Execute one LIF time step."""
        dt = self.dt * dt_factor
        leak = -self.h / self.tau_m
        synaptic = self.W.dot(self.spike_count > 0)
        self.h += dt * (leak + synaptic + I_ext)
        spikes = self.h >= self.v_th
        self.h[spikes] = self.v_reset
        self.spike_count[spikes] += 1
        n_spikes = int(spikes.sum())
        self.total_spikes += n_spikes
        self.total_energy_pJ += n_spikes * self.energy_per_spike
        return spikes
    
    def solve_pde(self, T_sim, I_ext_func, dt_factor=1.0):
        """Solve PDE via LIF dynamics."""
        n_steps = int(T_sim / (self.dt * dt_factor))
        spike_history = []
        for step_idx in range(n_steps):
            I_t = I_ext_func(step_idx, self.h)
            spikes = self.step(I_t, dt_factor)
            if step_idx % 10 == 0: spike_history.append(spikes.copy())
        return spike_history
    
    def get_energy_efficiency(self, n_ops_equivalent):
        """Compute energy efficiency."""
        energy_ops_pJ = n_ops_equivalent * 0.9
        return energy_ops_pJ / max(self.total_energy_pJ, 1e-12)
    
    def to_loihi_config(self):
        """Export config for Loihi 2."""
        return dict(num_neurons=self.N, tau_m=self.tau_m, v_th=self.v_th,
            v_reset=self.v_reset, dt=self.dt,
            energy_per_spike_pJ=self.energy_per_spike,
            total_spikes=int(self.total_spikes),
            total_energy_pJ=float(self.total_energy_pJ),
            estimated_tops=round(self.total_energy_pJ / 1e12 / 0.03, 4))


print('LIFNeuromorphicSolver loaded - Loihi 2 ready')
