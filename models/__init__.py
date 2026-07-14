# Hyper-CAD-BEV v6.5-Sparse
# Sparse Query BEV Perception on Riemannian Manifolds
from .riemannian import RiemannianManifold
from .pde_terrain import (OffroadTerrainGenerator, AnisotropicDiffusionField,
    ReactionDiffusionPDE, ImplicitBEVField)
from .lif_neuromorphic import LIFNeuromorphicSolver
from .admm_optimizer import ManifoldADMM, OptimizerBenchmark
