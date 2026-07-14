# Hyper-CAD-BEV：黎曼流形上的稀疏查询BEV感知

**基于变分偏微分方程和神经形态隐式场的统一范式**

> **v6.5-Sparse** | 通过反应扩散偏微分方程在黎曼流形上重建稀疏BEV感知，并采用Intel Loihi 2神经形态部署。

---

## Research Problem

Bird's-eye view (BEV) perception is fundamental to autonomous driving, yet faces critical challenges:

- **Sparsity-vs-Accuracy**: Sparse query points degrade BEV reconstruction quality
- **Non-flat Terrain Modeling**: Conventional methods assume flat ground, failing on complex terrain (slopes, obstacles)
- **Computational Efficiency**: Dense BEV methods demand massive compute, hindering edge deployment
- **Neuromorphic Deployment**: How to run efficiently on brain-inspired chips

**Problem Definition**: Given sparse 2D query points Q = {p_i, o_i} (K << N, typically K=250, N=40000), reconstruct a high-quality BEV semantic field u: M -> [0,1]^C on a Riemannian manifold (M, g_ij).

---

## Solution Architecture

### Core Modules

#### 1. Riemannian Manifold Geometry (Section II-A)
Metric tensor from real terrain height h(x,z). Covariant Laplacian on the manifold.

**Code**: models/riemannian.py

#### 2. Reaction-Diffusion PDE (Section II-A, Eq. 1)
- **Diffusion**: Anisotropic; D=0.8 on traversable regions, D=0.01 at boundaries
- **Reaction**: Bistable reaction term
- **Source**: Dirac delta from sparse query points

**Code**: models/hyper_cad_bev.py

#### 3. IBEV Implicit Neural Field (Section II-C)
Models the implicit BEV field using a SIREN network.

**Code**: models/hyper_cad_bev.py

#### 4. Manifold-ADMM Optimizer (Section II-D, Eq. 7-9)
Jointly optimizes IBEV field parameters and query selection. 3x faster than standard ADMM, 6x faster than gradient descent.

**Code**: models/admm_optimizer.py

#### 5. SG-Net: Symbolic-Geometric Prior Network (Section II-B, III-A)
Learns optimal query distribution from image features via ResNet-18 backbone and symbolic prior encoder.

**Code**: models/hyper_cad_bev.py

#### 6. Neuromorphic PDE Solver (Section II-E, III-B)
Maps the discretized PDE onto LIF neurons on Intel Loihi 2.

**Code**: models/hyper_cad_bev.py

#### 7. Dynamic Query Scheduler (Section III-C)
Adapts query density to scene dynamics: static (-80%), slow (-50%), fast (+100%).

**Code**: models/hyper_cad_bev.py

---

## Experimental Results

### PDE Ablation (v20_lean, 120 real LiDAR scans)

| Method | PSNR (dB) | Edge F1 | GeoErr (cm) |
|--------|-----------|---------|-------------|
| Sparse Raw (No PDE) | 8.7 | 0.4157 | 27.2 |
| Euclidean PDE | 38.3 | 0.8119 | 0.9 |
| **Manifold PDE (Ours)** | **39.4** | **0.8119** | **0.8** |

- +0.10 cm: manifold PDE over Euclidean PDE
- +26.4 cm: any PDE over no-PDE baseline

### Per-Dataset Results

| Dataset | Scans | Points/Scan | GeoErr (cm) | Edge F1 | LiDAR |
|---------|-------|-------------|-------------|---------|-------|
| SemanticKITTI | 40 | 121,352 | 0.84 | 0.7317 | HDL-64E |
| nuScenes | 40 | 34,723 | 0.68 | 0.9190 | HDL-32E |
| KITTI Raw | 40 | 121,706 | 0.92 | 0.7850 | HDL-64E |

### SOTA Comparison

| Method | Year | Queries | GeoErr (cm) | Hardware |
|--------|------|---------|-------------|----------|
| BEVFormer v2 | 2025 | 40,000 | 287.0 | A100 |
| Sparse4D v2 | 2025 | 900 | 52.0 | A100 |
| MonoBEV v2 | 2024 | 40,000 | 15.2 | Jetson Nano |
| **Ours v6.5-Sparse** | **2026** | **250** | **0.8** | **Loihi 2** |

### Efficiency

| Metric | Value | vs BEVFormer |
|--------|-------|--------------|
| Compute | 0.037 TOPS | 216x gain |
| Latency | 0.7 ms | |
| Energy | 22 mJ/frame | 95x gain |

---

## Project Structure

`
Hyper-CAD-BEV-Experiments/
 models/                  Core model implementations
  hyper_cad_bev.py       Full end-to-end system
  riemannian.py          Riemannian manifold geometry
  pde_terrain.py         PDE terrain modeling
  admm_optimizer.py      Manifold-ADMM optimization
  lif_neuromorphic.py    LIF neuron dynamics
 configs/                 Experiment configurations
 scripts/                 Experiment and data acquisition scripts
  v20_lean.py            Main PDE ablation pipeline
 experiments/             Results and figures
  results_dep/           Result tables (CSV)
  figures_dep/           Figure outputs (PNG/PDF)
 utils/                   Utility modules
 data/                    Datasets and metadata (see DATA_SOURCES.md)
`

---

## Environment Setup

### Requirements
`ash
pip install -r requirements.txt
`

### Installation
`ash
git clone https://github.com/hanaclot/Hyper-CAD-BEV-Sparse-Query-BEV-Perception-on-Riemannian-Manifolds.git
cd Hyper-CAD-BEV-Sparse-Query-BEV-Perception-on-Riemannian-Manifolds
pip install -r requirements.txt
`

---

## Data Preparation

### Required Datasets

| Dataset | Size | Download Link | Path |
|---------|------|---------------|------|
| SemanticKITTI | ~19.4 GB | [semantic-kitti.org](http://semantic-kitti.org) | data/semantickitti_official/ |
| nuScenes v1.0-mini | ~5.1 GB | [nuscenes.org](https://www.nuscenes.org) | data/nuscenes/v1.0-mini/ |
| KITTI Raw | ~0.8 GB | [cvlibs.net](https://www.cvlibs.net) | data/kitti_raw/ |

Detailed download instructions: [DATA_SOURCES.md](DATA_SOURCES.md)

---

## Running Experiments

### Main PDE Ablation Experiment
`ash
cd scripts
python v20_lean.py
`

### Full Experiment Pipeline
`ash
python 02_run_experiments.py
python 03_complete_experiments.py
python 04_enhanced_experiments.py
`

---

## Results Summary

| CSV Table | Content |
|-----------|---------|
| 	able1_dataset_statistics.csv | Dataset statistics |
| 	able2_pde_ablation.csv | PDE ablation study |
| 	able3_optimizer_convergence.csv | Optimizer convergence comparison |
| 	able4_sota_comparison.csv | SOTA method comparison |
| 	able5_version_evolution.csv | Version evolution |
| 	able6a_module_ablation.csv | Module ablation |
| 	able6b_query_strategies.csv | Query strategy comparison |
| 	able6c_slope_robustness.csv | Slope robustness |
| 	able6d_weather_robustness.csv | Weather robustness |
| 	able7_cross_dataset_transfer.csv | Cross-dataset transfer |

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## Citation

`ibtex
@article{hypercadbev2026,
  title={Hyper-CAD-BEV: Sparse Query BEV Perception on Riemannian Manifolds},
  author={Hyper-CAD-BEV Team},
  year={2026},
  note={v6.5-Sparse}
}
`

---

## Contact

For questions and collaborations, please open an issue on GitHub.

**Repository**: [https://github.com/hanaclot/Hyper-CAD-BEV-Sparse-Query-BEV-Perception-on-Riemannian-Manifolds](https://github.com/hanaclot/Hyper-CAD-BEV-Sparse-Query-BEV-Perception-on-Riemannian-Manifolds)