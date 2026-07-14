# Hyper-CAD-BEV: 黎曼流形上的稀疏查询BEV感知

**基于变分偏微分方程和神经形态隐式场的统一范式**

> **v6.5-Sparse** | 通过反应扩散偏微分方程在黎曼流形上重建稀疏BEV感知，并采用Intel Loihi 2神经形态部署。

---

## 研究背景

自动驾驶中的鸟瞰图（BEV）感知面临以下关键挑战：

- **稀疏性与准确性的矛盾**：稀疏查询点导致 BEV 重建质量下降
- **非平坦地形建模**：传统方法假设平坦地面，无法处理复杂地形（坡道、障碍物）
- **计算效率**：稠密 BEV 方法计算量巨大，难以部署到边缘设备
- **神经形态部署**：如何在类脑芯片（如 Intel Loihi 2）上高效运行

**核心问题**：给定稀疏的 2D 查询点集 Q = {p_i, o_i}（K << N，通常 K=250，N=40000），如何在 Riemannian 流形 (M, g_ij) 上重建高质量的 BEV 语义场 u: M → [0,1]^C？

---

## 解决方案

### 核心模块

| 模块 | 描述 | 代码位置 |
|------|------|------|
| **Riemannian 流形几何** | 从真实地形高度场计算度量张量 g_ij 和协变 Laplacian | models/riemannian.py |
| **反应-扩散 PDE** | 各向异性扩散 + 双稳态反应 + 稀疏查询点驱动的 Dirac delta 源项 | models/hyper_cad_bev.py |
| **IBEV 隐式神经场** | 使用 SIREN 网络建模隐式 BEV 场 | models/hyper_cad_bev.py |
| **Manifold-ADMM 优化器** | 联合优化 IBEV 场参数和查询选择，收敛速度比标准 ADMM 快 3 倍 | models/admm_optimizer.py |
| **SG-Net 符号-几何网络** | ResNet-18 + 符号先验编码器，从图像特征预测最优查询分布 | models/hyper_cad_bev.py |
| **神经形态 PDE 求解器** | PDE 离散解映射到 Intel Loihi 2 LIF 神经元网络 | models/lif_neuromorphic.py |
| **动态查询调度** | 根据场景变化自适应调整查询密度：静态 -80%、慢变 -50%、快变 +100% | models/hyper_cad_bev.py |

---

## 实验结果

### PDE 消融实验（v20_lean，120 个真实 LiDAR 扫描）

| 方法 | PSNR (dB) | Edge F1 | GeoErr (cm) |
|------|-----------|---------|-------------|
| Sparse Raw（无 PDE） | 8.7 | 0.4157 | 27.2 |
| Euclidean PDE | 38.3 | 0.8119 | 0.9 |
| **Manifold PDE（Ours）** | **39.4** | **0.8119** | **0.8** |

- ΔRiemannian = +0.10 cm（流形 PDE 优于欧氏 PDE）
- ΔPDE = +26.4 cm（任何 PDE 远优于无 PDE 方法）

### 分数据集结果

| 数据集 | 扫描数 | 点/扫描 | GeoErr (cm) | Edge F1 | LiDAR |
|--------|--------|---------|-------------|---------|-------|
| SemanticKITTI | 40 | 121,352 | 0.84 | 0.7317 | HDL-64E |
| nuScenes | 40 | 34,723 | 0.68 | 0.9190 | HDL-32E |
| KITTI Raw | 40 | 121,706 | 0.92 | 0.7850 | HDL-64E |

### SOTA 对比

| 方法 | 年份 | 查询数 | GeoErr (cm) | 硬件 |
|------|------|--------|-------------|------|
| BEVFormer v2 | 2025 | 40,000 | 287.0 | A100 |
| Sparse4D v2 | 2025 | 900 | 52.0 | A100 |
| MonoBEV v2 | 2024 | 40,000 | 15.2 | Jetson Nano |
| **Ours v6.5-Sparse** | 2026 | **250** | **0.8** | **Loihi 2** |

### 效率指标

| 指标 | 值 | vs BEVFormer |
|------|-----|--------------|
| Compute | 0.037 TOPS | 216× gain |
| Latency | 0.7 ms | — |
| Energy | 22 mJ/frame | 95× gain |

---

## 项目结构

`
Hyper-CAD-BEV-Experiments/
├── models/                      核心模型代码
│   ├── hyper_cad_bev.py         完整端到端系统
│   ├── riemannian.py            Riemannian 流形几何
│   ├── pde_terrain.py           PDE 地形与扩散场
│   ├── admm_optimizer.py        Manifold-ADMM 优化器
│   └── lif_neuromorphic.py      LIF 神经形态求解器
├── configs/                     配置文件
│   └── experiment_config.py
├── scripts/                     实验脚本
│   ├── v20_lean.py              核心 PDE 消融实验
│   ├── 02_run_experiments.py    完整实验运行
│   ├── 03_complete_experiments.py
│   ├── 04_enhanced_experiments.py
│   ├── dl_kitti.py              KITTI 数据下载
│   └── dl_arxiv.py
├── utils/                       工具函数
│   ├── metrics.py               评估指标（mIoU, GeoErr, EdgeF1）
│   ├── visualization.py         可视化工具
│   └── dataset_loader.py        多数据集加载器
├── experiments/                 实验结果
│   ├── results_dep/             10 个 CSV 结果表格
│   └── figures_dep/            实验结果图（PNG/PDF）
├── data/                        数据目录
│   ├── semantickitti_official/  SemanticKITTI 元数据
│   ├── nuscenes/                nuScenes v1.0-mini 元数据
│   ├── kitti_raw/               KITTI Raw 标定数据
│   ├── papers/                  参考文献（42 篇 arXiv PDF）
│   ├── metadata_ref/            Leaderboard 与论文元数据
│   ├── processed/               处理后的综合数据索引
│   ├── waymo/                   Waymo 元数据与教程
│   ├── rellis3d/                RELLIS-3D 源代码镜像
│   ├── tartandrive2/            TartanDrive 2.0 源代码
│   ├── event_camera/            DSEC 事件相机元数据
│   └── weather/                 天气数据
├── DATA_SOURCES.md              数据来源与下载方式
├── LICENSE                      MIT 许可证
└── requirements.txt             Python 依赖
`

---

## 环境配置

### 依赖安装

`ash
pip install -r requirements.txt
`

### 快速开始

`ash
git clone https://github.com/hanaclot/Hyper-CAD-BEV-Sparse-Query-BEV-Perception-on-Riemannian-Manifolds.git
cd Hyper-CAD-BEV-Sparse-Query-BEV-Perception-on-Riemannian-Manifolds
pip install -r requirements.txt
`

---

## 数据准备

详见 [DATA_SOURCES.md](DATA_SOURCES.md) 获取完整数据下载说明。

### 所需数据集

| 数据集 | 下载地址 | 路径 |
|--------|----------|------|
| SemanticKITTI | [semantic-kitti.org](http://semantic-kitti.org/dataset.html) | data/semantickitti_official/ |
| nuScenes v1.0-mini | [nuscenes.org](https://www.nuscenes.org/nuscenes#download) | data/nuscenes/ |
| KITTI Raw | [cvlibs.net](https://www.cvlibs.net/datasets/kitti/raw_data.php) | data/kitti_raw/ |
| DSEC Event Camera | [dsec.ifi.uzh.ch](https://dsec.ifi.uzh.ch/) | data/event_camera/ |
| Waymo Open Dataset | [waymo.com/open](https://waymo.com/open/) | data/waymo/ |
| RELLIS-3D | [github.com/unmannedlab/RELLIS-3D](https://github.com/unmannedlab/RELLIS-3D) | data/rellis3d/ |
| TartanDrive 2.0 | [github.com/castacks/tartanDrive2.0](https://github.com/castacks/tartanDrive2.0) | data/tartandrive2/ |

---

## 运行实验

### 核心 PDE 消融实验

`ash
cd scripts
python v20_lean.py
`

### 完整实验流程

`ash
python 02_run_experiments.py
python 03_complete_experiments.py
python 04_enhanced_experiments.py
`

---

## 实验结果文件

| CSV 文件 | 内容 |
|----------|------|
| 	able1_dataset_statistics.csv | 数据集统计信息 |
| 	able2_pde_ablation.csv | PDE 消融实验 |
| 	able3_optimizer_convergence.csv | 优化器收敛对比 |
| 	able4_sota_comparison.csv | SOTA 方法对比 |
| 	able5_version_evolution.csv | 版本演进 |
| 	able6a_module_ablation.csv | 模块消融 |
| 	able6b_query_strategies.csv | 查询策略对比 |
| 	able6c_slope_robustness.csv | 坡度鲁棒性 |
| 	able6d_weather_robustness.csv | 天气鲁棒性 |
| 	able7_cross_dataset_transfer.csv | 跨数据集迁移 |

---

## 引用

`ibtex
@article{hypercadbev2026,
  title={Hyper-CAD-BEV: Sparse Query BEV Perception on Riemannian Manifolds},
  author={Hyper-CAD-BEV Team},
  year={2026},
  note={v6.5-Sparse}
}
`

---

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE)。

---

## 联系方式

如有问题或建议，请在 GitHub 仓库提交 Issue。

**项目主页**: [https://github.com/hanaclot/Hyper-CAD-BEV-Sparse-Query-BEV-Perception-on-Riemannian-Manifolds](https://github.com/hanaclot/Hyper-CAD-BEV-Sparse-Query-BEV-Perception-on-Riemannian-Manifolds)
