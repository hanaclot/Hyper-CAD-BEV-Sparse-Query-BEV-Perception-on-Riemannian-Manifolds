# -*- coding: utf-8 -*-
"""
基于论文手稿第 II-III-IV 节
"""
import os
from pathlib import Path

# ==================== 路径配置 ====================
PROJECT_ROOT = Path(r"E:\Hyper-CAD-BEV-Experiments")
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA = DATA_ROOT / "raw"
PROCESSED_DATA = DATA_ROOT / "processed"

# ==================== 数据集配置 ====================
DATASET_CONFIG = {
    "name": "Rural-Manifold-Dataset",
    "description": "Global-scale dynamic terrain manifold dataset",
    "total_hours": 120,
    "num_scenarios": 15,
    "scenarios": [
        "flat_dirt_road", "moderate_gravel_slope", "steep_rocky_terrain",
        "muddy_trail", "grassland_path", "forest_clearing",
        "riverbank", "construction_site", "agricultural_field",
        "mountain_pass", "desert_track", "snow_covered_path",
        "wet_road", "dusty_trail", "night_operation"
    ],
    "sensors": {
        "event_camera": {"resolution": "1280x720", "fps": "variable"},
        "lidar": {"type": "Velodyne HDL-64E", "range": "120m"},
        "rgb_camera": {"resolution": "1920x1080", "fps": 30},
    },
    "splits": {
        "train": 10,  # scenarios
        "val": 2,
        "test": 3,
    }
}

# ==================== 模型配置 ====================
MODEL_CONFIG = {
    "name": "Hyper-CAD-BEV-v6.5-Sparse",
    "version": "6.5",
    
    # Riemannian流形配置
    "manifold": {
        "type": "riemannian_2d",
        "grid_resolution": (200, 200),  # N = X * Z
        "metric_tensor_init": "identity",
        "curvature_range": (-0.5, 0.5),  # 地形曲率范围 (坡道)
    },
    
    # 变分PDE配置
    "pde": {
        "type": "reaction_diffusion",
        "diffusion_coeff": {
            "drivable": 0.8,
            "obstacle_boundary": 0.01,
        },
        "reaction_rate": 0.5,
        "alpha_pde": 0.1,  # PDE正则化权重
    },
    
    # 隐式神经场配置
    "implicit_field": {
        "network_type": "siren",  # Sinusoidal Representation Network
        "hidden_dim": 256,
        "num_layers": 5,
        "omega_0": 30.0,
        "output_dim": 20,  # 语义类别数
    },
    
    # 稀疏查询配置
    "sparse_query": {
        "num_queries": 250,  # K
        "query_dim": 128,
        "hessian_threshold": 0.01,
    },
    
    # Manifold-ADMM配置
    "admm": {
        "rho": 1.0,
        "mu": 0.1,  # 稀疏性惩罚
        "eta": 0.01,  # 步长
        "max_iterations": 50,
        "convergence_tol": 1e-4,
    },
    
    # SG-Net配置
    "sg_net": {
        "backbone": "resnet18",
        "pretrained": True,
        "output_query_dim": 250,
        "symbolic_prior": {
            "road_width": 3.5,  # meters
            "ridge_width": 0.5,
            "vehicle_size": [4.5, 2.0],  # length, width
        }
    },
    
    # 神经形态映射配置
    "neuromorphic": {
        "chip": "Loihi_2",
        "neuron_model": "LIF",  # Leaky Integrate-and-Fire
        "membrane_time_constant": 20.0,  # ms
        "threshold_voltage": 1.0,
        "reset_voltage": 0.0,
        "refractory_period": 2.0,  # ms
    },
    
    # 动态查询调度
    "query_scheduling": {
        "static_area_reduction": 0.8,
        "slow_varying_reduction": 0.5,
        "rapid_varying_boost": 2.0,
    }
}

# ==================== 训练配置 ====================
TRAIN_CONFIG = {
    "epochs": 100,
    "batch_size": 4,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "lr_scheduler": {
        "type": "cosine",
        "warmup_epochs": 5,
        "min_lr": 1e-6,
    },
    "optimizer": "adamw",
    "loss_weights": {
        "ibev": 1.0,
        "pde": 0.1,
        "prior": 0.05,
    },
    "early_stopping": {
        "patience": 15,
        "metric": "val_miou",
    },
}

# ==================== SOTA对比方法配置 ====================
BASELINE_METHODS = {
    "BEVFormer_v2": {
        "year": 2025,
        "type": "dense_multi_camera",
        "hardware": "A100",
        "params": {"num_queries": 900, "num_layers": 6},
    },
    "BEVDet_v3": {
        "year": 2025,
        "type": "dense_multi_camera",
        "hardware": "A100",
        "params": {"depth_net": "supervised"},
    },
    "MonoBEV_v2": {
        "year": 2024,
        "type": "monocular_bev",
        "hardware": "Jetson_Nano",
        "params": {"vanishing_point": True},
    },
    "SingleBEV": {
        "year": 2024,
        "type": "monocular_bev",
        "hardware": "Jetson_Nano",
        "params": {"direct_generation": True},
    },
    "HyperCAD_BEV_v5_2": {
        "year": 2025,
        "type": "monocular_bev",
        "hardware": "Allwinner_V853",
        "params": {"zero_calibration": True},
    },
    "NeuBEV": {
        "year": 2025,
        "type": "neuromorphic_bev",
        "hardware": "Loihi_2",
        "params": {"snn_based": True},
    },
    "HyperCAD_BEV_v6_0_Neuro": {
        "year": 2026,
        "type": "neuromorphic_bev",
        "hardware": "Loihi_2",
        "params": {"pde_neuromorphic": True},
    },
    "HyperCAD_BEV_v6_5_Sparse": {
        "year": 2026,
        "type": "neuromorphic_bev",
        "hardware": "Loihi_2",
        "params": {"manifold_sparse_query": True},
    },
}

# ==================== 评估指标配置 ====================
EVAL_METRICS = {
    "semantic": ["mIoU", "per_class_IoU", "pixel_accuracy"],
    "geometric": ["geometric_error_cm", "depth_MAE", "depth_RMSE"],
    "efficiency": ["TOPS", "latency_ms", "energy_mJ_per_frame", "mIoU_per_J"],
    "robustness": ["slope_mIoU", "weather_mIoU", "illumination_mIoU"],
}

# ==================== 鲁棒性测试场景 ====================
ROBUSTNESS_CONDITIONS = {
    "terrain_slope": [0, 15, 25],  # degrees
    "weather": ["sunny", "overcast", "light_rain", "moderate_rain", "dust_storm"],
    "illumination": ["daylight", "dusk", "night_0.1lux"],
}

# ==================== 消融实验配置 ====================
ABLATION_CONFIGS = {
    "full_v65_sparse": {
        "riemannian": True, "pde": True, "admm": True,
        "neuromorphic": True, "dynamic_query": True,
    },
    "w/o_riemannian": {
        "riemannian": False, "pde": True, "admm": True,
        "neuromorphic": True, "dynamic_query": True,
    },
    "w/o_pde": {
        "riemannian": True, "pde": False, "admm": True,
        "neuromorphic": True, "dynamic_query": True,
    },
    "w/o_admm": {
        "riemannian": True, "pde": True, "admm": False,
        "neuromorphic": True, "dynamic_query": True,
    },
    "w/o_neuromorphic": {
        "riemannian": True, "pde": True, "admm": True,
        "neuromorphic": False, "dynamic_query": True,
    },
    "w/o_dynamic_query": {
        "riemannian": True, "pde": True, "admm": True,
        "neuromorphic": True, "dynamic_query": False,
    },
}

# ==================== 查询策略对比 ====================
QUERY_STRATEGIES = {
    "dense_query": {"type": "full_grid", "num_queries": 40000},
    "uniform_random": {"type": "random", "num_queries": 250},
    "edge_based": {"type": "edge", "num_queries": 250},
    "hessian_guided": {"type": "hessian_theoretical", "num_queries": 250},
    "sg_net_predicted": {"type": "sg_net", "num_queries": 250},
}

