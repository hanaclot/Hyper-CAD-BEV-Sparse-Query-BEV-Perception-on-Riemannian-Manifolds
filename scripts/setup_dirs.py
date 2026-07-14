import os
# Create all needed directories
dirs = [
    "E:/Hyper-CAD-BEV-Experiments/data/semantickitti_official",
    "E:/Hyper-CAD-BEV-Experiments/data/kitti",
    "E:/Hyper-CAD-BEV-Experiments/data/rellis3d",
    "E:/Hyper-CAD-BEV-Experiments/data/tartandrive2",
    "E:/Hyper-CAD-BEV-Experiments/data/event_camera",
    "E:/Hyper-CAD-BEV-Experiments/data/nuscenes",
    "E:/Hyper-CAD-BEV-Experiments/data/waymo",
    "E:/Hyper-CAD-BEV-Experiments/data/raw",
    "E:/Hyper-CAD-BEV-Experiments/data/processed",
    "E:/Hyper-CAD-BEV-Experiments/data/crawled",
    "D:/HyperCAD_BEV_2026/data",
    "D:/HyperCAD_BEV_2026/data/raw",
    "D:/HyperCAD_BEV_2026/data/processed",
    "D:/HyperCAD_BEV_2026/experiments",
    "D:/HyperCAD_BEV_2026/experiments/results",
    "D:/HyperCAD_BEV_2026/experiments/figures",
    "D:/HyperCAD_BEV_2026/experiments/checkpoints",
    "D:/HyperCAD_BEV_2026/experiments/logs",
]
for d in dirs:
    os.makedirs(d, exist_ok=True)
