import numpy as np
import os, json, struct
from pathlib import Path

# ===== SemanticKITTI Data Loader =====
# Data from http://semantic-kitti.org/

SEMANTIC_KITTI_CLASSES = [
    'unlabeled','car','bicycle','motorcycle','truck','other-vehicle',
    'person','bicyclist','motorcyclist','road','parking','sidewalk',
    'other-ground','building','fence','vegetation','trunk','terrain',
    'pole','traffic-sign'
]

LEARNING_MAP = {
    0:0,1:0,10:1,11:2,13:5,15:3,16:5,18:4,20:5,30:6,31:7,32:8,
    40:9,44:10,48:11,49:12,50:13,51:14,52:0,60:0,70:15,71:16,
    72:17,80:18,81:19,99:0,252:1,253:7,254:7,255:8,256:5,257:5,258:7,259:7
}

class SemanticKITTIDataset:
    def __init__(self, data_root='D:/HyperCAD_BEV_2026/data/semantickitti_lidar',
                 labels_root='D:/HyperCAD_BEV_2026/data/semantickitti',
                 sequences=None):
        self.data_root = Path(data_root)
        self.labels_root = Path(labels_root)
        self.sequences = sequences or ['00']
        self.scans = []
        for seq in self.sequences:
            scan_dir = self.data_root / 'sequences' / seq / 'velodyne'
            if scan_dir.exists():
                for f in sorted(scan_dir.glob('*.bin')):
                    self.scans.append((str(f), seq, f.stem))
        print(f'[SemanticKITTI] Loaded {len(self.scans)} scans from seqs {self.sequences}')

    def load_scan(self, idx):
        path, seq, stem = self.scans[idx]
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        # Try load label
        label_path = self.labels_root / 'sequences' / seq / 'labels' / f'{stem}.label'
        if label_path.exists():
            labels = np.fromfile(label_path, dtype=np.uint32)
            labels = labels & 0xFFFF
            labels = np.vectorize(LEARNING_MAP.get, otypes=[np.int32])(labels)
        else:
            labels = np.zeros(len(points), dtype=np.int32)
        return {'points': points, 'labels': labels, 'seq': seq, 'frame': stem}

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, idx):
        return self.load_scan(idx)

# ===== nuScenes Data Loader =====
# Data from https://www.nuscenes.org/

class NuScenesMiniDataset:
    def __init__(self, data_root=None):
        if data_root is None:
            candidates = [
                Path(r"E:\Hyper-CAD-BEV-Experiments\data\nuscenes\v1.0-mini"),
                Path(r"D:\HyperCAD_BEV_2026\data\nuscenes\v1.0-mini"),
            ]
            for c in candidates:
                if c.exists():
                    data_root = c
                    break
            if data_root is None:
                data_root = Path("D:/HyperCAD_BEV_2026/data")
        self.root = Path(data_root)
        self.samples_lidar_dir = self.root / "samples" / "LIDAR_TOP"
        self.sweeps_lidar_dir = self.root / "sweeps" / "LIDAR_TOP"
        self._build_index()

    def _build_index(self):
        lidar_files = []
        for d in [self.samples_lidar_dir, self.sweeps_lidar_dir]:
            if d.exists():
                lidar_files.extend(sorted(d.glob("*.pcd.bin")))
        self.lidar_files = lidar_files
        print(f"[nuScenes] Found {len(self.lidar_files)} LiDAR .pcd.bin files")

    def load_lidar_sweep(self, path_or_idx):
        if isinstance(path_or_idx, int):
            path = self.lidar_files[path_or_idx]
        else:
            path = Path(path_or_idx)
        try:
            points = np.fromfile(str(path), dtype=np.float32).reshape(-1, 5)
            return points[:, :4]
        except Exception as e:
            print(f"  [nuScenes] WARN: {path}: {e}")
            return np.zeros((1, 4), dtype=np.float32)

    def __len__(self):
        return len(self.lidar_files)

    def __getitem__(self, idx):
        path = self.lidar_files[idx]
        points = self.load_lidar_sweep(idx)
        return {"points": points, "path": str(path), "source": "nuscenes"}

# ===== Unified Multi-Source Data Loader =====

class MultiSourceBEVDataset:
    def __init__(self, config=None):
        self.datasets = []
        self.dataset_weights = []
        # Load SemanticKITTI
        if Path(r'E:\Hyper-CAD-BEV-Experiments\data\semantickitti_official\dataset\sequences').exists() or Path('D:/HyperCAD_BEV_2026/data/semantickitti_lidar').exists():
            self.datasets.append(SemanticKITTIDataset())
            self.dataset_weights.append(0.6)
        # Load nuScenes
        if Path(r'E:\Hyper-CAD-BEV-Experiments\data\nuscenes\v1.0-mini\samples\LIDAR_TOP').exists() or Path('D:/HyperCAD_BEV_2026/data/sweeps').exists():
            self.datasets.append(NuScenesMiniDataset())
            self.dataset_weights.append(0.4)
        self.total = sum(len(d) for d in self.datasets)
        print(f'[MultiSourceBEV] Total: {self.total} samples from {len(self.datasets)} datasets')

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        # Simple round-robin
        ds_idx = 0
        local_idx = idx
        for i, ds in enumerate(self.datasets):
            if local_idx < len(ds):
                ds_idx = i
                break
            local_idx -= len(ds)
        return self.datasets[ds_idx][local_idx % len(self.datasets[ds_idx])]
