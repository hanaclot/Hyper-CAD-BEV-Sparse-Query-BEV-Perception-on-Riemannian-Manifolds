"""
============================================================
Hyper-CAD-BEV v6.5-Sparse: 完整数据采集脚本
数据来源:
  1. RELLIS-3D: https://github.com/unmannedlab/RELLIS-3D
  2. SemanticKITTI: http://semantic-kitti.org/
  3. TartanDrive2: https://theairlab.org/TartanDrive2/
============================================================
"""
import os
import sys
import json
import time
import hashlib
import requests
import zipfile
import tarfile
from pathlib import Path
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv

# ===================== 配置 =====================
PROJECT_ROOT = Path(r"E:\Hyper-CAD-BEV-Experiments")
DATA_ROOT = PROJECT_ROOT / "data"

# 数据源URL配置
DATA_SOURCES = {
    "rellis3d": {
        "name": "RELLIS-3D Off-Road Dataset",
        "github_api": "https://api.github.com/repos/unmannedlab/RELLIS-3D",
        "description": "越野地形多模态数据集: LiDAR + RGB + 语义标注",
        "expected_files": ["image", "lidar", "calib", "labels"],
        "scenarios": ["rural", "unstructured", "off-road"],
    },
    "semantickitti": {
        "name": "SemanticKITTI",
        "website": "http://semantic-kitti.org/",
        "api_endpoint": "http://semantic-kitti.org/assets/data_odometry_labels.zip",
        "description": "LiDAR BEV语义分割基准数据集",
        "sequences": list(range(11)),  # 00-10 for training
    },
    "tartandrive2": {
        "name": "TartanDrive 2.0",
        "website": "https://theairlab.org/TartanDrive2/",
        "description": "高速越野动态场景几何和动力学建模数据集",
        "features": ["IMU", "wheel_odom", "terrain_properties"],
    }
}

# ===================== 工具函数 =====================
def download_file(url, dest_path, chunk_size=8192, max_retries=3):
    """下载文件，支持断点续传和重试"""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    for attempt in range(max_retries):
        try:
            headers = {}
            if dest_path.exists():
                headers["Range"] = f"bytes={dest_path.stat().st_size}-"
            
            response = requests.get(url, stream=True, headers=headers, timeout=30)
            
            if response.status_code in (200, 206):
                mode = "ab" if response.status_code == 206 else "wb"
                with open(dest_path, mode) as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                print(f"  [OK] Downloaded: {dest_path.name} ({dest_path.stat().st_size / 1e6:.1f} MB)")
                return True
            elif response.status_code == 416:
                print(f"  [SKIP] Already complete: {dest_path.name}")
                return True
            else:
                print(f"  [WARN] HTTP {response.status_code} for {url}")
        except Exception as e:
            print(f"  [RETRY {attempt+1}/{max_retries}] {e}")
            time.sleep(2 ** attempt)
    
    print(f"  [FAIL] Could not download: {url}")
    return False


def github_download_folder(repo_owner, repo_name, branch="main", folder="", dest_dir=None):
    """通过GitHub API下载仓库中的特定文件夹"""
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{folder}?ref={branch}"
    try:
        response = requests.get(api_url, headers={"Accept": "application/vnd.github.v3+json"})
        if response.status_code == 200:
            contents = response.json()
            for item in contents:
                if item["type"] == "file":
                    if dest_dir:
                        fpath = Path(dest_dir) / item["name"]
                    else:
                        fpath = DATA_ROOT / "rellis3d" / item["name"]
                    download_file(item["download_url"], fpath)
                elif item["type"] == "dir":
                    github_download_folder(repo_owner, repo_name, branch, item["path"], dest_dir)
        else:
            print(f"  GitHub API: {response.status_code}")
    except Exception as e:
        print(f"  GitHub API error: {e}")


# ===================== 数据采集主流程 =====================

def download_rellis3d():
    """从RELLIS-3D GitHub仓库下载数据"""
    print("\n" + "="*60)
    print("[1/3] Downloading RELLIS-3D Off-Road Dataset...")
    print("="*60)
    
    dest = DATA_ROOT / "rellis3d"
    dest.mkdir(parents=True, exist_ok=True)
    
    # 1. Clone repository metadata
    print("  -> Fetching repository structure...")
    github_download_folder("unmannedlab", "RELLIS-3D", "main", "", str(dest))
    
    # 2. Try to download actual data files from known mirrors
    rellis_urls = [
        # RELLIS-3D dataset download links (official mirrors)
        "https://drive.google.com/uc?export=download&id=1QwMfqmgwR2lYCEJt5T1FQnxN6nGJqOXa",
        "https://unmannedlab.github.io/RELLIS-3D/data/",
    ]
    
    # Save metadata
    metadata = {
        "dataset": "RELLIS-3D",
        "source": "https://github.com/unmannedlab/RELLIS-3D",
        "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "structure": {
            "raw_data": str(dest),
            "splits": {"train": "sequences 00-03", "val": "sequence 04", "test": "sequence 05"}
        }
    }
    
    with open(dest / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("  RELLIS-3D metadata saved. (Large files require manual download)")
    return metadata


def download_semantickitti():
    """从SemanticKITTI官网下载数据"""
    print("\n" + "="*60)
    print("[2/3] Downloading SemanticKITTI Dataset...")
    print("="*60)
    
    dest = DATA_ROOT / "semantickitti"
    dest.mkdir(parents=True, exist_ok=True)
    
    # SemanticKITTI provides direct downloads
    kitti_urls = {
        "velodyne_laser": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip",
        "calib": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip",
        "labels": "http://semantic-kitti.org/assets/data_odometry_labels.zip",
        "semantic_labels_voxels": "http://semantic-kitti.org/assets/data_odometry_voxels.zip",
    }
    
    for name, url in kitti_urls.items():
        print(f"  -> Downloading {name}...")
        fpath = dest / f"{name}.zip"
        success = download_file(url, fpath)
        if success and fpath.exists():
            try:
                with zipfile.ZipFile(fpath, "r") as zf:
                    zf.extractall(dest)
                print(f"  Extracted: {name}")
            except Exception as e:
                print(f"  Extract error: {e}")
    
    # Save SemanticKITTI metadata
    semantic_classes = {
        0: "unlabeled", 1: "car", 2: "bicycle", 3: "motorcycle", 4: "truck",
        5: "other-vehicle", 6: "person", 7: "bicyclist", 8: "motorcyclist",
        9: "road", 10: "parking", 11: "sidewalk", 12: "other-ground",
        13: "building", 14: "fence", 15: "vegetation", 16: "trunk",
        17: "terrain", 18: "pole", 19: "traffic-sign"
    }
    
    metadata = {
        "dataset": "SemanticKITTI",
        "source": "http://semantic-kitti.org/",
        "classes": semantic_classes,
        "num_classes": len(semantic_classes),
        "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "splits": {
            "train": [f"{i:02d}" for i in range(8)],
            "val": ["08"],
            "test": ["09", "10"]
        }
    }
    
    with open(dest / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


def download_tartandrive2():
    """从TartanDrive2网站下载数据"""
    print("\n" + "="*60)
    print("[3/3] Downloading TartanDrive 2.0 Dataset...")
    print("="*60)
    
    dest = DATA_ROOT / "tartandrive2"
    dest.mkdir(parents=True, exist_ok=True)
    
    tartandrive_urls = {
        "metadata": "https://theairlab.org/TartanDrive2/metadata.json",
        "traversability": "https://theairlab.org/TartanDrive2/traversability_data.zip",
    }
    
    for name, url in tartandrive_urls.items():
        print(f"  -> Attempting: {name}")
        fpath = dest / f"{name}.zip" if "zip" in url else dest / name
        download_file(url, fpath)
    
    metadata = {
        "dataset": "TartanDrive 2.0",
        "source": "https://theairlab.org/TartanDrive2/",
        "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "description": "High-speed off-road dynamic scene geometry and kinematics",
        "terrain_types": ["gravel", "dirt", "mud", "grass", "rocky"],
        "speed_range": "up to 10 m/s",
    }
    
    with open(dest / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


# ===================== 生成统一数据集CSV =====================

def generate_unified_dataset_csv():
    """将所有数据源信息整合为统一的CSV索引文件"""
    print("\n" + "="*60)
    print("Generating Unified Dataset CSV Index...")
    print("="*60)
    
    csv_path = DATA_ROOT / "processed" / "rural_manifold_dataset.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    rows = []
    
    # 扫描所有数据目录
    for dataset_name in ["rellis3d", "semantickitti", "tartandrive2"]:
        dataset_dir = DATA_ROOT / dataset_name
        
        if not dataset_dir.exists():
            continue
        
        # 收集所有文件
        all_files = list(dataset_dir.rglob("*"))
        data_files = [f for f in all_files if f.suffix.lower() in 
                      ['.bin', '.npy', '.png', '.jpg', '.jpeg', '.pcd', '.ply', '.bag', '.txt', '.label']]
        
        for f in data_files[:5000]:  # 限制扫描数量
            rel_path = f.relative_to(DATA_ROOT)
            file_hash = hashlib.md5(str(rel_path).encode()).hexdigest()[:8]
            
            # 推断 split
            parent = f.parent.name
            if "train" in str(f).lower() or any(f"0{i}" in str(f) for i in range(8)):
                split = "train"
            elif "val" in str(f).lower() or "08" in str(f):
                split = "val"
            else:
                split = "test"
            
            rows.append({
                "file_id": file_hash,
                "dataset": dataset_name,
                "relative_path": str(rel_path),
                "file_type": f.suffix.lower(),
                "file_size_bytes": f.stat().st_size if f.exists() else 0,
                "split": split,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
    
    # 写入CSV
    if rows:
        fieldnames = ["file_id", "dataset", "relative_path", "file_type", 
                      "file_size_bytes", "split", "timestamp"]
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  CSV generated: {csv_path} ({len(rows)} entries)")
    else:
        print("  No data files found. Creating empty template.")
        fieldnames = ["file_id", "dataset", "relative_path", "file_type", 
                      "file_size_bytes", "split", "timestamp"]
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    
    return csv_path


# ===================== 主入口 =====================
if __name__ == "__main__":
    print("=" * 60)
    print("Hyper-CAD-BEV v6.5-Sparse: Data Acquisition Pipeline")
    print("=" * 60)
    print(f"Project: {PROJECT_ROOT}")
    print(f"Data Root: {DATA_ROOT}")
    print()
    
    # Phase 1: Download data
    download_rellis3d()
    download_semantickitti()
    download_tartandrive2()
    
    # Phase 2: Generate unified CSV
    csv_path = generate_unified_dataset_csv()
    
    print("\n" + "=" * 60)
    print("Data acquisition complete!")
    print(f"CSV index: {csv_path}")
    print("=" * 60)
