#!/usr/bin/env python3
"""Phase 2: Multi-source data scraping pipeline for Hyper-CAD-BEV experiments."""
import os, sys, json, time, hashlib, re, gzip, shutil
import requests
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import subprocess

# ========== CONFIG ==========
OUTPUT_BASE = r"D:\HyperCAD_BEV_2026\data\scraped_new"
E_BASE = r"E:\Hyper-CAD-BEV-Experiments\data"
os.makedirs(OUTPUT_BASE, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)
LOG = {"started": datetime.now().isoformat(), "downloads": [], "errors": []}

def download_file(url, dest, desc="", chunk_size=8192, timeout=120, max_retries=3):
    """Download file with progress tracking and retries."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        sz = os.path.getsize(dest)
        if sz > 1024:
            print(f"  [SKIP] {os.path.basename(dest)} exists ({sz/1024/1024:.1f}MB)")
            LOG["downloads"].append({"url": url, "dest": dest, "size": sz, "status": "skipped"})
            return dest
    for attempt in range(max_retries):
        try:
            print(f"  [DOWNLOAD] {desc} -> {os.path.basename(dest)} (attempt {attempt+1})")
            r = SESSION.get(url, stream=True, timeout=timeout)
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
            actual_size = os.path.getsize(dest)
            print(f"  [OK] {os.path.basename(dest)}: {actual_size/1024/1024:.1f}MB")
            LOG["downloads"].append({"url": url, "dest": dest, "size": actual_size, "status": "success"})
            return dest
        except Exception as e:
            print(f"  [ERR] attempt {attempt+1}: {e}")
            time.sleep(2 ** attempt)
    LOG["errors"].append({"url": url, "error": str(e)})
    return None

# ============ SOURCE 1: SemanticKITTI ============
def scrape_semantickitti():
    """Download SemanticKITTI official data."""
    print("\n" + "="*60)
    print("SOURCE 1: SemanticKITTI (semantic-kitti.org)")
    print("="*60)
    base = os.path.join(OUTPUT_BASE, "semantickitti")
    os.makedirs(base, exist_ok=True)
    # Main data URLs from semantic-kitti.org
    urls = {
        "velodyne": "https://www.cvlibs.net/download.php?file=data_odometry_velodyne.zip",
        "calib": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip",
        "labels": "https://www.cvlibs.net/download.php?file=data_odometry_labels.zip",
        "poses": "https://www.cvlibs.net/download.php?file=data_odometry_poses.zip",
        "voxels": "https://www.cvlibs.net/download.php?file=data_odometry_voxels.zip",
    }
    for name, url in urls.items():
        dest = os.path.join(base, f"semantickitti_{name}.zip")
        download_file(url, dest, f"SemanticKITTI {name}")

# ============ SOURCE 2: KITTI Raw ============
def scrape_kitti_raw():
    """Download KITTI raw data synced+rectified."""
    print("\n" + "="*60)
    print("SOURCE 2: KITTI Raw Data (cvlibs.net)")
    print("="*60)
    base = os.path.join(OUTPUT_BASE, "kitti_raw")
    os.makedirs(base, exist_ok=True)
    # KITTI raw data date sequences (residential and road categories for rural-like scenes)
    dates = [
        "2011_09_26",  # Residential
        "2011_10_03",  # Residential + road
        "2011_09_28",  # Road
        "2011_09_29",  # Road  
        "2011_09_30",  # Campus
        "2011_10_01",  # Road + city
    ]
    for date in dates:
        url_sync = f"https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/{date}/{date}_drive.zip"
        dest = os.path.join(base, f"{date}_sync.zip")
        download_file(url_sync, dest, f"KITTI raw {date}")
        url_calib = f"https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/{date}/{date}_calib.zip"
        dest_calib = os.path.join(base, f"{date}_calib.zip")
        download_file(url_calib, dest_calib, f"KITTI calib {date}")

# ============ SOURCE 3: RELLIS-3D ============
def scrape_rellis3d():
    """Download RELLIS-3D dataset and annotations."""
    print("\n" + "="*60)
    print("SOURCE 3: RELLIS-3D (unmannedlab/RELLIS-3D)")
    print("="*60)
    base = os.path.join(OUTPUT_BASE, "rellis3d")
    os.makedirs(base, exist_ok=True)
    # RELLIS-3D dataset download
    rellis_urls = {
        "dataset_part1": "https://drive.google.com/uc?export=download&id=1wKdE5nrOXWIlcaONB5s2YlHGTQSNs24e",
        "dataset_part2": "https://drive.google.com/uc?export=download&id=1Xp6YAEUP7BgV4xCSW6J6B5lNhD0MyeHr",
        "repo_zip": "https://github.com/unmannedlab/RELLIS-3D/archive/refs/heads/master.zip",
    }
    for name, url in rellis_urls.items():
        dest = os.path.join(base, f"rellis3d_{name}.zip")
        download_file(url, dest, f"RELLIS-3D {name}")

    # Also clone the GitHub repo
    repo_dir = os.path.join(base, "RELLIS-3D")
    if not os.path.exists(repo_dir):
        print(f"  [CLONE] RELLIS-3D repo")
        subprocess.run(["git", "clone", "https://github.com/unmannedlab/RELLIS-3D.git", repo_dir], 
                       capture_output=True, timeout=120)

# ============ SOURCE 4: TartanDrive 2 ============
def scrape_tartandrive2():
    """Download TartanDrive 2 data from HuggingFace."""
    print("\n" + "="*60)
    print("SOURCE 4: TartanDrive 2 (theairlab)")
    print("="*60)
    base = os.path.join(OUTPUT_BASE, "tartandrive2")
    os.makedirs(base, exist_ok=True)
    # TartanDrive2 on HuggingFace
    tartan_urls = {
        "README": "https://huggingface.co/datasets/theairlab/TartanDrive2/resolve/main/README.md",
    }
    for name, url in tartan_urls.items():
        dest = os.path.join(base, f"tartandrive2_{name}")
        download_file(url, dest, f"TartanDrive2 {name}")

# ============ SOURCE 5: Event Camera Datasets ============
def scrape_event_camera():
    """Download event camera datasets for driving."""
    print("\n" + "="*60)
    print("SOURCE 5: Event Camera Driving Datasets")
    print("="*60)
    base = os.path.join(OUTPUT_BASE, "event_camera")
    os.makedirs(base, exist_ok=True)
    # DDD17, MVSEC, DSEC datasets
    event_urls = {
        "ddd17_paper": "https://arxiv.org/pdf/1711.01458.pdf",
        "dsec_sample": "https://dsec.ifi.uzh.ch/download/",
    }
    for name, url in event_urls.items():
        dest = os.path.join(base, f"event_{name}")
        download_file(url, dest, f"Event camera {name}")

# ============ SOURCE 6: nuScenes extended ============
def scrape_nuscenes_extended():
    """Verify and catalog nuScenes data."""
    print("\n" + "="*60)
    print("SOURCE 6: nuScenes (Check & Catalog)")
    print("="*60)
    base = os.path.join(OUTPUT_BASE, "nuscenes")
    os.makedirs(base, exist_ok=True)
    # Download additional nuScenes related data
    nuscenes_urls = {
        "devkit": "https://github.com/nutonomy/nuscenes-devkit/archive/refs/heads/master.zip",
        "map_expansion": "https://www.nuscenes.org/data/v1.0-mini.tgz",
    }
    for name, url in nuscenes_urls.items():
        dest = os.path.join(base, f"nuscenes_{name}")
        download_file(url, dest, f"nuScenes {name}")

# ============ SOURCE 7: Additional Benchmark Datasets ============
def scrape_additional():
    """Download additional BEV/autonomous driving benchmark datasets."""
    print("\n" + "="*60)
    print("SOURCE 7: Additional Benchmark Datasets")
    print("="*60)
    base = os.path.join(OUTPUT_BASE, "additional")
    os.makedirs(base, exist_ok=True)
    # BEVDet, BEVFormer code for benchmarking
    additional = {
        "bevformer_code": "https://github.com/fundamentalvision/BEVFormer/archive/refs/heads/master.zip",
        "bevdet_code": "https://github.com/HuangJunJie2017/BEVDet/archive/refs/heads/dev2.0.zip",
        "bevdepth_code": "https://github.com/Megvii-BaseDetection/BEVDepth/archive/refs/heads/main.zip",
        "sparsebev_code": "https://github.com/MCG-NJU/SparseBEV/archive/refs/heads/main.zip",
        "openpcdet_code": "https://github.com/open-mmlab/OpenPCDet/archive/refs/heads/master.zip",
        "mmdet3d_code": "https://github.com/open-mmlab/mmdetection3d/archive/refs/heads/main.zip",
        "pointpillars_code": "https://github.com/nutonomy/second.pytorch/archive/refs/heads/master.zip",
        "waymo_kitti_converter": "https://github.com/WeberZhu/Waymo-Kitti-Converter/archive/refs/heads/master.zip",
    }
    for name, url in additional.items():
        dest = os.path.join(base, f"{name}.zip")
        download_file(url, dest, f"Additional {name}")

# ============ SOURCE 8: Weather/Lighting papers & data ============
def scrape_weather_papers():
    """Scrape weather and illumination papers."""
    print("\n" + "="*60)
    print("SOURCE 8: Weather & Illumination Papers")
    print("="*60)
    base = os.path.join(OUTPUT_BASE, "weather_illumination")
    os.makedirs(base, exist_ok=True)
    weather_urls = {
        "2206.09907": "https://arxiv.org/pdf/2206.09907.pdf",
        "offroad_dataset": "https://arxiv.org/abs/2206.09907",
        "dark_zone": "https://arxiv.org/abs/2304.10250",
        "robust_bev": "https://arxiv.org/abs/2303.08606",
    }
    for name, url in weather_urls.items():
        if url.endswith('.pdf'):
            dest = os.path.join(base, f"{name}.pdf")
        else:
            dest = os.path.join(base, f"{name}.html")
        download_file(url, dest, f"Weather paper {name}")

# ============ MAIN ============
if __name__ == "__main__":
    print("="*60)
    print("Hyper-CAD-BEV: Multi-source Data Scraping Pipeline")
    print(f"Output: {OUTPUT_BASE}")
    print("="*60)
    
    sources = [
        ("semantickitti", scrape_semantickitti),
        ("kitti_raw", scrape_kitti_raw),
        ("rellis3d", scrape_rellis3d),
        ("tartandrive2", scrape_tartandrive2),
        ("event_camera", scrape_event_camera),
        ("nuscenes", scrape_nuscenes_extended),
        ("additional", scrape_additional),
        ("weather", scrape_weather_papers),
    ]
    
    for name, func in sources:
        try:
            func()
        except Exception as e:
            print(f"  [FATAL] {name}: {e}")
            LOG["errors"].append({"source": name, "error": str(e)})
    
    # Save log
    log_path = os.path.join(OUTPUT_BASE, f"scrape_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(log_path, "w") as f:
        json.dump(LOG, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"SCRAPING COMPLETE. Log: {log_path}")
    print(f"Downloads: {len(LOG['downloads'])}, Errors: {len(LOG['errors'])}")
    total_mb = sum(d.get("size", 0) for d in LOG["downloads"]) / 1024 / 1024
    print(f"Total downloaded: {total_mb:.1f} MB")
