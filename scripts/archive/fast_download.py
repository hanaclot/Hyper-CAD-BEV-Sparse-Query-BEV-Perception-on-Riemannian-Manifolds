#!/usr/bin/env python3
"""
Fast parallel downloader - focuses on large datasets
Uses concurrent.futures for parallel downloads
"""
import os, json, time, hashlib, ssl, sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
import urllib.error

ssl._create_default_https_context = ssl._create_unverified_context

E_DRIVE = r"E:\Hyper-CAD-BEV-Experiments"
DATA_DIR = os.path.join(E_DRIVE, "data")

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'

DOWNLOAD_TASKS = []

def add_task(url, dest, desc, priority=1):
    DOWNLOAD_TASKS.append({"url": url, "dest": dest, "desc": desc, "priority": priority})

def download_one(task):
    url, dest, desc = task["url"], task["dest"], task["desc"]
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    
    if os.path.exists(dest) and os.path.getsize(dest) > 10000:
        sz = os.path.getsize(dest)
        print(f"  [SKIP] {desc}: {sz/1e6:.1f}MB exists")
        return {"desc": desc, "size": sz, "status": "skip"}
    
    for attempt in range(2):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
                with open(dest, 'wb') as f:
                    f.write(data)
                sz = len(data)
                print(f"  [OK] {desc}: {sz/1e6:.1f}MB")
                return {"desc": desc, "size": sz, "status": "ok"}
        except Exception as e:
            if attempt == 1:
                print(f"  [FAIL] {desc}: {str(e)[:80]}")
                return {"desc": desc, "size": 0, "status": "fail", "error": str(e)[:100]}
            time.sleep(2)
    return {"desc": desc, "size": 0, "status": "fail"}

# Build task list
print("Building download task list...")

# SemanticKITTI - primary data source 
sk_base = "http://semantic-kitti.org/assets"
sk_dir = os.path.join(DATA_DIR, "semantickitti_official")
add_task(f"{sk_base}/data_odometry_voxels_all.zip", os.path.join(sk_dir, "voxels_all.zip"), "SK voxels ALL sequences", 1)
add_task(f"{sk_base}/data_odometry_calib.zip", os.path.join(sk_dir, "calib.zip"), "SK calibration", 2)
add_task(f"{sk_base}/data_odometry_poses.zip", os.path.join(sk_dir, "poses.zip"), "SK poses GT", 2)

# KITTI Odometry
kitti_base = "https://s3.eu-central-1.amazonaws.com/avg-kitti"
kitti_dir = os.path.join(DATA_DIR, "kitti")
# Try the main velodyne data file
add_task(f"{kitti_base}/data_odometry_velodyne.zip", os.path.join(kitti_dir, "odometry_velodyne.zip"), "KITTI Odometry Velodyne", 1)
add_task(f"{kitti_base}/data_odometry_calib.zip", os.path.join(kitti_dir, "odometry_calib.zip"), "KITTI Odometry Calib", 2)
add_task(f"{kitti_base}/data_odometry_gray.zip", os.path.join(kitti_dir, "odometry_gray.zip"), "KITTI Odometry Gray", 3)
add_task(f"{kitti_base}/data_object_image_2.zip", os.path.join(kitti_dir, "object_image_2.zip"), "KITTI Object Image2", 4)

# nuScenes mini - big file
nusc_dir = os.path.join(DATA_DIR, "nuscenes")
add_task("https://www.nuscenes.org/data/v1.0-mini.tgz", os.path.join(nusc_dir, "v1.0-mini.tgz"), "nuScenes mini", 1)

# GitHub repos - code bases
gh_dir = os.path.join(DATA_DIR, "github_repos")
for name, repo in [
    ("BEVFormer", "fundamentalvision/BEVFormer"),
    ("BEVDet", "HuangJunJie2017/BEVDet"),
    ("Sparse4D", "linxc06700/Sparse4D"),
    ("SparseBEV", "Megvii-BaseDetection/SparseBEV"),
    ("mmdetection3d", "open-mmlab/mmdetection3d"),
    ("OpenPCDet", "open-mmlab/OpenPCDet"),
    ("TartanDrive2", "castacks/TartanDrive2"),
]:
    add_task(f"https://github.com/{repo}/archive/refs/heads/main.zip", os.path.join(gh_dir, f"{name}.zip"), f"GitHub: {name}", 2)

# ArXiv papers
arxiv_dir = os.path.join(DATA_DIR, "arxiv_supplementary")
for aid, name in [
    ("2203.17270", "BEVFormer"), ("2404.06892", "SparseAD"),
    ("1711.01458", "EventCamera"), ("2408.16096", "LoihiFusion"),
    ("2206.09907", "WeatherOffroad"), ("2311.11722", "Sparse4D"),
    ("2308.09244", "SparseBEV"), ("2112.11790", "BEVDet"),
    ("2205.02833", "BEVerse"), ("2305.09910", "FB-BEV"),
    ("2211.12786", "PolarBEV"), ("2304.12345", "PETRv2"),
]:
    add_task(f"https://arxiv.org/pdf/{aid}.pdf", os.path.join(arxiv_dir, f"{aid}_{name}.pdf"), f"ArXiv: {name}", 3)

print(f"Total tasks: {len(DOWNLOAD_TASKS)}")

# Execute with thread pool
results = []
start = time.time()
with ThreadPoolExecutor(max_workers=6) as executor:
    futures = {executor.submit(download_one, t): t for t in DOWNLOAD_TASKS}
    for f in as_completed(futures):
        r = f.result()
        results.append(r)

total_bytes = sum(r["size"] for r in results)
ok_count = sum(1 for r in results if r["status"] == "ok")
skip_count = sum(1 for r in results if r["status"] == "skip")
fail_count = sum(1 for r in results if r["status"] == "fail")
elapsed = time.time() - start

print("\n" + "="*60)
print(f"DOWNLOAD COMPLETE: {elapsed:.0f}s")
print(f"  OK: {ok_count}, Skipped: {skip_count}, Failed: {fail_count}")
print(f"  Total new data: {total_bytes/1e6:.1f}MB")

# Calculate full data size
total_on_disk = 0
for root, dirs, files in os.walk(DATA_DIR):
    for f in files:
        fp = os.path.join(root, f)
        if os.path.exists(fp):
            total_on_disk += os.path.getsize(fp)
print(f"  Total data on disk: {total_on_disk/1e9:.2f}GB")

# Save summary
with open(os.path.join(DATA_DIR, "download_summary.json"), 'w') as f:
    json.dump({"results": results, "total_bytes": total_bytes, "elapsed": elapsed, "disk_total": total_on_disk}, f, indent=2)