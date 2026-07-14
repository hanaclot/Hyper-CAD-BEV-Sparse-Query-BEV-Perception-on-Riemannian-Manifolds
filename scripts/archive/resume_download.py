#!/usr/bin/env python3
"""Sequential downloader with resume support for large files"""
import os, sys, time, json, ssl, hashlib
from datetime import datetime
import urllib.request
import urllib.error

ssl._create_default_https_context = ssl._create_unverified_context
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

class ResumeDownloader:
    def __init__(self):
        self.log = []
        self.total_downloaded = 0
    
    def download(self, url, dest_path, desc):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        if os.path.exists(dest_path):
            existing = os.path.getsize(dest_path)
            if existing > 50000:
                print(f"  [EXISTS] {desc}: {existing/1e6:.1f}MB")
                return existing
        
        print(f"  [START] {desc}")
        print(f"  URL: {url[:100]}")
        
        try:
            req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
            with urllib.request.urlopen(req, timeout=300) as resp:
                total = int(resp.headers.get('Content-Length', 0))
                data_list = []
                downloaded = 0
                chunk_size = 1024 * 1024  # 1MB
                
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    data_list.append(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded * 100.0 / total
                        print(f"    {downloaded/1e6:.1f}/{total/1e6:.1f}MB ({pct:.0f}%)", end='\r')
                
                all_data = b''.join(data_list)
                with open(dest_path, 'wb') as f:
                    f.write(all_data)
                
                actual = os.path.getsize(dest_path)
                print(f"\n  [DONE] {desc}: {actual/1e6:.1f}MB")
                self.total_downloaded += actual
                return actual
                
        except Exception as e:
            print(f"\n  [FAIL] {desc}: {str(e)[:120]}")
            return 0

e = r"E:\Hyper-CAD-BEV-Experiments\data"

tasks = [
    # Priority 1: SemanticKITTI
    ("http://semantic-kitti.org/assets/data_odometry_voxels_all.zip", f"{e}\\semantickitti_official\\voxels_all.zip", "SK voxels all (3.3GB)"),
    ("http://semantic-kitti.org/assets/data_odometry_calib.zip", f"{e}\\semantickitti_official\\calib.zip", "SK calib"),
    ("http://semantic-kitti.org/assets/data_odometry_poses.zip", f"{e}\\semantickitti_official\\poses.zip", "SK poses"),
    
    # Priority 2: KITTI Odometry
    ("https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip", f"{e}\\kitti\\odom_calib.zip", "KITTI Odom Calib"),
    ("https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_gray.zip", f"{e}\\kitti\\odom_gray.zip", "KITTI Odom Gray"),
    
    # Priority 3: GitHub repos
    ("https://github.com/fundamentalvision/BEVFormer/archive/refs/heads/main.zip", f"{e}\\github_repos\\BEVFormer.zip", "BEVFormer repo"),
    ("https://github.com/open-mmlab/mmdetection3d/archive/refs/heads/main.zip", f"{e}\\github_repos\\mmdet3d.zip", "MMDetection3D"),
    ("https://github.com/open-mmlab/OpenPCDet/archive/refs/heads/master.zip", f"{e}\\github_repos\\OpenPCDet.zip", "OpenPCDet"),
    ("https://github.com/castacks/TartanDrive2/archive/refs/heads/main.zip", f"{e}\\github_repos\\TartanDrive2.zip", "TartanDrive2"),
    ("https://github.com/Megvii-BaseDetection/SparseBEV/archive/refs/heads/main.zip", f"{e}\\github_repos\\SparseBEV.zip", "SparseBEV"),
    
    # Priority 4: ArXiv
    ("https://arxiv.org/pdf/2203.17270.pdf", f"{e}\\arxiv_supplementary\\2203_BEVFormer.pdf", "BEVFormer paper"),
    ("https://arxiv.org/pdf/2404.06892.pdf", f"{e}\\arxiv_supplementary\\2404_SparseAD.pdf", "SparseAD paper"),
    ("https://arxiv.org/pdf/1711.01458.pdf", f"{e}\\arxiv_supplementary\\1711_EventCamera.pdf", "EventCamera paper"),
    ("https://arxiv.org/pdf/2206.09907.pdf", f"{e}\\arxiv_supplementary\\2206_Weather.pdf", "Weather paper"),
    ("https://arxiv.org/pdf/2408.16096.pdf", f"{e}\\arxiv_supplementary\\2408_LoihiFusion.pdf", "LoihiFusion paper"),
    ("https://arxiv.org/pdf/2311.11722.pdf", f"{e}\\arxiv_supplementary\\2311_Sparse4D.pdf", "Sparse4D paper"),
    ("https://arxiv.org/pdf/2308.09244.pdf", f"{e}\\arxiv_supplementary\\2308_SparseBEV.pdf", "SparseBEV paper"),
    ("https://arxiv.org/pdf/2112.11790.pdf", f"{e}\\arxiv_supplementary\\2112_BEVDet.pdf", "BEVDet paper"),
]

dler = ResumeDownloader()
start = time.time()

for url, dest, desc in tasks:
    try:
        dler.download(url, dest, desc)
    except Exception as ex:
        print(f"  [ERR] {desc}: {ex}")

# Summary
total_on_disk = sum(
    os.path.getsize(os.path.join(r,f)) 
    for r,_,fs in os.walk(e) for f in fs if os.path.exists(os.path.join(r,f))
)

elapsed = time.time() - start
print(f"\n{'='*50}")
print(f"DOWNLOAD COMPLETE ({elapsed:.0f}s)")
print(f"  New downloaded: {dler.total_downloaded/1e9:.2f}GB")
print(f"  Total on disk: {total_on_disk/1e9:.2f}GB")