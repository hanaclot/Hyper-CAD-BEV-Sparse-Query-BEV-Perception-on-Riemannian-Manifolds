# -*- coding: utf-8 -*-
"""Fast parallel downloader for small files + staged large files"""
import urllib.request, json, time, os, threading
from pathlib import Path
from datetime import datetime

DATA_ROOT = Path(r"E:\Hyper-CAD-BEV-Experiments\data")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
LOG = []
lock = threading.Lock()

def log(msg):
    t = datetime.now().strftime("%H:%M:%S")
    with lock:
        print(f"[{t}] {msg}")
        LOG.append(f"[{t}] {msg}")

def download_file(url, dest, desc, timeout=120):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 1000:
        mb = dest.stat().st_size / 1e6
        log(f"SKIP {desc}: exists {mb:.1f}MB")
        return dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            total_size = r.getheader("Content-Length")
            data = b""
            chunk_size = 128 * 1024  # 128KB chunks
            while True:
                chunk = r.read(chunk_size)
                if not chunk:
                    break
                data += chunk
        with open(dest, "wb") as f:
            f.write(data)
        mb = len(data) / 1e6
        log(f"DONE {desc}: {mb:.1f}MB")
        return len(data)
    except Exception as e:
        log(f"FAIL {desc}: {str(e)[:80]}")
        return 0

# PHASE 1: All ArXiv papers concurrently (small files ~0.5-2MB each)
log("="*50)
log("PHASE 1: ArXiv Papers (small, fast)")

arxiv = DATA_ROOT / "acquired" / "arxiv"
arxiv_list = [
    ("https://arxiv.org/pdf/2203.17270.pdf", "BEVFormer_2203.17270.pdf", "BEVFormer"),
    ("https://arxiv.org/pdf/2404.06892.pdf", "SparseAD_2404.06892.pdf", "SparseAD"),
    ("https://arxiv.org/pdf/1711.01458.pdf", "EventCamera_1711.01458.pdf", "EventCamera"),
    ("https://arxiv.org/pdf/2408.16096.pdf", "LoihiFusion_2408.16096.pdf", "LoihiFusion"),
    ("https://arxiv.org/pdf/2206.09907.pdf", "Weather_2206.09907.pdf", "Weather"),
    ("https://arxiv.org/pdf/2308.09244.pdf", "SparseBEV_2308.09244.pdf", "SparseBEV"),
    ("https://arxiv.org/pdf/2112.11790.pdf", "BEVDet_2112.11790.pdf", "BEVDet"),
    ("https://arxiv.org/pdf/2311.11722.pdf", "Sparse4D_2311.11722.pdf", "Sparse4D"),
    ("https://arxiv.org/pdf/2203.05625.pdf", "Petr3D_2203.05625.pdf", "Petr3D"),
    ("https://arxiv.org/pdf/2206.10092.pdf", "BEVDepth_2206.10092.pdf", "BEVDepth"),
    ("https://arxiv.org/pdf/1812.05784.pdf", "PointPillars_1812.05784.pdf", "PointPillars"),
    ("https://arxiv.org/pdf/1904.01416.pdf", "SemanticKITTI_1904.01416.pdf", "SK_paper"),
    ("https://arxiv.org/pdf/2011.07717.pdf", "RELLIS3D_2011.07717.pdf", "RELLIS3D_paper"),
    ("https://arxiv.org/pdf/2204.04615.pdf", "TartanDrive_2204.04615.pdf", "TartanDrive"),
    ("https://arxiv.org/pdf/1903.11027.pdf", "nuScenes_1903.11027.pdf", "nuScenes"),
    ("https://arxiv.org/pdf/1204.4087.pdf", "KITTI_1204.4087.pdf", "KITTI_original"),
]

total = 0
for url, fname, desc in arxiv_list:
    sz = download_file(url, arxiv / fname, desc)
    total += sz
    time.sleep(0.3)

log(f"ArXiv total: {total/1e6:.1f}MB")

# PHASE 2: GitHub repos (fast, small zips)
log("="*50)
log("PHASE 2: GitHub Repos")

github = DATA_ROOT / "github_repos"
github_repos = [
    ("unmannedlab/RELLIS-3D", "RELLIS-3D"),
    ("PRBonn/semantic-kitti-api", "SK-API"),
    ("fundamentalvision/BEVFormer", "BEVFormer_code"),
    ("nutonomy/nuscenes-devkit", "nuScenes-devkit"),
    ("open-mmlab/mmdetection3d", "MMDetection3D"),
    ("mit-han-lab/torchsparse", "TorchSparse"),
    ("traveller59/second.pytorch", "SECOND"),
    ("open-mmlab/OpenPCDet", "OpenPCDet"),
    ("HorizonRobotics/Sparse4D", "Sparse4D_code"),
    ("Megvii-BaseDetection/BEVDepth", "BEVDepth_code"),
    ("HuangJunJie2017/BEVDet", "BEVDet_code"),
]

for repo, name in github_repos:
    for branch in ["main", "master"]:
        url = f"https://api.github.com/repos/{repo}/zipball/{branch}"
        dest = github / f"{name}.zip"
        sz = download_file(url, dest, f"GitHub {name}", timeout=60)
        if sz > 500:
            total += sz
            break
        time.sleep(1)
    time.sleep(1)

log(f"ArXiv + GitHub total: {total/1e6:.1f}MB")

# Save
with open(DATA_ROOT / "processed" / "download_phase1_log.json", "w") as f:
    json.dump({"total_bytes": total, "log": LOG}, f, indent=2)
log(f"PHASE 1+2 complete: {total/1e6:.1f}MB")