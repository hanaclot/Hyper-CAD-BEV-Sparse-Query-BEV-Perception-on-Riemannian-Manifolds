# -*- coding: utf-8 -*-
"""Phase 2: Comprehensive Download Script - Target 10GB+ real data"""
import urllib.request, json, time, zipfile, tarfile, shutil, sys
from pathlib import Path
from datetime import datetime

DATA_ROOT = Path(r"E:\Hyper-CAD-BEV-Experiments\data")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
LOG = []

def log(msg):
    t = datetime.now().strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    print(line)
    LOG.append(line)

def download(url, dest, desc, timeout=600, retries=2):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 1000:
        mb = dest.stat().st_size / 1e6
        log(f"  SKIP (exists {mb:.1f}MB): {desc}")
        return dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    for i in range(retries + 1):
        try:
            log(f"  DOWNLOAD [{i+1}/{retries+1}]: {desc}")
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            with open(dest, "wb") as f:
                f.write(data)
            mb = len(data) / 1e6
            log(f"  DONE: {mb:.1f}MB -> {dest.name}")
            return len(data)
        except Exception as e:
            log(f"  FAIL [{i+1}]: {str(e)[:80]}")
            if i < retries:
                time.sleep(10)
    return 0

def extract(zip_path, extract_dir):
    zip_path = Path(zip_path)
    extract_dir = Path(extract_dir)
    try:
        if zip_path.suffix == ".zip":
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
            log(f"  EXTRACTED: {zip_path.name} -> {extract_dir}")
            return True
        elif ".tar" in zip_path.suffixes or zip_path.suffix in [".tgz", ".gz"]:
            import tarfile
            with tarfile.open(zip_path, "r:*") as tf:
                tf.extractall(extract_dir)
            log(f"  EXTRACTED: {zip_path.name} -> {extract_dir}")
            return True
    except Exception as e:
        log(f"  EXTRACT FAIL: {str(e)[:80]}")
        return False

total_bytes = 0

# =============================================
# 1. SEMANTICKITTI (biggest source)
# =============================================
log("=" * 60)
log("SECTION 1: SemanticKITTI Downloads")
log("=" * 60)

sk = DATA_ROOT / "semantickitti_official"

# 1a. Labels (179MB)
sz = download("http://semantic-kitti.org/assets/data_odometry_labels.zip",
              sk / "labels.zip", "SemanticKITTI Labels")
total_bytes += sz

# 1b. Voxels (694MB)
sz = download("http://semantic-kitti.org/assets/data_odometry_voxels.zip",
              sk / "voxels.zip", "SemanticKITTI Voxels")
total_bytes += sz

# 1c. Voxels ALL (3.3GB) - KEY SOURCE
sz = download("http://semantic-kitti.org/assets/data_odometry_voxels_all.zip",
              sk / "voxels_all.zip", "SemanticKITTI All Voxels (3.3GB)")
total_bytes += sz

# 1d. Velodyne laser data ALL sequences (unknown size, large)
sz = download("http://www.cvlibs.net/download.php?file=data_odometry_velodyne.zip",
              sk / "velodyne_all.zip", "KITTI Odometry Velodyne ALL sequences")
total_bytes += sz

# 1e. Calibration
sz = download("http://www.cvlibs.net/download.php?file=data_odometry_calib.zip",
              sk / "calib.zip", "KITTI Odometry Calibration")
total_bytes += sz

# 1f. Poses
sz = download("http://www.cvlibs.net/download.php?file=data_odometry_poses.zip",
              sk / "poses.zip", "KITTI Odometry Poses")
total_bytes += sz

# =============================================
# 2. KITTI RAW DATA SUBSETS
# =============================================
log("=" * 60)
log("SECTION 2: KITTI Raw Data")
log("=" * 60)

kitti_raw = DATA_ROOT / "kitti_raw"
kitti_raw_drives = [
    "2011_09_26_drive_0001", "2011_09_26_drive_0002",
    "2011_09_26_drive_0005", "2011_09_26_drive_0009",
    "2011_09_26_drive_0011", "2011_09_26_drive_0013",
    "2011_09_26_drive_0014", "2011_09_26_drive_0015",
    "2011_09_26_drive_0018", "2011_09_26_drive_0019",
    "2011_09_26_drive_0020", "2011_09_26_drive_0022",
    "2011_09_26_drive_0023", "2011_09_26_drive_0027",
    "2011_09_26_drive_0028", "2011_09_26_drive_0029",
    "2011_09_26_drive_0032", "2011_09_26_drive_0035",
    "2011_09_26_drive_0036", "2011_09_26_drive_0039",
    "2011_09_26_drive_0046", "2011_09_26_drive_0048",
    "2011_09_26_drive_0051", "2011_09_26_drive_0052",
    "2011_09_26_drive_0056", "2011_09_26_drive_0057",
    "2011_09_26_drive_0059", "2011_09_26_drive_0060",
    "2011_09_26_drive_0061", "2011_09_26_drive_0064",
]

# Download first 8 drives (smaller size, enough for experiments)
for drive in kitti_raw_drives[:8]:
    url = f"https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/{drive}/{drive}_sync.zip"
    sz = download(url, kitti_raw / f"{drive}_sync.zip", f"KITTI Raw {drive}")
    total_bytes += sz
    time.sleep(3)

# KITTI raw calibration
sz = download("https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_calib.zip",
              kitti_raw / "2011_09_26_calib.zip", "KITTI Raw Calib")
total_bytes += sz

# =============================================
# 3. ARXIV PAPERS (for reference)
# =============================================
log("=" * 60)
log("SECTION 3: ArXiv Papers")
log("=" * 60)

arxiv = DATA_ROOT / "acquired" / "arxiv"
arxiv_ids = {
    "BEVFormer_2203.17270": "2203.17270",
    "SparseAD_2404.06892": "2404.06892",
    "EventCamera_1711.01458": "1711.01458",
    "LoihiFusion_2408.16096": "2408.16096",
    "WeatherRobustness_2206.09907": "2206.09907",
    "SparseBEV_2308.09244": "2308.09244",
    "BEVDet_2112.11790": "2112.11790",
    "Sparse4D_2311.11722": "2311.11722",
    "Petr3D_2203.05625": "2203.05625",
    "BEVDepth_2206.10092": "2206.10092",
    "PointPillars_1812.05784": "1812.05784",
    "NeuromorphicReview_2006.14567": "2006.14567",
    "RELLIS_2011.07717": "2011.07717",
    "TartanDrive_2204.04615": "2204.04615",
    "nuScenes_1903.11027": "1903.11027",
    "Waymo_1912.04838": "1912.04838",
    "KITTI_1204.4087": "1204.4087",
    "SemanticKITTI_1904.01416": "1904.01416",
}
for name, pid in arxiv_ids.items():
    url = f"https://arxiv.org/pdf/{pid}.pdf"
    sz = download(url, arxiv / f"{name}.pdf", f"ArXiv {name}")
    total_bytes += sz
    time.sleep(1)

# =============================================
# 4. GITHUB REPOS (as zip)
# =============================================
log("=" * 60)
log("SECTION 4: GitHub Repositories")
log("=" * 60)

github = DATA_ROOT / "github_repos"
repos = [
    ("unmannedlab/RELLIS-3D", "RELLIS-3D"),
    ("PRBonn/semantic-kitti-api", "SemanticKITTI-API"),
    ("fundamentalvision/BEVFormer", "BEVFormer"),
    ("nutonomy/nuscenes-devkit", "nuScenes-devkit"),
    ("waymo-research/waymo-open-dataset", "Waymo-Open-Dataset"),
    ("open-mmlab/mmdetection3d", "MMDetection3D"),
    ("mit-han-lab/torchsparse", "TorchSparse"),
    ("traveller59/second.pytorch", "SECOND-PointPillars"),
]
for repo, name in repos:
    for branch in ["main", "master"]:
        url = f"https://github.com/{repo}/archive/refs/heads/{branch}.zip"
        sz = download(url, github / f"{name}_{branch}.zip", f"GitHub {name}")
        if sz > 1000:
            break
    time.sleep(2)

# =============================================
# 5. TARTANDRIVE2 SAMPLE DATA
# =============================================
log("=" * 60)
log("SECTION 5: TartanDrive2")
log("=" * 60)

td2 = DATA_ROOT / "raw" / "tartandrive2"
# TartanDrive2 sample trajectories
td2_urls = [
    "https://github.com/castacks/TartanDrive2.0/archive/refs/heads/main.zip",
]
for url in td2_urls:
    sz = download(url, td2 / "td2_repo.zip", "TartanDrive2 Repo")
    total_bytes += sz

# =============================================
# 6. RELLIS-3D
# =============================================
log("=" * 60)
log("SECTION 6: RELLIS-3D")
log("=" * 60)

r3d = DATA_ROOT / "raw" / "rellis3d"
rellis_urls = [
    "https://github.com/unmannedlab/RELLIS-3D/archive/refs/heads/master.zip",
]
for url in rellis_urls:
    sz = download(url, r3d / "rellis3d_repo.zip", "RELLIS-3D Repo")
    total_bytes += sz

# =============================================
# 7. NUSCENES MINI (if accessible)
# =============================================
log("=" * 60)
log("SECTION 7: nuScenes Mini")
log("=" * 60)

ns = DATA_ROOT / "raw" / "nuscenes"
# nuScenes mini is ~4GB, publicly available
ns_url = "https://www.nuscenes.org/data/v1.0-mini.tgz"
sz = download(ns_url, ns / "v1.0-mini.tgz", "nuScenes Mini")
total_bytes += sz

# =============================================
# SUMMARY
# =============================================
log("=" * 60)
log(f"TOTAL DOWNLOADED: {total_bytes/1e9:.2f} GB")
log(f"TOTAL DOWNLOADED: {total_bytes/1e6:.1f} MB")
log("=" * 60)

log_path = DATA_ROOT / "processed" / f"download_summary_{datetime.now():%Y%m%d_%H%M%S}.json"
with open(log_path, "w", encoding="utf-8") as f:
    json.dump({"total_bytes": total_bytes, "total_gb": total_bytes/1e9, "log": LOG}, f, indent=2)
print(f"Summary saved to {log_path}")