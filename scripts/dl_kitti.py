"""Targeted KITTI/SemanticKITTI data downloader for Hyper-CAD-BEV."""
import requests, os, time, json, sys
from datetime import datetime

OUT = r"D:\HyperCAD_BEV_2026\data\scraped_new"
os.makedirs(OUT, exist_ok=True)
LOG = {"started": datetime.now().isoformat(), "items": []}
S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0"})

def dl(url, dest, desc="", timeout=300):
    d = os.path.dirname(dest)
    os.makedirs(d, exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        sz = os.path.getsize(dest)
        print(f"SKIP {dest} ({sz/1024/1024:.0f}MB)")
        return dest
    print(f"DOWNLOAD {desc} -> {os.path.basename(dest)}")
    for i in range(3):
        try:
            r = S.get(url, stream=True, timeout=timeout)
            if r.status_code == 200:
                with open(dest + ".part", "wb") as f:
                    for c in r.iter_content(8192):
                        if c: f.write(c)
                os.rename(dest + ".part", dest)
                sz = os.path.getsize(dest)
                print(f"  OK {sz/1024/1024:.0f}MB")
                LOG["items"].append({"file": os.path.basename(dest), "size": sz, "status": "ok"})
                return dest
            else:
                print(f"  HTTP {r.status_code}")
        except Exception as e:
            print(f"  err {i}: {str(e)[:80]}")
            time.sleep(5)
    LOG["items"].append({"file": os.path.basename(dest), "status": "failed"})
    return None

# === KITTI Odometry (SemanticKITTI base) ===
print("="*60)
print("KITTI Odometry Data")
print("="*60)
kitti_base = os.path.join(OUT, "kitti_odometry")

# Small files first (calibration, poses, devkit)
dl("https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip",
   os.path.join(kitti_base, "data_odometry_calib.zip"), "odometry calib")
dl("https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip",
   os.path.join(kitti_base, "data_odometry_poses.zip"), "odometry poses")
dl("https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_odometry.zip",
   os.path.join(kitti_base, "devkit_odometry.zip"), "devkit")

# SemanticKITTI labels (small ~180MB) 
dl("https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_labels.zip",
   os.path.join(kitti_base, "data_odometry_labels.zip"), "odometry labels")

# Velodyne laser data - try getting it
dl("https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip",
   os.path.join(kitti_base, "data_odometry_velodyne.zip"), "odometry velodyne")

# === KITTI Raw Data - specific sequences ===
print("="*60)
print("KITTI Raw Data (selected sequences)")
print("="*60)
raw_base = os.path.join(OUT, "kitti_raw")

# Residential area sequences - good for rural scenes
raw_dates = ["2011_09_26", "2011_10_03", "2011_09_28"]
for date in raw_dates:
    # Download sync data for each date
    url = f"https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/{date}/{date}_drive.zip"
    dl(url, os.path.join(raw_base, f"{date}_sync.zip"), f"KITTI raw {date}")
    
    # Calibration
    url_calib = f"https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/{date}/{date}_calib.zip"
    dl(url_calib, os.path.join(raw_base, f"{date}_calib.zip"), f"KITTI calib {date}")

# === GitHub Repos ===
print("="*60)
print("GitHub Repositories")
print("="*60)
gh_base = os.path.join(OUT, "github")
repos = {
    "RELLIS-3D": "https://github.com/unmannedlab/RELLIS-3D/archive/refs/heads/master.zip",
    "SemanticKITTI-API": "https://github.com/PRBonn/semantic-kitti-api/archive/refs/heads/master.zip",
    "BEVFormer": "https://github.com/fundamentalvision/BEVFormer/archive/refs/heads/master.zip",
    "OpenPCDet": "https://github.com/open-mmlab/OpenPCDet/archive/refs/heads/master.zip",
    "mmdetection3d": "https://github.com/open-mmlab/mmdetection3d/archive/refs/heads/main.zip",
    "Sparse4D": "https://github.com/HorizonRobotics/Sparse4D/archive/refs/heads/main.zip",
    "BEVDet": "https://github.com/HuangJunJie2017/BEVDet/archive/refs/heads/dev2.0.zip",
    "nuscenes-devkit": "https://github.com/nutonomy/nuscenes-devkit/archive/refs/heads/master.zip",
}
for name, url in repos.items():
    dl(url, os.path.join(gh_base, f"{name}.zip"), f"GitHub {name}")

# === Arxiv Papers ===
print("="*60)
print("Arxiv Papers")
print("="*60)
arxiv_base = os.path.join(OUT, "arxiv")
papers = {
    "2203.17270_BEVFormer": "https://arxiv.org/pdf/2203.17270.pdf",
    "1711.01458_EventCamera": "https://arxiv.org/pdf/1711.01458.pdf",
    "2404.06892_SparseAD": "https://arxiv.org/pdf/2404.06892.pdf",
    "2408.16096_LoihiFusion": "https://arxiv.org/pdf/2408.16096.pdf",
    "2206.09907_WeatherOffroad": "https://arxiv.org/pdf/2206.09907.pdf",
    "2311.11722_Sparse4D": "https://arxiv.org/pdf/2311.11722.pdf",
    "2308.09244_SparseBEV": "https://arxiv.org/pdf/2308.09244.pdf",
    "2112.11790_BEVDet": "https://arxiv.org/pdf/2112.11790.pdf",
    "1905.01235_PointPillars": "https://arxiv.org/pdf/1905.01235.pdf",
    "2106.12978_NeRF": "https://arxiv.org/pdf/2106.12978.pdf",
    "2205.15016_RiemannianDL": "https://arxiv.org/pdf/2205.15016.pdf",
}
for name, url in papers.items():
    dl(url, os.path.join(arxiv_base, f"{name}.pdf"), f"Arxiv {name}")

# === Sensor-specific data ===
print("="*60)
print("Sensor-specific data")
print("="*60)
sensor_base = os.path.join(OUT, "sensor_data")
# Download LiDAR sample data and event camera samples
sensor_urls = {
    "event_camera_survey": "https://arxiv.org/pdf/1912.08432.pdf",
    "lidar_odometry_benchmark": "https://arxiv.org/pdf/1904.01669.pdf",
    "multimodal_bev": "https://arxiv.org/pdf/2305.09910.pdf",
}
for name, url in sensor_urls.items():
    dl(url, os.path.join(sensor_base, f"{name}.pdf"), f"Sensor {name}")

# Save
lp = os.path.join(OUT, f"download_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
with open(lp, "w") as f: json.dump(LOG, f, indent=2, default=str)
ok = sum(1 for x in LOG["items"] if x["status"]=="ok")
total_mb = sum(x["size"] for x in LOG["items"] if x["status"]=="ok") / 1024 / 1024
print(f"\nDONE: {ok}/{len(LOG['items'])} files, {total_mb:.0f}MB total")
print(f"Log: {lp}")
