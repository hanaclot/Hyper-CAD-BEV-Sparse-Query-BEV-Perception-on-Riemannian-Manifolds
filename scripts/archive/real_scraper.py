# real_scraper.py - Real data scraper, no fabrication
import os, sys, json, time, hashlib
from pathlib import Path
from datetime import datetime
import urllib.request
import urllib.error

E = Path(r"E:\Hyper-CAD-BEV-Experiments")
DATA = E / "data"
LOG_FILE = E / "REAL_SCRAPE_LOG.txt"

def log(msg):
    t = datetime.now().strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def download_file(url, dest_path, desc="", timeout=120):
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:120.0) Gecko/20100101 Firefox/120.0"
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        with open(dest, "wb") as f:
            f.write(data)
        sha = hashlib.sha256(data).hexdigest()[:16]
        log(f"  OK {desc}: {len(data):,} bytes -> {dest.name} [sha256:{sha}]")
        return True, len(data), sha
    except Exception as e:
        log(f"  FAIL {desc}: {e}")
        return False, 0, ""

# PHASE 1: RELLIS-3D
log("=" * 60)
log("PHASE 1: RELLIS-3D REAL DATA")
rellis_dir = DATA / "rellis3d_real"
rellis_dir.mkdir(exist_ok=True)
rellis_urls = [
    ("https://drive.google.com/uc?export=download&id=1NQQNIBm1qG0Z9eHX7NVrF2NFOSYHE09L",
     rellis_dir / "RELLIS-3D_sample.tar.gz", "RELLIS-3D Google Drive sample"),
    ("https://github.com/unmannedlab/RELLIS-3D/releases/download/v1.0/RELLIS-3D-sample.tar.gz",
     rellis_dir / "RELLIS-3D_release.tar.gz", "RELLIS-3D GitHub release"),
]
for url, path, desc in rellis_urls:
    download_file(url, path, desc)

# PHASE 2: TartanDrive 2
log("=" * 60)
log("PHASE 2: TartanDrive 2 REAL DATA")
tartan_dir = DATA / "tartandrive2_real"
tartan_dir.mkdir(exist_ok=True)
tartan_urls = [
    ("https://huggingface.co/datasets/theairlab/TartanDrive2/resolve/main/README.md",
     tartan_dir / "TartanDrive2_README.md", "TartanDrive2 HF README"),
]
for url, path, desc in tartan_urls:
    download_file(url, path, desc)

# PHASE 3: KITTI Raw
log("=" * 60)
log("PHASE 3: KITTI Raw REAL DATA")
kitti_dir = DATA / "kitti_raw_real"
kitti_dir.mkdir(exist_ok=True)
kitti_urls = [
    ("https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_raw_data.zip",
     kitti_dir / "kitti_devkit_raw.zip", "KITTI devkit raw"),
    ("https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip",
     kitti_dir / "kitti_2011_09_26_drive_0001_sync.zip", "KITTI raw sample 0001"),
]
for url, path, desc in kitti_urls:
    download_file(url, path, desc)

# PHASE 4: Event Camera DVS
log("=" * 60)
log("PHASE 4: Event Camera DVS REAL DATA")
event_dir = DATA / "event_camera_real"
event_dir.mkdir(exist_ok=True)
event_urls = [
    ("https://rpg.ifi.uzh.ch/datasets/davis/shapes_translation.zip",
     event_dir / "dvs_shapes_translation.zip", "DVS shapes_translation"),
    ("https://rpg.ifi.uzh.ch/datasets/davis/shapes_rotation.zip",
     event_dir / "dvs_shapes_rotation.zip", "DVS shapes_rotation"),
]
for url, path, desc in event_urls:
    download_file(url, path, desc)

# PHASE 5: Weather data (Open-Meteo free API)
log("=" * 60)
log("PHASE 5: Weather Data")
weather_dir = DATA / "weather_real"
weather_dir.mkdir(exist_ok=True)
weather_urls = [
    ("https://archive-api.open-meteo.com/v1/archive?latitude=52.52&longitude=13.41&start_date=2023-01-01&end_date=2023-12-31&daily=temperature_2m_mean,precipitation_sum,cloud_cover_mean,wind_speed_10m_max&timezone=Europe/Berlin",
     weather_dir / "berlin_2023_weather.json", "Berlin 2023 weather"),
    ("https://archive-api.open-meteo.com/v1/archive?latitude=40.44&longitude=-79.99&start_date=2023-06-01&end_date=2023-12-31&daily=temperature_2m_mean,precipitation_sum,cloud_cover_mean,wind_speed_10m_max&timezone=America/New_York",
     weather_dir / "pittsburgh_2023_weather.json", "Pittsburgh 2023 weather"),
]
for url, path, desc in weather_urls:
    download_file(url, path, desc)

log("=" * 60)
log("REAL SCRAPE COMPLETE. See REAL_SCRAPE_LOG.txt for details.")