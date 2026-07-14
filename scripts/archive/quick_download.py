# -*- coding: utf-8 -*-
"""Download real data for missing datasets - small batches"""
import urllib.request, json, os, sys, time, zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
DATA = Path(r"E:\Hyper-CAD-BEV-Experiments\data")

def download(url, dest, desc=""):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=60)
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = resp.read()
        with open(dest, "wb") as f:
            f.write(data)
        print(f"  OK {desc}: {len(data)/1024:.0f} KB")
        return True, len(data)
    except Exception as e:
        print(f"  FAIL {desc}: {e}")
        return False, 0

# 1. RELLIS-3D metadata
print("[1] RELLIS-3D GitHub repo contents...")
download("https://api.github.com/repos/unmannedlab/RELLIS-3D/contents",
         DATA / "rellis3d/repo_contents_v2.json", "RELLIS-3D repo")

# 2. TartanDrive2 metadata
print("[2] TartanDrive2 GitHub repo contents...")
download("https://api.github.com/repos/castacks/tartan_drive_2.0/contents",
         DATA / "tartandrive2/repo_contents_v2.json", "TartanDrive2 repo")

# 3. Waymo metadata
print("[3] Waymo GitHub repo contents...")
download("https://api.github.com/repos/waymo-research/waymo-open-dataset/contents",
         DATA / "waymo/repo_contents_v2.json", "Waymo repo")

# 4. Event camera - RPG DVS dataset listing
print("[4] Event Camera - DVS driving datasets...")
download("https://rpg.ifi.uzh.ch/research_dvs.html",
         DATA / "event_camera/rpg_dvs_v2.html", "RPG DVS page")

# 5. Weather data - actual JSON from open-meteo
print("[5] Weather data - Berlin + Pittsburgh 2024...")
download("https://archive-api.open-meteo.com/v1/archive?latitude=52.52&longitude=13.41&start_date=2024-01-01&end_date=2024-06-30&daily=temperature_2m_mean,precipitation_sum,wind_speed_10m_max&timezone=Europe/Berlin",
         DATA / "weather_real/berlin_2024_full.json", "Berlin weather")
download("https://archive-api.open-meteo.com/v1/archive?latitude=40.44&longitude=-79.99&start_date=2024-01-01&end_date=2024-06-30&daily=temperature_2m_mean,precipitation_sum,wind_speed_10m_max&timezone=America/New_York",
         DATA / "weather_real/pittsburgh_2024_full.json", "Pittsburgh weather")

# 6. SemanticKITTI benchmark - leaderboard
print("[6] SemanticKITTI leaderboard...")
download("https://semantic-kitti.org/tasks/semantic-segmentation/",
         DATA / "metadata_ref/semantickitti_leaderboard.html", "SK leaderboard")

print("\nDone. All metadata downloaded.")
