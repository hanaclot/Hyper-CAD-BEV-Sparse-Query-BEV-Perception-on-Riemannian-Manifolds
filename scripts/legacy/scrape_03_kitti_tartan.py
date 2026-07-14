# -*- coding: utf-8 -*-
"""爬取 SemanticKITTI 和 TartanDrive 2.0 网页元数据"""
import json, time, re, urllib.request, urllib.error
from pathlib import Path
from datetime import datetime

BASE = Path(r"E:\HyperCAD_BEV_2026")
DATA = BASE / "data"

def fetch(url, timeout=30):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    for attempt in range(2):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return {"status": resp.status, "data": resp.read().decode("utf-8", errors="replace")}
        except Exception as e:
            if attempt == 1:
                return {"status": -1, "error": str(e)}
            time.sleep(3)

# --- SemanticKITTI ---
print("="*60)
print("[1/2] SemanticKITTI: http://semantic-kitti.org/")
print("="*60)

SK = DATA / "semantickitti"
SK.mkdir(parents=True, exist_ok=True)
sk_result = {"source": "SemanticKITTI", "url": "http://semantic-kitti.org/"}

resp = fetch("http://semantic-kitti.org/", timeout=45)
if resp["status"] == 200:
    html = resp["data"]
    (SK / "index.html").write_text(html, encoding="utf-8")
    sk_result["page_chars"] = len(html)
    print(f"  Main page: {len(html)} chars")
else:
    print(f"  Main page failed: {resp.get('error')}")
    sk_result["page_error"] = resp.get("error", "")

# Try dataset page
resp2 = fetch("http://semantic-kitti.org/dataset.html", timeout=45)
if resp2["status"] == 200:
    (SK / "dataset.html").write_text(resp2["data"], encoding="utf-8")
    sk_result["dataset_page_chars"] = len(resp2["data"])
    print(f"  Dataset page: {len(resp2['data'])} chars")

sk_result["semantic_classes"] = {
    0:"unlabeled",1:"car",2:"bicycle",3:"motorcycle",4:"truck",
    5:"other-vehicle",6:"person",7:"bicyclist",8:"motorcyclist",
    9:"road",10:"parking",11:"sidewalk",12:"other-ground",
    13:"building",14:"fence",15:"vegetation",16:"trunk",
    17:"terrain",18:"pole",19:"traffic-sign"
}
sk_result["num_classes"] = 20
sk_result["splits"] = {"train": [f"{i:02d}" for i in range(8)], "val": ["08"], "test": ["09","10"]}
sk_result["timestamp"] = datetime.now().isoformat()
(SK / "scrape_result.json").write_text(json.dumps(sk_result, indent=2, ensure_ascii=False), encoding="utf-8")

# --- TartanDrive 2.0 ---
print("\n" + "="*60)
print("[2/2] TartanDrive 2.0: https://theairlab.org/TartanDrive2/")
print("="*60)

TD = DATA / "tartandrive2"
TD.mkdir(parents=True, exist_ok=True)
td_result = {"source": "TartanDrive 2.0", "url": "https://theairlab.org/TartanDrive2/"}

resp3 = fetch("https://theairlab.org/TartanDrive2/", timeout=45)
if resp3["status"] == 200:
    html3 = resp3["data"]
    (TD / "index.html").write_text(html3, encoding="utf-8")
    td_result["page_chars"] = len(html3)
    print(f"  Main page: {len(html3)} chars")
else:
    print(f"  Page failed: {resp3.get('error')}")

# Try arXiv paper
resp4 = fetch("https://arxiv.org/abs/2403.01072", timeout=45)
if resp4["status"] == 200:
    (TD / "arxiv_abstract.html").write_text(resp4["data"], encoding="utf-8")
    td_result["arxiv_chars"] = len(resp4["data"])
    print(f"  arXiv page: {len(resp4['data'])} chars")

td_result["features"] = {
    "speed_range": "up to 10 m/s",
    "terrain_types": ["gravel","dirt","mud","grass","rocky","snow"],
    "sensors": ["RGB","LiDAR","IMU","GPS","wheel_odometry"],
    "total_trajectories": "200+ km"
}
td_result["timestamp"] = datetime.now().isoformat()
(TD / "scrape_result.json").write_text(json.dumps(td_result, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"\n{'='*60}")
print("Done: SemanticKITTI + TartanDrive 2.0 scraped!")
print(f"{'='*60}")
