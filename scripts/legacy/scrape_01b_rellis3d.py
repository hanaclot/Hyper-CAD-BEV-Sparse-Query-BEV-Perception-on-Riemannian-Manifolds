# -*- coding: utf-8 -*-
"""RELLIS-3D 源码文件爬取 (直接从 raw.githubusercontent.com)"""
import json, time, urllib.request, urllib.error
from pathlib import Path

RELLIS = Path(r"E:\HyperCAD_BEV_2026\data\rellis3d")
RELLIS.mkdir(parents=True, exist_ok=True)

def fetch_raw(url, timeout=20):
    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            if attempt == 2:
                print(f"  FAIL {url}: {e}")
                return None
            time.sleep(2)

# Source files to download
files = [
    "config.py", "rellis.py", "cityscapes.py", "cityscapes_labels.py",
    "train.py", "loss.py", "DualTaskLoss.py", "GatedSpatialConv.py",
    "Resnet.py", "SEresnext.py", "gscnn.py", "wider_resnet.py",
    "joint_transforms.py", "transforms.py", "edge_utils.py",
    "optimizer.py", "mynn.py", "AttrDict.py", "misc.py", "__init__.py",
    "custom_functional.py", "f_boundary.py", "image_page.py",
    "run_gscnn.sh", "run_gscnn_eval.sh", "Dockerfile", ".gitignore",
    "bibtex.txt", "gscnn.txt", "index.html", "LICENSE"
]

BASE = "https://raw.githubusercontent.com/unmannedlab/RELLIS-3D/main"
count = 0
for fname in files:
    content = fetch_raw(f"{BASE}/{fname}")
    if content is None:
        content = fetch_raw(f"{BASE.replace('main','master')}/{fname}")
    if content:
        (RELLIS / fname).write_text(content, encoding="utf-8")
        count += 1
        print(f"  OK: {fname} ({len(content)} chars)")

print(f"\n[RELLIS-3D] Downloaded {count}/{len(files)} files")

# Update result
result = {"files_downloaded": count, "source": "RELLIS-3D"}
existing = RELLIS / "scrape_result.json"
if existing.exists():
    old = json.loads(existing.read_text(encoding="utf-8"))
    old.update(result)
    result = old
result["files_downloaded"] = count
existing.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print("[RELLIS-3D] Complete!")
