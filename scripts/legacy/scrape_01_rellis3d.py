# -*- coding: utf-8 -*-
"""RELLIS-3D 完整爬取 - 独立脚本"""
import json, time, re, urllib.request, urllib.error
from pathlib import Path
from datetime import datetime

BASE = Path(r"E:\HyperCAD_BEV_2026")
DATA = BASE / "data"
RELLIS = DATA / "rellis3d"
RELLIS.mkdir(parents=True, exist_ok=True)

def fetch(url, headers=None, timeout=30):
    if headers is None:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return {"status": resp.status, "data": resp.read()}
        except urllib.error.HTTPError as e:
            if attempt == 2: return {"status": e.code, "error": str(e)}
        except Exception as e:
            if attempt == 2: return {"status": -1, "error": str(e)}
            time.sleep(1)

result = {"source": "RELLIS-3D", "url": "https://github.com/unmannedlab/RELLIS-3D"}

# API metadata (already done)
api_resp = fetch("https://api.github.com/repos/unmannedlab/RELLIS-3D", 
                  {"Accept": "application/vnd.github.v3+json", "User-Agent": "HyperCAD"})
if api_resp.get("status") == 200:
    repo = json.loads(api_resp["data"].decode("utf-8"))
    result["stars"] = repo.get("stargazers_count", "?")
    result["forks"] = repo.get("forks_count", "?")
    result["description"] = repo.get("description", "")
    result["language"] = repo.get("language", "")
    result["topics"] = repo.get("topics", [])
    with open(RELLIS / "repo_metadata.json", "w") as f:
        json.dump(repo, f, indent=2)
    print(f"[RELLIS-3D] Stars={result['stars']} Forks={result['forks']}")

# README
for branch in ["main", "master"]:
    r = fetch(f"https://raw.githubusercontent.com/unmannedlab/RELLIS-3D/{branch}/README.md")
    if r.get("status") == 200:
        text = r["data"].decode("utf-8", errors="replace")
        with open(RELLIS / "README.md", "w", encoding="utf-8") as f:
            f.write(text)
        result["readme_chars"] = len(text)
        print(f"[RELLIS-3D] README: {len(text)} chars")
        break

# Key source files
files = ["config.py","rellis.py","cityscapes.py","cityscapes_labels.py","train.py","loss.py",
         "DualTaskLoss.py","GatedSpatialConv.py","Resnet.py","SEresnext.py","gscnn.py",
         "wider_resnet.py","joint_transforms.py","transforms.py","edge_utils.py",
         "optimizer.py","mynn.py","AttrDict.py","misc.py","__init__.py"]
count = 0
for fname in files:
    for branch in ["main", "master"]:
        r = fetch(f"https://raw.githubusercontent.com/unmannedlab/RELLIS-3D/{branch}/{fname}")
        if r.get("status") == 200:
            text = r["data"].decode("utf-8", errors="replace")
            with open(RELLIS / fname, "w", encoding="utf-8") as f:
                f.write(text)
            count += 1
            break
    if count % 5 == 0:
        print(f"[RELLIS-3D] Downloaded {count}/{len(files)} files...")

print(f"[RELLIS-3D] Complete: {count}/{len(files)} source files downloaded")

result["files_downloaded"] = count
result["timestamp"] = datetime.now().isoformat()
with open(RELLIS / "scrape_result.json", "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print("[RELLIS-3D] DONE")
