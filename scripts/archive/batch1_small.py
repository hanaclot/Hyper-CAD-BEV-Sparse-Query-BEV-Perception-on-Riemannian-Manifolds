"""Batch 1: Small files - arxiv papers, GitHub repos, calibration files."""
import requests, os, time, json
from datetime import datetime

OUT = r"D:\HyperCAD_BEV_2026\data\scraped_new"
os.makedirs(OUT, exist_ok=True)
LOG = {"started": datetime.now().isoformat(), "items": []}
S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0"})
S.timeout = 60

def dl(url, dest, desc=""):
    d = os.path.dirname(dest)
    os.makedirs(d, exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        print(f"SKIP {desc} ({os.path.getsize(dest)/1024:.0f}KB)")
        return True
    print(f"GET {desc}")
    for i in range(3):
        try:
            r = S.get(url, timeout=120)
            if r.status_code == 200:
                with open(dest, "wb") as f: f.write(r.content)
                print(f"  OK {len(r.content)/1024:.0f}KB")
                LOG["items"].append({"name": desc, "size": len(r.content), "status": "ok"})
                return True
            else:
                print(f"  HTTP {r.status_code}")
                if r.status_code == 404:
                    LOG["items"].append({"name": desc, "status": "404"})
                    return False
        except Exception as e:
            print(f"  err: {str(e)[:60]}")
            time.sleep(3)
    LOG["items"].append({"name": desc, "status": "failed"})
    return False

# === Arxiv papers (small, fast) ===
for name, paper_id in [
    ("BEVFormer", "2203.17270"), ("EventCamera", "1711.01458"),
    ("SparseAD", "2404.06892"), ("LoihiFusion", "2408.16096"),
    ("WeatherOffroad", "2206.09907"), ("Sparse4D_v2", "2311.11722"),
    ("SparseBEV", "2308.09244"), ("BEVDet", "2112.11790"),
    ("PointPillars", "1905.01235"), ("RiemannianDL", "2205.15016"),
    ("BEVDepth", "2206.10092"), ("PolarBEV", "2211.12786"),
    ("Detr3D", "2110.06922"), ("PETRv2", "2304.12345"),
    ("FB-BEV", "2305.09910"), ("NeuralFields", "2106.12978"),
    ("EventSurvey", "1912.08432"), ("LiDARSurvey", "1904.01669"),
]:
    dl(f"https://arxiv.org/pdf/{paper_id}.pdf", 
       os.path.join(OUT, "arxiv", f"{paper_id}_{name}.pdf"), f"arxiv:{name}")

# === GitHub repos (medium, ~500KB-5MB each) ===
repos = [
    ("unmannedlab/RELLIS-3D", "RELLIS-3D"),
    ("PRBonn/semantic-kitti-api", "SemanticKITTI-API"),
    ("fundamentalvision/BEVFormer", "BEVFormer"),
    ("open-mmlab/OpenPCDet", "OpenPCDet"),
    ("open-mmlab/mmdetection3d", "mmdetection3d"),
    ("HorizonRobotics/Sparse4D", "Sparse4D"),
    ("HuangJunJie2017/BEVDet", "BEVDet"),
    ("nutonomy/nuscenes-devkit", "nuscenes-devkit"),
    ("MCG-NJU/SparseBEV", "SparseBEV"),
    ("Megvii-BaseDetection/BEVDepth", "BEVDepth"),
    ("traveller59/second.pytorch", "SECOND"),
    ("mit-han-lab/torchsparse", "TorchSparse"),
    ("sshaoshuai/Pointnet2.PyTorch", "PointNet2"),
    ("XuyangBai/SuperGaussian", "SuperGaussian"),
]
for repo, name in repos:
    dl(f"https://github.com/{repo}/archive/refs/heads/master.zip", 
       os.path.join(OUT, "github", f"{name}.zip"), f"gh:{name}")

lp = os.path.join(OUT, f"batch1_log.json")
with open(lp, "w") as f: json.dump(LOG, f, indent=2, default=str)
ok = sum(1 for x in LOG["items"] if x["status"]=="ok")
print(f"\nBATCH1: {ok}/{len(LOG['items'])} done")
