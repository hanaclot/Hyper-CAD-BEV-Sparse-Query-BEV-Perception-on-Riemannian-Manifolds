import urllib.request, json, time, os
from pathlib import Path

DATA = Path(r"E:\Hyper-CAD-BEV-Experiments\data")
H = {"User-Agent": "Mozilla/5.0"}

arxiv = DATA / "acquired" / "arxiv"
arxiv.mkdir(parents=True, exist_ok=True)

papers = [
    ("BEVFormer", "2203.17270"), ("SparseAD", "2404.06892"),
    ("EventCamera", "1711.01458"), ("LoihiFusion", "2408.16096"),
    ("Weather", "2206.09907"), ("SparseBEV", "2308.09244"),
    ("BEVDet", "2112.11790"), ("Sparse4D", "2311.11722"),
    ("Petr3D", "2203.05625"), ("BEVDepth", "2206.10092"),
    ("PointPillars", "1812.05784"), ("RELLIS3D", "2011.07717"),
    ("TartanDrive", "2204.04615"), ("nuScenes", "1903.11027"),
    ("KITTI", "1204.4087"), ("SemanticKITTI", "1904.01416"),
]

ok = 0; fail = 0
for name, pid in papers:
    dst = arxiv / f"{name}_{pid}.pdf"
    if dst.exists() and dst.stat().st_size > 1000:
        ok += 1
        continue
    try:
        url = f"https://arxiv.org/pdf/{pid}.pdf"
        req = urllib.request.Request(url, headers=H)
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        with open(dst, "wb") as f:
            f.write(data)
        kb = len(data)/1024
        print(f"  OK: {name}: {kb:.0f}KB")
        ok += 1
    except Exception as e:
        print(f"  FAIL: {name}: {str(e)[:60]}")
        fail += 1
    time.sleep(0.5)

print(f"\nDone: {ok} OK, {fail} FAIL")