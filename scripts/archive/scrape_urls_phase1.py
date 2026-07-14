# -*- coding: utf-8 -*-
"""Phase 1: Scrape data source websites to find actual download URLs"""
import urllib.request, urllib.error, re, json, time
from pathlib import Path

OUT = Path(r"E:\Hyper-CAD-BEV-Experiments\data\scraped\discovered_urls.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
results = {}

def fetch(url, name, timeout=60):
    print(f"[FETCH] {name}: {url}")
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            html = r.read().decode('utf-8', errors='replace')
            print(f"  OK: {len(html)} chars")
            return html
    except Exception as e:
        print(f"  ERROR: {e}")
        return ""

def extract_urls(html, base_domain=""):
    """Extract all URLs matching common data file patterns"""
    patterns = [
        r'(https?://[^\s<>"\']+\.(?:zip|tar\.gz|tgz|h5|npy|npz|bin|pcd|las|laz|bag|pkl|pt|onnx))',
        r'(https?://[^\s<>"\']+\.(?:pdf))',
        r'(https?://[^\s<>"\']+download[^\s<>"\']*)',
        r'(https?://[^\s<>"\']+dataset[^\s<>"\']*)',
    ]
    found = set()
    for pat in patterns:
        for m in re.finditer(pat, html, re.IGNORECASE):
            u = m.group(1)
            if len(u) < 500 and not u.endswith(('.css','.js','.png','.jpg','.svg','.ico','.gif','.woff')):
                found.add(u)
    return sorted(found)

# 1. SemanticKITTI
html = fetch("http://semantic-kitti.org/dataset.html", "SemanticKITTI-dataset")
results["semantickitti_dataset"] = {"urls": extract_urls(html), "html_len": len(html)}

# 2. SemanticKITTI main page  
html = fetch("http://semantic-kitti.org/", "SemanticKITTI-main")
results["semantickitti_main"] = {"urls": extract_urls(html), "html_len": len(html)}

# 3. RELLIS-3D GitHub
html = fetch("https://github.com/unmannedlab/RELLIS-3D", "RELLIS-3D-GitHub")
results["rellis3d_github"] = {"urls": extract_urls(html), "html_len": len(html)}

# 4. RELLIS-3D README raw
html = fetch("https://raw.githubusercontent.com/unmannedlab/RELLIS-3D/master/README.md", "RELLIS-3D-README")
results["rellis3d_readme"] = {"urls": extract_urls(html), "html_len": len(html)}

# 5. TartanDrive2
html = fetch("https://theairlab.org/TartanDrive2/", "TartanDrive2")
results["tartandrive2"] = {"urls": extract_urls(html), "html_len": len(html)}

# 6. nuScenes
html = fetch("https://www.nuscenes.org/download", "nuScenes-download")
results["nuscenes_download"] = {"urls": extract_urls(html), "html_len": len(html)}

# 7. KITTI raw
html = fetch("https://www.cvlibs.net/datasets/kitti/raw_data.php", "KITTI-raw")
results["kitti_raw"] = {"urls": extract_urls(html), "html_len": len(html)}

# 8. KITTI odometry
html = fetch("https://www.cvlibs.net/datasets/kitti/eval_odometry.php", "KITTI-odometry")
results["kitti_odometry"] = {"urls": extract_urls(html), "html_len": len(html)}

# 9. Waymo Open Dataset
html = fetch("https://waymo.com/open/", "Waymo")
results["waymo"] = {"urls": extract_urls(html), "html_len": len(html)}

# Save all findings
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n[DONE] Saved to {OUT}")
print(f"Total sources: {len(results)}")
for k, v in results.items():
    print(f"  {k}: {len(v.get('urls',[]))} URLs found")