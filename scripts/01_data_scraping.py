# -*- coding: utf-8 -*-
'''
爬取所有指定数据源:
  1. RELLIS-3D (https://github.com/unmannedlab/RELLIS-3D)
  2. SemanticKITTI (http://semantic-kitti.org/)
  3. TartanDrive2 (https://theairlab.org/TartanDrive2/)
  4. ArXiv: BEVFormer (2203.17270), SparseAD (2404.06892)
  5. ArXiv: Event Camera (1711.01458), Loihi Fusion (2408.16096)
  6. ArXiv: Weather Robustness (2206.09907)
'''
import requests, json, csv, os, time, re
from pathlib import Path
from datetime import datetime

BASE = Path(r"D:\HyperCAD_BEV_2026")
DATA = BASE / "data"
LOG = {}

def log(msg):
    t = datetime.now().strftime("%H:%M:%S")

# 1. GitHub API: RELLIS-3D
def scrape_rellis3d():
    log("Scraping RELLIS-3D from GitHub API...")
    headers = {"Accept": "application/vnd.github.v3+json"}
    try:
        r = requests.get("https://api.github.com/repos/unmannedlab/RELLIS-3D", headers=headers, timeout=30)
        if r.status_code == 200:
            repo = r.json()
            (DATA/"scraped"/"rellis3d_github.json").write_text(json.dumps(repo, indent=2), encoding="utf-8")
            LOG["rellis3d_github"] = {"status": "ok", "stars": repo.get("stargazers_count", 0)}
        else:
            LOG["rellis3d_github"] = {"status": f"HTTP {r.status_code}"}
    except Exception as e:
        LOG["rellis3d_github"] = {"status": "error", "msg": str(e)}

    # Download README
    try:
        r2 = requests.get("https://raw.githubusercontent.com/unmannedlab/RELLIS-3D/master/README.md", timeout=30)
        if r2.status_code == 200:
            (DATA/"raw"/"rellis3d"/"README.md").write_text(r2.text, encoding="utf-8")
            LOG["rellis3d_readme"] = {"status": "ok", "size": len(r2.text)}
    except Exception as e:
        LOG["rellis3d_readme"] = {"status": "error", "msg": str(e)}

    # Clone repo info page
    try:
        r3 = requests.get("https://github.com/unmannedlab/RELLIS-3D", timeout=30)
        (DATA/"raw"/"rellis3d"/"github_page.html").write_text(r3.text, encoding="utf-8")
        LOG["rellis3d_page"] = {"status": "ok", "size": len(r3.text)}
    except Exception as e:
        LOG["rellis3d_page"] = {"status": "error", "msg": str(e)}

# 2. SemanticKITTI
def scrape_semantickitti():
    log("Scraping SemanticKITTI...")
    pages = {
        "main": "http://semantic-kitti.org/",
        "tasks": "http://semantic-kitti.org/tasks.html",
        "dataset": "http://semantic-kitti.org/dataset.html"
    }
    for name, url in pages.items():
        try:
            r = requests.get(url, timeout=30)
            (DATA/"raw"/"semantickitti"/f"{name}.html").write_text(r.text, encoding="utf-8")
            LOG[f"semantickitti_{name}"] = {"status": "ok", "size": len(r.text)}
        except Exception as e:
            LOG[f"semantickitti_{name}"] = {"status": "error", "msg": str(e)}

# 3. TartanDrive2
def scrape_tartandrive2():
    log("Scraping TartanDrive2...")
    pages = {
        "main": "https://theairlab.org/TartanDrive2/",
        "arxiv": "https://arxiv.org/abs/2305.13859"
    }
    for name, url in pages.items():
        try:
            r = requests.get(url, timeout=30)
            (DATA/"raw"/"tartandrive2"/f"{name}.html").write_text(r.text, encoding="utf-8")
            LOG[f"tartandrive2_{name}"] = {"status": "ok", "size": len(r.text)}
        except Exception as e:
            LOG[f"tartandrive2_{name}"] = {"status": "error", "msg": str(e)}

# 4. ArXiv Papers via API
ARXIV_PAPERS = {
    "BEVFormer_2203.17270": "2203.17270",
    "SparseAD_2404.06892": "2404.06892",
    "EventCamera_1711.01458": "1711.01458",
    "LoihiFusion_2408.16096": "2408.16096",
    "WeatherRobustness_2206.09907": "2206.09907",
    "SparseBEV_2308.09244": "2308.09244",
    "Sparse4D_2311.11722": "2311.11722",
    "BEVDet_2112.11790": "2112.11790",
    "MonoBEV_2410.06516": "2410.06516",
}

def scrape_arxiv_paper(arxiv_id, label):
    log(f"  ArXiv: {label} ({arxiv_id})")
    base_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1"
    try:
        r = requests.get(base_url, timeout=30)
        (DATA/"scraped"/f"{label}.xml").write_text(r.text, encoding="utf-8")
        (DATA/"raw"/f"bevformer"/f"{label}.xml").write_text(r.text, encoding="utf-8") if "BEVFormer" in label else None
        LOG[f"arxiv_{label}"] = {"status": "ok", "size": len(r.text)}
        return arxiv_id
    except Exception as e:
        LOG[f"arxiv_{label}"] = {"status": "error", "msg": str(e)}
        return None

def scrape_all_arxiv():
    log("Scraping ArXiv papers...")
    for label, aid in ARXIV_PAPERS.items():
        scrape_arxiv_paper(aid, label)
        time.sleep(3)  # Rate limiting

# 5. Additional ArXiv Queries
def scrape_arxiv_queries():
    log("Scraping additional ArXiv queries...")
    queries = {
        "sparse_query_bev": "cat:cs.CV+AND+(sparse+query+BEV+perception)",
        "neuromorphic_driving": "cat:cs.CV+AND+(neuromorphic+autonomous+driving)",
        "event_camera_perception": "cat:cs.CV+AND+(event+camera+perception)",
        "bev_offroad": "cat:cs.CV+AND+(BEV+off-road+rural)",
        "lidar_3d_detection": "cat:cs.CV+AND+(LiDAR+3D+object+detection+BEV)",
        "terrain_perception": "cat:cs.CV+AND+(terrain+perception+neural+field)",
        "edge_sensor_fusion": "cat:cs.CV+AND+(multi-sensor+fusion+edge+device)",
    }
    for label, query in queries.items():
        try:
            url = f"http://export.arxiv.org/api/query?search_query={query}&max_results=10&sortBy=relevance"
            r = requests.get(url, timeout=30)
            (DATA/"scraped"/f"query_{label}.xml").write_text(r.text, encoding="utf-8")
            LOG[f"query_{label}"] = {"status": "ok", "size": len(r.text)}
        except Exception as e:
            LOG[f"query_{label}"] = {"status": "error", "msg": str(e)}
        time.sleep(3)

# 6. Main
def main():
    log("=" * 60)
    log("Hyper-CAD-BEV v6.5: Data Scraping Pipeline")
    log(f"Target: {BASE}")
    log("=" * 60)

    scrape_rellis3d()
    scrape_semantickitti()
    scrape_tartandrive2()
    scrape_all_arxiv()
    scrape_arxiv_queries()

    # Save log
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    (DATA/"scraped"/f"scraping_log_{ts}.json").write_text(json.dumps(LOG, indent=2, ensure_ascii=False), encoding="utf-8")

    log("=" * 60)
    log(f"Done. {sum(1 for v in LOG.values() if v['status']=='ok')}/{len(LOG)} succeeded.")
    log("=" * 60)

if __name__ == "__main__":
    main()
