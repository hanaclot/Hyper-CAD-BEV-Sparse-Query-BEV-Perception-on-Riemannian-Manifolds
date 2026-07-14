# -*- coding: utf-8 -*-
"""
Deep Scraper for Hyper-CAD-BEV v6.5-Sparse Experiment
=====================================================
Scrapes ALL 8 data sources with full audit trail.
Every data point must be traceable to its source URL.
ZERO synthetic data generation.

Sources:
1. RELLIS-3D GitHub (https://github.com/unmannedlab/RELLIS-3D)
2. SemanticKITTI (http://semantic-kitti.org/)
3. TartanDrive2 (https://theairlab.org/TartanDrive2/)
4. arXiv:2203.17270 (BEVFormer)
5. arXiv:2404.06892 (SparseAD)
6. arXiv:1711.01458 (Event Camera)
7. arXiv:2408.16096v1 (Loihi Fusion)
8. arXiv:2206.09907 (Weather Robustness)
"""
import sys, os, json, csv, hashlib, time as time_mod
from datetime import datetime, timezone
from pathlib import Path
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
import re

PROJECT = Path(r"E:\HyperCAD_BEV_Sparse")
SCRAPED = PROJECT / "data" / "scraped"
RAW = PROJECT / "data" / "raw"
PROCESSED = PROJECT / "data" / "processed"
for d in [SCRAPED, RAW, PROCESSED]:
    d.mkdir(parents=True, exist_ok=True)

SCRAPE_LOG = []
UA = "HyperCAD-BEV-Research/1.0 (academic data collection; contact: gao.zihan@kust.edu.cn)"

def log(msg):
    t = datetime.now().strftime("%H:%M:%S")
    print(f"[{t}] {msg}")
    SCRAPE_LOG.append({"time": t, "msg": msg})

def safe_fetch(url, timeout=30):
    """Fetch URL with error handling"""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            return data, resp.status
    except Exception as e:
        log(f"  WARN: {url} -> {e}")
        return None, None

def save_content(path, content, is_bytes=False):
    mode = "wb" if is_bytes else "w"
    enc = None if is_bytes else "utf-8"
    with open(path, mode, encoding=enc) as f:
        f.write(content)

def content_hash(content):
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()[:16]

# =============================================
# SOURCE 1: RELLIS-3D GitHub Repository
# =============================================
def scrape_rellis3d():
    log("=== SOURCE 1: RELLIS-3D GitHub ===")
    outdir = RAW / "rellis3d"
    outdir.mkdir(parents=True, exist_ok=True)
    
    sources = [
        ("https://github.com/unmannedlab/RELLIS-3D", "github_main.html"),
        ("https://raw.githubusercontent.com/unmannedlab/RELLIS-3D/master/README.md", "README.md"),
        ("https://api.github.com/repos/unmannedlab/RELLIS-3D", "api_repo.json"),
    ]
    
    results = {}
    for url, fname in sources:
        log(f"  Fetching: {url}")
        data, status = safe_fetch(url)
        if data:
            path = outdir / fname
            if isinstance(data, bytes):
                save_content(path, data, is_bytes=True)
            else:
                save_content(path, data)
            h = content_hash(data if isinstance(data,str) else data.decode("utf-8","ignore"))
            log(f"  -> {fname} ({len(data)} bytes, hash={h}, status={status})")
            results[fname] = {"size": len(data), "hash": h, "status": status}
    
    # Parse GitHub API for real repo metrics
    api_path = outdir / "api_repo.json"
    if api_path.exists():
        with open(api_path, "r", encoding="utf-8") as f:
            api_data = json.load(f)
        repo_metrics = {
            "stars": api_data.get("stargazers_count", 0),
            "forks": api_data.get("forks_count", 0),
            "open_issues": api_data.get("open_issues_count", 0),
            "size_kb": api_data.get("size", 0),
            "language": api_data.get("language", ""),
            "description": api_data.get("description", ""),
            "created_at": api_data.get("created_at", ""),
            "updated_at": api_data.get("updated_at", ""),
        }
        log(f"  Repo: stars={repo_metrics['stars']}, forks={repo_metrics['forks']}")
        results["repo_metrics"] = repo_metrics
    
    # Scrape dataset statistics from README
    readme_path = outdir / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            readme = f.read()
        # Extract numbers from README
        dataset_info = {}
        for pat, key in [
            (r"(\d+[,.]?\d*)\s*(?:hours|hrs)", "hours"),
            (r"(\d+[,.]?\d*)\s*(?:GB|TB)", "size_gb"),
            (r"(\d+[,.]?\d*)\s*(?:images|frames)", "num_images"),
            (r"(\d+)\s*(?:classes|categories)", "num_classes"),
            (r"(\d+[,.]?\d*)\s*(?:LiDAR|laser)\s*(?:scans|points)", "lidar_points"),
        ]:
            m = re.search(pat, readme, re.IGNORECASE)
            if m:
                dataset_info[key] = m.group(1)
        results["dataset_info"] = dataset_info
        log(f"  Dataset info: {dataset_info}")
    
    # Save scrape result
    with open(outdir / "deep_scrape_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results

# =============================================
# SOURCE 2: SemanticKITTI
# =============================================
def scrape_semantickitti():
    log("=== SOURCE 2: SemanticKITTI ===")
    outdir = RAW / "semantickitti"
    outdir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    urls = {
        "main": "http://semantic-kitti.org/",
        "dataset": "http://semantic-kitti.org/dataset.html",
        "tasks": "http://semantic-kitti.org/tasks.html",
    }
    
    for name, url in urls.items():
        log(f"  Fetching: {url}")
        data, status = safe_fetch(url, timeout=60)
        if data:
            path = outdir / f"{name}.html"
            html = data.decode("utf-8", errors="ignore")
            save_content(path, html)
            h = content_hash(html)
            log(f"  -> {name}.html ({len(html)} bytes, hash={h}, status={status})")
            results[name] = {"size": len(html), "hash": h, "status": status}
            
            # Parse leaderboard from tasks page
            if name == "tasks":
                # Extract benchmark numbers
                bench_data = {}
                # Look for table with IoU, accuracy numbers
                metrics = re.findall(r"(\d{2}\.\d{1,2})\s*(?:%|percent)", html)
                if metrics:
                    bench_data["found_percentages"] = metrics[:20]
                # Look for method names
                methods = re.findall(r'(?:>|<td>|<th>)\s*([A-Z][A-Za-z0-9\-+\s]{2,40})\s*(?:<|$)', html)
                if methods:
                    bench_data["found_methods"] = [m.strip() for m in methods if len(m.strip())>5][:20]
                results["benchmark_data"] = bench_data
                log(f"  Found {len(metrics)} metrics, {len(methods)} method names")
    
    with open(outdir / "deep_scrape_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results

# =============================================
# SOURCE 3: TartanDrive2
# =============================================
def scrape_tartandrive2():
    log("=== SOURCE 3: TartanDrive2 ===")
    outdir = RAW / "tartandrive2"
    outdir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    urls = {
        "main": "https://theairlab.org/TartanDrive2/",
        "arxiv": "https://arxiv.org/abs/2206.09907",
    }
    
    for name, url in urls.items():
        log(f"  Fetching: {url}")
        data, status = safe_fetch(url, timeout=60)
        if data:
            path = outdir / f"{name}.html"
            html = data.decode("utf-8", errors="ignore") if isinstance(data, bytes) else data
            save_content(path, html)
            h = content_hash(html)
            log(f"  -> {name}.html ({len(html)} bytes, hash={h}, status={status})")
            results[name] = {"size": len(html), "hash": h, "status": status}
    
    # Also fetch arxiv metadata
    arxiv_xml_url = "https://export.arxiv.org/api/query?id_list=2206.09907"
    data, status = safe_fetch(arxiv_xml_url, timeout=30)
    if data:
        path = outdir / "arxiv_metadata.xml"
        xml_str = data.decode("utf-8", errors="ignore")
        save_content(path, xml_str)
        results["arxiv_metadata"] = {"size": len(xml_str), "status": status}
    
    with open(outdir / "deep_scrape_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results

# =============================================
# SOURCES 4-8: arXiv Papers
# =============================================
ARXIV_PAPERS = {
    "bevformer": {
        "id": "2203.17270",
        "name": "BEVFormer: Learning Birds-Eye-View Representation from Multi-Camera Images",
        "url": "https://arxiv.org/abs/2203.17270",
        "dir": SCRAPED / "arxiv" / "bevformer"
    },
    "sparsead": {
        "id": "2404.06892",
        "name": "SparseAD: Sparse Query-Centric Paradigm for End-to-End Autonomous Driving",
        "url": "https://arxiv.org/abs/2404.06892",
        "dir": SCRAPED / "arxiv" / "sparsead"
    },
    "event_camera": {
        "id": "1711.01458",
        "name": "Event-based Vision: A Survey",
        "url": "https://arxiv.org/abs/1711.01458",
        "dir": SCRAPED / "arxiv" / "event_camera"
    },
    "loihi_fusion": {
        "id": "2408.16096",
        "name": "Edge Multi-Sensor Fusion: Neuromorphic Approaches",
        "url": "https://arxiv.org/abs/2408.16096",
        "dir": SCRAPED / "arxiv" / "loihi_fusion"
    },
    "weather_robustness": {
        "id": "2206.09907",
        "name": "Off-Road Free Space Detection with Weather and Lighting Variations",
        "url": "https://arxiv.org/abs/2206.09907",
        "dir": SCRAPED / "arxiv" / "weather_robustness"
    },
}

def scrape_arxiv_papers():
    log("=== SOURCES 4-8: arXiv Papers ===")
    all_results = {}
    
    for key, paper in ARXIV_PAPERS.items():
        log(f"  Paper: {paper['name']} ({paper['id']})")
        paper["dir"].mkdir(parents=True, exist_ok=True)
        results = {}
        
        # 1. API metadata
        api_url = f"https://export.arxiv.org/api/query?id_list={paper['id']}"
        data, status = safe_fetch(api_url, timeout=30)
        if data:
            xml_str = data.decode("utf-8", errors="ignore")
            path = paper["dir"] / "metadata.xml"
            save_content(path, xml_str)
            log(f"    metadata.xml: {len(xml_str)} bytes (status={status})")
            results["metadata"] = {"size": len(xml_str), "status": status}
            
            # Parse XML
            try:
                root = ET.fromstring(xml_str)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                title = root.find(".//atom:title", ns)
                summary = root.find(".//atom:summary", ns)
                if title is not None:
                    results["title"] = title.text.strip()
                if summary is not None:
                    results["abstract"] = summary.text.strip()[:500]
            except Exception as e:
                log(f"    XML parse warn: {e}")
        
        # 2. Abstract page
        data, status = safe_fetch(paper["url"], timeout=30)
        if data:
            html = data.decode("utf-8", errors="ignore")
            path = paper["dir"] / "abstract_page.html"
            save_content(path, html)
            log(f"    abstract_page.html: {len(html)} bytes (status={status})")
            results["abstract_page"] = {"size": len(html), "status": status}
            
            # Extract any quantitative results from abstract
            metrics = re.findall(r"(\d{1,3}\.\d{1,2})\s*(?:%|percent|mIoU|mAP|AP)", html)
            if metrics:
                results["extracted_metrics"] = metrics[:15]
        
        # 3. PDF (if not already downloaded)
        pdf_path = paper["dir"] / "paper.pdf"
        if not pdf_path.exists():
            pdf_url = f"https://arxiv.org/pdf/{paper['id']}.pdf"
            data, status = safe_fetch(pdf_url, timeout=120)
            if data:
                save_content(pdf_path, data, is_bytes=True)
                log(f"    paper.pdf: {len(data)} bytes (status={status})")
                results["pdf"] = {"size": len(data), "status": status}
        
        # Save result
        with open(paper["dir"] / "scrape_result.json", "w", encoding="utf-8") as f:
            json.dump({**results, "scraped_at": datetime.now(timezone.utc).isoformat()}, 
                     f, indent=2, ensure_ascii=False)
        
        all_results[key] = results
    
    return all_results

# =============================================
# AGGREGATION: Create Master Data Index
# =============================================
def create_master_index(all_results):
    log("=== Creating Master Data Index ===")
    
    rows = []
    source_idx = 0
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # Walk all scraped data
    for root, dirs, files in os.walk(SCRAPED):
        for f in files:
            path = Path(root) / f
            if path.suffix in [".json", ".html", ".xml", ".pdf", ".txt", ".md"]:
                rel = path.relative_to(PROJECT)
                size = path.stat().st_size
                source_idx += 1
                rows.append({
                    "index": source_idx,
                    "relative_path": str(rel),
                    "filename": f,
                    "type": path.suffix.lstrip("."),
                    "size_bytes": size,
                    "size_mb": round(size / 1e6, 4),
                    "scraped_at": timestamp,
                })
    
    # Also walk raw data
    for root, dirs, files in os.walk(RAW):
        for f in files:
            path = Path(root) / f
            if path.suffix not in [".jpg", ".png", ".gif"]:  # skip images
                rel = path.relative_to(PROJECT)
                size = path.stat().st_size
                source_idx += 1
                rows.append({
                    "index": source_idx,
                    "relative_path": str(rel),
                    "filename": f,
                    "type": path.suffix.lstrip("."),
                    "size_bytes": size,
                    "size_mb": round(size / 1e6, 4),
                    "scraped_at": timestamp,
                })
    
    # Write index
    csv_path = PROCESSED / f"master_data_index_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index","relative_path","filename","type","size_bytes","size_mb","scraped_at"])
        writer.writeheader()
        writer.writerows(rows)
    
    log(f"  Master index: {len(rows)} data files indexed -> {csv_path.name}")
    
    # Summary by type
    type_summary = {}
    for row in rows:
        t = row["type"]
        if t not in type_summary:
            type_summary[t] = {"count": 0, "total_mb": 0}
        type_summary[t]["count"] += 1
        type_summary[t]["total_mb"] += row["size_mb"]
    
    for t, s in sorted(type_summary.items()):
        log(f"    {t}: {s['count']} files, {s['total_mb']:.2f} MB")
    
    return rows

# =============================================
# MAIN
# =============================================
def main():
    log("="*60)
    log("Hyper-CAD-BEV v6.5-Sparse DEEP SCRAPER")
    log(f"Started: {datetime.now(timezone.utc).isoformat()}")
    log("="*60)
    
    all_results = {}
    
    # Deep scrape all sources
    all_results["rellis3d"] = scrape_rellis3d()
    all_results["semantickitti"] = scrape_semantickitti()
    all_results["tartandrive2"] = scrape_tartandrive2()
    all_results["arxiv"] = scrape_arxiv_papers()
    
    # Create master index
    master_index = create_master_index(all_results)
    
    # Save scrape log
    log_path = PROCESSED / f"deep_scrape_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "started": datetime.now(timezone.utc).isoformat(),
            "log_entries": SCRAPE_LOG,
            "results_summary": {k: type(v).__name__ for k, v in all_results.items()},
            "total_files_indexed": len(master_index),
        }, f, indent=2, ensure_ascii=False)
    
    log("="*60)
    log(f"DEEP SCRAPING COMPLETE: {len(SCRAPE_LOG)} log entries")
    log(f"Log saved to: {log_path}")
    log("="*60)

if __name__ == "__main__":
    main()
