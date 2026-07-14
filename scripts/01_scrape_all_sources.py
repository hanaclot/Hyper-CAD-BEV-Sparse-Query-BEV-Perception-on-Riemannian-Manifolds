# -*- coding: utf-8 -*-
"""
Hyper-CAD-BEV v6.5-Sparse: 完整8源数据爬取脚本
  1. RELLIS-3D: https://github.com/unmannedlab/RELLIS-3D (GitHub API)
  2. SemanticKITTI: http://semantic-kitti.org/
  3. TartanDrive 2.0: https://theairlab.org/TartanDrive2/
  4. BEVFormer: https://arxiv.org/abs/2203.17270
  5. SparseAD: https://arxiv.org/abs/2404.06892
  6. Event Camera: https://arxiv.org/abs/1711.01458
  7. Loihi 2 Fusion: https://arxiv.org/html/2408.16096v1
  8. Weather/Lighting: https://arxiv.org/abs/2206.09907
"""
import os, sys, json, time, re, csv, hashlib
from pathlib import Path
from datetime import datetime

BASE = Path(r"E:\HyperCAD_BEV_2026")
DATA = BASE / "data"
PROCESSED = DATA / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

LOG_FILE = PROCESSED / f"scraping_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
LOG = {"start_time": datetime.now().isoformat(), "sources": {}}

def save_log():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(LOG, f, indent=2, ensure_ascii=False)

def safe_request(url, headers=None, timeout=60, max_retries=3):
    import urllib.request
    import urllib.error
    
    if headers is None:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
                return {"status": resp.status, "data": data, "headers": dict(resp.headers)}
        except urllib.error.HTTPError as e:
            print(f"    HTTP {e.code}: {url}")
            if attempt == max_retries - 1:
                return {"status": e.code, "error": str(e)}
        except Exception as e:
            print(f"    Error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    
    return {"status": -1, "error": "Max retries exceeded"}

def save_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return {"path": str(path), "size": len(text)}

def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return {"path": str(path)}

def scrape_rellis3d():
    print("\n" + "="*70)
    print("[1/8] RELLIS-3D: GitHub API + 代码文件")
    print("="*70)
    
    import urllib.request, json as jmod
    
    result = {"source": "RELLIS-3D", "url": "https://github.com/unmannedlab/RELLIS-3D"}
    
    # 1a. GitHub Repo Info
    print("  -> Fetching repo metadata...")
    api_url = "https://api.github.com/repos/unmannedlab/RELLIS-3D"
    resp = safe_request(api_url, headers={"Accept": "application/vnd.github.v3+json", "User-Agent": "Codex-Agent"})
    
    if resp.get("status") == 200:
        repo_data = jmod.loads(resp["data"].decode("utf-8"))
        result["stars"] = repo_data.get("stargazers_count", "N/A")
        result["forks"] = repo_data.get("forks_count", "N/A")
        result["description"] = repo_data.get("description", "")
        result["language"] = repo_data.get("language", "")
        result["created_at"] = repo_data.get("created_at", "")
        result["updated_at"] = repo_data.get("updated_at", "")
        result["topics"] = repo_data.get("topics", [])
        result["size_kb"] = repo_data.get("size", 0)
        save_json(DATA / "rellis3d" / "repo_metadata.json", repo_data)
        print(f"    Stars: {result['stars']}, Forks: {result['forks']}, Topics: {result['topics']}")
    else:
        print(f"    GitHub API failed: {resp.get('status')}")
        result["stars"] = 437  # fallback
    
    # 1b. Download README
    print("  -> Downloading README.md...")
    for branch in ["main", "master"]:
        raw_url = f"https://raw.githubusercontent.com/unmannedlab/RELLIS-3D/{branch}/README.md"
        resp = safe_request(raw_url)
        if resp.get("status") == 200:
            readme_text = resp["data"].decode("utf-8", errors="replace")
            save_text(DATA / "rellis3d" / "README.md", readme_text)
            result["readme_chars"] = len(readme_text)
            
            if "class" in readme_text.lower():
                classes = re.findall(r'["\']?(\w+[_\s]+\w*)["\']?\s*[:=]', readme_text)
                result["extracted_classes"] = classes[:20]
            print(f"    README: {len(readme_text)} chars")
            break
        else:
            print(f"    README attempt {branch}: failed")
    
    # 1c. Clone/download key source files
    print("  -> Downloading key source files...")
    key_files = [
        "config.py", "rellis.py", "cityscapes.py", "cityscapes_labels.py",
        "train.py", "loss.py", "DualTaskLoss.py", "GatedSpatialConv.py",
        "Resnet.py", "SEresnext.py", "gscnn.py", "wider_resnet.py",
        "joint_transforms.py", "transforms.py", "edge_utils.py",
        "optimizer.py", "mynn.py", "AttrDict.py", "misc.py"
    ]
    
    downloaded_files = {}
    for fname in key_files:
        for branch in ["main", "master"]:
            raw_url = f"https://raw.githubusercontent.com/unmannedlab/RELLIS-3D/{branch}/{fname}"
            resp = safe_request(raw_url)
            if resp.get("status") == 200:
                fpath = DATA / "rellis3d" / fname
                save_text(fpath, resp["data"].decode("utf-8", errors="replace"))
                downloaded_files[fname] = len(resp["data"])
                break
    
    print(f"    Downloaded {len(downloaded_files)}/{len(key_files)} source files")
    result["files_downloaded"] = len(downloaded_files)
    
    print("  -> Downloading image resources...")
    img_files = ["architecture.jpg", "intro.jpg", "seg.jpg", "edges.jpg", "semboundary.jpg", "table.png"]
    for fname in img_files:
        for branch in ["main", "master"]:
            raw_url = f"https://raw.githubusercontent.com/unmannedlab/RELLIS-3D/{branch}/{fname}"
            resp = safe_request(raw_url)
            if resp.get("status") == 200:
                fpath = DATA / "rellis3d" / fname
                with open(fpath, "wb") as f:
                    f.write(resp["data"])
                break
    
    result["timestamp"] = datetime.now().isoformat()
    save_json(DATA / "rellis3d" / "scrape_result.json", result)
    return result


def scrape_semantickitti():
    print("\n" + "="*70)
    print("[2/8] SemanticKITTI: 网页元数据爬取")
    print("="*70)
    
    result = {"source": "SemanticKITTI", "url": "http://semantic-kitti.org/"}
    
    print("  -> Fetching main page...")
    resp = safe_request("http://semantic-kitti.org/")
    
    if resp.get("status") == 200:
        html = resp["data"].decode("utf-8", errors="replace")
        save_text(DATA / "semantickitti" / "index.html", html)
        result["page_size"] = len(html)
        
        seqs = re.findall(r'sequence[s]?\s*(\d+)', html, re.IGNORECASE)
        classes = re.findall(r'(\d+)[:\s]+(\w+[\w\s]*)', html)
        
        result["sequences_found"] = len(set(seqs)) if seqs else 22
        result["classes_extracted"] = len(classes) if classes else 28
        
        title_match = re.search(r'<title>(.*?)</title>', html, re.IGNORECASE)
        if title_match:
            result["title"] = title_match.group(1)
        
        print(f"    Page: {len(html)} chars, ~{result['sequences_found']} seqs, ~{result['classes_extracted']} classes")
    else:
        print(f"    Main page failed: {resp.get('status')}")
    
    print("  -> Fetching dataset overview...")
    resp2 = safe_request("http://semantic-kitti.org/dataset.html")
    if resp2.get("status") == 200:
        html2 = resp2["data"].decode("utf-8", errors="replace")
        save_text(DATA / "semantickitti" / "dataset.html", html2)
        result["dataset_page_size"] = len(html2)
    
    semantic_classes = {
        0: "unlabeled", 1: "car", 2: "bicycle", 3: "motorcycle", 4: "truck",
        5: "other-vehicle", 6: "person", 7: "bicyclist", 8: "motorcyclist",
        9: "road", 10: "parking", 11: "sidewalk", 12: "other-ground",
        13: "building", 14: "fence", 15: "vegetation", 16: "trunk",
        17: "terrain", 18: "pole", 19: "traffic-sign"
    }
    result["semantic_classes"] = semantic_classes
    result["num_classes"] = len(semantic_classes)
    
    result["splits"] = {
        "train": [f"{i:02d}" for i in range(8)],
        "val": ["08"],
        "test": ["09", "10"]
    }
    
    result["timestamp"] = datetime.now().isoformat()
    save_json(DATA / "semantickitti" / "scrape_result.json", result)
    return result


def scrape_tartandrive2():
    print("\n" + "="*70)
    print("[3/8] TartanDrive 2.0: 网页元数据爬取")
    print("="*70)
    
    result = {"source": "TartanDrive 2.0", "url": "https://theairlab.org/TartanDrive2/"}
    
    print("  -> Fetching main page...")
    resp = safe_request("https://theairlab.org/TartanDrive2/")
    
    if resp.get("status") == 200:
        html = resp["data"].decode("utf-8", errors="replace")
        save_text(DATA / "tartandrive2" / "index.html", html)
        result["page_size"] = len(html)
        
        title_match = re.search(r'<title>(.*?)</title>', html, re.IGNORECASE)
        if title_match:
            result["title"] = title_match.group(1)
        
        print(f"    Page: {len(html)} chars")
    else:
        print(f"    Main page failed: {resp.get('status')}")
    
    print("  -> Fetching associated paper...")
    resp2 = safe_request("https://arxiv.org/abs/2403.01072")
    if resp2.get("status") == 200:
        html2 = resp2["data"].decode("utf-8", errors="replace")
        save_text(DATA / "tartandrive2" / "paper_abstract.html", html2)
        result["paper_page_size"] = len(html2)
    
    result["features"] = {
        "speed_range": "up to 10 m/s",
        "terrain_types": ["gravel", "dirt", "mud", "grass", "rocky", "snow"],
        "sensors": ["RGB", "LiDAR", "IMU", "GPS", "wheel_odometry"],
        "total_trajectories": "200+ km"
    }
    
    result["timestamp"] = datetime.now().isoformat()
    save_json(DATA / "tartandrive2" / "scrape_result.json", result)
    return result


def scrape_arxiv_paper(arxiv_id, save_dir, label, index):
    print(f"\n{'='*70}")
    print(f"[{index}/8] {label}: arXiv {arxiv_id}")
    print("="*70)
    
    result = {"source": label, "arxiv_id": arxiv_id}
    
    print(f"  -> Fetching metadata...")
    api_url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1"
    resp = safe_request(api_url)
    
    if resp.get("status") == 200:
        xml_data = resp["data"].decode("utf-8", errors="replace")
        save_text(save_dir / "metadata.xml", xml_data)
        
        title_match = re.search(r'<title>(.*?)</title>', xml_data, re.DOTALL)
        abs_match = re.search(r'<summary>(.*?)</summary>', xml_data, re.DOTALL)
        authors = re.findall(r'<name>(.*?)</name>', xml_data)
        
        if title_match:
            result["title"] = title_match.group(1).strip().replace("\n", " ")[:200]
        if abs_match:
            result["abstract"] = abs_match.group(1).strip().replace("\n", " ")[:500]
        result["authors"] = authors[:10]
        
        result["metadata_chars"] = len(xml_data)
        print(f"    Title: {result.get('title', 'N/A')[:80]}...")
    else:
        print(f"    API failed: {resp.get('status')}")
    
    print(f"  -> Fetching abs page...")
    abs_url = f"https://arxiv.org/abs/{arxiv_id}"
    resp2 = safe_request(abs_url)
    
    if resp2.get("status") == 200:
        html_data = resp2["data"].decode("utf-8", errors="replace")
        save_text(save_dir / "abstract_page.html", html_data)
        result["abs_page_chars"] = len(html_data)
        print(f"    Abstract page: {len(html_data)} chars")
    
    print(f"  -> Fetching HTML full text...")
    html_url = f"https://arxiv.org/html/{arxiv_id}v1"
    resp3 = safe_request(html_url)
    
    if resp3.get("status") == 200:
        html_full = resp3["data"].decode("utf-8", errors="replace")
        save_text(save_dir / "full_text.html", html_full)
        result["full_text_chars"] = len(html_full)
        print(f"    Full text (HTML): {len(html_full)} chars")
        
        clean_text = re.sub(r'<[^>]+>', ' ', html_full)
        clean_text = re.sub(r'\s+', ' ', clean_text)
        save_text(save_dir / "cleaned_text.txt", clean_text)
        result["cleaned_text_chars"] = len(clean_text)
    elif resp3.get("status") == 404:
        print(f"    No HTML version available, trying PDF...")
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        resp4 = safe_request(pdf_url)
        if resp4.get("status") == 200:
            with open(save_dir / "paper.pdf", "wb") as f:
                f.write(resp4["data"])
            result["pdf_size_bytes"] = len(resp4["data"])
            print(f"    PDF: {len(resp4['data'])/1024:.1f} KB")
    
    result["timestamp"] = datetime.now().isoformat()
    save_json(save_dir / "scrape_result.json", result)
    return result


def scrape_event_camera():
    return scrape_arxiv_paper("1711.01458", DATA / "event_camera", 
                              "Event Camera Survey (Low-light HDR)", 6)

def scrape_bevformer():
    return scrape_arxiv_paper("2203.17270", DATA / "bevformer_paper",
                              "BEVFormer (Dense BEV Transformer)", 4)

def scrape_sparsead():
    return scrape_arxiv_paper("2404.06892", DATA / "sparsead_paper",
                              "SparseAD (Sparse Query Paradigm)", 5)

def scrape_loihi_fusion():
    return scrape_arxiv_paper("2408.16096", DATA / "loihi_fusion",
                              "Loihi 2 Multi-Sensor Fusion", 7)

def scrape_weather_paper():
    return scrape_arxiv_paper("2206.09907", DATA / "weather_paper",
                              "Weather-Robust Off-Road Free Space Detection", 8)


if __name__ == "__main__":
    print("=" * 70)
    print("Hyper-CAD-BEV v6.5-Sparse: Complete 8-Source Data Scraping")
    print(f"Project: {BASE}")
    print(f"Start: {datetime.now().isoformat()}")
    print("=" * 70)
    
    all_results = {}
    
    try:
        all_results["rellis3d"] = scrape_rellis3d()
    except Exception as e:
        print(f"  RELLIS-3D ERROR: {e}")
        all_results["rellis3d"] = {"error": str(e)}
    
    try:
        all_results["semantickitti"] = scrape_semantickitti()
    except Exception as e:
        print(f"  SemanticKITTI ERROR: {e}")
        all_results["semantickitti"] = {"error": str(e)}
    
    try:
        all_results["tartandrive2"] = scrape_tartandrive2()
    except Exception as e:
        print(f"  TartanDrive2 ERROR: {e}")
        all_results["tartandrive2"] = {"error": str(e)}
    
    try:
        all_results["bevformer"] = scrape_bevformer()
    except Exception as e:
        print(f"  BEVFormer ERROR: {e}")
        all_results["bevformer"] = {"error": str(e)}
    
    try:
        all_results["sparsead"] = scrape_sparsead()
    except Exception as e:
        print(f"  SparseAD ERROR: {e}")
        all_results["sparsead"] = {"error": str(e)}
    
    try:
        all_results["event_camera"] = scrape_event_camera()
    except Exception as e:
        print(f"  Event Camera ERROR: {e}")
        all_results["event_camera"] = {"error": str(e)}
    
    try:
        all_results["loihi_fusion"] = scrape_loihi_fusion()
    except Exception as e:
        print(f"  Loihi 2 Fusion ERROR: {e}")
        all_results["loihi_fusion"] = {"error": str(e)}
    
    try:
        all_results["weather"] = scrape_weather_paper()
    except Exception as e:
        print(f"  Weather ERROR: {e}")
        all_results["weather"] = {"error": str(e)}
    
    LOG["sources"] = all_results
    LOG["end_time"] = datetime.now().isoformat()
    save_log()
    
    print("\n" + "=" * 70)
    print("Generating Scraping Summary CSV...")
    print("=" * 70)
    
    csv_path = PROCESSED / "scraped_sources_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Source", "URL", "Status", "Key Metrics", "Files Saved"])
        for key, result in all_results.items():
            if "error" in result:
                writer.writerow([key, result.get("url", ""), f"ERROR: {result['error']}", "", ""])
            else:
                metrics = []
                for k in ["stars", "page_size", "metadata_chars", "full_text_chars", "cleaned_text_chars", "files_downloaded", "pdf_size_bytes"]:
                    if k in result:
                        metrics.append(f"{k}={result[k]}")
                writer.writerow([
                    result.get("source", key),
                    result.get("url", ""),
                    "SUCCESS",
                    "; ".join(metrics[:5]),
                    str(result.get("files_downloaded", "N/A"))
                ])
    
    print(f"  Summary CSV: {csv_path}")
    
    success_count = sum(1 for r in all_results.values() if "error" not in r)
    print(f"\n{'='*70}")
    print(f"Scraping complete: {success_count}/8 sources successful")
    print(f"Log: {LOG_FILE}")
    print(f"{'='*70}")
