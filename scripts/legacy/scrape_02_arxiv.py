# -*- coding: utf-8 -*-
import json, time, re, urllib.request, urllib.error
from pathlib import Path
from datetime import datetime

BASE = Path(r"E:\HyperCAD_BEV_2026")
DATA = BASE / "data"

def fetch(url, timeout=30):
    headers = {"User-Agent": "HyperCAD-BEV/1.0 (mailto:research@example.com)"}
    for attempt in range(2):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return {"status": resp.status, "data": resp.read().decode("utf-8", errors="replace")}
        except Exception as e:
            if attempt == 1:
                return {"status": -1, "error": str(e)}
            time.sleep(2)

def scrape_arxiv(arxiv_id, save_dir, label):
    print(f"\n--- {label} (arXiv:{arxiv_id}) ---")
    save_dir.mkdir(parents=True, exist_ok=True)
    result = {"source": label, "arxiv_id": arxiv_id}
    
    # API query
    api = f"https://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1"
    resp = fetch(api, timeout=45)
    
    if resp["status"] == 200:
        xml = resp["data"]
        (save_dir / "metadata.xml").write_text(xml, encoding="utf-8")
        result["metadata_chars"] = len(xml)
        
        # Extract
        t = re.search(r'<title>(.*?)</title>', xml, re.DOTALL)
        a = re.search(r'<summary>(.*?)</summary>', xml, re.DOTALL)
        if t: result["title"] = t.group(1).strip().replace("\n"," ")[:200]
        if a: result["abstract"] = a.group(1).strip().replace("\n"," ")[:500]
        authors = re.findall(r'<name>(.*?)</name>', xml)
        result["authors"] = authors[:8]
        print(f"  Title: {result.get('title','?')[:100]}")
        print(f"  Authors: {len(authors)} total")
    else:
        print(f"  API FAILED: {resp.get('error','?')}")
        result["api_error"] = resp.get("error","")
    
    # Abstract page
    resp2 = fetch(f"https://arxiv.org/abs/{arxiv_id}")
    if resp2["status"] == 200:
        (save_dir / "abstract_page.html").write_text(resp2["data"], encoding="utf-8")
        result["abs_chars"] = len(resp2["data"])
        print(f"  Abs page: {len(resp2['data'])} chars")
    
    # Try HTML version
    resp3 = fetch(f"https://arxiv.org/html/{arxiv_id}v1")
    if resp3["status"] == 200:
        html = resp3["data"]
        (save_dir / "full_text.html").write_text(html, encoding="utf-8")
        result["full_chars"] = len(html)
        clean = re.sub(r'<[^>]+>',' ', html)
        clean = re.sub(r'\s+',' ', clean)
        (save_dir / "cleaned_text.txt").write_text(clean, encoding="utf-8")
        result["clean_chars"] = len(clean)
        print(f"  Full HTML: {len(html)} chars, cleaned: {len(clean)} chars")
    else:
        print(f"  No HTML version")
    
    result["timestamp"] = datetime.now().isoformat()
    (save_dir / "scrape_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result

# Scrape all 5 arXiv papers
results = {}
for aid, dname, label in [
    ("2203.17270", "bevformer_paper", "BEVFormer"),
    ("2404.06892", "sparsead_paper", "SparseAD"),
    ("1711.01458", "event_camera", "Event Camera"),
    ("2408.16096", "loihi_fusion", "Loihi 2 Fusion"),
    ("2206.09907", "weather_paper", "Weather/Lighting"),
]:
    try:
        results[dname] = scrape_arxiv(aid, DATA / dname, label)
    except Exception as e:
        print(f"ERROR {label}: {e}")
        results[dname] = {"error": str(e)}

# Summary
PROCESSED = DATA / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)
LOG = {"scraped_at": datetime.now().isoformat(), "results": results}
(PROCESSED / "arxiv_scrape_log.json").write_text(json.dumps(LOG, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"\n=== Done: {sum(1 for r in results.values() if 'error' not in r)}/5 arXiv papers scraped ===")
