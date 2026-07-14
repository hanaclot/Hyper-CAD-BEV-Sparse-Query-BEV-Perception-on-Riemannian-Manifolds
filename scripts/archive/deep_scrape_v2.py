# -*- coding: utf-8 -*-
import json, os, sys, time, csv, requests, re
from datetime import datetime
import xml.etree.ElementTree as ET

PROJECT_ROOT = r"E:\Hyper-CAD-BEV-Experiments"
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "crawled")
os.makedirs(DATA_DIR, exist_ok=True)

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

log("="*60)
log("DEEP SCRAPE: Real Benchmark Data Collection")
log("="*60)

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# 1. SemanticKITTI leaderboard
log("1/7: SemanticKITTI leaderboard...")
sk_url = "http://semantic-kitti.org/tasks.html"
try:
    resp = requests.get(sk_url, headers=headers, timeout=30)
    html = resp.text
    sk_entries = []
    table_pat = r"<table[^>]*>(.*?)</table>"
    tables = re.findall(table_pat, html, re.DOTALL | re.IGNORECASE)
    for tm in tables:
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", tm, re.DOTALL)
        for row in rows:
            tds = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)
            if len(tds) >= 2:
                vals = [re.sub(r"<[^>]+>", "", td).strip() for td in tds]
                if vals[0] and vals[0] not in ["Method","Rank","#"]:
                    sk_entries.append({"method": vals[0], "metric": vals[1] if len(vals)>1 else "", "all": vals})
    with open(os.path.join(DATA_DIR, "semantickitti", "tasks_deep.html"), "w", encoding="utf-8") as f:
        f.write(html)
    save_json({"entries": sk_entries, "total": len(sk_entries), "source": sk_url}, os.path.join(DATA_DIR, "semantickitti", "leaderboard_deep.json"))
    log(f"  SK leaderboard: {len(sk_entries)} entries from tasks.html")
except Exception as e:
    log(f"  SK error: {e}")

# 2. RELLIS-3D
log("2/7: RELLIS-3D GitHub...")
try:
    api_url = "https://api.github.com/repos/unmannedlab/RELLIS-3D"
    resp = requests.get(api_url, headers=headers, timeout=30)
    repo = resp.json()
    save_json({"stars": repo.get("stargazers_count",0), "forks": repo.get("forks_count",0),
               "open_issues": repo.get("open_issues_count",0), "description": repo.get("description",""),
               "updated_at": repo.get("updated_at",""), "topics": repo.get("topics",[])},
              os.path.join(DATA_DIR, "rellis3d", "repo_analysis.json"))
    log(f"  RELLIS-3D: {repo.get('stargazers_count',0)} stars")
except Exception as e:
    log(f"  RELLIS error: {e}")

# 3. ArXiv deep papers
log("3/7: arXiv deep metadata...")
arxiv_ids = [
    ("2206.09907","Weather-Robust Off-road Detection"),
    ("2203.17270","BEVFormer"),
    ("2404.06892","SparseAD"),
    ("1711.01458","Event Camera Survey"),
    ("2408.16096","Loihi-2 Sensor Fusion"),
    ("2309.15654","Neuromorphic BEV"),
    ("2308.09244","SparseBEV"),
    ("2409.09350","OPUS Sparse Occupancy"),
    ("2211.14710","3DPPE"),
    ("2304.08463","Wide-Baseline Views"),
    ("2104.13283","Riemannian DL"),
    ("2404.01586","Reaction-Diffusion MFC"),
    ("2512.08237","Fast-BEV++"),
]
arxiv_deep = {}
for pid, desc in arxiv_ids:
    try:
        url = f"https://export.arxiv.org/api/query?id_list={pid}&max_results=1"
        resp = requests.get(url, headers=headers, timeout=30)
        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
        entry = root.find("atom:entry", ns)
        if entry is not None:
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)
            pub = entry.find("atom:published", ns)
            authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns) if a.find("atom:name", ns) is not None]
            cats = [c.get("term") for c in entry.findall("atom:category", ns)]
            arxiv_deep[pid] = {
                "title": title_el.text.strip() if title_el is not None else desc,
                "summary": (summary_el.text[:500] if summary_el is not None else ""),
                "published": pub.text if pub is not None else "",
                "authors": authors,
                "categories": cats,
            }
            log(f"  {pid}: {title_el.text[:60] if title_el is not None else desc}")
        time.sleep(0.5)
    except Exception as e:
        arxiv_deep[pid] = {"title": desc, "error": str(e)[:200]}
        log(f"  {pid}: ERROR {str(e)[:60]}")

save_json(arxiv_deep, os.path.join(DATA_DIR, "arxiv", "paper_metadata_deep.json"))
log(f"  arXiv: {len(arxiv_deep)} papers indexed")

# 4. TartanDrive2
log("4/7: TartanDrive2...")
try:
    url = "https://theairlab.org/TartanDrive2/"
    resp = requests.get(url, headers=headers, timeout=30)
    html = resp.text
    metrics = {}
    patterns = {"hours": r"(\d+[\d,]*)\s*hours?", "km": r"(\d+[\d,]*)\s*km", "trajectories": r"(\d+[\d,]*)\s*trajector", "frames": r"(?i)(\d+[\d,]*)\s*frames?", "sensors": r"(\d+[\d,]*)\s*sensors?"}
    for k, p in patterns.items():
        m = re.search(p, html)
        if m: metrics[k] = m.group(1)
    with open(os.path.join(DATA_DIR, "tartandrive2", "page_deep.html"), "w", encoding="utf-8") as f:
        f.write(html)
    save_json({"metrics": metrics, "html_size": len(html)}, os.path.join(DATA_DIR, "tartandrive2", "dataset_stats.json"))
    log(f"  TartanDrive2 metrics: {metrics}")
except Exception as e:
    log(f"  TartanDrive2: {e}")

# 5. Event camera papers
log("5/7: Event camera related papers...")
try:
    url = "http://export.arxiv.org/api/query?search_query=all:event+camera+driving+autonomous+benchmark&max_results=20&sortBy=relevance"
    resp = requests.get(url, headers=headers, timeout=30)
    root = ET.fromstring(resp.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers = []
    for entry in root.findall("atom:entry", ns):
        t = entry.find("atom:title", ns)
        s = entry.find("atom:summary", ns)
        papers.append({"title": t.text.strip() if t is not None else "", "summary": (s.text[:200] if s is not None else "")})
    save_json({"papers": papers, "count": len(papers)}, os.path.join(DATA_DIR, "event_camera", "related_papers.json"))
    log(f"  Event camera: {len(papers)} papers")
except Exception as e:
    log(f"  Event camera: {e}")

# 6. Loihi-2 neuromorphic
log("6/7: Loihi-2 neuromorphic fusion...")
try:
    url = "http://export.arxiv.org/api/query?search_query=all:loihi+neuromorphic+sensor+fusion+perception&max_results=15&sortBy=relevance"
    resp = requests.get(url, headers=headers, timeout=30)
    root = ET.fromstring(resp.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers = []
    for entry in root.findall("atom:entry", ns):
        t = entry.find("atom:title", ns)
        papers.append({"title": t.text.strip() if t is not None else ""})
    save_json({"papers": papers, "count": len(papers)}, os.path.join(DATA_DIR, "neuromorphic", "loihi_related.json"))
    log(f"  Loihi-2: {len(papers)} papers")
except Exception as e:
    log(f"  Loihi: {e}")

# 7. Weather robustness + Sparse query
log("7/7: Weather robustness and sparse query...")
try:
    weather_url = "http://export.arxiv.org/api/query?search_query=all:off-road+weather+robust+BEV+perception&max_results=15"
    resp = requests.get(weather_url, headers=headers, timeout=30)
    root = ET.fromstring(resp.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers = []
    for entry in root.findall("atom:entry", ns):
        t = entry.find("atom:title", ns)
        papers.append({"title": t.text.strip() if t is not None else ""})
    save_json({"papers": papers, "count": len(papers)}, os.path.join(DATA_DIR, "weather_robustness", "related_papers.json"))
    log(f"  Weather robustness: {len(papers)} papers")

    sparse_url = "http://export.arxiv.org/api/query?search_query=all:sparse+query+BEV+3D+detection&max_results=20&sortBy=relevance"
    resp2 = requests.get(sparse_url, headers=headers, timeout=30)
    root2 = ET.fromstring(resp2.text)
    papers2 = []
    for entry in root2.findall("atom:entry", ns):
        t = entry.find("atom:title", ns)
        papers2.append({"title": t.text.strip() if t is not None else ""})
    save_json({"papers": papers2, "count": len(papers2)}, os.path.join(DATA_DIR, "benchmarks", "sparse_query_papers.json"))
    log(f"  Sparse query: {len(papers2)} papers")
except Exception as e:
    log(f"  Weather/sparse: {e}")

log("\n" + "="*60)
log("DEEP SCRAPE COMPLETE - ALL DATA FROM REAL PUBLIC SOURCES")
log("="*60)
