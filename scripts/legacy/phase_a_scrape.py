import urllib.request, json, os, sys, time, hashlib, xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = r"E:\Hyper-CAD-BEV-Experiments"
CRAWLED_DIR = os.path.join(PROJECT_ROOT, "data", "crawled")
os.makedirs(CRAWLED_DIR, exist_ok=True)

def fetch_json(url, timeout=30, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json"
            })
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            if i == retries - 1:
                print(f"  FAIL: {url} - {e}")
                return None
            time.sleep(2)

def fetch_text(url, timeout=30):
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  FAIL: {url} - {e}")
        return None

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_text(text, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

print("=" * 70)
print("Phase A: Deep Data Scraping - Expanding Real Data Volume")
print(f"Start: {datetime.now().isoformat()}")
print("=" * 70)

# A1: SemanticKITTI - Deep crawl leaderboard & dataset details
print("\n--- A1: SemanticKITTI Deep Crawl ---")
sk_dir = os.path.join(CRAWLED_DIR, "semantickitti")
os.makedirs(sk_dir, exist_ok=True)

tasks = ["semantic-segmentation", "panoptic-segmentation", "4d-panoptic-segmentation",
         "moving-object-segmentation", "scene-completion"]
for task in tasks:
    url = f"http://semantic-kitti.org/tasks/{task}.html"
    html = fetch_text(url)
    if html:
        save_text(html, os.path.join(sk_dir, f"task_{task}.html"))
        print(f"  Fetched: {task} ({len(html)} chars)")
    time.sleep(0.5)

dataset_html = fetch_text("http://semantic-kitti.org/dataset.html")
if dataset_html:
    save_text(dataset_html, os.path.join(sk_dir, "dataset_full.html"))
    print(f"  Dataset page: {len(dataset_html)} chars")

for lb in ["semantic_single", "semantic_multi", "panoptic", "panoptic4d", "mos", "completion"]:
    lb_path = os.path.join(sk_dir, f"{lb}.json")
    if os.path.exists(lb_path):
        with open(lb_path) as f:
            data = json.load(f)
        if isinstance(data, dict) and "data" in data:
            print(f"  Leaderboard [{lb}]: {len(data['data'])} entries")
        elif isinstance(data, list):
            print(f"  Leaderboard [{lb}]: {len(data)} entries")

# A2: RELLIS-3D GitHub
print("\n--- A2: RELLIS-3D Deep GitHub API Crawl ---")
rellis_dir = os.path.join(CRAWLED_DIR, "rellis3d")
os.makedirs(rellis_dir, exist_ok=True)

repo_data = fetch_json("https://api.github.com/repos/unmannedlab/RELLIS-3D")
if repo_data:
    save_json(repo_data, os.path.join(rellis_dir, "github_repo.json"))
    print(f"  Stars: {repo_data.get('stargazers_count')}, Forks: {repo_data.get('forks_count')}")

tree_data = fetch_json("https://api.github.com/repos/unmannedlab/RELLIS-3D/git/trees/main?recursive=1")
if tree_data:
    save_json(tree_data, os.path.join(rellis_dir, "full_file_tree.json"))
    files = tree_data.get("tree", [])
    py_files = [f for f in files if f["path"].endswith(".py")]
    print(f"  Files: {len(files)}, Python: {len(py_files)}")

commits_data = fetch_json("https://api.github.com/repos/unmannedlab/RELLIS-3D/commits?per_page=50")
if commits_data:
    save_json(commits_data, os.path.join(rellis_dir, "recent_commits.json"))
    print(f"  Recent commits: {len(commits_data)}")

# A3: TartanDrive2
print("\n--- A3: TartanDrive2 Deep Crawl ---")
tartan_dir = os.path.join(CRAWLED_DIR, "tartandrive2")
os.makedirs(tartan_dir, exist_ok=True)

tartan_html = fetch_text("https://theairlab.org/TartanDrive2/")
if tartan_html:
    save_text(tartan_html, os.path.join(tartan_dir, "website_full_v2.html"))
    print(f"  Website: {len(tartan_html)} chars")

info_html = fetch_text("https://theairlab.org/dataset/tartan-drive-2/")
if info_html:
    save_text(info_html, os.path.join(tartan_dir, "dataset_info.html"))
    print(f"  Dataset info: {len(info_html)} chars")

# A4: ArXiv Expanded Paper Crawl
print("\n--- A4: ArXiv Expanded Paper Crawl ---")
arxiv_dir = os.path.join(CRAWLED_DIR, "arxiv")
os.makedirs(arxiv_dir, exist_ok=True)

search_queries = [
    ("rural+BEV+perception", "search_rural_bev_v2"),
    ("offroad+autonomous+driving+semantic+segmentation", "search_offroad_semantic"),
    ("LiDAR+BEV+3D+detection+terrain", "search_lidar_bev_terrain"),
    ("event+camera+autonomous+driving+perception", "search_event_camera_driving"),
    ("sparse+query+3D+detection+transformer", "search_sparse_query_detector"),
    ("riemannian+manifold+deep+learning+geometry", "search_riemannian_dl"),
    ("neuromorphic+spiking+neural+network+robotics", "search_neuromorphic_robotics"),
    ("partial+differential+equation+neural+field+BEV", "search_pde_neural_field"),
    ("edge+computing+autonomous+vehicle+perception", "search_edge_av_perception"),
    ("multi-sensor+fusion+LiDAR+camera+rural+off-road", "search_multisensor_offroad"),
    ("implicit+neural+representation+3D+perception", "search_inr_av"),
    ("variational+inference+computer+vision+3D", "search_variational_cv3d"),
]

for query, tag in search_queries:
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=30&sortBy=relevance"
    xml_str = fetch_text(url, timeout=45)
    if xml_str:
        save_text(xml_str, os.path.join(arxiv_dir, f"{tag}.xml"))
        count = xml_str.count("<entry>")
        print(f"  [{tag}]: {count} papers")
    time.sleep(1.5)

key_papers = [
    "2404.06892", "2203.17270", "2408.16096", "2206.09907",
    "1711.01458", "2308.09244", "2409.09350", "2211.14710",
    "2304.08463", "2404.01586", "2104.13283", "2512.08237",
]
for pid in key_papers:
    url = f"http://export.arxiv.org/api/query?id_list={pid}"
    xml_str = fetch_text(url)
    if xml_str:
        save_text(xml_str, os.path.join(arxiv_dir, f"{pid}_v2.xml"))
    time.sleep(1)

# A5: Event Camera benchmarks
print("\n--- A5: Event Camera Data ---")
event_dir = os.path.join(CRAWLED_DIR, "event_camera")
os.makedirs(event_dir, exist_ok=True)

event_html = fetch_text("https://arxiv.org/abs/1711.01458")
if event_html:
    save_text(event_html, os.path.join(event_dir, "survey_page_v2.html"))
    print(f"  Event camera survey: {len(event_html)} chars")

eq_xml = fetch_text("http://export.arxiv.org/api/query?search_query=all:event+camera+benchmark+autonomous+driving&start=0&max_results=20", timeout=45)
if eq_xml:
    save_text(eq_xml, os.path.join(event_dir, "benchmark_search.xml"))
    print(f"  Event benchmark search: {eq_xml.count('<entry>')} papers")

# A6: Edge fusion & Loihi papers
print("\n--- A6: Edge/Neuromorphic Computing Papers ---")
neuromorphic_dir = os.path.join(CRAWLED_DIR, "neuromorphic")
os.makedirs(neuromorphic_dir, exist_ok=True)

loihi_html = fetch_text("https://arxiv.org/html/2408.16096v1")
if loihi_html:
    save_text(loihi_html, os.path.join(neuromorphic_dir, "loihi_fusion_full.html"))
    print(f"  Loihi fusion HTML: {len(loihi_html)} chars")

loihi_xml = fetch_text("http://export.arxiv.org/api/query?search_query=all:Loihi+2+benchmark+energy+efficiency&start=0&max_results=15", timeout=45)
if loihi_xml:
    save_text(loihi_xml, os.path.join(neuromorphic_dir, "loihi_benchmarks.xml"))
    print(f"  Loihi benchmarks: {loihi_xml.count('<entry>')} papers")

snn_xml = fetch_text("http://export.arxiv.org/api/query?search_query=all:spiking+neural+network+survey+autonomous+perception&start=0&max_results=15", timeout=45)
if snn_xml:
    save_text(snn_xml, os.path.join(neuromorphic_dir, "snn_survey.xml"))
    print(f"  SNN survey: {snn_xml.count('<entry>')} papers")

# A7: Weather/illumination robustness papers
print("\n--- A7: Weather Robustness Papers ---")
weather_dir = os.path.join(CRAWLED_DIR, "weather_robustness")
os.makedirs(weather_dir, exist_ok=True)

weather_xml = fetch_text("http://export.arxiv.org/api/query?search_query=all:adverse+weather+robust+perception+autonomous+driving+BEV&start=0&max_results=20", timeout=45)
if weather_xml:
    save_text(weather_xml, os.path.join(weather_dir, "robustness_search.xml"))
    print(f"  Weather robustness: {weather_xml.count('<entry>')} papers")

# A8: Compile summary
print("\n" + "=" * 70)
print("Phase A Complete: Data Crawling Summary")
print("=" * 70)

total_files = 0
total_size = 0
for root, dirs, files in os.walk(CRAWLED_DIR):
    for f in files:
        fp = os.path.join(root, f)
        total_files += 1
        total_size += os.path.getsize(fp)

summary = {
    "phase": "A - Deep Data Crawling",
    "timestamp": datetime.now().isoformat(),
    "total_files_crawled": total_files,
    "total_size_bytes": total_size,
    "total_size_mb": round(total_size / (1024 * 1024), 2),
    "directories": {
        "semantickitti": len(os.listdir(sk_dir)),
        "rellis3d": len(os.listdir(rellis_dir)),
        "tartandrive2": len(os.listdir(tartan_dir)),
        "arxiv": len(os.listdir(arxiv_dir)),
        "event_camera": len(os.listdir(event_dir)),
        "neuromorphic": len(os.listdir(neuromorphic_dir)),
        "weather_robustness": len(os.listdir(weather_dir)),
    },
}
save_json(summary, os.path.join(CRAWLED_DIR, "phase_a_summary.json"))
print(json.dumps(summary, indent=2))
print("\nDone.")
