import sys, json, os, requests, time
sys.stdout.reconfigure(encoding="utf-8")
session = requests.Session()
session.headers.update({"User-Agent": "Hyper-CAD-BEV-Research/1.0 (academic)"})
sk_dir = r"E:\Hyper-CAD-BEV-Experiments\data\crawled\semantickitti"
os.makedirs(sk_dir, exist_ok=True)

# Fetch all SemanticKITTI JSON data files
json_files = [
    "semantic_single.json", "semantic_multi.json",
    "panoptic.json", "panoptic4d.json", "mos.json", "completion.json"
]
for jf in json_files:
    url = f"http://semantic-kitti.org/data/{jf}"
    print(f"Fetching: {url}...")
    try:
        r = session.get(url, timeout=30)
        if r.status_code == 200:
            data = r.json()
            out_path = os.path.join(sk_dir, jf)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            # Show summary
            if "data" in data:
                print(f"  OK: {len(data['data'])} entries, keys: {list(data.keys())}")
                if data["data"]:
                    first = data["data"][0]
                    print(f"  First entry keys: {list(first.keys())[:10]}")
            else:
                print(f"  OK: keys={list(data.keys())}, size={len(r.text)}")
        else:
            print(f"  FAIL: HTTP {r.status_code}")
    except Exception as e:
        print(f"  ERROR: {e}")
    time.sleep(1)
print("\nDone!")
