import sys, re
sys.stdout.reconfigure(encoding="utf-8")
path = r"E:\Hyper-CAD-BEV-Experiments\data\crawled\semantickitti\leaderboard_semseg.html"
with open(path, "r", encoding="utf-8", errors="ignore") as f:
    html = f.read()
# Extract the last script fully
scripts = re.findall(r'<script[^>]*>(.*?)</script>', html, re.DOTALL|re.IGNORECASE)
last_script = scripts[-1]
print(f"Last script size: {len(last_script)}")
# Look for ajax, fetch, data URLs
for pat in ["ajaxURL", "ajax", "fetch", ".json", "data/", "_data"]:
    hits = [m for m in re.finditer(pat, last_script, re.IGNORECASE)]
    for h in hits:
        start = max(0, h.start()-30)
        end = min(len(last_script), h.end()+100)
        print(f"\n  '{pat}' at {h.start()}: ...{last_script[start:end]}...")

# Also search for "id_list" or "benchmark"
for pat in ["id_list", "benchmark", "leaderboard", "semseg"]:
    hits = list(re.finditer(pat, last_script, re.IGNORECASE))
    if hits:
        for h in hits[:3]:
            start = max(0, h.start()-30)
            end = min(len(last_script), h.end()+100)
            print(f"\n  '{pat}': ...{last_script[start:end]}...")
