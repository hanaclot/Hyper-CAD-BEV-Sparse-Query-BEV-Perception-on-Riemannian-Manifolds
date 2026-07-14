import sys, os, re
sys.stdout.reconfigure(encoding="utf-8")
path = r"E:\Hyper-CAD-BEV-Experiments\data\crawled\semantickitti\leaderboard_semseg.html"
with open(path, "r", encoding="utf-8", errors="ignore") as f:
    html = f.read()
print(f"HTML size: {len(html)} chars")
# Find all data-related patterns
for pattern in ["dataTable", "leaderboard", ".json", "ajax", "fetch(", "XMLHttpRequest", "tbody", "thead"]:
    count = len(re.findall(pattern, html, re.IGNORECASE))
    print(f"  '{pattern}': {count} occurrences")

# Look for method/pipeline names
methods = re.findall(r'(SalsaNext|DarkNet|RangeNet|KPConv|SPVCNN|Cylinder3D|PolarNet|SqueezeSeg|TangentConv|RandLA|PolarStream|LiLaNet|SMAC-Seg)[^<]{0,40}', html, re.IGNORECASE)
print(f"\nMethod names found ({len(methods)}):")
for m in methods[:20]:
    print(f"  {m.strip()}")

# Look for numbers near mIoU or accuracy
mIoU_patterns = re.findall(r'(\d{2}\.\d{1,2})\s*[%]', html)
print(f"\nmIoU-like values: {mIoU_patterns[:20]}")

# Check for script sections
scripts = re.findall(r'<script[^>]*>(.*?)</script>', html, re.DOTALL|re.IGNORECASE)
print(f"\nScript sections: {len(scripts)}")
for s in scripts[:3]:
    if 'leaderboard' in s.lower() or 'table' in s.lower():
        print(f"  Script ({len(s)} chars): contains leaderboard/table data")
        # Extract potential JSON data
        jsons = re.findall(r'\[{.*?}\]', s, re.DOTALL)
        print(f"  Potential JSON arrays: {len(jsons)}")

# Try to find hidden data
data_divs = re.findall(r'<div[^>]*class="[^"]*"[^>]*>', html)
print(f"\nDiv elements: {len(data_divs)}")
for d in data_divs:
    if 'table' in d.lower() or 'data' in d.lower() or 'result' in d.lower() or 'leader' in d.lower():
        print(f"  {d[:150]}")
