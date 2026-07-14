import sys, os, re
sys.stdout.reconfigure(encoding="utf-8")
path = r"E:\Hyper-CAD-BEV-Experiments\data\crawled\semantickitti\leaderboard_semseg.html"
with open(path, "r", encoding="utf-8", errors="ignore") as f:
    html = f.read()
print(f"HTML size: {len(html)} chars")
patterns = ["dataTable", "leaderboard", "json", "ajax", "tbody", "thead"]
for p in patterns:
    c = html.lower().count(p.lower())
    print(f"  '{p}': {c} occurrences")
# Print first 2000 chars to understand format
print("\n--- First 2000 chars ---")
print(html[:2000])
print("\n--- Last 1000 chars ---")
print(html[-1000:])
