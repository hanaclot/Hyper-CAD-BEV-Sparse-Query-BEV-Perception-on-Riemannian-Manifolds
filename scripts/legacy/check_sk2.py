import sys, re
sys.stdout.reconfigure(encoding="utf-8")
path = r"E:\Hyper-CAD-BEV-Experiments\data\crawled\semantickitti\leaderboard_semseg.html"
with open(path, "r", encoding="utf-8", errors="ignore") as f:
    html = f.read()
# Find JSON file references
urls = re.findall(r'["]https?://[^"]*\.json[^"]*["]', html)
print("JSON URLs found:")
for u in urls:
    print(f"  {u}")
# Find data URLs
urls2 = re.findall(r"[']https?://[^']*\.json[^']*[']", html)
for u in urls2:
    print(f"  {u}")
# Find all URLs in script sections
scripts = re.findall(r'<script[^>]*>(.*?)</script>', html, re.DOTALL|re.IGNORECASE)
print(f"\nScript sections: {len(scripts)}")
for i, s in enumerate(scripts):
    if 'tabulator' in s.lower() or 'json' in s.lower() or 'data' in s.lower():
        print(f"\n--- Script {i} ({len(s)} chars) ---")
        # Find URLs
        urls_in_script = re.findall(r'(https?://[^\s\"<>]+)', s)
        for u in urls_in_script[:20]:
            print(f"  URL: {u}")
        # Show the first part
        if len(s) > 500:
            print(s[:500])
        else:
            print(s)
