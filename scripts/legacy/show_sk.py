import sys, json
sys.stdout.reconfigure(encoding="utf-8")
with open(r"E:\Hyper-CAD-BEV-Experiments\data\crawled\semantickitti\semantic_single.json", "r", encoding="utf-8") as f:
    data = json.load(f)
entries = sorted(data["data"], key=lambda x: float(x.get("miou", 0)))
print("SemanticKITTI Leaderboard (sorted by mIoU):")
for i, e in enumerate(entries):
    print(f"  {i+1}. {e['approach']}: mIoU={e.get('miou','?')}, acc={e.get('accuracy','?')}")
print(f"\nDataset statistics page:")
import requests
r = requests.get("http://semantic-kitti.org/dataset.html", timeout=15)
import re
# Find numbers like "23,201" or "43,552"
nums = re.findall(r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)', r.text)
print(f"  Numbers found: {nums[:20]}")
