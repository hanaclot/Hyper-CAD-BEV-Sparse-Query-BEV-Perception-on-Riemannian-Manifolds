import sys, json
sys.stdout.reconfigure(encoding="utf-8")
with open(r"E:\Hyper-CAD-BEV-Experiments\data\crawled\semantickitti\semantic_single.json", "r", encoding="utf-8") as f:
    data = json.load(f)
print(f"Last modified: {data['last_modified']}")
print(f"Total entries: {len(data['data'])}")
print("\nTop 10 entries (mIoU sorted):")
# Sort by mIoU descending
entries = sorted(data["data"], key=lambda x: x.get("miou", 0), reverse=True)
for i, e in enumerate(entries[:10]):
    print(f"  {i+1}. {e['approach']}: mIoU={e.get('miou','?')}%, acc={e.get('accuracy','?')}%")
print(f"\nAll approaches:")
for e in entries:
    print(f"  {e['approach']}: mIoU={e.get('miou','?')}%, acc={e.get('accuracy','?')}%")
