import sys, json
sys.stdout.reconfigure(encoding="utf-8")
with open(r"E:\Hyper-CAD-BEV-Experiments\data\crawled\rellis3d\file_tree.json", "r", encoding="utf-8") as f:
    tree = json.load(f)
files = tree["files"]
print(f"Total files: {len(files)}")
# Find edge-related files
for f in files:
    path = f["path"]
    if "edge" in path.lower() or "boundary" in path.lower() or "gated" in path.lower() or "loss" in path.lower() or "gscnn" in path.lower():
        print(f"  {path} ({f.get('size','?')} bytes)")
# Also look for any .py files in root or key dirs
py_files = [f for f in files if f["path"].endswith(".py")]
print(f"\nPython files: {len(py_files)}")
for f in py_files:
    print(f"  {f['path']}")
