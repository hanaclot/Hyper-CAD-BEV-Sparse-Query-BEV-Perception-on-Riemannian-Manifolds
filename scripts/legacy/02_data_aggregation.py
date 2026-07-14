# -*- coding: utf-8 -*-
import json, csv, hashlib
from pathlib import Path
from datetime import datetime

BASE = Path(r"E:\HyperCAD_BEV_2026")
DATA = BASE / "data"
PROCESSED = DATA / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# 1. Collect all scrape results
sources_summary = []
for dname in ["rellis3d", "semantickitti", "tartandrive2", 
              "event_camera", "bevformer_paper", "sparsead_paper",
              "loihi_fusion", "weather_paper"]:
    d = DATA / dname
    sr = d / "scrape_result.json"
    if sr.exists():
        s = json.loads(sr.read_text(encoding="utf-8"))
        files = list(d.glob("*"))
        s["file_count"] = len(files)
        s["total_bytes"] = sum(f.stat().st_size for f in files if f.is_file())
        sources_summary.append(s)
    else:
        sources_summary.append({"source": dname, "error": "no scrape_result.json"})

# 2. Build file manifest CSV
manifest_rows = []
for dname in ["rellis3d", "semantickitti", "tartandrive2",
              "event_camera", "bevformer_paper", "sparsead_paper",
              "loihi_fusion", "weather_paper"]:
    d = DATA / dname
    if not d.exists():
        continue
    for f in d.rglob("*"):
        if f.is_file():
            rel = f.relative_to(DATA)
            fid = hashlib.md5(str(rel).encode()).hexdigest()[:8]
            manifest_rows.append({
                "file_id": fid,
                "source": dname,
                "relative_path": str(rel),
                "file_type": f.suffix.lower(),
                "file_size": f.stat().st_size,
                "last_modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
            })

# Write manifest CSV
man_path = PROCESSED / "rural_manifold_dataset_index.csv"
with open(man_path, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["file_id","source","relative_path","file_type","file_size","last_modified"])
    w.writeheader()
    w.writerows(manifest_rows)
print(f"Manifest CSV: {man_path} ({len(manifest_rows)} files)")

# 3. Sources summary CSV
sum_path = PROCESSED / "scraped_sources_summary.csv"
with open(sum_path, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["Source","URL","Status","Key Metrics","Files"])
    for s in sources_summary:
        src = s.get("source", "?")
        url = s.get("url", "")
        status = "ERROR" if "error" in s else "SUCCESS"
        if "error" in s:
            metrics = s["error"]
        else:
            metrics_parts = []
            for k in ["stars","page_chars","metadata_chars","full_chars","clean_chars","files_downloaded","readme_chars"]:
                if k in s: metrics_parts.append(f"{k}={s[k]}")
            metrics = "; ".join(metrics_parts[:4])
        files = s.get("file_count", s.get("files_downloaded", "N/A"))
        w.writerow([src, url, status, metrics, files])

print(f"Summary CSV: {sum_path}")

# 4. Save aggregated metadata
agg = {
    "project": "Hyper-CAD-BEV v6.5-Sparse",
    "created": datetime.now().isoformat(),
    "data_sources": len(sources_summary),
    "total_files": len(manifest_rows),
    "total_bytes": sum(r["file_size"] for r in manifest_rows),
    "sources": sources_summary
}
(PROCESSED / "aggregated_metadata.json").write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"\nAggregation complete!")
print(f"  Sources: {len(sources_summary)}")
print(f"  Total files: {len(manifest_rows)}")
print(f"  Total size: {agg['total_bytes']/1024:.1f} KB")
