import urllib.request, json, time
HEADERS = {"User-Agent": "Mozilla/5.0"}
urls_to_test = [
    ("http://semantic-kitti.org/assets/data_odometry_labels.zip", "SK_labels"),
    ("http://semantic-kitti.org/assets/data_odometry_voxels.zip", "SK_voxels"),
    ("http://semantic-kitti.org/assets/data_odometry_voxels_all.zip", "SK_voxels_all"),
    ("http://www.cvlibs.net/download.php?file=data_odometry_calib.zip", "KITTI_calib"),
    ("http://www.cvlibs.net/download.php?file=data_odometry_velodyne.zip", "KITTI_velodyne_all"),
    ("https://arxiv.org/pdf/2206.09907.pdf", "weather_paper"),
    ("https://arxiv.org/pdf/1711.01458.pdf", "event_camera"),
    ("https://arxiv.org/pdf/2404.06892.pdf", "sparsead"),
    ("https://arxiv.org/pdf/2203.17270.pdf", "bevformer"),
    ("https://arxiv.org/pdf/2408.16096.pdf", "loihi_fusion"),
]
results = {}
for url, name in urls_to_test:
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=15) as r:
            cl = r.getheader("Content-Length")
            size_str = f"{int(cl)/1e6:.1f}MB" if cl else "unknown"
            results[name] = {"status": "ok", "size": size_str, "url": url}
    except Exception as e:
        results[name] = {"status": "fail", "error": str(e)[:100]}
with open(r"E:\Hyper-CAD-BEV-Experiments\data\scraped\url_connectivity.json", "w") as f:
    json.dump(results, f, indent=2)
