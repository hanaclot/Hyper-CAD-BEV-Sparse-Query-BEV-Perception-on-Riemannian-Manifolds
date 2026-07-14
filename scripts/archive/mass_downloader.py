#!/usr/bin/env python3
# Mass data downloader for Hyper-CAD-BEV experiment
# Downloads from real websites only - NO generated data

import os, sys, json, time, hashlib, urllib.request, urllib.error, ssl

ssl._create_default_https_context = ssl._create_unverified_context

DOWNLOAD_DIR = "E:/Hyper-CAD-BEV-Experiments/data"
WORK_DIR = "D:/HyperCAD_BEV_2026/data"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(WORK_DIR, exist_ok=True)

SOURCES = [
    # === SemanticKITTI (http://semantic-kitti.org/) ===
    {
        "name": "semantickitti_voxels_all",
        "url": "http://semantic-kitti.org/assets/data_odometry_voxels_all.zip",
        "dest": f"{DOWNLOAD_DIR}/semantickitti_official/data_odometry_voxels_all.zip",
        "min_size_mb": 3000
    },
    {
        "name": "semantickitti_calib",
        "url": "http://semantic-kitti.org/assets/data_odometry_calib.zip",
        "dest": f"{DOWNLOAD_DIR}/semantickitti_official/data_odometry_calib.zip",
        "min_size_mb": 1
    },
    {
        "name": "semantickitti_poses",
        "url": "http://semantic-kitti.org/assets/data_odometry_poses.zip",
        "dest": f"{DOWNLOAD_DIR}/semantickitti_official/data_odometry_poses.zip",
        "min_size_mb": 1
    },
    # === SemanticKITTI raw sequences (sequence 00-10) ===
    {
        "name": "semantickitti_velodyne_all",
        "url": "http://semantic-kitti.org/assets/data_odometry_velodyne.zip",
        "dest": f"{DOWNLOAD_DIR}/semantickitti_official/data_odometry_velodyne.zip",
        "min_size_mb": 500
    },
    {
        "name": "semantickitti_labels",
        "url": "http://semantic-kitti.org/assets/data_odometry_labels.zip",
        "dest": f"{DOWNLOAD_DIR}/semantickitti_official/data_odometry_labels.zip",
        "min_size_mb": 50
    },
    # === KITTI (https://www.cvlibs.net/datasets/kitti/) ===
    {
        "name": "kitti_odometry_gray",
        "url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_gray.zip",
        "dest": f"{DOWNLOAD_DIR}/kitti/data_odometry_gray.zip",
        "min_size_mb": 20000
    },
    {
        "name": "kitti_odometry_color",
        "url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_color.zip",
        "dest": f"{DOWNLOAD_DIR}/kitti/data_odometry_color.zip",
        "min_size_mb": 60000
    },
    {
        "name": "kitti_odometry_velodyne",
        "url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip",
        "dest": f"{DOWNLOAD_DIR}/kitti/data_odometry_velodyne.zip",
        "min_size_mb": 20000
    },
    {
        "name": "kitti_object_image2",
        "url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip",
        "dest": f"{DOWNLOAD_DIR}/kitti/data_object_image_2.zip",
        "min_size_mb": 10000
    },
    {
        "name": "kitti_object_velodyne",
        "url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip",
        "dest": f"{DOWNLOAD_DIR}/kitti/data_object_velodyne.zip",
        "min_size_mb": 2000
    },
    {
        "name": "kitti_raw_data",
        "url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip",
        "dest": f"{DOWNLOAD_DIR}/kitti_raw/kitti_raw_0926_0001_sync.zip",
        "min_size_mb": 100
    },
    {
        "name": "kitti_devkit",
        "url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_object.zip",
        "dest": f"{DOWNLOAD_DIR}/kitti/devkit_object.zip",
        "min_size_mb": 1
    },
]

print(f"Starting mass download: {len(SOURCES)} sources")


# ===== DOWNLOAD CORE =====
class ChunkedDownloader:
    def __init__(self, chunk_size=8*1024*1024):
        self.chunk_size = chunk_size
        self.total_downloaded = 0
        self.failed = []
        self.success = []
        self.skipped = []

    def download_url(self, source, timeout=1800):
        url = source['url']
        dest = source['dest']
        name = source['name']
        min_size = source.get('min_size_mb', 0) * 1024 * 1024

        dest_dir = os.path.dirname(dest)
        os.makedirs(dest_dir, exist_ok=True)

        # Check if already downloaded successfully
        if os.path.exists(dest):
            existing_size = os.path.getsize(dest)
            if existing_size >= min_size:
                print(f"[SKIP] {name}: already {existing_size/1024/1024:.1f} MB")
                self.skipped.append(name)
                return f"SKIPPED ({existing_size/1024/1024:.1f} MB)"
            else:
                print(f"[RESUME] {name}: incomplete ({existing_size/1024/1024:.1f} MB), re-downloading...")
                os.remove(dest)

        print(f"[DOWNLOAD] {name}: {url}")

        retries = 3
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    total_size = int(response.headers.get('Content-Length', 0))
                    print(f"  Total size: {total_size/1024/1024:.1f} MB ({total_size} bytes)")

                    downloaded = 0
                    with open(dest, 'wb') as f:
                        while True:
                            chunk = response.read(self.chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)
                            pct = (downloaded / total_size * 100) if total_size else 0
                            if downloaded % (50*1024*1024) < self.chunk_size:
                                print(f"  Progress: {downloaded/1024/1024:.1f} MB ({pct:.0f}%)", end='\r')

                final_size = os.path.getsize(dest)
                if final_size >= min_size:
                    self.total_downloaded += final_size
                    self.success.append(name)
                    print(f"\n[OK] {name}: {final_size/1024/1024:.1f} MB")
                    return f"OK ({final_size/1024/1024:.1f} MB)"
                else:
                    print(f"\n[WARN] {name}: too small ({final_size/1024/1024:.1f} MB), expected >{min_size/1024/1024:.1f} MB")
                    self.failed.append(name)
                    return f"TOO SMALL ({final_size/1024/1024:.1f} MB)"
            except Exception as e:
                print(f"\n[RETRY {attempt+1}/{retries}] {name}: {e}")
                time.sleep(5)

        self.failed.append(name)
        return f"FAILED after {retries} retries"

    def run_all(self, sources):
        print("="*80)
        print(f"Starting batch download of {len(sources)} sources")
        print("="*80)
        start = time.time()

        results = {}
        for i, src in enumerate(sources):
            print(f"\n[{i+1}/{len(sources)}] {src['name']}")
            results[src['name']] = self.download_url(src)
            elapsed = time.time() - start
            print(f"  Total elapsed: {elapsed/60:.1f} min, Downloaded: {self.total_downloaded/1024/1024/1024:.2f} GB")

        print("\n" + "="*80)
        print("DOWNLOAD SUMMARY")
        print("="*80)
        print(f"Total time: {(time.time()-start)/60:.1f} min")
        print(f"Total downloaded: {self.total_downloaded/1024/1024/1024:.2f} GB")
        print(f"Success: {len(self.success)}, Skipped: {len(self.skipped)}, Failed: {len(self.failed)}")
        if self.failed:
            print(f"Failed: {self.failed}")
        print("="*80)

        with open(f"{WORK_DIR}/download_results.json", "w") as f:
            json.dump({
                "results": results,
                "total_gb": self.total_downloaded/1024/1024/1024,
                "success": self.success,
                "failed": self.failed,
                "skipped": self.skipped
            }, f, indent=2)

        return results


if __name__ == "__main__":
    downloader = ChunkedDownloader()
    downloader.run_all(SOURCES)
