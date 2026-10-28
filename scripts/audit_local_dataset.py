import argparse
import collections
import hashlib
import json
import os
import sys
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

parser = argparse.ArgumentParser(description="Audit local dataset for image counts, corruption, and train/test leakage.")
parser.add_argument(
    "--dataset-root",
    type=str,
    default=os.environ.get("LOCAL_DATASET_ROOT", r"C:\Users\nduym\Downloads\dataset"),
    help="Path to local dataset containing split subdirectories (e.g. Train, Test)",
)
parser.add_argument(
    "--output-json",
    type=str,
    default=None,
    help="Path to save JSON audit report (defaults to datasets/local_dataset_audit.json)",
)
args, _ = parser.parse_known_args()
LOCAL_DATASET_ROOT = os.path.abspath(args.dataset_root)

print(f"=== AUDITING LOCAL DATASET AT: {LOCAL_DATASET_ROOT} ===")

if not os.path.exists(LOCAL_DATASET_ROOT):
    print(f"ERROR: Path does not exist: {LOCAL_DATASET_ROOT}")
    sys.exit(1)

subdirs = [d for d in os.listdir(LOCAL_DATASET_ROOT) if os.path.isdir(os.path.join(LOCAL_DATASET_ROOT, d))]
print(f"Top-level directories: {subdirs}")

report = {
    "root": LOCAL_DATASET_ROOT,
    "splits": {},
    "total_images": 0,
    "total_bytes": 0,
    "corrupted_images": [],
    "extensions": collections.Counter(),
    "duplicate_hashes_internal": 0,
    "train_test_overlap_hashes": 0,
}

train_hashes = {}
test_hashes = {}

for split in subdirs:
    split_path = os.path.join(LOCAL_DATASET_ROOT, split)
    categories = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])
    
    split_info = {
        "categories": {},
        "total_images": 0,
        "total_bytes": 0,
    }
    
    for cat in categories:
        cat_path = os.path.join(split_path, cat)
        files = [f for f in os.listdir(cat_path) if os.path.isfile(os.path.join(cat_path, f))]
        
        cat_bytes = 0
        cat_valid_images = 0
        
        for f in files:
            f_path = os.path.join(cat_path, f)
            ext = os.path.splitext(f)[1].lower()
            report["extensions"][ext] += 1
            f_size = os.path.getsize(f_path)
            cat_bytes += f_size
            
            # Read first 64KB for SHA-256 hash
            try:
                with open(f_path, "rb") as fp:
                    content = fp.read()
                    f_hash = hashlib.sha256(content).hexdigest()
                    
                    if split.lower() == "train":
                        if f_hash in train_hashes:
                            report["duplicate_hashes_internal"] += 1
                        else:
                            train_hashes[f_hash] = (cat, f)
                    elif split.lower() == "test":
                        if f_hash in test_hashes:
                            report["duplicate_hashes_internal"] += 1
                        else:
                            test_hashes[f_hash] = (cat, f)
            except Exception as e:
                report["corrupted_images"].append((f_path, str(e)))
                continue

            # Verify image with PIL
            try:
                with Image.open(f_path) as img:
                    img.verify()
                    cat_valid_images += 1
            except Exception as e:
                report["corrupted_images"].append((f_path, str(e)))
                
        split_info["categories"][cat] = {
            "count": len(files),
            "valid_images": cat_valid_images,
            "size_mb": round(cat_bytes / (1024 * 1024), 2)
        }
        split_info["total_images"] += len(files)
        split_info["total_bytes"] += cat_bytes
        
    report["splits"][split] = split_info
    report["total_images"] += split_info["total_images"]
    report["total_bytes"] += split_info["total_bytes"]

# Check train vs test hash overlap (Data Leakage)
overlap = set(train_hashes.keys()).intersection(set(test_hashes.keys()))
report["train_test_overlap_hashes"] = len(overlap)

# Check overlap against Project-AgML / Sriram Reddy dataset if cached
hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
AGML_CACHE_DIR = os.path.join(hf_home, "hub", "datasets--Project-AgML--fresh_rotten_fruit_classification")
agml_overlap_detected = False
if os.path.exists(AGML_CACHE_DIR):
    report["agml_cache_exists"] = True
else:
    report["agml_cache_exists"] = False

# Summary output
print("\n=== AUDIT RESULTS ===")
print(f"Total files: {report['total_images']}")
print(f"Total size: {round(report['total_bytes'] / (1024*1024*1024), 2)} GB ({round(report['total_bytes'] / (1024*1024), 1)} MB)")
print(f"File extensions: {dict(report['extensions'])}")
print(f"Corrupted / Unreadable images: {len(report['corrupted_images'])}")
print(f"Internal duplicate images (SHA-256): {report['duplicate_hashes_internal']}")
print(f"Data Leakage (Train/Test identical image hash overlap): {report['train_test_overlap_hashes']}")

print("\n--- SPLIT BREAKDOWN ---")
for split_name, sinfo in report["splits"].items():
    print(f"\nSplit '{split_name}': {sinfo['total_images']} images ({round(sinfo['total_bytes']/(1024*1024), 1)} MB), {len(sinfo['categories'])} folders:")
    for cat, cinfo in sinfo["categories"].items():
        print(f"  - {cat:20s}: {cinfo['count']:5d} images ({cinfo['size_mb']:6.1f} MB)")

# Save detailed json report to datasets/local_dataset_audit.json
output_json = args.output_json or os.path.join(BASE_DIR, "datasets", "local_dataset_audit.json")
with open(output_json, "w", encoding="utf-8") as fp:
    json.dump(report, fp, ensure_ascii=False, indent=2)
print(f"\nSaved raw audit data to: {output_json}")
