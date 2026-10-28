import hashlib
import io
import json
import os
import random
import sys
from PIL import Image
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

# Force UTF-8 on Windows stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Fix random seed for reproducible split
random.seed(42)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "fruits")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# 16 classes from Fruits-360 mapping
FRUITS_360_MAP = {
    "tao": ["Apple Braeburn", "Apple Crimson Snow", "Apple Golden", "Apple Granny Smith", "Apple Pink Lady", "Apple Red", "Apple Red Delicious", "Apple Red Yellow"],
    "cam_sanh": ["Orange", "Clementine", "Mandarine"],
    "chuoi": ["Banana", "Banana Lady Finger", "Banana Red"],
    "xoai": ["Mango", "Mango Red"],
    "nho": ["Grape Blue", "Grape Pink", "Grape White"],
    "dau_tay": ["Strawberry", "Strawberry Wedge"],
    "dua_hau": ["Watermelon"],
    "thanh_long": ["Pitahaya Red", "Cactus fruit"],
    "dua": ["Pineapple", "Pineapple Mini"],
    "bo": ["Avocado", "Avocado ripe"],
    "du_du": ["Papaya"],
    "oi": ["Guava"],
    "buoi": ["Pomelo Sweetie"],
    "vai": ["Lychee"],
    "chom_chom": ["Rambutan"],
    "mang_cut": ["Mangostan"]
}

# 4 pending classes for user to supplement (according to classes.json)
PENDING_CLASSES = ["sau_rieng", "mit", "man", "mang_cau"]

# Load Fruits-360 Parquet
print("Loading PedroSampaio/fruits-360 test parquet...")
f360 = hf_hub_download(repo_id="PedroSampaio/fruits-360", filename="data/test-00000-of-00001-0d294abe3826b2e6.parquet", repo_type="dataset")
table360 = pq.read_table(f360)
hf_meta360 = json.loads(table360.schema.metadata.get(b'huggingface', b'{}').decode('utf-8'))
f360_names = hf_meta360['info']['features']['label']['names']
f360_name_to_idx = {n: i for i, n in enumerate(f360_names)}

# Load Darthraider Parquet for real-world Banana & Mango
print("Loading darthraider/fruit-ripeness-detection-dataset train parquet...")
f_darth = hf_hub_download(repo_id="darthraider/fruit-ripeness-detection-dataset", filename="data/train-00000-of-00001.parquet", repo_type="dataset")
table_darth = pq.read_table(f_darth)
# Label 0: Raw_Banana, 1: Raw_Mango, 2: Ripe_Banana, 3: Ripe_Mango

# Extract images per class
collected_images = {k: [] for k in FRUITS_360_MAP.keys()}
seen_hashes = set()

def process_image_bytes(img_bytes, target_size=(224, 224)):
    try:
        h = hashlib.sha256(img_bytes).hexdigest()
        if h in seen_hashes:
            return None
        seen_hashes.add(h)
        with Image.open(io.BytesIO(img_bytes)) as img:
            img = img.convert("RGB")
            img = img.resize(target_size, Image.Resampling.BILINEAR)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            return buf.getvalue()
    except Exception as e:
        return None

# 1. Extract from Fruits-360
print("\nExtracting from Fruits-360...")
labels_360 = table360['label'].to_pylist()
images_360 = table360['image']

# Reverse index for quick lookup
target_name_to_class_id = {}
for cid, fnames in FRUITS_360_MAP.items():
    for fn in fnames:
        if fn in f360_name_to_idx:
            target_name_to_class_id[f360_name_to_idx[fn]] = cid

for idx, lbl in enumerate(labels_360):
    if lbl in target_name_to_class_id:
        cid = target_name_to_class_id[lbl]
        # Max 200 images from Fruits-360 per class to leave room for real-world images
        if len(collected_images[cid]) < 220:
            img_dict = images_360[idx].as_py()
            if img_dict and 'bytes' in img_dict and img_dict['bytes']:
                processed = process_image_bytes(img_dict['bytes'])
                if processed:
                    collected_images[cid].append(processed)

# 2. Extract from Darthraider for Chuối and Xoài (add real-world diversity)
print("\nExtracting real-world scenes from Darthraider for Chuối and Xoài...")
labels_darth = table_darth['label'].to_pylist()
images_darth = table_darth['image']

for idx, lbl in enumerate(labels_darth):
    # 0: Raw_Banana, 2: Ripe_Banana -> chuoi
    if lbl in (0, 2) and len(collected_images["chuoi"]) < 300:
        img_dict = images_darth[idx].as_py()
        if img_dict and 'bytes' in img_dict and img_dict['bytes']:
            processed = process_image_bytes(img_dict['bytes'])
            if processed:
                collected_images["chuoi"].append(processed)
    # 1: Raw_Mango, 3: Ripe_Mango -> xoai
    elif lbl in (1, 3) and len(collected_images["xoai"]) < 300:
        img_dict = images_darth[idx].as_py()
        if img_dict and 'bytes' in img_dict and img_dict['bytes']:
            processed = process_image_bytes(img_dict['bytes'])
            if processed:
                collected_images["xoai"].append(processed)

# 3. Save to train/val/test splits (70% / 15% / 15%)
print("\nSaving images to train / val / test splits...")
summary = {}

for cid, img_list in collected_images.items():
    random.shuffle(img_list)
    total = len(img_list)
    n_train = int(total * 0.70)
    n_val = int(total * 0.15)
    n_test = total - n_train - n_val

    train_imgs = img_list[:n_train]
    val_imgs = img_list[n_train:n_train + n_val]
    test_imgs = img_list[n_train + n_val:]

    # Directories
    c_train_dir = os.path.join(TRAIN_DIR, cid)
    c_val_dir = os.path.join(VAL_DIR, cid)
    c_test_dir = os.path.join(TEST_DIR, cid)
    os.makedirs(c_train_dir, exist_ok=True)
    os.makedirs(c_val_dir, exist_ok=True)
    os.makedirs(c_test_dir, exist_ok=True)

    for i, b in enumerate(train_imgs):
        with open(os.path.join(c_train_dir, f"{cid}_{i+1:04d}.jpg"), "wb") as f:
            f.write(b)
    for i, b in enumerate(val_imgs):
        with open(os.path.join(c_val_dir, f"{cid}_{i+1:04d}.jpg"), "wb") as f:
            f.write(b)
    for i, b in enumerate(test_imgs):
        with open(os.path.join(c_test_dir, f"{cid}_{i+1:04d}.jpg"), "wb") as f:
            f.write(b)

    summary[cid] = {
        "status": "ready",
        "train": len(train_imgs),
        "val": len(val_imgs),
        "test": len(test_imgs),
        "total": total
    }
    print(f"  Class {cid:12s}: Total {total:3d} (Train: {len(train_imgs):3d}, Val: {len(val_imgs):2d}, Test: {len(test_imgs):2d})")

# 4. Create placeholder directories for 4 pending classes
for pid in PENDING_CLASSES:
    os.makedirs(os.path.join(TRAIN_DIR, pid), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, pid), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, pid), exist_ok=True)
    summary[pid] = {
        "status": "pending_user_supplement",
        "train": 0,
        "val": 0,
        "test": 0,
        "total": 0
    }
    print(f"  Class {pid:12s}: PENDING (Placeholder created)")

# Save summary JSON
with open(os.path.join(DATASET_DIR, "dataset_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\nDataset preparation completed successfully!")
print(f"Summary saved to: {os.path.join(DATASET_DIR, 'dataset_summary.json')}")
