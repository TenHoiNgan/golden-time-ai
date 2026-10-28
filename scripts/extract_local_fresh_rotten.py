import argparse
import hashlib
import io
import json
import os
import random
import sys
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T

# UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

random.seed(42)

parser = argparse.ArgumentParser(description="Extract clean fresh/rotten images from local dataset.")
parser.add_argument(
    "--source-dir",
    type=str,
    default=os.environ.get("LOCAL_DATASET_ROOT", r"C:\Users\nduym\Downloads\dataset"),
    help="Path to raw source dataset directory containing Train/Test folders",
)
args, _ = parser.parse_known_args()
LOCAL_DATASET_ROOT = os.path.abspath(args.source_dir)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "fruits")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

TARGET_MAPPING = {
    "tao": {
        "fresh": ["freshapples"],
        "rotten": ["rottenapples"]
    },
    "cam_sanh": {
        "fresh": ["freshoranges"],
        "rotten": ["rottenoranges"]
    },
    "chuoi": {
        "fresh": ["freshbanana"],
        "rotten": ["rottenbanana"]
    }
}

# Laplacian kernel for blur score
kernel = torch.tensor([[[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]]], dtype=torch.float32)
to_tensor = T.ToTensor()

seen_hashes = set()

# Pre-load hashes of existing images in datasets/fruits to ensure zero duplicate with Fruits-360
for root, _, files in os.walk(DATASET_DIR):
    for f in files:
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            f_path = os.path.join(root, f)
            try:
                with open(f_path, "rb") as fp:
                    h = hashlib.sha256(fp.read()).hexdigest()
                    seen_hashes.add(h)
            except Exception:
                pass

print(f"Loaded {len(seen_hashes)} existing image hashes to prevent cross-dataset duplicates.")

def assess_image_quality(file_path):
    """
    Đánh giá chất lượng ảnh:
    - Loại bỏ ảnh lỗi / không đọc được.
    - Loại bỏ ảnh trùng lặp SHA-256.
    - Loại bỏ ảnh quá nhỏ (< 150x150).
    - Loại bỏ ảnh quá tối (< 35) hoặc cháy sáng (> 225).
    - Tính điểm độ nét Laplacian Variance.
    """
    try:
        with open(file_path, "rb") as fp:
            data = fp.read()
            h = hashlib.sha256(data).hexdigest()
            if h in seen_hashes:
                return None
            seen_hashes.add(h)

        with Image.open(io.BytesIO(data)) as img:
            w, h = img.size
            if w < 150 or h < 150:
                return None  # Độ phân giải quá thấp

            gray = img.convert("L")
            t = to_tensor(gray).unsqueeze(0)
            mean_brightness = float(t.mean().item() * 255)
            if mean_brightness < 35.0 or mean_brightness > 225.0:
                return None  # Quá tối hoặc quá chói

            edge = F.conv2d(t, kernel)
            blur_score = float(torch.var(edge).item() * 10000)
            
            # Chuẩn hóa về 224x224 RGB
            rgb_img = img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            buf = io.BytesIO()
            rgb_img.save(buf, format="JPEG", quality=90)
            processed_bytes = buf.getvalue()

            return {
                "blur_score": blur_score,
                "bytes": processed_bytes,
                "orig_path": file_path
            }
    except Exception:
        return None

# Tiến hành quét, đánh giá và lấy mẫu cho 3 loại quả
TARGET_PER_SUBGROUP = 150  # 150 fresh + 150 rotten = 300 ảnh chất lượng cao nhất cho mỗi loại quả

extracted_summary = {}

for class_id, groups in TARGET_MAPPING.items():
    print(f"\n=== Đang xử lý lớp: {class_id} ===")
    class_extracted = {"fresh": [], "rotten": []}
    
    for state in ["fresh", "rotten"]:
        folder_names = groups[state]
        candidates = []
        
        # Quét trong cả Train và Test của C:\Users\nduym\Downloads\dataset
        for split_dir in ["Train", "Test"]:
            for fname in folder_names:
                target_folder = os.path.join(LOCAL_DATASET_ROOT, split_dir, fname)
                if not os.path.exists(target_folder):
                    continue
                for f in os.listdir(target_folder):
                    f_path = os.path.join(target_folder, f)
                    if os.path.isfile(f_path):
                        res = assess_image_quality(f_path)
                        if res:
                            candidates.append(res)
                            
        # Sắp xếp theo điểm sắc nét blur_score giảm dần (chọn ảnh nét nhất trước)
        candidates.sort(key=lambda x: x["blur_score"], reverse=True)
        selected = candidates[:TARGET_PER_SUBGROUP]
        class_extracted[state] = selected
        print(f"  [{state.upper()}]: Quét {len(candidates)} ảnh đạt chuẩn -> Chọn {len(selected)} ảnh sắc nét nhất (Blur score từ {selected[-1]['blur_score']:.1f} đến {selected[0]['blur_score']:.1f})")

    # Chia Train (70%), Val (15%), Test (15%) cho từng state
    saved_counts = {"train": 0, "val": 0, "test": 0}
    for state in ["fresh", "rotten"]:
        items = class_extracted[state]
        random.shuffle(items)
        n_total = len(items)
        n_train = int(n_total * 0.70)
        n_val = int(n_total * 0.15)
        n_test = n_total - n_train - n_val
        
        train_items = items[:n_train]
        val_items = items[n_train:n_train + n_val]
        test_items = items[n_train + n_val:]
        
        c_train_dir = os.path.join(TRAIN_DIR, class_id)
        c_val_dir = os.path.join(VAL_DIR, class_id)
        c_test_dir = os.path.join(TEST_DIR, class_id)
        
        # Ghi tệp
        for i, item in enumerate(train_items):
            fname = f"{class_id}_{state}_{i+1:04d}.jpg"
            with open(os.path.join(c_train_dir, fname), "wb") as fp:
                fp.write(item["bytes"])
            saved_counts["train"] += 1
            
        for i, item in enumerate(val_items):
            fname = f"{class_id}_{state}_{i+1:04d}.jpg"
            with open(os.path.join(c_val_dir, fname), "wb") as fp:
                fp.write(item["bytes"])
            saved_counts["val"] += 1
            
        for i, item in enumerate(test_items):
            fname = f"{class_id}_{state}_{i+1:04d}.jpg"
            with open(os.path.join(c_test_dir, fname), "wb") as fp:
                fp.write(item["bytes"])
            saved_counts["test"] += 1
            
    extracted_summary[class_id] = saved_counts
    print(f"  -> Đã bổ sung vào dataset: Train +{saved_counts['train']}, Val +{saved_counts['val']}, Test +{saved_counts['test']}")

# Cập nhật lại dataset_summary.json
summary_file = os.path.join(DATASET_DIR, "dataset_summary.json")
if os.path.exists(summary_file):
    with open(summary_file, "r", encoding="utf-8") as fp:
        summary_data = json.load(fp)
else:
    summary_data = {}

# Đếm lại số tệp thực tế trong từng folder của datasets/fruits
for root_split, split_key in [(TRAIN_DIR, "train"), (VAL_DIR, "val"), (TEST_DIR, "test")]:
    for cid in os.listdir(root_split):
        c_path = os.path.join(root_split, cid)
        if os.path.isdir(c_path):
            count = len([f for f in os.listdir(c_path) if f.lower().endswith(".jpg")])
            if cid not in summary_data:
                summary_data[cid] = {"status": "ready" if count > 0 else "pending_user_supplement"}
            summary_data[cid][split_key] = count

for cid, info in summary_data.items():
    info["total"] = info.get("train", 0) + info.get("val", 0) + info.get("test", 0)
    info["status"] = "ready" if info["total"] > 0 else "pending_user_supplement"

with open(summary_file, "w", encoding="utf-8") as fp:
    json.dump(summary_data, fp, ensure_ascii=False, indent=2)

print(f"\nCập nhật thành công siêu dữ liệu dataset tại: {summary_file}")
