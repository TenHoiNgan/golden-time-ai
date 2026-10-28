"""
Script đánh giá hiệu năng mô hình nhận diện trái cây trên tập kiểm thử độc lập (Test Set).
Hỗ trợ tính toán: Accuracy, Per-class Precision/Recall/F1, Macro-F1, Weighted-F1 và Confusion Matrix.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.models.model_loader import FruitVisionModel
from training.train import SelectedClassesDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình nhận diện trái cây")
    parser.add_argument("--test-dir", type=str, default="./datasets/fruits/test", help="Thư mục chứa tập ảnh test")
    parser.add_argument("--model-path", type=str, default="./models/fruit_model.pt", help="Đường dẫn file trọng số mô hình")
    parser.add_argument("--meta-path", type=str, default="./models/model_meta.json", help="Đường dẫn file metadata")
    parser.add_argument("--output-json", type=str, default=None, help="Đường dẫn lưu kết quả đánh giá dạng JSON")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def calculate_metrics(
    y_true: List[int],
    y_pred: List[int],
    num_classes: int,
    class_names: List[str],
) -> Dict:
    """
    Tính toán chi tiết các chỉ số phân loại:
    - Overall Top-1 Accuracy
    - Per-class Precision, Recall, F1-score
    - Macro-F1, Weighted-F1
    - Confusion Matrix và Top cặp lớp bị nhầm lẫn
    """
    total = len(y_true)
    if total == 0:
        return {
            "total_samples": 0,
            "overall_accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "per_class": {},
            "confusion_matrix": [],
            "top_confused_pairs": [],
        }

    # Xây dựng Confusion Matrix: row = true, col = pred
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            matrix[t][p] += 1

    per_class = {}
    f1_list = []
    weights_list = []

    for i in range(num_classes):
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(num_classes) if r != i)
        fn = sum(matrix[i][c] for c in range(num_classes) if c != i)
        support = sum(matrix[i])

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        c_name = class_names[i] if i < len(class_names) else f"class_{i}"
        per_class[c_name] = {
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
        f1_list.append(f1)
        weights_list.append(support)

    correct_total = sum(matrix[i][i] for i in range(num_classes))
    overall_acc = (correct_total / total) if total > 0 else 0.0

    macro_f1 = (sum(f1_list) / len(f1_list)) if f1_list else 0.0
    weighted_f1 = (
        sum(f * w for f, w in zip(f1_list, weights_list)) / total
    ) if total > 0 else 0.0

    # Tìm các cặp nhầm lẫn nhiều nhất (true != pred)
    confused_pairs = []
    for r in range(num_classes):
        for c in range(num_classes):
            if r != c and matrix[r][c] > 0:
                true_name = class_names[r] if r < len(class_names) else f"class_{r}"
                pred_name = class_names[c] if c < len(class_names) else f"class_{c}"
                confused_pairs.append({
                    "true_class": true_name,
                    "pred_class": pred_name,
                    "count": matrix[r][c],
                })
    confused_pairs.sort(key=lambda x: x["count"], reverse=True)

    return {
        "total_samples": total,
        "correct_predictions": correct_total,
        "overall_accuracy": round(overall_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "top_confused_pairs": confused_pairs[:10],
    }


def evaluate(args):
    print("=" * 70)
    print("Bắt đầu đánh giá mô hình trên tập kiểm thử độc lập (Test Set)")
    print(f"Mô hình: {args.model_path} | Tập test: {args.test_dir}")
    print("=" * 70)

    if not os.path.exists(args.test_dir):
        print(f"\n[THÔNG BÁO] Không tìm thấy thư mục kiểm thử: {args.test_dir}")
        print("Vui lòng chuẩn bị tập test theo cấu trúc thư mục lớp để đánh giá.")
        return

    if not os.path.exists(args.model_path):
        print(f"\n[THÔNG BÁO] Không tìm thấy tệp checkpoint: {args.model_path}")
        print("Vui lòng chạy script huấn luyện train.py trước.")
        return

    # Tải metadata bắt buộc để xác định chính xác ánh xạ lớp
    if not os.path.exists(args.meta_path):
        raise FileNotFoundError(
            f"Không tìm thấy metadata mô hình: {args.meta_path}. "
            "Cần có metadata để xác định đúng thứ tự ánh xạ lớp (class mapping) sang tập test."
        )

    with open(args.meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    class_names = meta.get("classes")
    class_ids = meta.get("class_ids")
    folders = meta.get("folders")
    class_to_idx = meta.get("class_to_idx")

    if not class_names or not isinstance(class_names, list):
        raise ValueError(f"Metadata tại {args.meta_path} không hợp lệ: thiếu trường 'classes'.")

    # Tái tạo class_to_idx nếu checkpoint cũ chưa có trường này
    if not class_to_idx:
        ref_keys = folders or class_ids
        if not ref_keys or len(ref_keys) != len(class_names):
            raise ValueError(f"Metadata tại {args.meta_path} không thể suy diễn ánh xạ 'class_to_idx' an toàn.")
        class_to_idx = {k: idx for idx, k in enumerate(ref_keys)}

    num_classes = len(class_names)
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_dataset = SelectedClassesDataset(args.test_dir, class_to_idx=class_to_idx, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = FruitVisionModel(num_classes=num_classes, pretrained=False)
    state_dict = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(args.device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    metrics = calculate_metrics(y_true, y_pred, num_classes, class_names)

    print(f"\nKẾT QUẢ ĐÁNH GIÁ CHUẨN:")
    print(f"Tổng số ảnh kiểm thử: {metrics['total_samples']}")
    print(f"Độ chính xác (Top-1 Accuracy): {metrics['overall_accuracy'] * 100:.2f}%")
    print(f"Macro-F1 Score:                {metrics['macro_f1'] * 100:.2f}%")
    print(f"Weighted-F1 Score:             {metrics['weighted_f1'] * 100:.2f}%\n")

    print(f"{'Loại trái cây':<22} | {'Mẫu':<6} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 68)
    for c_name, p_data in metrics["per_class"].items():
        print(
            f"{c_name:<22} | "
            f"{p_data['support']:<6} | "
            f"{p_data['precision']*100:<9.1f}% | "
            f"{p_data['recall']*100:<9.1f}% | "
            f"{p_data['f1']*100:<9.1f}%"
        )

    if metrics["top_confused_pairs"]:
        print(f"\nTOP CÁC CẶP LỚP HAY BỊ NHẦM LẪN NHẤT:")
        for pair in metrics["top_confused_pairs"]:
            print(f"  - Quả thật: '{pair['true_class']}' nhầm thành '{pair['pred_class']}' ({pair['count']} ảnh)")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"\nĐã lưu kết quả đánh giá chi tiết vào: {args.output_json}")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
