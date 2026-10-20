"""
Script đánh giá hiệu năng mô hình nhận diện trái cây trên tập kiểm thử độc lập (Test Set).
"""

import argparse
import json
import os
import sys
from typing import Dict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.models.model_loader import FruitVisionModel


def parse_args():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình nhận diện trái cây")
    parser.add_argument("--test-dir", type=str, default="./datasets/fruits/test", help="Thư mục chứa tập ảnh test")
    parser.add_argument("--model-path", type=str, default="./models/fruit_model.pt", help="Đường dẫn file trọng số mô hình")
    parser.add_argument("--meta-path", type=str, default="./models/model_meta.json", help="Đường dẫn file metadata")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def evaluate(args):
    print("=" * 60)
    print("Bắt đầu đánh giá mô hình trên tập kiểm thử")
    print(f"Mô hình: {args.model_path} | Tập test: {args.test_dir}")
    print("=" * 60)

    if not os.path.exists(args.test_dir):
        print(f"Không tìm thấy thư mục kiểm thử: {args.test_dir}")
        print("Vui lòng chuẩn bị tập test theo cấu trúc thư mục lớp để đánh giá.")
        return

    if not os.path.exists(args.model_path):
        print(f"Không tìm thấy tệp checkpoint: {args.model_path}")
        print("Vui lòng chạy script huấn luyện train.py trước.")
        return

    # Tải metadata nếu có
    class_names = None
    if os.path.exists(args.meta_path):
        with open(args.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            class_names = meta.get("classes")

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_dataset = datasets.ImageFolder(args.test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if class_names is None:
        class_names = test_dataset.classes

    num_classes = len(class_names)
    model = FruitVisionModel(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.to(args.device)
    model.eval()

    total = 0
    correct = 0
    class_correct = {i: 0 for i in range(num_classes)}
    class_total = {i: 0 for i in range(num_classes)}

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            for p, l in zip(preds, labels):
                l_idx = l.item()
                if p == l:
                    class_correct[l_idx] += 1
                class_total[l_idx] += 1

    overall_acc = (correct / total) * 100.0 if total > 0 else 0.0
    print(f"\nKết quả đánh giá tổng thể:")
    print(f"Tổng số ảnh kiểm thử: {total}")
    print(f"Độ chính xác (Top-1 Accuracy): {overall_acc:.2f}%\n")

    print(f"{'Loại trái cây':<25} | {'Số mẫu':<10} | {'Chính xác':<12}")
    print("-" * 52)
    for idx, c_name in enumerate(class_names):
        c_tot = class_total[idx]
        c_cor = class_correct[idx]
        acc_str = f"{(c_cor / c_tot) * 100:.1f}%" if c_tot > 0 else "N/A"
        print(f"{c_name:<25} | {c_tot:<10} | {acc_str:<12}")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
