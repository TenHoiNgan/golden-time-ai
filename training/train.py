"""
Script huấn luyện và fine-tuning mô hình nhận diện trái cây Golden Time.
Kiến trúc sử dụng: MobileNetV3 với Transfer Learning.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Thêm đường dẫn app vào sys.path để tái sử dụng model definition
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.models.model_loader import GOLDEN_TIME_FRUITS, FruitVisionModel


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình nhận diện trái cây Golden Time")
    parser.add_argument("--data-dir", type=str, default="./datasets/fruits", help="Thư mục chứa dữ liệu ảnh theo cấu trúc lớp")
    parser.add_argument("--output-dir", type=str, default="./models", help="Thư mục lưu checkpoint sau huấn luyện")
    parser.add_argument("--epochs", type=int, default=10, help="Số lượt huấn luyện (epochs)")
    parser.add_argument("--batch-size", type=int, default=32, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Tốc độ học (learning rate)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Thiết bị tính toán (cpu / cuda)")
    return parser.parse_args()


def get_data_loaders(data_dir: str, batch_size: int) -> Tuple[Dict[str, DataLoader], Dict[str, int], list]:
    """Khởi tạo data transformations và loaders cho tập train và val."""
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(
            f"Không tìm thấy thư mục {train_dir} hoặc {val_dir}. "
            f"Vui lòng tham khảo hướng dẫn cấu trúc dataset tại datasets/README.md."
        )

    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }

    image_datasets = {
        "train": datasets.ImageFolder(train_dir, data_transforms["train"]),
        "val": datasets.ImageFolder(val_dir, data_transforms["val"]),
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == "train"), num_workers=2)
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes

    return dataloaders, dataset_sizes, class_names


def train_model(args):
    print("=" * 60)
    print("Bắt đầu quy trình huấn luyện mô hình Golden Time AI")
    print(f"Thiết bị: {args.device} | Epochs: {args.epochs} | Batch size: {args.batch_size}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        dataloaders, dataset_sizes, class_names = get_data_loaders(args.data_dir, args.batch_size)
        num_classes = len(class_names)
        print(f"Tập train: {dataset_sizes['train']} ảnh | Tập val: {dataset_sizes['val']} ảnh")
        print(f"Các lớp nhận diện ({num_classes}): {class_names}")
    except FileNotFoundError as e:
        print(f"\n[THÔNG BÁO QUAN TRỌNG VỀ DỮ LIỆU]:\n{e}\n")
        print("Mẹo: Dịch vụ AI hiện tại đang sử dụng mô hình nền tảng MobileNetV3 (ImageNet-1K).")
        print("Khi bạn đã thu thập đủ bộ ảnh có nhãn, chạy lại lệnh này để fine-tune mô hình.")
        return

    # Khởi tạo mô hình
    model = FruitVisionModel(num_classes=num_classes, pretrained=True)
    model.to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    best_model_weights = None
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 20)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = model.state_dict()

    time_elapsed = time.time() - start_time
    print(f"\nHuấn luyện hoàn tất trong {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Độ chính xác Validation tốt nhất: {best_acc:.4f}")

    # Lưu checkpoint
    if best_model_weights is not None:
        save_path = os.path.join(args.output_dir, "fruit_model.pt")
        torch.save(best_model_weights, save_path)
        print(f"Đã lưu trọng số mô hình tốt nhất vào: {save_path}")

        meta_path = os.path.join(args.output_dir, "model_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "classes": class_names,
                "best_val_accuracy": float(best_acc),
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, ensure_ascii=False, indent=2)
        print(f"Đã lưu metadata mô hình tại: {meta_path}")


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
