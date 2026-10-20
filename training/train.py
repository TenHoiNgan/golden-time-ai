"""
Script huấn luyện và fine-tuning mô hình nhận diện trái cây Golden Time.
Kiến trúc sử dụng: MobileNetV3 với Transfer Learning.
"""

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Thêm đường dẫn app vào sys.path để tái sử dụng model definition và registry
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.core.registry import fruit_registry
from app.models.model_loader import FruitVisionModel


def set_seed(seed: int = 42):
    """Thiết lập random seed cố định để đảm bảo kết quả huấn luyện có thể tái lập."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình nhận diện trái cây Golden Time")
    parser.add_argument("--data-dir", type=str, default="./datasets/fruits", help="Thư mục chứa dữ liệu ảnh theo cấu trúc lớp")
    parser.add_argument("--output-dir", type=str, default="./models", help="Thư mục lưu checkpoint sau huấn luyện")
    parser.add_argument("--epochs", type=int, default=10, help="Số lượt huấn luyện (epochs)")
    parser.add_argument("--batch-size", type=int, default=32, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Tốc độ học (learning rate)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed tái lập kết quả")
    parser.add_argument("--patience", type=int, default=5, help="Số epoch chờ trước khi dừng sớm (early stopping)")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0, help="Số epoch đóng băng backbone ban đầu")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Thiết bị tính toán (cpu / cuda)")
    return parser.parse_args()


def get_data_loaders(data_dir: str, batch_size: int) -> Tuple[Dict[str, DataLoader], Dict[str, int], List[str], List[str], List[str]]:
    """
    Khởi tạo data transformations và loaders cho tập train và val.
    Ánh xạ thư mục theo đúng fruit_registry để đảm bảo thứ tự index chính xác.
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(
            f"Không tìm thấy thư mục {train_dir} hoặc {val_dir}. "
            f"Vui lòng tham khảo hướng dẫn cấu trúc dataset tại datasets/README.md."
        )

    # 1. Kiểm tra cấu trúc thư mục đối chiếu với registry
    matched, missing, unexpected = fruit_registry.validate_dataset_structure(train_dir)
    if unexpected:
        print(f"[CẢNH BÁO REGISTRY] Phát hiện thư mục không có trong registry: {unexpected}")
    if missing:
        print(f"[THÔNG TIN REGISTRY] {len(missing)} lớp chưa có dữ liệu trong train: {missing[:5]}...")

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

    raw_folders = image_datasets["train"].classes
    if len(raw_folders) == 0:
        raise ValueError(f"Thư mục {train_dir} không chứa thư mục con lớp nào.")

    # Ánh xạ từ folder name sang display_name và class_id chính xác
    display_names = []
    class_ids = []
    for f in raw_folders:
        fruit_cls = fruit_registry.get_by_folder(f)
        if fruit_cls:
            display_names.append(fruit_cls.display_name)
            class_ids.append(fruit_cls.class_id)
        else:
            display_names.append(f)
            class_ids.append(f)

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == "train"), num_workers=2)
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}

    return dataloaders, dataset_sizes, raw_folders, display_names, class_ids


def train_model(args):
    set_seed(args.seed)

    print("=" * 60)
    print("Bắt đầu quy trình huấn luyện mô hình Golden Time AI")
    print(f"Thiết bị: {args.device} | Epochs: {args.epochs} | Batch size: {args.batch_size} | Seed: {args.seed}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        dataloaders, dataset_sizes, raw_folders, display_names, class_ids = get_data_loaders(args.data_dir, args.batch_size)
        num_classes = len(raw_folders)
        print(f"Tập train: {dataset_sizes['train']} ảnh | Tập val: {dataset_sizes['val']} ảnh")
        print(f"Các lớp nhận diện ({num_classes}): {display_names}")
    except (FileNotFoundError, ValueError) as e:
        print(f"\n[THÔNG BÁO QUAN TRỌNG VỀ DỮ LIỆU]:\n{e}\n")
        print("Mẹo: Dịch vụ AI hiện tại đang sử dụng mô hình nền tảng MobileNetV3 (ImageNet-1K).")
        print("Khi bạn đã thu thập đủ bộ ảnh có nhãn, chạy lại lệnh này để fine-tune mô hình.")
        return

    # Khởi tạo mô hình với số lớp động
    model = FruitVisionModel(num_classes=num_classes, pretrained=True)
    model.to(args.device)

    # Đóng băng backbone nếu có yêu cầu
    if args.freeze_backbone_epochs > 0:
        for param in model.backbone.features.parameters():
            param.requires_grad = False
        print(f"Đã đóng băng backbone cho {args.freeze_backbone_epochs} epoch đầu tiên.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    best_model_weights = None
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 20)

        # Mở khóa backbone sau khi hết số epoch freeze
        if args.freeze_backbone_epochs > 0 and epoch == args.freeze_backbone_epochs + 1:
            for param in model.backbone.features.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.freeze_backbone_epochs)
            print("Đã mở khóa backbone để fine-tune toàn mạng với learning rate giảm 10x.")

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

            if phase == "val":
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_weights = model.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f"\n[EARLY STOPPING] Dừng sớm tại epoch {epoch} do {args.patience} epoch không cải thiện validation accuracy.")
            break

    time_elapsed = time.time() - start_time
    print(f"\nHuấn luyện hoàn tất trong {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Độ chính xác Validation tốt nhất: {best_acc:.4f}")

    # Lưu checkpoint an toàn
    if best_model_weights is not None:
        save_path = os.path.join(args.output_dir, "fruit_model.pt")
        # Giữ backup an toàn nếu file cũ đã tồn tại
        if os.path.exists(save_path):
            backup_path = f"{save_path}.bak"
            try:
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                os.rename(save_path, backup_path)
                print(f"Đã sao lưu checkpoint trước đó vào: {backup_path}")
            except Exception as e:
                print(f"Không thể sao lưu file cũ: {e}")

        torch.save(best_model_weights, save_path)
        print(f"Đã lưu trọng số mô hình tốt nhất vào: {save_path}")

        meta_path = os.path.join(args.output_dir, "model_meta.json")
        metadata = {
            "version": "1.0.0",
            "num_classes": num_classes,
            "classes": display_names,
            "class_ids": class_ids,
            "folders": raw_folders,
            "best_val_accuracy": float(best_acc),
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "base_model": "mobilenet_v3_small",
            "seed": args.seed,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "epochs": args.epochs,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"Đã lưu metadata mô hình tại: {meta_path}")


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
