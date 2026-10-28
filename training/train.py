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
from typing import Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

# Thêm đường dẫn app vào sys.path để tái sử dụng model definition và registry
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.core.registry import fruit_registry
from app.models.model_loader import FruitVisionModel

# 15 lớp baseline đạt chuẩn >= 200 ảnh theo registry (bảo đảm thứ tự registry)
BASELINE_15_CLASSES: List[str] = [
    "tao",
    "cam_sanh",
    "chuoi",
    "xoai",
    "nho",
    "dau_tay",
    "thanh_long",
    "dua",
    "bo",
    "sau_rieng",
    "mit",
    "du_du",
    "oi",
    "vai",
    "mang_cau",
]

VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


class SelectedClassesDataset(torch.utils.data.Dataset):
    """
    Dataset loader chỉ nạp các lớp được chọn tường minh, dùng chung một class_to_idx cố định.
    Bỏ qua hoàn toàn các lớp ngoài danh sách (như 'man' hay các lớp chưa đủ dữ liệu),
    tránh crash khi gặp thư mục rỗng và bảo đảm target index luôn trong phạm vi 0..N-1.
    """

    def __init__(
        self,
        root_dir: str,
        class_to_idx: Dict[str, int],
        transform=None,
    ):
        self.root_dir = os.path.abspath(root_dir)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.classes = sorted(list(class_to_idx.keys()), key=lambda c: class_to_idx[c])
        self.samples: List[Tuple[str, int]] = []

        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Thư mục không tồn tại: {self.root_dir}")

        for folder_name, target_idx in self.class_to_idx.items():
            folder_path = os.path.join(self.root_dir, folder_name)
            if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
                raise FileNotFoundError(
                    f"Thiếu thư mục lớp baseline bắt buộc: '{folder_name}' trong {self.root_dir}"
                )

            image_files = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith(VALID_IMAGE_EXTENSIONS)
            ]
            if len(image_files) == 0:
                raise ValueError(
                    f"Lớp baseline '{folder_name}' không có ảnh hợp lệ nào trong {self.root_dir}"
                )

            for img_path in sorted(image_files):
                self.samples.append((img_path, target_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, target = self.samples[idx]
        with Image.open(img_path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


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
    parser.add_argument(
        "--classes",
        nargs="+",
        default=BASELINE_15_CLASSES,
        help="Danh sách class_id cần huấn luyện (mặc định: 15 lớp baseline)",
    )
    return parser.parse_args()


def get_data_loaders(
    data_dir: str,
    batch_size: int,
    selected_classes: Optional[List[str]] = None,
) -> Tuple[Dict[str, DataLoader], Dict[str, int], List[str], List[str], List[str], Dict[str, int]]:
    """
    Khởi tạo data transformations và loaders cho tập train và val.
    Ánh xạ thư mục theo đúng fruit_registry và class_to_idx cố định.
    Chỉ nạp các lớp trong selected_classes (mặc định: BASELINE_15_CLASSES),
    bỏ qua các thư mục khác (như 'man' hoặc các lớp không thuộc baseline).
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(
            f"Không tìm thấy thư mục {train_dir} hoặc {val_dir}. "
            f"Vui lòng tham khảo hướng dẫn cấu trúc dataset tại datasets/README.md."
        )

    # 1. Xác định danh sách class mục tiêu
    target_classes = selected_classes if selected_classes is not None else BASELINE_15_CLASSES

    # 2. Xây dựng mapping nhất quán từ registry (giữ đúng thứ tự registry)
    ordered_folders: List[str] = []
    display_names: List[str] = []
    class_ids: List[str] = []
    class_to_idx: Dict[str, int] = {}

    for c in target_classes:
        fruit_cls = fruit_registry.get_by_id(c) or fruit_registry.get_by_folder(c)
        if fruit_cls:
            f_name = fruit_cls.folder_name
            d_name = fruit_cls.display_name
            c_id = fruit_cls.class_id
        else:
            f_name = c
            d_name = c
            c_id = c

        if f_name not in class_to_idx:
            idx = len(class_to_idx)
            class_to_idx[f_name] = idx
            ordered_folders.append(f_name)
            display_names.append(d_name)
            class_ids.append(c_id)

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
        "train": SelectedClassesDataset(train_dir, class_to_idx, data_transforms["train"]),
        "val": SelectedClassesDataset(val_dir, class_to_idx, data_transforms["val"]),
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == "train"), num_workers=2)
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}

    return dataloaders, dataset_sizes, ordered_folders, display_names, class_ids, class_to_idx


def train_model(args):
    set_seed(args.seed)

    print("=" * 60)
    print("Bắt đầu quy trình huấn luyện mô hình Golden Time AI")
    print(f"Thiết bị: {args.device} | Epochs: {args.epochs} | Batch size: {args.batch_size} | Seed: {args.seed}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        dataloaders, dataset_sizes, raw_folders, display_names, class_ids, class_to_idx = get_data_loaders(
            args.data_dir, args.batch_size, selected_classes=args.classes
        )
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
            "class_to_idx": class_to_idx,
            "best_val_accuracy": float(best_acc),
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "base_model": "mobilenet_v3_small",
            "seed": args.seed,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "epochs": args.epochs,
            "dataset_split_plan": "datasets/durian_split_plan.json",
            "dataset_manifest_p4": "datasets/manifest_phase4_sau_rieng.json",
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"Đã lưu metadata mô hình tại: {meta_path}")


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
