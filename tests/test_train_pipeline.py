import json
import os
import sys
from PIL import Image
import pytest
import torch
from torchvision import transforms

from app.models.model_loader import FruitVisionModel, ModelLoader
from app.schemas.analysis import AnalysisResponse
from training.train import (
    BASELINE_15_CLASSES,
    SelectedClassesDataset,
    get_data_loaders,
    parse_args,
    set_seed,
)
from training.evaluate import calculate_metrics, evaluate


def test_parse_args_defaults():
    """Kiểm tra các tham số mặc định của parser, bao gồm 15 lớp baseline."""
    sys_argv_backup = sys.argv
    sys.argv = ["train.py"]
    try:
        args = parse_args()
        assert args.epochs == 10
        assert args.batch_size == 32
        assert args.lr == 1e-3
        assert args.seed == 42
        assert args.patience == 5
        assert args.freeze_backbone_epochs == 0
        assert args.classes == BASELINE_15_CLASSES
        assert len(args.classes) == 15
    finally:
        sys.argv = sys_argv_backup


def test_set_seed_reproducibility():
    """Kiểm tra thiết lập random seed tạo tensor ngẫu nhiên giống hệt nhau."""
    set_seed(1234)
    t1 = torch.randn(5, 5)
    set_seed(1234)
    t2 = torch.randn(5, 5)
    assert torch.equal(t1, t2)


def test_get_data_loaders_missing_dir(tmp_path):
    """Kiểm tra get_data_loaders báo lỗi khi không tìm thấy train/val."""
    with pytest.raises(FileNotFoundError, match="Không tìm thấy thư mục"):
        get_data_loaders(str(tmp_path), batch_size=4)


def test_get_data_loaders_mapping_with_registry(tmp_path):
    """Kiểm tra get_data_loaders ánh xạ đúng folder_name sang display_name và class_id."""
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"

    # Tạo 2 class: tao và cam_sanh
    for d in [train_dir, val_dir]:
        for c in ["tao", "cam_sanh"]:
            class_folder = d / c
            class_folder.mkdir(parents=True)
            img = Image.new("RGB", (100, 100), color="red")
            img.save(str(class_folder / "test.jpg"))

    dataloaders, dataset_sizes, raw_folders, display_names, class_ids, class_to_idx = get_data_loaders(
        str(tmp_path), batch_size=2, selected_classes=["tao", "cam_sanh"]
    )

    assert "cam_sanh" in raw_folders
    assert "tao" in raw_folders
    assert "Cam sành" in display_names
    assert "Táo" in display_names
    assert "cam_sanh" in class_ids
    assert "tao" in class_ids
    assert class_to_idx == {"tao": 0, "cam_sanh": 1}
    assert dataset_sizes["train"] == 2
    assert dataset_sizes["val"] == 2


def test_dataset_loader_ignores_empty_non_baseline_folder(tmp_path):
    """
    Xác minh: Thư mục 'man/' (rỗng, 0 ảnh) hoặc các lớp ngoài baseline
    được bỏ qua hoàn toàn và KHÔNG gây crash pipeline.
    """
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"

    for d in [train_dir, val_dir]:
        # Tạo lớp baseline 'tao' có ảnh
        tao_dir = d / "tao"
        tao_dir.mkdir(parents=True)
        img = Image.new("RGB", (100, 100), color="green")
        img.save(str(tao_dir / "sample.jpg"))

        # Tạo thư mục rỗng 'man' ngoài danh sách chọn
        man_dir = d / "man"
        man_dir.mkdir(parents=True)

    # get_data_loaders chỉ yêu cầu nạp 'tao'
    dataloaders, dataset_sizes, raw_folders, display_names, class_ids, class_to_idx = get_data_loaders(
        str(tmp_path), batch_size=2, selected_classes=["tao"]
    )

    assert raw_folders == ["tao"]
    assert display_names == ["Táo"]
    assert class_to_idx == {"tao": 0}
    assert dataset_sizes["train"] == 1
    assert dataset_sizes["val"] == 1


def test_missing_baseline_class_raises_clear_error(tmp_path):
    """Xác minh: Thiếu thư mục hoặc lớp baseline 0 ảnh sẽ báo lỗi rõ ràng."""
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"

    # Chỉ tạo 'tao', thiếu 'chuoi'
    for d in [train_dir, val_dir]:
        tao_dir = d / "tao"
        tao_dir.mkdir(parents=True)
        img = Image.new("RGB", (100, 100), color="blue")
        img.save(str(tao_dir / "sample.jpg"))

    with pytest.raises(FileNotFoundError, match="Thiếu thư mục lớp baseline bắt buộc"):
        get_data_loaders(str(tmp_path), batch_size=2, selected_classes=["tao", "chuoi"])


def test_empty_baseline_class_raises_value_error(tmp_path):
    """Xác minh: Lớp baseline có thư mục nhưng 0 ảnh sẽ báo ValueError rõ ràng."""
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"

    for d in [train_dir, val_dir]:
        tao_dir = d / "tao"
        tao_dir.mkdir(parents=True)
        img = Image.new("RGB", (100, 100), color="yellow")
        img.save(str(tao_dir / "sample.jpg"))

        # Thư mục chuoi tồn tại nhưng không có ảnh nào
        chuoi_dir = d / "chuoi"
        chuoi_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="không có ảnh hợp lệ nào"):
        get_data_loaders(str(tmp_path), batch_size=2, selected_classes=["tao", "chuoi"])


def test_target_indices_strictly_bounded_0_to_14(tmp_path):
    """Xác minh: Tất cả target index sinh ra luôn nằm nghiêm ngặt trong phạm vi 0..N-1."""
    root_dir = tmp_path / "split"
    # Giả lập 15 lớp baseline
    class_to_idx = {c: idx for idx, c in enumerate(BASELINE_15_CLASSES)}

    for c in BASELINE_15_CLASSES:
        cdir = root_dir / c
        cdir.mkdir(parents=True)
        img = Image.new("RGB", (50, 50), color="white")
        img.save(str(cdir / "img.jpg"))

    ds = SelectedClassesDataset(str(root_dir), class_to_idx=class_to_idx, transform=transforms.ToTensor())
    assert len(ds) == 15

    targets = set()
    for _, target in ds:
        targets.add(target)
        assert 0 <= target <= 14

    assert targets == set(range(15))


def test_evaluate_requires_valid_metadata(tmp_path):
    """Xác minh: evaluate báo FileNotFoundError hoặc ValueError rõ ràng khi metadata không hợp lệ."""
    import argparse
    dummy_args = argparse.Namespace(
        test_dir=str(tmp_path / "test"),
        model_path=str(tmp_path / "model.pt"),
        meta_path=str(tmp_path / "non_existent_meta.json"),
        output_json=None,
        batch_size=16,
        device="cpu",
    )

    # 1. Thiếu test_dir
    # evaluate in ra thông báo và return an toàn
    evaluate(dummy_args)

    # 2. Tạo test_dir và model_path giả lập, nhưng thiếu meta_path
    (tmp_path / "test").mkdir()
    (tmp_path / "model.pt").touch()

    with pytest.raises(FileNotFoundError, match="Không tìm thấy metadata mô hình"):
        evaluate(dummy_args)

    # 3. Meta_path tồn tại nhưng thiếu 'classes'
    bad_meta = tmp_path / "bad_meta.json"
    with open(bad_meta, "w", encoding="utf-8") as f:
        json.dump({"version": "1.0.0"}, f)
    dummy_args.meta_path = str(bad_meta)

    with pytest.raises(ValueError, match="Metadata.*không hợp lệ"):
        evaluate(dummy_args)


def test_api_schema_contract_preserved():
    """Xác minh: Schema AnalysisResponse vẫn bảo toàn 9 trường hợp đồng bắt buộc kể cả sweetnessLabel."""
    sample = {
        "fruitName": "Sầu riêng",
        "qualityScore": 95,
        "freshnessScore": 92,
        "ripenessLabel": "Vừa chín",
        "sweetnessLabel": "Ngọt béo đậm đà",
        "useWithin": "2–3 ngày",
        "qualityWarning": None,
        "suggestion": "Bảo quản nhiệt độ phòng thoáng mát",
        "confidence": 0.965,
    }
    resp = AnalysisResponse(**sample)
    assert resp.fruitName == "Sầu riêng"
    assert resp.sweetnessLabel == "Ngọt béo đậm đà"
    assert resp.confidence == 0.965
    assert set(resp.model_dump().keys()) == {
        "fruitName", "qualityScore", "freshnessScore", "ripenessLabel",
        "sweetnessLabel", "useWithin", "qualityWarning", "suggestion", "confidence"
    }
