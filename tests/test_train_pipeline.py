import os
from PIL import Image
import pytest
import torch
from training.train import parse_args, set_seed, get_data_loaders


def test_parse_args_defaults():
    """Kiểm tra các tham số mặc định của parser."""
    import sys
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
            # Tạo 1 ảnh giả lập
            img = Image.new("RGB", (100, 100), color="red")
            img.save(str(class_folder / "test.jpg"))

    dataloaders, dataset_sizes, raw_folders, display_names, class_ids = get_data_loaders(
        str(tmp_path), batch_size=2
    )

    assert "cam_sanh" in raw_folders
    assert "tao" in raw_folders
    assert "Cam sành" in display_names
    assert "Táo" in display_names
    assert "cam_sanh" in class_ids
    assert "tao" in class_ids
    assert dataset_sizes["train"] == 2
    assert dataset_sizes["val"] == 2
