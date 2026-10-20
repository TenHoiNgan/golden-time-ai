import json
import pytest
from app.core.registry import FruitRegistry, FruitClass, fruit_registry


def test_registry_loaded_default():
    """Kiểm tra registry mặc định nạp đủ 20 lớp và không có lỗi."""
    assert fruit_registry.total_classes == 20
    assert "tao" in fruit_registry.get_class_ids()
    assert "sau_rieng" in fruit_registry.get_class_ids()
    assert "mang_cut" in fruit_registry.get_class_ids()

    tao = fruit_registry.get_by_id("tao")
    assert tao is not None
    assert tao.display_name == "Táo"
    assert tao.folder_name == "tao"


def test_registry_lookups():
    """Kiểm tra các phương thức tra cứu theo folder và alias."""
    cam = fruit_registry.get_by_folder("cam_sanh")
    assert cam is not None
    assert cam.class_id == "cam_sanh"

    # Tra cứu qua alias
    matched_durian = fruit_registry.match_any("durian")
    assert matched_durian is not None
    assert matched_durian.class_id == "sau_rieng"

    matched_apple = fruit_registry.match_any("apple")
    assert matched_apple is not None
    assert matched_apple.class_id == "tao"

    assert fruit_registry.match_any("khong_ton_tai") is None


def test_registry_rejects_duplicate_id(tmp_path):
    """Kiểm tra registry từ chối file có class_id trùng lặp."""
    bad_data = [
        {"class_id": "tao", "display_name": "Táo Đỏ", "folder_name": "tao_do"},
        {"class_id": "tao", "display_name": "Táo Xanh", "folder_name": "tao_xanh"},
    ]
    file = tmp_path / "bad.json"
    file.write_text(json.dumps(bad_data), encoding="utf-8")

    with pytest.raises(ValueError, match="Trùng lặp class_id"):
        FruitRegistry(str(file))


def test_registry_rejects_duplicate_folder(tmp_path):
    """Kiểm tra registry từ chối file có folder_name trùng lặp."""
    bad_data = [
        {"class_id": "tao_1", "display_name": "Táo 1", "folder_name": "tao"},
        {"class_id": "tao_2", "display_name": "Táo 2", "folder_name": "tao"},
    ]
    file = tmp_path / "bad.json"
    file.write_text(json.dumps(bad_data), encoding="utf-8")

    with pytest.raises(ValueError, match="Trùng lặp folder_name"):
        FruitRegistry(str(file))


def test_registry_validate_dataset_folders(tmp_path):
    """Kiểm tra hàm validate_dataset_structure phân loại đúng matched, missing, unexpected."""
    # Tạo cấu trúc thư mục giả lập
    (tmp_path / "tao").mkdir()
    (tmp_path / "chuoi").mkdir()
    (tmp_path / "thu_muc_la").mkdir()

    reg = fruit_registry
    matched, missing, unexpected = reg.validate_dataset_structure(str(tmp_path))

    assert "tao" in matched
    assert "chuoi" in matched
    assert "thu_muc_la" in unexpected
    assert "cam_sanh" in missing
