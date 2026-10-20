import json
import torch
import pytest
from app.models.model_loader import FruitVisionModel, ModelLoader, model_loader


def test_fruit_vision_model_dynamic_classes():
    """Kiểm tra khởi tạo FruitVisionModel với các số lượng lớp khác nhau."""
    for n_cls in [5, 10, 20, 50, 100]:
        model = FruitVisionModel(num_classes=n_cls, pretrained=False)
        dummy_input = torch.zeros(2, 3, 224, 224)
        out = model(dummy_input)
        assert out.shape == (2, n_cls)


def test_model_loader_properties():
    """Kiểm tra các thuộc tính động của model_loader."""
    assert model_loader.is_loaded is True
    assert model_loader.num_classes >= 10
    assert len(model_loader.class_names) == model_loader.num_classes
    assert "Táo" in model_loader.class_names
    assert "Cam sành" in model_loader.class_names


def test_custom_checkpoint_with_matching_meta(tmp_path):
    """Kiểm tra nạp checkpoint tùy biến thành công khi metadata khớp số lớp."""
    num_classes = 4
    custom_classes = ["Táo", "Cam", "Chuối", "Xoài"]
    custom_ids = ["tao", "cam", "chuoi", "xoai"]

    model = FruitVisionModel(num_classes=num_classes, pretrained=False)
    weights_file = tmp_path / "custom_model.pt"
    meta_file = tmp_path / "model_meta.json"

    torch.save(model.state_dict(), str(weights_file))
    meta_file.write_text(
        json.dumps({
            "classes": custom_classes,
            "class_ids": custom_ids,
            "num_classes": num_classes,
        }),
        encoding="utf-8",
    )

    # Khởi tạo loader trỏ tới checkpoint này
    loader = ModelLoader.__new__(ModelLoader)
    loader._model = None
    loader._is_custom = False
    loader._device = torch.device("cpu")

    # Giả lập đường dẫn settings
    from unittest.mock import patch
    with patch("app.models.model_loader.settings.MODEL_WEIGHTS_PATH", str(weights_file)):
        loader._init_model()

    assert loader.is_custom is True
    assert loader.num_classes == 4
    assert loader.class_names == custom_classes


def test_custom_checkpoint_shape_mismatch_fallback(tmp_path):
    """Kiểm tra checkpoint có số lớp lệch với metadata bị phát hiện và fallback an toàn."""
    model = FruitVisionModel(num_classes=10, pretrained=False)
    weights_file = tmp_path / "mismatch_model.pt"
    meta_file = tmp_path / "model_meta.json"

    torch.save(model.state_dict(), str(weights_file))
    # Metadata khai báo chỉ 4 lớp, nhưng weights có 10 lớp
    meta_file.write_text(
        json.dumps({
            "classes": ["A", "B", "C", "D"],
            "class_ids": ["a", "b", "c", "d"],
            "num_classes": 4,
        }),
        encoding="utf-8",
    )

    loader = ModelLoader.__new__(ModelLoader)
    loader._model = None
    loader._is_custom = False
    loader._device = torch.device("cpu")

    from unittest.mock import patch
    with patch("app.models.model_loader.settings.MODEL_WEIGHTS_PATH", str(weights_file)):
        loader._init_model()

    # Phải fallback về base ImageNet model, không được gán is_custom=True
    assert loader.is_custom is False
    assert loader.is_loaded is True
