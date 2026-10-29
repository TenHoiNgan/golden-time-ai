import logging
import os
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)

from app.core.config import settings
from app.core.registry import fruit_registry

logger = logging.getLogger(__name__)

# Danh mục trái cây mục tiêu cơ sở (tương thích ngược với hệ thống Golden Time cũ)
GOLDEN_TIME_FRUITS = fruit_registry.get_display_names()[:10]

# Ánh xạ nhãn ImageNet sang loại quả của Golden Time
# 948: Granny Smith, 950: orange, 951: lemon, 952: fig, 953: pineapple, 954: banana, 955: jackfruit, 957: pomegranate, 990: strawberry
IMAGENET_FRUIT_MAP: Dict[int, str] = {
    948: "Táo",            # Granny Smith (apple)
    950: "Cam sành",       # orange
    951: "Cam sành",       # lemon/citrus
    953: "Dứa",            # pineapple
    954: "Chuối",          # banana
    957: "Lựu",            # pomegranate
    990: "Dâu tây",        # strawberry
}

# Các từ khóa tiếng Anh/Việt tương ứng trong ImageNet class names
KEYWORD_TO_FRUIT = [
    ("apple", "Táo"),
    ("orange", "Cam sành"),
    ("lemon", "Cam sành"),
    ("citrus", "Cam sành"),
    ("banana", "Chuối"),
    ("jackfruit", "Mít"),
    ("mango", "Xoài cát Hòa Lộc"),
    ("grape", "Nho"),
    ("strawberry", "Dâu tây"),
    ("watermelon", "Dưa hấu"),
    ("dragon fruit", "Thanh long"),
    ("pitaya", "Thanh long"),
    ("pineapple", "Dứa"),
    ("avocado", "Bơ"),
]


import json


class FruitVisionModel(nn.Module):
    """Mô hình phân loại trái cây dựa trên kiến trúc MobileNetV3 hỗ trợ số lớp động."""

    def __init__(self, num_classes: Optional[int] = None, pretrained: bool = True):
        super().__init__()
        target_classes = num_classes if num_classes is not None else fruit_registry.total_classes
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights)
        in_features = self.backbone.classifier[3].in_features
        # Thay thế lớp classifier cuối cho số lớp trái cây mục tiêu
        self.backbone.classifier[3] = nn.Linear(in_features, target_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ModelLoader:
    """Quản lý tải và nạp mô hình vào bộ nhớ với ánh xạ nhãn động."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance._model = None
            cls._instance._is_custom = False
            cls._instance._classes = fruit_registry.get_display_names()
            cls._instance._class_ids = fruit_registry.get_class_ids()
            cls._instance._meta = None
            cls._instance._device = torch.device(settings.DEVICE if torch.cuda.is_available() and settings.DEVICE == "cuda" else "cpu")
            cls._instance._init_model()
        return cls._instance

    def _init_model(self):
        """Khởi tạo mô hình: kiểm tra checkpoint custom và metadata trước, nếu không có thì tải pretrained MobileNetV3."""
        weights_path = settings.MODEL_WEIGHTS_PATH or os.path.join(
            os.path.dirname(__file__), "..", "..", "models", "fruit_model.pt"
        )
        weights_path = os.path.abspath(weights_path)
        meta_path = os.path.join(os.path.dirname(weights_path), "model_meta.json")

        if os.path.exists(weights_path):
            try:
                logger.info("Đang nạp mô hình fine-tuned từ %s...", weights_path)
                state_dict = torch.load(weights_path, map_location=self._device)
                classifier_weight = state_dict.get("backbone.classifier.3.weight")
                if classifier_weight is None:
                    raise ValueError("Không tìm thấy trọng số backbone.classifier.3.weight trong checkpoint.")

                out_features = classifier_weight.shape[0]

                # 1. Đọc metadata nếu có
                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    classes = meta.get("classes", [])
                    if len(classes) != out_features:
                        raise ValueError(
                            f"Lệch số lớp: checkpoint có {out_features} lớp nhưng metadata khai báo {len(classes)} lớp."
                        )
                    self._classes = classes
                    self._class_ids = meta.get("class_ids", [])
                    self._meta = meta
                elif out_features == len(GOLDEN_TIME_FRUITS):
                    logger.warning("Không tìm thấy model_meta.json. Sử dụng danh mục 10 lớp cơ sở tương thích ngược.")
                    self._classes = GOLDEN_TIME_FRUITS
                    self._class_ids = fruit_registry.get_class_ids()[:len(GOLDEN_TIME_FRUITS)]
                else:
                    raise ValueError(
                        f"Thiếu file metadata model_meta.json cho checkpoint tùy biến {out_features} lớp. "
                        "Không thể tự động suy diễn thứ tự nhãn an toàn."
                    )

                model = FruitVisionModel(num_classes=out_features, pretrained=False)
                model.load_state_dict(state_dict)
                model.to(self._device)
                model.eval()
                self._model = model
                self._is_custom = True
                logger.info("Đã nạp thành công mô hình fine-tuned với %d lớp nhận diện!", out_features)
                return
            except Exception as e:
                logger.warning("Không nạp được checkpoint custom (%s). Chuyển sang mô hình pretrained cơ sở.", e)

        try:
            logger.info("Đang nạp mô hình pretrained MobileNetV3 (ImageNet-1K)...")
            # Tải MobileNetV3 pretrained gốc
            weights = MobileNet_V3_Small_Weights.DEFAULT
            model = mobilenet_v3_small(weights=weights)
            model.to(self._device)
            model.eval()
            self._model = model
            self._is_custom = False
            self._classes = fruit_registry.get_display_names()
            self._class_ids = fruit_registry.get_class_ids()
            self._categories = weights.meta.get("categories", [])
            logger.info("Đã nạp thành công mô hình MobileNetV3 pretrained!")
        except Exception as e:
            logger.error("Lỗi khi nạp mô hình MobileNetV3: %s", e)
            self._model = None

    @property
    def model(self):
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def is_custom(self) -> bool:
        return self._is_custom

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def class_names(self) -> List[str]:
        return self._classes

    @property
    def class_ids(self) -> List[str]:
        return self._class_ids

    @property
    def num_classes(self) -> int:
        return len(self._classes)

    @property
    def metadata(self) -> Optional[Dict]:
        return self._meta


model_loader = ModelLoader()
