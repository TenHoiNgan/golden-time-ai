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

logger = logging.getLogger(__name__)

# Danh mục trái cây mục tiêu của hệ thống Golden Time
GOLDEN_TIME_FRUITS = [
    "Táo",
    "Cam sành",
    "Chuối",
    "Xoài cát Hòa Lộc",
    "Nho",
    "Dâu tây",
    "Dưa hấu",
    "Thanh long",
    "Dứa",
    "Bơ",
]

# Ánh xạ nhãn ImageNet sang loại quả của Golden Time
# 948: Granny Smith, 950: orange, 951: lemon, 952: fig, 953: pineapple, 954: banana, 955: jackfruit, 957: pomegranate, 990: strawberry
IMAGENET_FRUIT_MAP: Dict[int, str] = {
    948: "Táo",            # Granny Smith (apple)
    950: "Cam sành",       # orange
    951: "Cam sành",       # lemon/citrus
    953: "Dứa",            # pineapple
    954: "Chuối",          # banana
    955: "Thanh long",     # jackfruit / tropical
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
    ("mango", "Xoài cát Hòa Lộc"),
    ("grape", "Nho"),
    ("strawberry", "Dâu tây"),
    ("watermelon", "Dưa hấu"),
    ("dragon fruit", "Thanh long"),
    ("pitaya", "Thanh long"),
    ("pineapple", "Dứa"),
    ("avocado", "Bơ"),
]


class FruitVisionModel(nn.Module):
    """Mô hình phân loại trái cây dựa trên kiến trúc MobileNetV3."""

    def __init__(self, num_classes: int = len(GOLDEN_TIME_FRUITS), pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights)
        in_features = self.backbone.classifier[3].in_features
        # Thay thế lớp classifier cuối cho số lớp trái cây mục tiêu
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ModelLoader:
    """Quản lý tải và nạp mô hình vào bộ nhớ."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance._model = None
            cls._instance._is_custom = False
            cls._instance._device = torch.device(settings.DEVICE if torch.cuda.is_available() and settings.DEVICE == "cuda" else "cpu")
            cls._instance._init_model()
        return cls._instance

    def _init_model(self):
        """Khởi tạo mô hình: kiểm tra checkpoint custom trước, nếu không có thì tải pretrained MobileNetV3."""
        weights_path = settings.MODEL_WEIGHTS_PATH or os.path.join(
            os.path.dirname(__file__), "..", "..", "models", "fruit_model.pt"
        )
        weights_path = os.path.abspath(weights_path)

        if os.path.exists(weights_path):
            try:
                logger.info("Đang nạp mô hình fine-tuned từ %s...", weights_path)
                model = FruitVisionModel(num_classes=len(GOLDEN_TIME_FRUITS), pretrained=False)
                state_dict = torch.load(weights_path, map_location=self._device)
                model.load_state_dict(state_dict)
                model.to(self._device)
                model.eval()
                self._model = model
                self._is_custom = True
                logger.info("Đã nạp thành công mô hình fine-tuned!")
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


model_loader = ModelLoader()
