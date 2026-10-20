import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from app.models.predictors import (
    FruitClassifier,
    RipenessEstimator,
    FreshnessQualityEvaluator,
    SweetnessEstimator,
)
from app.core.config import settings
from app.core.registry import fruit_registry


class TestPredictors:
    def test_fruit_classifier_custom_model_dynamic_classes(self):
        """Kiểm tra FruitClassifier với mô hình tùy biến và danh sách lớp động."""
        classifier = FruitClassifier()
        mock_loader = MagicMock()
        mock_loader.is_loaded = True
        mock_loader.is_custom = True
        mock_loader.class_names = ["Táo", "Cam", "Chuối", "Bơ", "Sầu riêng"]

        # Giả lập output logits: class index 4 (Sầu riêng) có logit cao nhất
        mock_model = MagicMock()
        logits = torch.tensor([[1.0, 2.0, 0.5, 0.1, 10.0]])
        mock_model.return_value = logits
        mock_loader.model = mock_model
        classifier.loader = mock_loader

        dummy_tensor = torch.zeros(1, 3, 224, 224)
        fruit_name, conf = classifier.predict(dummy_tensor, {})

        assert fruit_name == "Sầu riêng"
        assert conf > 0.9

    def test_fruit_classifier_low_confidence_unknown(self):
        """Kiểm tra khi độ tin cậy thấp hơn ngưỡng CONFIDENCE_THRESHOLD thì trả về Chưa nhận diện được."""
        classifier = FruitClassifier()
        mock_loader = MagicMock()
        mock_loader.is_loaded = True
        mock_loader.is_custom = True
        mock_loader.class_names = ["Táo", "Cam", "Chuối", "Xoài", "Bơ"]

        # Phân bố đều qua 5 lớp khiến mỗi lớp có xác suất ~0.20 < settings.CONFIDENCE_THRESHOLD (0.25)
        mock_model = MagicMock()
        logits = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]])
        mock_model.return_value = logits
        mock_loader.model = mock_model
        classifier.loader = mock_loader

        dummy_tensor = torch.zeros(1, 3, 224, 224)
        fruit_name, conf = classifier.predict(dummy_tensor, {})

        assert fruit_name == "Chưa nhận diện được"
        assert conf < settings.CONFIDENCE_THRESHOLD

    def test_ripeness_estimator_known_and_extended_fruits(self):
        """Kiểm tra ước tính độ chín cho cả trái cây cơ bản và trái cây mở rộng từ profile."""
        estimator = RipenessEstimator()

        # Unknown fruit
        assert estimator.estimate("Chưa nhận diện được", {}) == "Không xác định"

        # Chuối
        assert estimator.estimate("Chuối", {"green_ratio": 0.5}) == "Chưa chín (vỏ xanh)"
        assert estimator.estimate("Chuối", {"dark_spot_ratio": 0.3}) == "Quá chín (có đốm nâu)"
        assert estimator.estimate("Chuối", {"yellow_orange_ratio": 0.6}) == "Chín tới (vàng ươm)"

        # Quả mở rộng từ registry profile: Bơ (avocado)
        assert estimator.estimate("Bơ", {"green_ratio": 0.45}) == "Còn xanh"
        assert estimator.estimate("Bơ", {"dark_spot_ratio": 0.30}) == "Quá chín"

    def test_freshness_evaluator_and_storage_tip_from_profile(self):
        """Kiểm tra đánh giá chất lượng và gợi ý bảo quản từ registry profile."""
        evaluator = FreshnessQualityEvaluator()

        # Unknown fruit
        q, f, use_within, warning, tip = evaluator.evaluate("Chưa nhận diện được", 0.3, "Không xác định", {})
        assert q == 0
        assert f == 0
        assert use_within == "Không xác định"
        assert warning is not None
        assert "chụp lại ảnh" in tip

        # Extended fruit: Sầu riêng
        q, f, use_within, warning, tip = evaluator.evaluate(
            "Sầu riêng", 0.95, "Vừa chín tới", {"dark_spot_ratio": 0.05, "avg_sat": 150.0, "avg_val": 150.0}
        )
        assert 15 <= q <= 100
        assert 20 <= f <= 100
        assert "gai" in tip or "tủ lạnh" in tip or "Bảo quản" in tip

    def test_sweetness_estimator_heuristic_and_registry_profile(self):
        """Kiểm tra ước tính độ ngọt định tính, disclaimer khoa học và profile lookup."""
        estimator = SweetnessEstimator()

        # Kiểm tra docstring có disclaimer khoa học về Brix
        assert "Brix" in estimator.__doc__
        assert "heuristic" in estimator.__doc__

        # Unknown
        assert estimator.estimate("Chưa nhận diện được", "Không xác định") == "Chưa có dữ liệu"

        # Quả cơ bản
        assert estimator.estimate("Chuối", "Quá chín") == "Ngọt đậm"
        assert estimator.estimate("Táo", "Chín đỏ") == "Ngọt thanh giòn"

        # Quả mở rộng từ classes.json profile
        assert estimator.estimate("Sầu riêng", "Vừa chín tới") == "Ngọt đậm béo ngậy"
        assert estimator.estimate("Mít", "Vừa chín tới") == "Ngọt đậm thơm nồng"
        assert estimator.estimate("Ổi", "Vừa chín tới") == "Ngọt thanh giòn"
