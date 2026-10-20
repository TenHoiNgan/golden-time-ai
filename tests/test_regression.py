import pytest
from unittest.mock import patch
from app.core.registry import fruit_registry
from app.models.model_loader import model_loader


class TestRegressionAndApiContract:
    """Bộ kiểm thử hồi quy bảo đảm tính tương thích tuyệt đối của API và Backend Golden Time."""

    EXPECTED_ANALYSIS_FIELDS = {
        "fruitName",
        "qualityScore",
        "freshnessScore",
        "ripenessLabel",
        "sweetnessLabel",
        "useWithin",
        "qualityWarning",
        "suggestion",
        "confidence",
    }

    def test_health_endpoint_contract(self, client):
        """Xác thực endpoint /health trả về đúng schema và danh sách supportedFruits động."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()

        assert data["status"] in ["UP", "DEGRADED"]
        assert "modelLoaded" in data
        assert "supportedFruits" in data
        assert isinstance(data["supportedFruits"], list)
        assert len(data["supportedFruits"]) > 0
        assert "modelName" in data
        assert "device" in data

    def test_analyze_endpoint_contract_known_fruit(self, client, sample_fruit_image_bytes):
        """Xác thực endpoint /api/v1/analyze trả về đúng 9 trường bắt buộc của Java Backend."""
        files = {
            "file": ("apple.png", sample_fruit_image_bytes, "image/png"),
        }
        response = client.post("/api/v1/analyze", files=files)
        assert response.status_code == 200
        data = response.json()

        # Kiểm tra đầy đủ 9 trường
        assert set(data.keys()) == self.EXPECTED_ANALYSIS_FIELDS

        # Kiểm tra kiểu dữ liệu và ràng buộc giá trị
        assert isinstance(data["fruitName"], str)
        assert len(data["fruitName"]) > 0

        assert isinstance(data["qualityScore"], int)
        assert 0 <= data["qualityScore"] <= 100

        assert isinstance(data["freshnessScore"], int)
        assert 0 <= data["freshnessScore"] <= 100

        assert isinstance(data["ripenessLabel"], str)
        assert data["sweetnessLabel"] is None or isinstance(data["sweetnessLabel"], str)
        assert isinstance(data["useWithin"], str)
        assert data["qualityWarning"] is None or isinstance(data["qualityWarning"], str)
        assert data["suggestion"] is None or isinstance(data["suggestion"], str)

        assert isinstance(data["confidence"], (int, float))
        assert 0.0 <= data["confidence"] <= 1.0

    def test_analyze_endpoint_contract_unknown_fruit(self, client, sample_banana_image_bytes):
        """Xác thực khi không nhận diện được quả, API vẫn giữ trọn vẹn 9 trường không gây vỡ backend."""
        # Giả lập predictor trả về Chưa nhận diện được
        with patch("app.services.inference_service.fruit_classifier.predict") as mock_predict:
            mock_predict.return_value = ("Chưa nhận diện được", 0.15)
            files = {
                "file": ("banana.jpg", sample_banana_image_bytes, "image/jpeg"),
            }
            response = client.post("/api/v1/analyze", files=files)
            assert response.status_code == 200
            data = response.json()

            assert set(data.keys()) == self.EXPECTED_ANALYSIS_FIELDS
            assert data["fruitName"] == "Chưa nhận diện được"
            assert data["qualityScore"] == 0
            assert data["freshnessScore"] == 0
            assert data["ripenessLabel"] == "Không xác định"
            assert data["sweetnessLabel"] == "Chưa có dữ liệu"
            assert data["useWithin"] == "Không xác định"
            assert data["qualityWarning"] is not None
            assert data["suggestion"] is not None
            assert data["confidence"] == 0.15

    def test_registry_contains_all_10_original_fruits(self):
        """Đảm bảo 10 loại trái cây ban đầu của Golden Time luôn hiện diện đầy đủ trong registry."""
        original_fruits = [
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
        registered_displays = fruit_registry.get_display_names()
        for fruit in original_fruits:
            assert fruit in registered_displays, f"Lớp quả gốc '{fruit}' bị thiếu trong registry!"

    def test_custom_model_dynamic_class_loading_simulation(self, client, sample_fruit_image_bytes):
        """Mô phỏng trường hợp nạp custom model 20 lớp và gọi API suy luận."""
        active_classes = fruit_registry.get_display_names()[:20]
        with patch.object(model_loader, "_is_custom", True), \
             patch.object(model_loader, "_classes", active_classes), \
             patch("app.services.inference_service.fruit_classifier.predict") as mock_predict:
            mock_predict.return_value = ("Sầu riêng", 0.92)

            files = {"file": ("test.png", sample_fruit_image_bytes, "image/png")}
            response = client.post("/api/v1/analyze", files=files)
            assert response.status_code == 200
            data = response.json()

            assert data["fruitName"] == "Sầu riêng"
            assert data["confidence"] == 0.92
            assert data["qualityScore"] > 0
