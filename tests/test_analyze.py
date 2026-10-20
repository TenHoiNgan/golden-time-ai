import io
from fastapi.testclient import TestClient


def test_analyze_valid_image(client: TestClient, sample_fruit_image_bytes: bytes):
    """Kiểm tra phân tích ảnh hoa quả hợp lệ thành công."""
    files = {
        "file": ("test_apple.png", sample_fruit_image_bytes, "image/png"),
    }
    response = client.post("/api/v1/analyze", files=files)
    assert response.status_code == 200

    data = response.json()

    # 1. Kiểm tra các trường bắt buộc theo hợp đồng dữ liệu
    assert "fruitName" in data
    assert isinstance(data["fruitName"], str)
    assert len(data["fruitName"]) > 0

    assert "qualityScore" in data
    assert isinstance(data["qualityScore"], int)
    assert 0 <= data["qualityScore"] <= 100

    assert "freshnessScore" in data
    assert isinstance(data["freshnessScore"], int)
    assert 0 <= data["freshnessScore"] <= 100

    assert "ripenessLabel" in data
    assert isinstance(data["ripenessLabel"], str)

    assert "confidence" in data
    assert isinstance(data["confidence"], (int, float))
    assert 0.0 <= data["confidence"] <= 1.0

    # 2. Kiểm tra trường độ ngọt sweetnessLabel
    assert "sweetnessLabel" in data
    assert isinstance(data["sweetnessLabel"], str)
    assert len(data["sweetnessLabel"]) > 0

    # 3. Kiểm tra các trường khuyến nghị
    assert "useWithin" in data
    assert "suggestion" in data
    assert "qualityWarning" in data


def test_analyze_compat_alias_endpoint(client: TestClient, sample_fruit_image_bytes: bytes):
    """Kiểm tra route tương thích /analyze hoạt động giống /api/v1/analyze."""
    files = {
        "file": ("test_apple.png", sample_fruit_image_bytes, "image/png"),
    }
    response = client.post("/analyze", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "fruitName" in data
    assert "sweetnessLabel" in data


def test_analyze_empty_file(client: TestClient):
    """Gửi tệp rỗng phải trả về 400 Bad Request."""
    files = {
        "file": ("empty.png", b"", "image/png"),
    }
    response = client.post("/api/v1/analyze", files=files)
    assert response.status_code == 400
    assert "rỗng" in response.json().get("detail", "").lower()


def test_analyze_non_image_file(client: TestClient, non_image_bytes: bytes):
    """Gửi tệp không phải ảnh phải trả về 400 Bad Request."""
    files = {
        "file": ("document.txt", non_image_bytes, "text/plain"),
    }
    response = client.post("/api/v1/analyze", files=files)
    assert response.status_code == 400
    assert "hợp lệ" in response.json().get("detail", "").lower()


def test_analyze_corrupted_image(client: TestClient, corrupted_image_bytes: bytes):
    """Gửi tệp ảnh bị hỏng cấu trúc phải trả về 400 Bad Request."""
    files = {
        "file": ("corrupt.jpg", corrupted_image_bytes, "image/jpeg"),
    }
    response = client.post("/api/v1/analyze", files=files)
    assert response.status_code == 400


def test_analyze_oversized_file(client: TestClient):
    """Gửi tệp vượt quá 10MB phải trả về 413 Payload Too Large."""
    # Tạo payload vượt quá 10MB
    large_payload = b"\x89PNG\r\n\x1a\n" + b"\x00" * (11 * 1024 * 1024)
    files = {
        "file": ("oversized.png", large_payload, "image/png"),
    }
    response = client.post("/api/v1/analyze", files=files)
    assert response.status_code == 413
    assert "vượt quá" in response.json().get("detail", "").lower()
