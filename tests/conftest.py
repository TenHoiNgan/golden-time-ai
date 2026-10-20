import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    """Khởi tạo TestClient của FastAPI."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_fruit_image_bytes():
    """Tạo tệp ảnh PNG hoa quả hợp lệ giả lập trong bộ nhớ."""
    img = Image.new("RGB", (256, 256), color=(220, 40, 40))  # Màu đỏ tượng trưng cho quả táo
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def sample_banana_image_bytes():
    """Tạo tệp ảnh JPEG hoa quả hợp lệ giả lập trong bộ nhớ (màu vàng chuối)."""
    img = Image.new("RGB", (256, 256), color=(240, 210, 30))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def non_image_bytes():
    """Tệp giả lập không phải ảnh (text)."""
    return b"This is a text file, definitely not a valid image."


@pytest.fixture
def corrupted_image_bytes():
    """Tệp giả mạo magic bytes của JPEG nhưng nội dung bị hỏng."""
    return b"\xff\xd8\xff\xe0" + b"\x00" * 30
