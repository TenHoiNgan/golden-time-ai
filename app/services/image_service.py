import io
from typing import Tuple, Optional
from PIL import Image
import torch
from torchvision import transforms
from fastapi import HTTPException, status

from app.core.config import settings

# ImageNet normalization standard
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ImageService:
    """Xử lý kiểm tra, xác thực và tiền xử lý ảnh cho mô hình suy luận."""

    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def validate_and_load(self, file_bytes: bytes, declared_content_type: Optional[str] = None) -> Image.Image:
        """
        Kiểm tra tính hợp lệ của tệp ảnh:
        - Giới hạn kích thước (mặc định 10MB).
        - Magic bytes định dạng ảnh thực tế.
        - Khả năng mở và giải mã bằng Pillow.
        """
        if not file_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tệp ảnh rỗng hoặc không có dữ liệu.",
            )

        max_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
        if len(file_bytes) > max_bytes:
            code = getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413)
            raise HTTPException(
                status_code=code,
                detail=f"Kích thước ảnh vượt quá giới hạn cho phép ({settings.MAX_IMAGE_SIZE_MB} MB).",
            )

        # Kiểm tra Magic bytes thực tế
        detected_format = self._detect_magic_format(file_bytes)
        if not detected_format:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Định dạng tệp không phải ảnh JPEG, PNG hoặc WebP hợp lệ.",
            )

        try:
            image = Image.open(io.BytesIO(file_bytes))
            image.verify()  # Kiểm tra tính toàn vẹn của tệp ảnh
            # Mở lại sau khi verify vì Pillow yêu cầu sau verify()
            image = Image.open(io.BytesIO(file_bytes))
            # Chuyển về RGB để loại bỏ kênh alpha hoặc ảnh grayscale
            if image.mode != "RGB":
                image = image.convert("RGB")
            return image
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tệp ảnh bị hỏng hoặc không thể giải mã nội dung ảnh.",
            )

    def preprocess_for_model(self, image: Image.Image, device: str = "cpu") -> torch.Tensor:
        """Chuyển đổi PIL Image sang PyTorch Tensor chuẩn hóa (1, 3, 224, 224)."""
        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0)  # Thêm batch dimension
        return tensor.to(device)

    def extract_visual_features(self, image: Image.Image) -> dict:
        """
        Trích xuất đặc trưng thị giác để hỗ trợ đánh giá độ chín và độ tươi:
        - Không gian màu HSV (Hue, Saturation, Value)
        - Độ đồng nhất bề mặt
        - Tỷ lệ vùng đốm tối (bruise / defect ratio)
        """
        # Resize nhỏ để phân tích nhanh
        thumb = image.resize((128, 128))
        hsv_img = thumb.convert("HSV")
        h_data, s_data, v_data = hsv_img.split()

        get_pixels = lambda img_channel: list(
            img_channel.get_flattened_data()
            if hasattr(img_channel, "get_flattened_data")
            else img_channel.getdata()
        )
        h_pixels = get_pixels(h_data)
        s_pixels = get_pixels(s_data)
        v_pixels = get_pixels(v_data)

        total_pixels = len(h_pixels)
        if total_pixels == 0:
            return {
                "avg_hue": 0.0,
                "avg_sat": 0.0,
                "avg_val": 0.0,
                "green_ratio": 0.0,
                "yellow_orange_ratio": 0.0,
                "red_ratio": 0.0,
                "dark_spot_ratio": 0.0,
            }

        # Hue trên Pillow HSV chạy từ 0 đến 255 (tương đương 0-360 độ: 1 deg ~ 0.71)
        # Đỏ: ~0-20 và 235-255
        # Cam/Vàng: ~20-65
        # Xanh lá: ~65-115
        green_count = 0
        yellow_orange_count = 0
        red_count = 0
        dark_spot_count = 0

        for h, s, v in zip(h_pixels, s_pixels, v_pixels):
            # Điểm thâm/sẫm màu: Value thấp (< 60) kết hợp Saturation vừa
            if v < 60:
                dark_spot_count += 1
            if s > 40:  # Chỉ xét khi có độ bão hòa màu nhất định
                if 65 <= h <= 115:
                    green_count += 1
                elif 20 <= h < 65:
                    yellow_orange_count += 1
                elif h < 20 or h >= 235:
                    red_count += 1

        return {
            "avg_hue": sum(h_pixels) / total_pixels,
            "avg_sat": sum(s_pixels) / total_pixels,
            "avg_val": sum(v_pixels) / total_pixels,
            "green_ratio": green_count / total_pixels,
            "yellow_orange_ratio": yellow_orange_count / total_pixels,
            "red_ratio": red_count / total_pixels,
            "dark_spot_ratio": dark_spot_count / total_pixels,
        }

    @staticmethod
    def _detect_magic_format(data: bytes) -> Optional[str]:
        """Kiểm tra byte signature của các định dạng ảnh phổ biến."""
        if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
            return "png"
        if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
            return "jpeg"
        if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return "webp"
        return None


image_service = ImageService()
