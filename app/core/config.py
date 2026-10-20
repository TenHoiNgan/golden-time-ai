import os
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Cấu hình dịch vụ AI nhận diện và phân tích trái cây Golden Time."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "Golden Time AI Fruit Intelligence Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Giới hạn tệp ảnh
    MAX_IMAGE_SIZE_MB: int = 10
    ALLOWED_CONTENT_TYPES: List[str] = [
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
    ]

    # Cấu hình Model AI
    MODEL_NAME: str = "mobilenet_v3_small"
    MODEL_WEIGHTS_PATH: Optional[str] = None
    DEVICE: str = "cpu"
    CONFIDENCE_THRESHOLD: float = 0.25

    # Bảo mật API (tùy chọn khớp với AI_SERVICE_API_KEY ở backend)
    API_KEY: Optional[str] = None

    # CORS
    CORS_ORIGINS: List[str] = ["*"]


settings = Settings()
