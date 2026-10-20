from fastapi import APIRouter

from app.core.config import settings
from app.models.model_loader import GOLDEN_TIME_FRUITS, model_loader
from app.schemas.analysis import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Kiểm tra trạng thái sẵn sàng của dịch vụ AI và mô hình."""
    is_ready = model_loader.is_loaded
    return HealthResponse(
        status="UP" if is_ready else "DEGRADED",
        modelLoaded=is_ready,
        modelName=settings.MODEL_NAME if not model_loader.is_custom else f"{settings.MODEL_NAME}-finetuned",
        device=str(model_loader.device),
        supportedFruits=GOLDEN_TIME_FRUITS,
    )
