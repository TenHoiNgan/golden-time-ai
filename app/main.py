import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.analyze import router as analyze_router
from app.api.health import router as health_router
from app.core.config import settings
from app.models.model_loader import model_loader

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("golden-time-ai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý vòng đời ứng dụng: khởi động mô hình AI khi bật server."""
    logger.info("Đang khởi động dịch vụ AI Golden Time...")
    # Kích hoạt nạp mô hình vào bộ nhớ
    _ = model_loader.model
    if model_loader.is_loaded:
        logger.info("Mô hình AI đã sẵn sàng phục vụ trên thiết bị: %s", model_loader.device)
    else:
        logger.warning("CẢNH BÁO: Mô hình AI chưa được tải thành công!")
    yield
    logger.info("Đang tắt dịch vụ AI Golden Time...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Dịch vụ AI độc lập nhận diện và phân tích chất lượng trái cây cho dự án Golden Time.",
    lifespan=lifespan,
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký các router
app.include_router(health_router)
app.include_router(analyze_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
