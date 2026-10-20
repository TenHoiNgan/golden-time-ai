from typing import Optional
from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile, status

from app.core.config import settings
from app.schemas.analysis import AnalysisResponse
from app.services.inference_service import inference_service

router = APIRouter(tags=["Analysis"])


def verify_api_key(authorization: Optional[str] = Header(None)) -> None:
    """Xác thực Bearer token nếu hệ thống có đặt cấu hình API_KEY."""
    if not settings.API_KEY:
        return  # Không yêu cầu xác thực nếu chưa cấu hình API key

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Thiếu Authorization header trong yêu cầu.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer" or parts[1] != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key không hợp lệ hoặc không có quyền truy cập.",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post(
    "/api/v1/analyze",
    response_model=AnalysisResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Nhận diện và phân tích chất lượng trái cây bằng AI",
    description="Nhận tệp ảnh hoa quả qua multipart/form-data field 'file' và trả về các chỉ số phân tích.",
)
@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    dependencies=[Depends(verify_api_key)],
    include_in_schema=False,  # Alias route tương thích
)
async def analyze_fruit_image(file: UploadFile = File(..., description="Tệp ảnh hoa quả (PNG, JPEG, WebP)")):
    """
    Endpoint chính tiếp nhận yêu cầu phân tích từ Backend Golden Time hoặc người dùng:
    - Nhận field 'file'.
    - Kiểm tra định dạng và kích thước.
    - Chạy suy luận qua mô hình.
    - Trả về JSON theo đúng hợp đồng (không có sweetnessLabel).
    """
    try:
        content = await file.read()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Không đọc được dữ liệu tệp tải lên.",
        )

    return inference_service.analyze(content, declared_content_type=file.content_type)
