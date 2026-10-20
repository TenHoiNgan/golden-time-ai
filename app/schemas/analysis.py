from typing import List, Optional
from pydantic import BaseModel, Field


class AnalysisResponse(BaseModel):
    """
    Hợp đồng dữ liệu phản hồi phân tích ảnh trái cây.
    Đồng bộ đầy đủ với FruitAnalysis.java ở backend và ScanResult ở frontend.
    """

    fruitName: str = Field(
        ...,
        description="Tên loại trái cây được nhận diện (tiếng Việt)",
        examples=["Táo", "Cam sành", "Chuối", "Xoài cát Hòa Lộc"],
    )
    qualityScore: int = Field(
        ...,
        ge=0,
        le=100,
        description="Điểm chất lượng tổng thể từ 0 đến 100",
        examples=[85],
    )
    freshnessScore: int = Field(
        ...,
        ge=0,
        le=100,
        description="Điểm độ tươi ước tính từ 0 đến 100",
        examples=[88],
    )
    ripenessLabel: str = Field(
        ...,
        description="Nhãn mức độ chín thực tế",
        examples=["Chín tới", "Vừa chín", "Chưa chín", "Quá chín"],
    )
    sweetnessLabel: Optional[str] = Field(
        None,
        description="Nhãn độ ngọt ước tính dựa trên loại quả và độ chín",
        examples=["Ngọt đậm", "Chua ngọt", "Ngọt thanh"],
    )
    useWithin: Optional[str] = Field(
        None,
        description="Thời gian khuyến nghị nên sử dụng",
        examples=["3–5 ngày", "1–2 ngày", "Nên dùng ngay"],
    )
    qualityWarning: Optional[str] = Field(
        None,
        description="Cảnh báo chất lượng hoặc khuyết tật nhận biết được",
        examples=["Vỏ có vết thâm nhẹ, nên dùng sớm"],
    )
    suggestion: Optional[str] = Field(
        None,
        description="Gợi ý bảo quản hoặc chế biến phù hợp",
        examples=["Bảo quản ngăn mát tủ lạnh 4–8°C để giữ độ giòn tươi ngon"],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Độ tin cậy của mô hình từ 0.0 đến 1.0",
        examples=[0.92],
    )


class HealthResponse(BaseModel):
    """Trạng thái sức khỏe của AI service."""

    status: str = Field(..., examples=["UP", "DEGRADED"])
    modelLoaded: bool = Field(..., examples=[True])
    modelName: str = Field(..., examples=["mobilenet_v3_small"])
    device: str = Field(..., examples=["cpu", "cuda"])
    supportedFruits: List[str] = Field(default_factory=list)
