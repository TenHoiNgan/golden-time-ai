import pytest
from pydantic import ValidationError

from app.schemas.analysis import AnalysisResponse, HealthResponse


def test_analysis_response_schema_valid():
    """Kiểm tra tạo AnalysisResponse hợp lệ."""
    data = {
        "fruitName": "Cam sành",
        "qualityScore": 88,
        "freshnessScore": 90,
        "ripenessLabel": "Vừa chín tới",
        "sweetnessLabel": "Chua ngọt mọng nước",
        "useWithin": "3–5 ngày",
        "qualityWarning": "Vỏ xanh bóng đẹp",
        "suggestion": "Bảo quản nơi thoáng mát",
        "confidence": 0.85,
    }
    model = AnalysisResponse(**data)
    dumped = model.model_dump()
    assert dumped["fruitName"] == "Cam sành"
    assert dumped["sweetnessLabel"] == "Chua ngọt mọng nước"


def test_analysis_response_schema_rejects_out_of_bounds_scores():
    """Kiểm tra validation chặn các điểm số ngoài khoảng 0-100."""
    with pytest.raises(ValidationError):
        AnalysisResponse(
            fruitName="Táo",
            qualityScore=150,  # Không hợp lệ > 100
            freshnessScore=80,
            ripenessLabel="Chín",
            confidence=0.9,
        )

    with pytest.raises(ValidationError):
        AnalysisResponse(
            fruitName="Táo",
            qualityScore=80,
            freshnessScore=-10,  # Không hợp lệ < 0
            ripenessLabel="Chín",
            confidence=0.9,
        )

    with pytest.raises(ValidationError):
        AnalysisResponse(
            fruitName="Táo",
            qualityScore=80,
            freshnessScore=80,
            ripenessLabel="Chín",
            confidence=1.5,  # Không hợp lệ > 1.0
        )
