import logging
from typing import Optional
from fastapi import HTTPException, status

from app.core.config import settings
from app.models.model_loader import model_loader
from app.models.predictors import (
    fruit_classifier,
    ripeness_estimator,
    freshness_evaluator,
    sweetness_estimator,
)
from app.schemas.analysis import AnalysisResponse
from app.services.image_service import image_service

logger = logging.getLogger(__name__)


class InferenceService:
    """Điều phối toàn bộ luồng xử lý và phân tích hình ảnh trái cây."""

    def __init__(self):
        self.image_service = image_service
        self.classifier = fruit_classifier
        self.ripeness = ripeness_estimator
        self.freshness = freshness_evaluator
        self.sweetness = sweetness_estimator
        self.loader = model_loader

    def analyze(self, image_bytes: bytes, declared_content_type: Optional[str] = None) -> AnalysisResponse:
        """
        Thực hiện toàn bộ quy trình:
        1. Kiểm tra kích thước và định dạng ảnh.
        2. Mở ảnh bằng PIL và trích xuất đặc trưng thị giác.
        3. Chuyển tensor qua mô hình AI để nhận diện loại quả và độ tin cậy.
        4. Ước tính mức độ chín và độ ngọt tương ứng.
        5. Đánh giá độ tươi, điểm chất lượng tổng thể, thời hạn dùng và gợi ý.
        6. Đóng gói kết quả trả về đúng hợp đồng tương thích với Backend Golden Time.
        """
        if not self.loader.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Mô hình AI chưa sẵn sàng phục vụ suy luận.",
            )

        # 1 & 2. Kiểm tra và mở ảnh
        image = self.image_service.validate_and_load(image_bytes, declared_content_type)
        visual_features = self.image_service.extract_visual_features(image)

        # 3. Chuyển tensor và dự đoán phân loại
        tensor = self.image_service.preprocess_for_model(image, device=str(self.loader.device))
        fruit_name, confidence = self.classifier.predict(tensor, visual_features)

        # 4. Ước tính độ chín và độ ngọt
        ripeness_label = self.ripeness.estimate(fruit_name, visual_features)
        sweetness_label = self.sweetness.estimate(fruit_name, ripeness_label)

        # 5. Đánh giá chất lượng và độ tươi
        (
            quality_score,
            freshness_score,
            use_within,
            quality_warning,
            suggestion,
        ) = self.freshness.evaluate(fruit_name, confidence, ripeness_label, visual_features)

        logger.info(
            "Phân tích thành công: quả=%s, conf=%.2f, chín=%s, ngọt=%s, tươi=%d, điểm=%d",
            fruit_name,
            confidence,
            ripeness_label,
            sweetness_label,
            freshness_score,
            quality_score,
        )

        # 6. Trả về đúng schema
        return AnalysisResponse(
            fruitName=fruit_name,
            qualityScore=quality_score,
            freshnessScore=freshness_score,
            ripenessLabel=ripeness_label,
            sweetnessLabel=sweetness_label,
            useWithin=use_within,
            qualityWarning=quality_warning,
            suggestion=suggestion,
            confidence=confidence,
        )


inference_service = InferenceService()
