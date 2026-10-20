import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F

from app.core.config import settings
from app.models.model_loader import (
    GOLDEN_TIME_FRUITS,
    IMAGENET_FRUIT_MAP,
    KEYWORD_TO_FRUIT,
    model_loader,
)


class FruitClassifier:
    """Thực hiện suy luận nhận diện loại trái cây từ tensor ảnh."""

    def __init__(self):
        self.loader = model_loader

    def predict(self, tensor: torch.Tensor, visual_features: dict) -> Tuple[str, float]:
        """
        Dự đoán loại trái cây và độ tin cậy.
        Hỗ trợ cả mô hình fine-tuned chuyên biệt và mô hình nền tảng MobileNetV3 ImageNet.
        """
        if not self.loader.is_loaded:
            raise RuntimeError("Mô hình AI chưa được tải vào bộ nhớ.")

        with torch.no_grad():
            outputs = self.loader.model(tensor)
            probabilities = F.softmax(outputs, dim=1)[0]

        # Trường hợp 1: Mô hình đã được fine-tune trực tiếp trên danh mục Golden Time
        if self.loader.is_custom:
            top_prob, top_idx = torch.topk(probabilities, 1)
            confidence = float(top_prob[0].item())
            idx = int(top_idx[0].item())
            if 0 <= idx < len(GOLDEN_TIME_FRUITS) and confidence >= settings.CONFIDENCE_THRESHOLD:
                return GOLDEN_TIME_FRUITS[idx], round(confidence, 4)
            return "Chưa nhận diện được", round(confidence, 4)

        # Trường hợp 2: Mô hình pretrained MobileNetV3 (ImageNet-1K)
        # Quét top-15 dự đoán tìm kiếm các lớp trái cây
        topk_probs, topk_indices = torch.topk(probabilities, 15)

        matched_fruit = None
        max_fruit_conf = 0.0

        for prob, idx in zip(topk_probs, topk_indices):
            class_id = int(idx.item())
            conf = float(prob.item())

            # 1. Khớp theo ImageNet index ánh xạ trực tiếp
            if class_id in IMAGENET_FRUIT_MAP:
                fruit_name = IMAGENET_FRUIT_MAP[class_id]
                if conf > max_fruit_conf:
                    max_fruit_conf = conf
                    matched_fruit = fruit_name
                continue

            # 2. Khớp theo tên class trong metadata ImageNet
            if hasattr(self.loader, "_categories") and 0 <= class_id < len(self.loader._categories):
                category_name = self.loader._categories[class_id].lower()
                for keyword, target_fruit in KEYWORD_TO_FRUIT:
                    if keyword in category_name:
                        if conf > max_fruit_conf:
                            max_fruit_conf = conf
                            matched_fruit = target_fruit
                        break

        # Nếu mô hình tìm thấy loại quả và độ tin cậy đạt ngưỡng
        if matched_fruit and max_fruit_conf >= settings.CONFIDENCE_THRESHOLD:
            # Chuẩn hóa confidence cho phù hợp với nhận diện trái cây đơn lẻ
            adjusted_conf = min(1.0, max_fruit_conf * 1.5 + 0.1)
            return matched_fruit, round(adjusted_conf, 4)

        # Dự phòng bằng đặc trưng thị giác màu sắc khi mô hình ImageNet bị phân tán bởi góc chụp
        color_detected = self._fallback_by_color(visual_features)
        if color_detected:
            fruit_name, color_conf = color_detected
            return fruit_name, round(color_conf, 4)

        # Không nhận diện được hoặc ảnh không phải trái cây
        highest_conf = float(topk_probs[0].item())
        return "Chưa nhận diện được", round(min(highest_conf, 0.35), 4)

    def _fallback_by_color(self, features: dict) -> Optional[Tuple[str, float]]:
        """Phát hiện bổ trợ dựa trên đặc trưng phân bố màu sắc đặc trưng của một số loại quả."""
        red = features.get("red_ratio", 0.0)
        yellow_orange = features.get("yellow_orange_ratio", 0.0)
        green = features.get("green_ratio", 0.0)
        sat = features.get("avg_sat", 0.0)

        # Chỉ áp dụng khi ảnh có độ bão hòa màu cao (trái cây chụp rõ)
        if sat < 70:
            return None

        if red > 0.45:
            return "Táo", 0.65
        if yellow_orange > 0.50:
            return "Chuối", 0.60
        if green > 0.45:
            return "Cam sành", 0.58
        return None


class RipenessEstimator:
    """Ước tính mức độ chín của quả dựa trên đặc trưng hình ảnh và chủng loại."""

    def estimate(self, fruit_name: str, features: dict) -> str:
        if fruit_name == "Chưa nhận diện được":
            return "Không xác định"

        green = features.get("green_ratio", 0.0)
        yellow_orange = features.get("yellow_orange_ratio", 0.0)
        red = features.get("red_ratio", 0.0)
        dark_spots = features.get("dark_spot_ratio", 0.0)

        # Chuối: phân biệt rất rõ qua màu xanh / vàng / đốm nâu
        if fruit_name == "Chuối":
            if green > 0.35:
                return "Chưa chín (vỏ xanh)"
            if dark_spots > 0.20:
                return "Quá chín (có đốm nâu)"
            if yellow_orange > 0.45:
                return "Chín tới (vàng ươm)"
            return "Vừa chín tới"

        # Táo: vỏ đỏ / xanh
        if fruit_name == "Táo":
            if dark_spots > 0.25:
                return "Quá chín"
            if red > 0.40:
                return "Chín đỏ đều"
            if green > 0.35:
                return "Chín xanh (giòn)"
            return "Vừa chín tới"

        # Cam: vỏ cam sành Việt Nam có thể chín khi vỏ còn ánh xanh
        if "Cam" in fruit_name:
            if yellow_orange > 0.35:
                return "Chín mọng nước"
            if green > 0.50:
                return "Chín xanh (đặc trưng cam sành)"
            return "Vừa chín tới"

        # Xoài
        if "Xoài" in fruit_name:
            if yellow_orange > 0.40:
                return "Chín vàng thơm"
            if green > 0.40:
                return "Xanh giòn"
            return "Vừa chín tới"

        # Dâu tây
        if "Dâu" in fruit_name:
            if red > 0.50:
                return "Chín đỏ mọng"
            if green > 0.20:
                return "Còn ương (chưa đỏ hẳn)"
            return "Vừa chín tới"

        # Mặc định theo mức độ điểm thâm và màu sắc chung
        if dark_spots > 0.25:
            return "Quá chín"
        if green > 0.40:
            return "Chưa chín hẳn"
        return "Vừa chín tới"


class FreshnessQualityEvaluator:
    """Đánh giá độ tươi, điểm chất lượng tổng thể, thời hạn khuyên dùng và gợi ý."""

    def evaluate(
        self,
        fruit_name: str,
        confidence: float,
        ripeness: str,
        features: dict,
    ) -> Tuple[int, int, str, Optional[str], Optional[str]]:
        if fruit_name == "Chưa nhận diện được":
            return (
                0,
                0,
                "Không xác định",
                "Ảnh mờ hoặc không nhận diện được loại trái cây rõ ràng.",
                "Vui lòng chụp lại ảnh rõ nét, đủ ánh sáng và cận cảnh loại trái cây.",
            )

        dark_spots = features.get("dark_spot_ratio", 0.0)
        sat = features.get("avg_sat", 100.0)
        val = features.get("avg_val", 120.0)

        # 1. Tính điểm độ tươi (Freshness Score: 0-100)
        # Độ tươi tỷ lệ nghịch với vết thâm/vùng tối và tỷ lệ thuận với độ rực rỡ màu sắc
        freshness_penalty = min(60, int(dark_spots * 180))
        vibrancy_bonus = min(20, int((sat / 255.0) * 15 + (val / 255.0) * 10))

        base_freshness = 85
        if "Quá chín" in ripeness:
            base_freshness -= 25
        elif "Chưa chín" in ripeness:
            base_freshness += 5

        freshness_score = max(20, min(100, base_freshness - freshness_penalty + vibrancy_bonus))

        # 2. Tính điểm chất lượng tổng thể (Quality Score: 0-100)
        # Kết hợp độ tươi, độ tin cậy nhận diện và mức độ chín
        ripeness_multiplier = 0.7 if "Quá chín" in ripeness else 1.0
        confidence_factor = 0.5 + 0.5 * min(1.0, confidence)
        raw_quality = freshness_score * ripeness_multiplier * confidence_factor
        quality_score = max(15, min(100, int(round(raw_quality))))

        # 3. Thời gian nên dùng (useWithin)
        if "Quá chín" in ripeness or freshness_score < 50:
            use_within = "Nên dùng ngay trong ngày"
        elif "Chín" in ripeness and "đỏ" in ripeness or "mọng" in ripeness:
            use_within = "1–2 ngày"
        elif "Chưa chín" in ripeness or "Xanh" in ripeness:
            use_within = "5–7 ngày"
        else:
            use_within = "3–5 ngày"

        # 4. Cảnh báo chất lượng (qualityWarning)
        quality_warning = None
        if dark_spots > 0.20:
            quality_warning = "Phát hiện vết thâm đốm trên bề mặt vỏ; nên kiểm tra kỹ trước khi ăn."
        elif "Quá chín" in ripeness:
            quality_warning = "Quả đã chín mềm, dễ bị nhũn hoặc suy giảm hương vị nếu để lâu."
        elif freshness_score >= 85 and quality_score >= 80:
            quality_warning = "Trái cây tươi ngon, bề mặt vỏ sáng đều, đạt chuẩn chất lượng."

        # 5. Gợi ý bảo quản và sử dụng (suggestion)
        suggestion = self._generate_suggestion(fruit_name, ripeness, freshness_score)

        return quality_score, freshness_score, use_within, quality_warning, suggestion

    def _generate_suggestion(self, fruit: str, ripeness: str, freshness: int) -> str:
        if "Chuối" in fruit:
            if "Chưa chín" in ripeness:
                return "Để chuối ở nhiệt độ phòng nơi thoáng mát, không cho vào tủ lạnh khi vỏ còn xanh."
            if "Quá chín" in ripeness:
                return "Có thể bóc vỏ làm sinh tố hoặc làm bánh chuối nướng để tận dụng vị ngọt tự nhiên."
            return "Treo nải chuối ở nơi thoáng khí để tránh bị thâm các mặt tiếp xúc."

        if "Táo" in fruit:
            return "Bảo quản trong ngăn mát tủ lạnh 4–8°C để giữ độ giòn và hạn chế mất nước."

        if "Cam" in fruit:
            return "Bảo quản nơi thoáng mát hoặc ngăn mát tủ lạnh; vắt lấy nước dùng tươi để giữ trọn vitamin C."

        if "Xoài" in fruit:
            if "Xanh" in ripeness:
                return "Ủ trong túi giấy ở nhiệt độ phòng 1–2 ngày nếu muốn xoài chín mềm ngọt."
            return "Gọt ăn trực tiếp hoặc bảo quản hộp kín trong tủ lạnh dùng trong 1–2 ngày."

        if "Dâu" in fruit:
            return "Không rửa trước khi bảo quản; lót khăn giấy vào hộp và giữ trong ngăn mát tủ lạnh."

        return "Bảo quản ở nhiệt độ mát mẻ, tránh ánh nắng trực tiếp để hoa quả giữ được độ tươi lâu nhất."


fruit_classifier = FruitClassifier()
ripeness_estimator = RipenessEstimator()
freshness_evaluator = FreshnessQualityEvaluator()


class SweetnessEstimator:
    """Ước tính nhãn độ ngọt dựa trên chủng loại trái cây và mức độ chín."""

    def estimate(self, fruit_name: str, ripeness: str) -> Optional[str]:
        if fruit_name == "Chưa nhận diện được":
            return "Chưa có dữ liệu"

        if "Chuối" in fruit_name:
            if "Quá chín" in ripeness:
                return "Ngọt đậm"
            if "Chưa chín" in ripeness:
                return "Chát nhẹ / ít ngọt"
            return "Ngọt tự nhiên"

        if "Táo" in fruit_name:
            if "Chín đỏ" in ripeness:
                return "Ngọt thanh giòn"
            if "xanh" in ripeness.lower():
                return "Chua ngọt giòn"
            return "Ngọt dịu"

        if "Cam" in fruit_name:
            if "mọng nước" in ripeness:
                return "Chua ngọt mọng nước"
            if "xanh" in ripeness.lower():
                return "Chua thanh"
            return "Chua ngọt dịu"

        if "Xoài" in fruit_name:
            if "vàng" in ripeness.lower() or "chín" in ripeness.lower():
                return "Ngọt lịm thơm"
            if "xanh" in ripeness.lower():
                return "Chua giòn"
            return "Chua ngọt"

        if "Dâu" in fruit_name:
            if "đỏ mọng" in ripeness:
                return "Chua ngọt dịu"
            return "Chua thanh"

        if "Nho" in fruit_name:
            return "Ngọt đậm"

        if "Dưa hấu" in fruit_name:
            return "Ngọt mát"

        if "Thanh long" in fruit_name:
            return "Ngọt thanh mát"

        if "Dứa" in fruit_name:
            return "Chua ngọt đậm đà"

        if "Bơ" in fruit_name:
            return "Béo ngậy (không ngọt)"

        return "Ngọt tự nhiên"


sweetness_estimator = SweetnessEstimator()
