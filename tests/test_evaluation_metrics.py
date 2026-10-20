import pytest
from training.evaluate import calculate_metrics


def test_calculate_metrics_perfect_prediction():
    """Kiểm tra khi tất cả dự đoán đều đúng 100%."""
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 1, 2, 2]
    class_names = ["Táo", "Cam", "Chuối"]

    metrics = calculate_metrics(y_true, y_pred, num_classes=3, class_names=class_names)
    assert metrics["total_samples"] == 6
    assert metrics["correct_predictions"] == 6
    assert metrics["overall_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["weighted_f1"] == 1.0
    assert len(metrics["top_confused_pairs"]) == 0

    assert metrics["per_class"]["Táo"]["precision"] == 1.0
    assert metrics["per_class"]["Táo"]["recall"] == 1.0


def test_calculate_metrics_with_known_errors():
    """
    Kiểm tra tính toán với ma trận nhầm lẫn xác định trước:
    Lớp 0 (Táo): 2 mẫu đúng, 1 mẫu đoán nhầm thành Cam (Lớp 1).
    Lớp 1 (Cam): 2 mẫu đúng, 0 mẫu sai.
    Ground truth: [0, 0, 0, 1, 1]
    Prediction:   [0, 0, 1, 1, 1]
    """
    y_true = [0, 0, 0, 1, 1]
    y_pred = [0, 0, 1, 1, 1]
    class_names = ["Táo", "Cam"]

    metrics = calculate_metrics(y_true, y_pred, num_classes=2, class_names=class_names)

    assert metrics["total_samples"] == 5
    assert metrics["correct_predictions"] == 4
    assert metrics["overall_accuracy"] == 0.8  # 4/5

    # Lớp 0: TP=2, FP=0, FN=1 -> Precision = 2/(2+0) = 1.0; Recall = 2/(2+1) = 0.6667
    p0 = metrics["per_class"]["Táo"]
    assert p0["tp"] == 2
    assert p0["fp"] == 0
    assert p0["fn"] == 1
    assert p0["precision"] == 1.0
    assert pytest.approx(p0["recall"], 0.01) == 0.6667

    # Lớp 1: TP=2, FP=1 (1 táo bị đoán nhầm thành cam), FN=0 -> Precision = 2/(2+1) = 0.6667, Recall = 2/(2+0) = 1.0
    p1 = metrics["per_class"]["Cam"]
    assert p1["tp"] == 2
    assert p1["fp"] == 1
    assert p1["fn"] == 0
    assert pytest.approx(p1["precision"], 0.01) == 0.6667
    assert p1["recall"] == 1.0

    # Kiểm tra cặp nhầm lẫn
    assert len(metrics["top_confused_pairs"]) == 1
    assert metrics["top_confused_pairs"][0]["true_class"] == "Táo"
    assert metrics["top_confused_pairs"][0]["pred_class"] == "Cam"
    assert metrics["top_confused_pairs"][0]["count"] == 1


def test_calculate_metrics_empty():
    """Kiểm tra xử lý danh sách rỗng không gây lỗi chia cho 0."""
    metrics = calculate_metrics([], [], num_classes=2, class_names=["A", "B"])
    assert metrics["total_samples"] == 0
    assert metrics["overall_accuracy"] == 0.0
    assert metrics["macro_f1"] == 0.0
