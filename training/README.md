# Hướng dẫn Huấn luyện & Đánh giá Mô hình AI Golden Time

Thư mục này chứa pipeline chuẩn hóa phục vụ huấn luyện (transfer learning/fine-tuning) và đánh giá độc lập mô hình thị giác máy tính nhận diện trái cây cho dự án Golden Time, hỗ trợ mở rộng linh hoạt từ 10 lên 20–100 lớp.

---

## 1. Kiến trúc Mô hình & Nguồn Danh mục Chuẩn

- **Backbone cơ sở:** `MobileNetV3-Small` (PyTorch Torchvision).
- **Lý do lựa chọn:**
  - Kích thước siêu nhẹ (~10 MB).
  - Tốc độ suy luận CPU xuất sắc (< 50ms/ảnh), phù hợp môi trường cloud container và chuyển đổi on-device (ONNX/TFLite).
  - Khả năng trích xuất đặc trưng hình ảnh mạnh mẽ từ pre-training trên ImageNet-1K.
- **Nguồn chân lý lớp (Single Source of Truth):**
  - Mọi lớp quả được định nghĩa tại [`app/core/classes.json`](../app/core/classes.json) và quản lý qua [`app/core/registry.py`](../app/core/registry.py).
  - Khi mở rộng lớp mới, chỉ cần thêm thông tin vào `classes.json` (tên hiển thị, tên thư mục, alias, profile độ chín và bảo quản). Không cần sửa đổi logic code trong `train.py` hay `evaluate.py`.

---

## 2. Tổ chức Dữ liệu Huấn luyện

Dữ liệu ảnh cần được tổ chức theo chuẩn `torchvision.datasets.ImageFolder` trong thư mục `datasets/fruits/`:

```text
datasets/fruits/
├── train/                  # Tập huấn luyện (~70%)
│   ├── bo/
│   ├── cam_sanh/
│   ├── chuoi/
│   ├── sau_rieng/
│   └── ...
├── val/                    # Tập kiểm định trong lúc train (~15%)
│   ├── bo/
│   ├── ...
└── test/                   # Tập kiểm thử độc lập cuối cùng (~15%)
    ├── bo/
    └── ...
```

### Yêu cầu bắt buộc:
1. Tên thư mục lớp trong `train/`, `val/`, `test/` phải đồng bộ với `folder_name` hoặc `aliases` trong `classes.json`.
2. Pipeline sẽ tự động kiểm tra tính nhất quán giữa các thư mục và ánh xạ chính xác index của `ImageFolder` (A-Z) sang `class_id` và `display_name` tiếng Việt, ngăn chặn hoàn toàn lỗi lệch nhãn ngầm.
3. Tuyệt đối không để ảnh bị trùng lặp (data leakage) giữa 3 tập `train`, `val`, và `test`.

---

## 3. Quy trình Huấn luyện (Fine-Tuning)

Để thực hiện huấn luyện mô hình với danh mục lớp tùy biến:

```bash
# Huấn luyện với tham số chuẩn hóa
python training/train.py \
  --data-dir ./datasets/fruits \
  --output-dir ./models \
  --epochs 20 \
  --batch-size 32 \
  --lr 0.001 \
  --patience 5 \
  --seed 42
```

### Các tham số chính:
- `--data-dir`: Đường dẫn thư mục dữ liệu chứa `train/` và `val/` (mặc định `./datasets/fruits`).
- `--output-dir`: Thư mục lưu checkpoint sau khi train (mặc định `./models`).
- `--epochs`: Số epoch tối đa (khuyến nghị 15–25 khi fine-tuning MobileNetV3).
- `--batch-size`: Kích thước mini-batch (16 hoặc 32).
- `--lr`: Learning rate ban đầu (mặc định 0.001 với optimizer AdamW và CosineAnnealingLR).
- `--patience`: Ngưỡng Early Stopping tính theo số epoch không cải thiện validation loss (mặc định 5).
- `--seed`: Giá trị Random Seed cố định để đảm bảo tính tái lập (mặc định 42).

### Cơ chế an toàn tự động:
- Tự động sao lưu checkpoint hiện có thành `fruit_model.pt.bak` trước khi ghi đè.
- Tự động tạo tệp `models/model_meta.json` ghi lại đúng thứ tự `classes`, `class_ids`, `folders`, `best_val_acc` và thời gian train.

---

## 4. Đánh giá Mô hình Độc lập (Evaluation Pipeline)

Sau khi huấn luyện, tiến hành đánh giá toàn diện mô hình trên tập `test/` độc lập:

```bash
python training/evaluate.py \
  --test-dir ./datasets/fruits/test \
  --model-path ./models/fruit_model.pt \
  --output-json ./reports/test_eval_report.json
```

### Các chỉ số được tính toán tự động:
1. **Top-1 Accuracy:** Tỷ lệ phân loại chính xác tổng thể.
2. **Macro-Average F1-Score:** Trung bình cộng F1 của tất cả các lớp, phản ánh công bằng hiệu năng trên cả các lớp ít mẫu.
3. **Weighted-Average F1-Score:** Trung bình F1 có trọng số theo số lượng mẫu thực tế.
4. **Per-Class Metrics:** Bảng chi tiết Precision, Recall, F1-Score và Support cho từng lớp quả.
5. **Confusion Matrix & Top Confused Pairs:** Thống kê ma trận nhầm lẫn và chỉ ra các cặp quả dễ bị nhận diện sai nhất (ví dụ Cam sành vs Bưởi da xanh) để định hướng thu thập bổ sung dữ liệu.
6. **Báo cáo JSON (`--output-json`):** Xuất toàn bộ kết quả ra tệp JSON phục vụ CI/CD gating và theo dõi chất lượng.

---

## 5. Tiêu chuẩn Phát hành Mô hình (Model Gating Criteria)

Một phiên bản mô hình chỉ đủ điều kiện đưa vào thư mục `models/releases/<version>/` khi vượt qua các ngưỡng chất lượng sau:
- `Top-1 Accuracy (test)` $\ge 0.88$
- `Macro-F1 (test)` $\ge 0.85$
- Không có bất kỳ lớp nào có `Recall` $< 0.70$
- Đã được kiểm tra an toàn bằng test hồi quy `tests/test_regression.py`.
