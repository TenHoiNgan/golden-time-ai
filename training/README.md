# Hướng dẫn Huấn luyện & Đánh giá Mô hình AI Golden Time

Thư mục này chứa pipeline huấn luyện transfer learning và fine-tuning mô hình thị giác máy tính nhận diện trái cây cho dự án Golden Time.

---

## 1. Kiến trúc Mô hình

- **Backbone cơ sở:** `MobileNetV3-Small` (PyTorch Torchvision).
- **Lý do lựa chọn:**
  - Kích thước siêu nhẹ (~10 MB).
  - Tốc độ suy luận cao trên CPU thông thường (< 50ms/ảnh).
  - Khả năng trích xuất đặc trưng hình ảnh xuất sắc được pretrained trên ImageNet-1K.
- **Lớp đầu ra:** Lớp Linear tùy biến tương ứng với số loại quả cần nhận diện trong danh mục Golden Time.

---

## 2. Chuẩn bị Dữ liệu

Dữ liệu ảnh cần được tổ chức theo cấu trúc chuẩn `torchvision.datasets.ImageFolder` trong thư mục `datasets/fruits/`:

```text
datasets/fruits/
├── train/
│   ├── Tao/
│   ├── Cam_sanh/
│   ├── Chuoi/
│   ├── Xoai_cat_Hoa_Loc/
│   ├── Nho/
│   └── Dau_tay/
├── val/
│   ├── Tao/
│   ├── ...
└── test/
    ├── Tao/
    ├── ...
```

Xem chi tiết quy cách thu thập và gắn nhãn ảnh tại [`datasets/README.md`](../datasets/README.md).

---

## 3. Các bước Huấn luyện (Fine-Tuning)

Để thực hiện huấn luyện mô hình với bộ dữ liệu của bạn:

```bash
# Huấn luyện trên CPU (hoặc tự động phát hiện GPU CUDA nếu có)
python training/train.py --data-dir ./datasets/fruits --epochs 15 --batch-size 32 --lr 0.001
```

**Các tham số chính:**
- `--data-dir`: Đường dẫn thư mục dữ liệu (mặc định `./datasets/fruits`).
- `--output-dir`: Nơi lưu checkpoint sau khi train (mặc định `./models`).
- `--epochs`: Số lượt học (khuyến nghị 10–20 lượt khi fine-tune).
- `--batch-size`: Kích thước batch (khuyến nghị 16 hoặc 32).
- `--lr`: Learning rate (khuyến nghị 1e-3 với optimizer AdamW).

Khi kết thúc, tệp trọng số tốt nhất sẽ tự động được lưu tại `models/fruit_model.pt` và metadata tại `models/model_meta.json`. Dịch vụ AI sẽ tự động nhận diện và nạp tệp này khi khởi động lại!

---

## 4. Đánh giá Mô hình trên Tập Test

Chạy script đánh giá độc lập:

```bash
python training/evaluate.py --test-dir ./datasets/fruits/test --model-path ./models/fruit_model.pt
```

Kết quả sẽ in ra bảng độ chính xác tổng thể (Top-1 Accuracy) và độ chính xác phân loại theo từng loại quả.
