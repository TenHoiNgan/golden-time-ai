# Thư mục Lưu trữ Trọng số Mô hình AI Golden Time

Thư mục này dùng để lưu trữ các tệp checkpoint trọng số mô hình đã được huấn luyện hoặc fine-tuned phục vụ suy luận nhận diện trái cây.

---

## 1. Cơ chế Nạp Mô hình Tự động (Dynamic Model Loading)

Khi dịch vụ `golden-time-ai` khởi động:
1. Hệ thống tự động kiểm tra xem có tệp `models/fruit_model.pt` hay không.
2. **Nếu có tệp checkpoint:**
   - Hệ thống tìm tệp siêu dữ liệu `models/model_meta.json` đi kèm.
   - Đọc danh sách lớp động (`classes`, `class_ids`), xác định số lớp thực tế (`num_classes`).
   - Khởi tạo kiến trúc `FruitVisionModel(num_classes=num_classes)` và nạp trọng số `state_dict`.
   - **Kiểm định an toàn (Shape Validation):** Xác thực số chiều output của tensor classifier `out_features == len(classes)`. Nếu không khớp, hệ thống từ chối nạp để ngăn lỗi lệch index và chuyển sang chế độ dự phòng an toàn.
   - Đặt cờ trạng thái `is_loaded=True, is_custom=True`.
3. **Nếu chưa có checkpoint tùy biến:**
   - Hệ thống tự động tải trọng số nền tảng **MobileNetV3-Small** (pretrained ImageNet-1K từ Torchvision).
   - Sử dụng cơ chế mapping thông minh (15 lớp trái cây tương ứng) kết hợp phân tích đặc trưng màu sắc HSV.

---

## 2. Chuẩn Lưu trữ và Quản lý Phiên bản (Release Artifact Specification)

Để đảm bảo tính tái lập (reproducibility), khả năng kiểm toán và quy trình rollback tin cậy khi mở rộng từ 10 lên 20–100 lớp, các phiên bản mô hình phát hành phải tuân thủ cấu trúc thư mục sau:

```text
models/
├── releases/
│   ├── v1.0.0/
│   │   ├── fruit_model.pt          # Trọng số mô hình (PyTorch state_dict)
│   │   ├── model_meta.json         # Danh sách classes, class_ids, val_acc, hyperparams
│   │   └── checksum.sha256         # Bảng mã băm SHA-256 xác thực tính toàn vẹn
│   └── v1.1.0/
│       ├── fruit_model.pt
│       ├── model_meta.json
│       └── checksum.sha256
├── fruit_model.pt                  # Checkpoint tích cực đang được nạp phục vụ API
├── fruit_model.pt.bak              # Bản lưu dự phòng tự động của lần train gần nhất
└── model_meta.json                 # Metadata của mô hình tích cực
```

### Cấu trúc tệp `model_meta.json` chuẩn:
```json
{
  "architecture": "mobilenet_v3_small",
  "num_classes": 20,
  "class_ids": ["bo", "buoi", "cam_sanh", "chom_chom", "chuoi", "..."],
  "classes": ["Bơ", "Bưởi", "Cam sành", "Chôm chôm", "Chuối", "..."],
  "folders": ["bo", "buoi", "cam_sanh", "chom_chom", "chuoi", "..."],
  "best_val_acc": 0.9425,
  "trained_epochs": 18,
  "trained_at": "2026-10-21T00:00:00Z"
}
```

---

## 3. Quy trình Triển khai và Rollback (Deployment & Rollback Procedure)

### Kích hoạt phiên bản mới:
1. Sao chép bộ artifact từ thư mục phát hành:
   ```bash
   cp models/releases/v1.1.0/fruit_model.pt models/fruit_model.pt
   cp models/releases/v1.1.0/model_meta.json models/model_meta.json
   ```
2. Khởi động lại dịch vụ hoặc gọi `/health` để xác minh danh sách `supportedFruits`.

### Rollback khẩn cấp khi phát hiện suy giảm chất lượng:
1. Nhanh chóng phục hồi từ bản phát hành ổn định trước đó:
   ```bash
   cp models/releases/v1.0.0/fruit_model.pt models/fruit_model.pt
   cp models/releases/v1.0.0/model_meta.json models/model_meta.json
   ```
   Hoặc phục hồi từ checkpoint backup tự động:
   ```bash
   cp models/fruit_model.pt.bak models/fruit_model.pt
   ```
2. Khởi động lại dịch vụ. Toàn bộ API endpoint tiếp tục vận hành bình thường mà không cần sửa code.

---

## 4. Quy định Vệ sinh Git (Git Hygiene & Secrets)

- **TUYỆT ĐỐI KHÔNG** commit các tệp nhị phân trọng số lớn (`.pt`, `.pth`, `.bin`, `.onnx`, `.tflite`) vào Git repository công khai.
- Tệp `.gitignore` của dự án đã cấu hình bỏ qua `models/*.pt`, `models/*.pth` và `models/releases/`.
- Lưu trữ trọng số chính thức trên Cloud Object Storage (S3, GCS, Cloudflare R2 hoặc Hugging Face Hub Private) và tải về môi trường staging/production qua script xác thực checksum SHA-256.
