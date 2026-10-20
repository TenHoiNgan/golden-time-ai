# Dịch vụ AI Nhận diện & Phân tích Trái cây — Golden Time

Dịch vụ AI độc lập (Microservice) phục vụ nhận diện chủng loại, ước tính mức độ chín, đánh giá độ tươi và tính toán điểm chất lượng của trái cây tươi từ hình ảnh cho hệ sinh thái **Golden Time**.

---

## 1. Yêu cầu Môi trường

- **Python:** 3.11+ (Khuyến nghị 3.11.x)
- **Hệ điều hành:** Windows 10/11, Ubuntu 20.04+, macOS
- **Phần cứng:**
  - Tối thiểu 2 GB RAM (chạy suy luận mượt mà trên CPU)
  - Tùy chọn GPU NVIDIA CUDA nếu muốn tăng tốc độ huấn luyện/suy luận batch lớn

---

## 2. Cài đặt Phụ thuộc

1. Tạo môi trường ảo (khuyến nghị):
```bash
python -m venv venv

# Kích hoạt trên Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Kích hoạt trên Linux/macOS:
source venv/bin/activate
```

2. Cài đặt các thư viện:
```bash
pip install -r requirements.txt
```

*(Nếu cài đặt riêng PyTorch CPU để tối ưu dung lượng tải: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`)*

---

## 3. Khởi chạy Dịch vụ

### Khởi chạy trực tiếp bằng Uvicorn:
```bash
# Chạy tại cổng 8000
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Dịch vụ sẽ sẵn sàng tại:
- **API Base:** `http://localhost:8000`
- **Tài liệu Swagger tương tác:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

### Khởi chạy bằng Docker:
```bash
docker build -t golden-time-ai .
docker run -p 8000:8000 golden-time-ai
```

---

## 4. API Endpoints & Hợp đồng Dữ liệu

### 4.1. `GET /health`
Kiểm tra trạng thái sẵn sàng của dịch vụ và mô hình:
```bash
curl -X GET http://localhost:8000/health
```

**Response mẫu:**
```json
{
  "status": "UP",
  "modelLoaded": true,
  "modelName": "mobilenet_v3_small",
  "device": "cpu",
  "supportedFruits": [
    "Táo",
    "Cam sành",
    "Chuối",
    "Xoài cát Hòa Lộc",
    "Nho",
    "Dâu tây",
    "Dưa hấu",
    "Thanh long",
    "Dứa",
    "Bơ"
  ]
}
```

### 4.2. `POST /api/v1/analyze` (hoặc alias `/analyze`)
Tiếp nhận tệp ảnh hoa quả qua `multipart/form-data` (field: `file`).

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "file=@/duong/dan/anh/qua_tao.jpg"
```

**Response mẫu:**
```json
{
  "fruitName": "Táo",
  "qualityScore": 86,
  "freshnessScore": 90,
  "ripenessLabel": "Chín đỏ đều",
  "sweetnessLabel": "Ngọt thanh giòn",
  "useWithin": "3–5 ngày",
  "qualityWarning": "Trái cây tươi ngon, bề mặt vỏ sáng đều, đạt chuẩn chất lượng.",
  "suggestion": "Bảo quản trong ngăn mát tủ lạnh 4–8°C để giữ độ giòn và hạn chế mất nước.",
  "confidence": 0.88
}
```

> [!NOTE]
> **Đồng bộ Hợp đồng Dữ liệu:**
> Toàn bộ các trường `fruitName`, `qualityScore`, `freshnessScore`, `ripenessLabel`, `sweetnessLabel`, `useWithin`, `qualityWarning`, `suggestion`, `confidence` được đồng bộ hoàn hảo với DTO `FruitAnalysis.java` của Backend và giao diện Frontend Golden Time. `sweetnessLabel` được ước tính dựa trên chủng loại trái cây và mức độ chín nhận diện được.

---

## 5. Kết nối với Backend Golden Time (`golden-time-be`)

Trong tệp `golden-time-be/src/main/resources/application.yml` hoặc thiết lập biến môi trường trước khi chạy Spring Boot:

```powershell
$env:AI_SERVICE_URL = "http://localhost:8000/api/v1/analyze"
$env:AI_SERVICE_API_KEY = "" # (Để trống nếu không đặt API_KEY trong AI service)
```

Khi đó, khi người dùng chụp ảnh tại giao diện `/scan`, backend Spring Boot sẽ tự động chuyển tiếp ảnh tới `golden-time-ai` và lưu kết quả vào cơ sở dữ liệu PostgreSQL.

---

## 6. Chạy Kiểm thử Tự động (Tests)

Bộ kiểm thử được viết bằng `pytest` bao quát các kịch bản: ảnh hợp lệ, tệp rỗng, tệp không phải ảnh, ảnh bị hỏng, kích thước vượt 10MB, tính hợp lệ của schema, và kiểm tra đầy đủ trường `sweetnessLabel`.

```bash
pytest -v
```

---

## 7. Pipeline Huấn luyện & Đánh giá (Training & Evaluation)

### Huấn luyện Fine-Tuning:
1. Chuẩn bị ảnh vào `datasets/fruits/train` và `datasets/fruits/val` (xem hướng dẫn tại [`datasets/README.md`](datasets/README.md)).
2. Chạy lệnh:
```bash
python training/train.py --epochs 15 --batch-size 32 --lr 0.001
```
Trọng số tốt nhất sẽ tự động được lưu tại `models/fruit_model.pt`.

### Đánh giá tập Test:
```bash
python training/evaluate.py --test-dir ./datasets/fruits/test --model-path ./models/fruit_model.pt
```

---

## 8. Các Lỗi Thường gặp & Khắc phục

| Lỗi | Nguyên nhân | Cách khắc phục |
| :--- | :--- | :--- |
| `HTTP 400: Định dạng tệp không phải ảnh JPEG, PNG hoặc WebP hợp lệ` | Tệp gửi lên không có magic byte của ảnh (ví dụ file txt hoặc pdf đổi đuôi). | Đảm bảo gửi ảnh đúng định dạng PNG, JPG hoặc WebP. |
| `HTTP 413: Kích thước ảnh vượt quá giới hạn cho phép` | Ảnh lớn hơn 10 MB. | Nén ảnh trước khi tải lên hoặc tăng `MAX_IMAGE_SIZE_MB` trong `.env`. |
| `HTTP 503: Mô hình AI chưa sẵn sàng phục vụ` | Lỗi khi khởi tạo weights của PyTorch. | Kiểm tra kết nối Internet lần đầu để Torchvision tải weights MobileNetV3 hoặc kiểm tra tệp `models/fruit_model.pt`. |
| `HTTP 401: Thiếu Authorization header` | Dịch vụ đã bật `AI_SERVICE_API_KEY` trong `.env` nhưng request không kèm token. | Bổ sung header `Authorization: Bearer <API_KEY>` trong request. |
