# AI Model Card — Golden Time Fruit Intelligence

Tài liệu thẻ mô hình (Model Card) mô tả chi tiết kiến trúc, nguồn gốc, phạm vi hỗ trợ, cơ sở khoa học và các giới hạn kỹ thuật của mô hình AI nhận diện và phân tích trái cây Golden Time.

---

## 1. Thông tin Tổng quan

- **Tên mô hình:** `GoldenTime-FruitVision-MobileNetV3`
- **Phiên bản:** `1.0.0`
- **Loại mô hình:** Phân loại ảnh (Image Classification) & Trích xuất đặc trưng thị giác (Visual Feature Extraction).
- **Kiến trúc mạng:** MobileNetV3-Small (Howard et al., 2019) kết hợp bộ suy luận đặc trưng màu sắc HSV (Hue, Saturation, Value) và phân tích vùng khuyết tật bề mặt (Surface Defect Estimation).
- **Khung phần mềm (Framework):** PyTorch 2.x, Torchvision 0.17+, Pillow, FastAPI.
- **Giấy phép (License):**
  - MobileNetV3 Backbone (Torchvision weights): Apache 2.0 (Được phép sử dụng thương mại).
  - FastAPI / Pillow: MIT / Historical Permission Notice and Disclaimer (HPND).

---

## 2. Phạm vi Trái cây Hỗ trợ (Supported Scope)

Mô hình được tối ưu cho các loại trái cây phổ biến tại thị trường Việt Nam theo danh mục Golden Time:

| Tên tiếng Việt | Tên khoa học / Tiếng Anh | Khả năng nhận diện |
| :--- | :--- | :--- |
| **Táo** | Apple (*Malus domestica*) | Rất cao |
| **Cam sành** | Orange / Green Citrus (*Citrus sinensis*) | Rất cao |
| **Chuối** | Banana (*Musa acuminata*) | Rất cao |
| **Xoài cát Hòa Lộc**| Mango (*Mangifera indica*) | Cao |
| **Nho** | Grape (*Vitis vinifera*) | Cao |
| **Dâu tây** | Strawberry (*Fragaria × ananassa*) | Rất cao |
| **Dưa hấu** | Watermelon (*Citrullus lanatus*) | Cao |
| **Thanh long** | Dragon Fruit / Pitaya (*Selenicereus undatus*) | Cao |
| **Dứa / Thơm** | Pineapple (*Ananas comosus*) | Rất cao |
| **Bơ** | Avocado (*Persea americana*) | Khá |

---

## 3. Các Tác vụ & Cơ sở Khoa học

### 3.1. Nhận diện Chủng loại (Fruit Classification)
- Sử dụng mạng nơ-ron tích chập sâu (Deep CNN) để trích xuất đặc trưng hình thái học (hình dạng, cấu trúc cuống, họa tiết bề mặt).
- Ánh xạ xác suất Softmax qua độ tin cậy `confidence` $\in [0.0, 1.0]$.
- Khi `confidence` $< 0.25$ hoặc ảnh không chứa trái cây, mô hình trả về nhãn *"Chưa nhận diện được"*.

### 3.2. Ước tính Mức độ Chín (Ripeness Estimation)
- **Cơ sở thị giác:** Dựa trên sự chuyển dịch sắc thái màu sắc đặc trưng của từng loại quả trong quá trình sinh tổng hợp chlorophyll và carotenoid/anthocyanin:
  - *Chuối:* Chuyển từ màu xanh lục (vỏ dày, nhiều tinh bột) sang màu vàng sáng (vừa chín tới) và xuất hiện đốm nâu đường/melanin (quá chín).
  - *Cam sành Việt Nam:* Nhận diện sắc thái xanh đậm sang xanh vàng/cam mọng nước.
  - *Táo & Dâu tây:* Đánh giá độ đồng đều của sắc tố đỏ.
- **Nhãn đầu ra:** *"Chưa chín"*, *"Vừa chín tới"*, *"Chín đỏ đều / Chín mọng"*, *"Quá chín"*, *"Không xác định"*.

### 3.3. Đánh giá Độ tươi & Điểm chất lượng (Freshness & Quality Score)
- **Độ tươi (`freshnessScore` 0–100):** Đánh giá dựa trên độ tươi sáng bề mặt (Saturation & Value) và tỷ lệ diện tích các đốm sẫm màu/vết bầm dập bất thường (`dark_spot_ratio`).
- **Điểm chất lượng (`qualityScore` 0–100):** Chỉ số tổng hợp giữa độ tươi, độ tin cậy của thuật toán phân loại và hệ số phạt nếu quả đã ở trạng thái quá chín hoặc nẫu hỏng.
- **Thời hạn khuyên dùng (`useWithin`):** Dự đoán số ngày tối ưu nên tiêu thụ dựa trên tốc độ thoái hóa sinh học tương ứng với mức độ chín hiện thời.

### 3.4. Ước tính Độ ngọt (`sweetnessLabel`)
- **Nguyên lý ước tính:** Độ ngọt được ước tính thông qua luật heuristic kết hợp giữa chủng loại trái cây được nhận diện và giai đoạn chín của quả (ví dụ: chuối chín đốm nâu có hàm lượng đường cao -> "Ngọt đậm", xoài xanh -> "Chua giòn", cam sành mọng nước -> "Chua ngọt mọng nước").
- **Lưu ý khoa học:** Đây là chỉ số tham khảo ước tính dựa trên mức độ chín bên ngoài, không thay thế cho phép đo trực tiếp độ Brix bằng thiết bị quang phổ khúc xạ kế.

---

## 4. Dữ liệu Huấn luyện & Đánh giá

- **Pre-training:** ImageNet-1K (1.28 triệu ảnh, 1000 lớp đối tượng tự nhiên).
- **Fine-tuning Pipeline:** Hỗ trợ nạp bộ dữ liệu tùy biến từ `datasets/fruits/` thông qua script `training/train.py`.
- **Tốc độ suy luận (Inference Latency):**
  - CPU Intel Core i5 / AMD Ryzen 5: ~35–55 ms / ảnh.
  - Bộ nhớ RAM tiêu thụ: ~180 MB.

---

## 5. Giới hạn Kỹ thuật & Khuyến cáo Sử dụng

1. **Điều kiện ánh sáng:** Ảnh chụp quá tối (thiếu sáng), ngược sáng mạnh, hoặc bị chói lóa ánh đèn flash sẽ làm sai lệch phân tích màu sắc HSV, dẫn đến giảm độ chính xác của ước tính độ tươi và độ chín.
2. **Quả được bọc trong bao bì nilon/xốp:** Lớp bao bọc hoặc tem nhãn quá lớn che khuất bề mặt quả có thể khiến mô hình nhận diện sai hoặc giảm điểm tươi.
3. **Món ăn đã qua chế biến:** Mô hình chỉ phân tích quả tươi nguyên vẹn; không hỗ trợ quả đã gọt vỏ, thái lát, nước ép hoặc hoa quả sấy khô.
4. **Trái cây nằm ngoài danh mục:** Khi đưa vào các vật thể không phải trái cây, mô hình sẽ trả về *"Chưa nhận diện được"* với điểm chất lượng 0.
