# Hướng dẫn Chuẩn bị Dataset Trái cây

Thư mục này dùng để lưu trữ dữ liệu ảnh phục vụ huấn luyện và kiểm thử mô hình AI nhận diện trái cây cho Golden Time.

---

## 1. Cấu trúc thư mục khuyến nghị

```text
datasets/fruits/
├── train/          # Tập huấn luyện (~70% số lượng ảnh)
│   ├── Tao/
│   ├── Cam_sanh/
│   ├── Chuoi/
│   ├── Xoai_cat_Hoa_Loc/
│   ├── Nho/
│   ├── Dau_tay/
│   ├── Dua_hau/
│   ├── Thanh_long/
│   ├── Dua/
│   └── Bo/
├── val/            # Tập kiểm định trong lúc train (~15% số lượng ảnh)
│   ├── Tao/
│   └── ...
└── test/           # Tập kiểm thử độc lập cuối cùng (~15% số lượng ảnh)
    ├── Tao/
    └── ...
```

---

## 2. Tiêu chuẩn ảnh đầu vào

- **Định dạng:** JPEG, PNG hoặc WebP.
- **Kích thước khuyến nghị:** Tối thiểu 300x300 pixels (tối ưu: 500x500 pixels đến 1200x1200 pixels).
- **Góc chụp:**
  - Chụp quả đơn lẻ trên nền phẳng, bàn, hoặc khay đựng hoa quả.
  - Đa dạng góc nhìn (nhìn thẳng, nhìn nghiêng, nhìn từ cuống quả).
  - Đa dạng điều kiện ánh sáng (ánh sáng ban ngày tự nhiên, ánh sáng đèn trong nhà, siêu thị).
- **Số lượng mẫu tối thiểu:** Khuyến nghị từ 150–300 ảnh/lớp quả để fine-tuning đạt độ chính xác > 90%.

---

## 3. Nguồn dữ liệu mở tham khảo

Bạn có thể tải các dataset trái cây mã nguồn mở phổ biến:
1. **Fruit-360 Dataset:** Bộ dữ liệu gồm hơn 90.000 ảnh hoa quả trên nền trắng (rất tốt làm pre-training).
2. **Vietnamese Fruits Dataset:** Bộ ảnh hoa quả nhiệt đới Việt Nam trên Kaggle.
3. **Roboflow Fruit Quality Dataset:** Cung cấp thêm nhãn phân loại quả tươi / quả hỏng (Fresh vs Rotten).

> [!NOTE]
> Không commit các tệp ảnh lớn vào Git repository. Chỉ commit cấu trúc thư mục hoặc script tải dữ liệu.
