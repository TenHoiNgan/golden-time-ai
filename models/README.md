# Thư mục Lưu trữ Trọng số Mô hình AI

Thư mục này dùng để lưu trữ các tệp checkpoint trọng số mô hình đã được huấn luyện hoặc fine-tuned:

- `fruit_model.pt`: Tệp PyTorch `state_dict` của mô hình fine-tuned trên danh mục trái cây Golden Time.
- `model_meta.json`: Tệp lưu danh sách các lớp quả (classes) và chỉ số accuracy tốt nhất.

---

## 1. Cơ chế nạp mô hình tự động của Dịch vụ AI

Khi dịch vụ `golden-time-ai` khởi động:
1. Hệ thống tự động kiểm tra xem có tệp `models/fruit_model.pt` hay không.
2. **Nếu có:** Nạp mô hình tùy biến này và đặt cờ `modelLoaded=True, is_custom=True`.
3. **Nếu chưa có:** Hệ thống tự động sử dụng mô hình nền tảng **MobileNetV3-Small** (pretrained trên ImageNet-1K từ Torchvision) và sử dụng cơ chế ánh xạ thông minh danh mục quả.

---

## 2. Lưu ý về Git

Các tệp `.pt`, `.pth`, `.bin`, `.onnx` có kích thước lớn không nên commit lên Git repo. Hãy sử dụng Git LFS hoặc tải từ cloud storage (S3/GCS/Hugging Face Hub) khi triển khai.
