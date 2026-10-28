# BÁO CÁO KHẢO SÁT & ĐÁNH GIÁ DATASET TRÁI CÂY (DATASET SELECTION AUDIT)
## ĐỐI CHIẾU 20 LỚP TRÁI CÂY GOLDEN TIME & 6 NGUỒN DATASET
### (Bao gồm khảo sát chi tiết nguồn cục bộ: `Densu341/Fresh-rotten-fruit`)

> **Dự án:** Golden Time — Phân hệ Trí tuệ Nhân tạo (`golden-time-ai`)  
> **Giai đoạn:** Phase 3 — Bước 1: Khảo sát, Kiểm toán & Chọn lọc Dữ liệu (Chỉ đọc, không xóa, không sửa, không sao chép)  
> **Thư mục local đã kiểm toán:** `Downloads/dataset` (`Densu341/Fresh-rotten-fruit`)  
> **Thư mục dataset dự kiến:** `golden-time-ai/datasets/fruits`  
> **Ngày lập báo cáo:** 27/10/2026  

---

## 1. Executive Summary (Tóm tắt Điều hành)

1. **Mục tiêu:** Mở rộng khảo sát danh mục lên **6 nguồn dataset** (5 nguồn Hugging Face trực tuyến + 1 nguồn đã tải sẵn về máy cục bộ tại `Downloads/dataset`), đánh giá độ phủ 20 lớp quả Golden Time và phân tích nhãn chất lượng (Fresh vs Rotten, Ripeness).
2. **Kết quả Kiểm toán Nguồn Cục bộ `Densu341/Fresh-rotten-fruit` (Read-only):**
   - Đã kiểm tra 100% tệp tại `Downloads/dataset`: Tổng cộng **30,357 tệp ảnh** (khớp với công bố 30.3k trên Hugging Face), dung lượng thực tế trên đĩa là **2.88 GB** (2,951.5 MB).
   - Định dạng ảnh: **25,819 tệp `.png`** và **4,538 tệp `.jpg`**. Không có tệp hỏng (0 corrupted files).
   - **Thực hư về "22 nhãn":** Dữ liệu **KHÔNG PHẢI** gồm 22 loại trái cây. Thực chất chỉ gồm **9 mặt hàng nông sản** được chia thành từng cặp trạng thái *Tươi (Fresh)* và *Hỏng (Rotten)*.
   - Trong 9 mặt hàng trên, có **6 loại là rau củ** (*Khổ qua, Ớt chuông, Dưa chuột, Đậu bắp, Khoai tây, Cà chua* — không thuộc danh mục Golden Time). **Chỉ có 3 loại trái cây thuộc Golden Time: Táo, Chuối, Cam**.
   - **Phát hiện nghiêm trọng về Data Leakage (Rò rỉ dữ liệu):** Trong số 6,738 ảnh của tập `Test`, có tới **4,039 ảnh (chiếm 59.9%) là bản sao trùng khớp 100% mã băm SHA-256 với tập `Train`**. Nếu sử dụng nguyên split gốc của tác giả để huấn luyện, mô hình sẽ bị rò rỉ dữ liệu trầm trọng, dẫn đến độ chính xác đánh giá trên test bị ảo hoàn toàn.
   - **Lỗi chính tả và thiếu sót thư mục trong tập `Test`:** Tập `Test` thiếu hẳn 2 loại rau củ (`bittergroud`, `capsicum`), đồng thời có lỗi gõ sai tên thư mục: `freshpatato` / `rottenpatato` (sai chính tả *potato*) và `freshtamto` / `rottentamto` (sai chính tả *tomato*).
3. **Phân tích Bản quyền Giấy phép `OpenRAIL`:**
   - Trang nguồn Hugging Face gắn tag `openrail`. Giấy phép OpenRAIL cho phép sử dụng thương mại, fine-tune và phân phối, nhưng có các ràng buộc nghiêm ngặt về hành vi (Behavioral Use Restrictions - Phụ lục A) và yêu cầu đính kèm văn bản license khi phân phối sản phẩm phái sinh. 
   - Ứng dụng phân tích chất lượng nông sản của Golden Time hoàn toàn tương thích với các mục đích cho phép của OpenRAIL. Đồng thời, qua đối chiếu trực quan và cấu trúc nhãn, phần ảnh Táo, Chuối, Cam trong dataset này có sự tương đồng lớn với bộ dữ liệu Kaggle của Sriram Reddy Kalluri (CC0 Public Domain). Tuy nhiên, để bảo đảm tính nghiêm ngặt về pháp lý, dự án vẫn xem xét tuân thủ đầy đủ các điều khoản OpenRAIL đã công bố.
4. **Khuyến nghị Xử lý:**
   - **Không sao chép nguyên trạng** thư mục `Downloads/dataset` vào dự án (để tránh 6 loại rau củ không liên quan, tránh 4,039 ảnh trùng lặp train-test và tránh phình to 2.88 GB bộ nhớ).
   - **Đề xuất trích xuất chọn lọc:** Chỉ lọc lấy ảnh của 3 lớp trái cây hợp lệ (`tao`, `cam_sanh`, `chuoi`), loại bỏ toàn bộ ảnh trùng lặp qua mã băm SHA-256, gom chung lại và tự chia lại tập Train (70%) / Val (15%) / Test (15%) độc lập để triệt tiêu hoàn toàn Data Leakage.

---

## 2. Bảng So sánh 6 Dataset & Phân tích Bản quyền (License Analysis)

| STT | Dataset ID / Đường dẫn | Revision / Vị trí | Format & Cấu trúc | Số Row / Ảnh công bố | Số ảnh Kiểm đếm Thực tế (Local) | Dung lượng trên đĩa | License công bố | Đánh giá Quyền Thương mại & Rủi ro Pháp lý | Kết luận Đề xuất |
|:---:|:---|:---:|:---|:---:|:---:|:---:|:---:|:---|:---:|
| 1 | **`VinayHajare/Fruits-30`** | `main` (Online) | Parquet ($224 \times 224$) | 826 ảnh (30 classes) | Chưa tải local | ~365 MB | Chưa rõ trên card | **Chưa duyệt sử dụng**. Số mẫu quá ít (< 30 ảnh/lớp). | ⚠️ **Loại khỏi train chính** |
| 2 | **`Project-AgML/fresh_rotten_fruit_classification`** | `main` (Online / Cached) | Parquet (gốc Kaggle Sriram Reddy) | 13,599 ảnh (6 classes) | 13,599 ảnh | ~1.8 – 2.0 GB | **Apache-2.0** / CC0 | **Được phép thương mại & nghiên cứu** (yêu cầu dẫn nguồn Attribution). | ✅ **Giữ lại** (Táo, Chuối, Cam) |
| 3 | **`Project-AgML/FruitVision_quality_classification`** | `main` (Online) | Parquet ($512 \times 512$) | ~83,500 rows (augmented) | Đã loại bỏ | ~3.5 – 4.5 GB | **`CC BY-NC-ND 4.0`** | ❌ **CẤM THƯƠNG MẠI & CẤM PHÁI SINH**. Rủi ro vi phạm bản quyền quốc tế. | 🚫 **Loại bỏ hoàn toàn** |
| 4 | **`TJIET/FruitBench`** | `main` (Online) | Parquet (Image-to-Text) | ~3,200 rows (16 classes) | Đã tạm dừng | ~0.5 – 1.0 GB | **Không công bố** | **Chưa duyệt sử dụng** (Mặc định *All Rights Reserved*). | ⏳ **Tạm dừng / Chờ xác minh** |
| 5 | **`darthraider/fruit-ripeness-detection-dataset`** | `main` (Online / Cached) | Parquet ($640 \times 480$) | 4,999 ảnh (Chuối, Xoài) | 4,999 ảnh | ~391 MB | **Apache-2.0** | **Được phép thương mại & nghiên cứu** (độ chín Chuối, Xoài). | ✅ **Giữ lại** (Chuối, Xoài) |
| 6 | **`Densu341/Fresh-rotten-fruit`** | **Local:** `Downloads/dataset` | Folder `Train/` & `Test/` (.png, .jpg) | 30.3k ảnh (18 classes) | **30,357 ảnh** sạch | **2.88 GB** | **OpenRAIL** | **Cho phép thương mại có điều kiện** (tuân thủ Use Restrictions). Cần lưu ý 59.9% test leakage. | 🔍 **Khai thác chọn lọc 3 lớp quả** (sau khi deduplicate) |

---

## 3. Khảo sát Chi tiết Nguồn Cục bộ `Densu341/Fresh-rotten-fruit`

*(Dữ liệu được trích xuất hoàn toàn bằng thao tác chỉ đọc (read-only) từ script [`scripts/audit_local_dataset.py`](file:///D:/golden-time/golden-time-ai/scripts/audit_local_dataset.py) và lưu tại [`datasets/local_dataset_audit.json`](file:///D:/golden-time/golden-time-ai/datasets/local_dataset_audit.json))*

### 3.1. Thống kê Chi tiết từng Thư mục Nhãn (Local Count Breakdown)

| Nhãn Thư mục | Mặt hàng Nông sản | Phân loại | Số ảnh trong `Train` | Dung lượng Train (MB) | Số ảnh trong `Test` | Dung lượng Test (MB) | Tổng số ảnh | Tương thích Golden Time? |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `freshapples` | Táo tươi | Trái cây | 2,424 | 267.1 | 791 | 80.6 | **3,215** | ✅ Trùng khớp (`tao`) |
| `rottenapples` | Táo hỏng | Trái cây | 3,248 | 428.5 | 988 | 128.7 | **4,236** | ✅ Trùng khớp (`tao`) |
| `freshbanana` | Chuối tươi | Trái cây | 2,468 | 294.4 | 892 | 108.7 | **3,360** | ✅ Trùng khớp (`chuoi`) |
| `rottenbanana` | Chuối hỏng | Trái cây | 2,932 | 476.7 | 900 | 151.2 | **3,832** | ✅ Trùng khớp (`chuoi`) |
| `freshoranges` | Cam tươi | Trái cây | 1,466 | 178.3 | 388 | 47.4 | **1,854** | ✅ Trùng khớp (`cam_sanh`) |
| `rottenoranges` | Cam hỏng | Trái cây | 1,595 | 249.0 | 403 | 63.4 | **1,998** | ✅ Trùng khớp (`cam_sanh`) |
| `freshbittergroud` | Khổ qua tươi | Rau củ | 327 | 7.5 | 0 *(Thiếu)* | 0.0 | **327** | ❌ Không thuộc 20 loại quả |
| `rottenbittergroud` | Khổ qua hỏng | Rau củ | 357 | 8.4 | 0 *(Thiếu)* | 0.0 | **357** | ❌ Không thuộc 20 loại quả |
| `freshcapsicum` | Ớt chuông tươi | Rau củ | 990 | 22.5 | 0 *(Thiếu)* | 0.0 | **990** | ❌ Không thuộc 20 loại quả |
| `rottencapsicum` | Ớt chuông hỏng | Rau củ | 901 | 19.8 | 0 *(Thiếu)* | 0.0 | **901** | ❌ Không thuộc 20 loại quả |
| `freshcucumber` | Dưa chuột tươi | Rau củ | 496 | 28.0 | 279 | 16.3 | **775** | ❌ Không thuộc 20 loại quả |
| `rottencucumber` | Dưa chuột hỏng | Rau củ | 421 | 24.0 | 255 | 15.2 | **676** | ❌ Không thuộc 20 loại quả |
| `freshokra` | Đậu bắp tươi | Rau củ | 635 | 38.3 | 370 | 22.7 | **1,005** | ❌ Không thuộc 20 loại quả |
| `rottenokra` | Đậu bắp hỏng | Rau củ | 338 | 19.8 | 224 | 12.9 | **562** | ❌ Không thuộc 20 loại quả |
| `freshpotato` / `patato` | Khoai tây tươi | Củ | 536 | 30.2 | 270 | 15.7 | **806** | ❌ Không thuộc 20 loại quả |
| `rottenpotato` / `patato` | Khoai tây hỏng | Củ | 802 | 46.4 | 370 | 21.8 | **1,172** | ❌ Không thuộc 20 loại quả |
| `freshtomato` / `tamto` | Cà chua tươi | Rau củ | 1,858 | 77.3 | 255 | 12.5 | **2,113** | ❌ Không thuộc 20 loại quả |
| `rottentomato` / `tamto` | Cà chua hỏng | Rau củ | 1,825 | 33.1 | 353 | 5.2 | **2,178** | ❌ Không thuộc 20 loại quả |
| **TỔNG CỘNG** | **9 mặt hàng** | **18 nhóm** | **23,619** | **2,249.3 MB** | **6,738** | **702.2 MB** | **30,357 ảnh** | **3/20 loại quả Golden Time** |

### 3.2. Đánh giá Tính Toàn vẹn Kỹ thuật & Rủi ro Chất lượng Dữ liệu
1. **Kiểm tra tệp hỏng (Image Corruption Check):**
   - Đã kiểm tra mở và xác thực header/payload qua thư viện PIL trên toàn bộ 30,357 ảnh.
   - Kết quả: **0 ảnh hỏng** (100% tệp ảnh đọc tốt).
2. **Trùng lặp nội bộ (Internal Duplicates):**
   - Quét mã băm SHA-256 toàn diện: Phát hiện **3,248 ảnh trùng lặp nội bộ** (các ảnh giống hệt nhau về nội dung nhị phân nhưng nằm ở các tên file khác nhau hoặc lặp lại).
3. **Data Leakage giữa Train và Test (Rủi ro Nghiêm trọng nhất):**
   - **4,039 ảnh trong tập `Test` có mã băm SHA-256 trùng khớp hoàn toàn với ảnh trong tập `Train`**.
   - Điều này chứng minh tập `Test` dựng sẵn của tác giả dataset không phải là tập test độc lập, mà được cắt ngẫu nhiên từ tập đã nhân bản ảnh. Nếu sử dụng trực tiếp split này, chỉ số kiểm thử mô hình sẽ bị sai lệch hoàn toàn.
4. **Sai sót Cấu trúc Thư mục của Tác giả:**
   - Tập `Train` có 18 thư mục (9 cặp fresh/rotten).
   - Tập `Test` chỉ có 14 thư mục (bị thiếu mất cặp `bittergroud` và `capsicum`).
   - Tên thư mục trong `Test` bị lỗi chính tả: `freshpatato` thay vì `freshpotato`, `freshtamto` thay vì `freshtomato`.

---

## 4. Ma trận Coverage 20 Lớp Quả Đối chiếu Nguồn Cục bộ

| STT | Class ID | Tên hiển thị | 5 Nguồn Hugging Face Trước | Nguồn Cục bộ `Densu341/Fresh-rotten-fruit` | Đánh giá Đóng góp của Nguồn Cục bộ |
|:---:|:---|:---|:---:|:---:|:---|
| 1 | `tao` | Táo | Có (~220 ảnh sạch) | **Có 7,451 ảnh** (3,215 Fresh, 4,236 Rotten) | 🟢 **Bổ sung cực lớn** nhãn độ tươi & thối hỏng |
| 2 | `cam_sanh` | Cam sành | Có (~220 ảnh) | **Có 3,852 ảnh** (1,854 Fresh, 1,998 Rotten) | 🟢 **Bổ sung cực lớn** nhãn độ tươi & thối hỏng |
| 3 | `chuoi` | Chuối | Có (~300 ảnh) | **Có 7,192 ảnh** (3,360 Fresh, 3,832 Rotten) | 🟢 **Bổ sung cực lớn** nhãn độ tươi & thối hỏng |
| 4 | `xoai` | Xoài cát Hòa Lộc | Có (~300 ảnh) | **0 ảnh (Không có)** | ⚪ Không đóng góp |
| 5 | `nho` | Nho | Có (~220 ảnh) | **0 ảnh (Không có)** | ⚪ Không đóng góp |
| 6 | `dau_tay` | Dâu tây | Có (~220 ảnh) | **0 ảnh (Không có)** | ⚪ Không đóng góp |
| 7 | `dua_hau` | Dưa hấu | Có (~157 ảnh) | **0 ảnh (Không có)** | ⚪ Không đóng góp |
| 8 | `thanh_long` | Thanh long | Có (~220 ảnh) | **0 ảnh (Không có)** | ⚪ Không đóng góp |
| 9 | `dua` | Dứa | Có (~220 ảnh) | **0 ảnh (Không có)** | ⚪ Không đóng góp |
| 10 | `bo` | Bơ | Có (~220 ảnh) | **0 ảnh (Không có)** | ⚪ Không đóng góp |
| 11 | `du_du` | Đu đủ | Có (~164 ảnh) | **0 ảnh (Không có)** | ⚪ Không đóng góp |
| 12 | `oi` | Ổi | Có (~166 ảnh) | **0 ảnh (Không có)** | ⚪ Không đóng góp |
| 13 | `buoi` | Bưởi | Có (~153 ảnh) | **0 ảnh (Không có)** | ⚪ Không đóng góp |
| 14 | `vai` | Vải | Có (~166 ảnh) | **0 ảnh (Không có)** | ⚪ Không đóng góp |
| 15 | `chom_chom` | Chôm chôm | Có (~164 ảnh) | **0 ảnh (Không có)** | ⚪ Không đóng góp |
| 16 | `mang_cut` | Măng cụt | Có (~102 ảnh) | **0 ảnh (Không có)** | ⚪ Không đóng góp |
| 17 | `sau_rieng` | Sầu riêng | 0 ảnh | **0 ảnh (Không có)** | ❌ Vẫn thiếu (Chờ nguồn bổ sung) |
| 18 | `mit` | Mít | 0 ảnh | **0 ảnh (Không có)** | ❌ Vẫn thiếu (Chờ nguồn bổ sung) |
| 19 | `nhan` | Nhãn | 0 ảnh | **0 ảnh (Không có)** | ❌ Vẫn thiếu (Chờ nguồn bổ sung) |
| 20 | `mang_cau` | Mãng cầu | 0 ảnh | **0 ảnh (Không có)** | ❌ Vẫn thiếu (Chờ nguồn bổ sung) |

> [!NOTE]
> **Nhận định:** Nguồn cục bộ `Densu341/Fresh-rotten-fruit` **không giúp tăng số lượng lớp quả** (vẫn chỉ có 3 loại quả Táo, Chuối, Cam đã có từ trước, 17 loại quả còn lại hoàn toàn vắng mặt). Tuy nhiên, nguồn này cung cấp một khối lượng ảnh **Fresh vs Rotten khổng lồ (hơn 18,000 ảnh quả thật)**, giúp nâng cao vượt bậc khả năng đánh giá độ tươi và phát hiện quả thối hỏng cho 3 loại quả tiêu thụ mạnh nhất Việt Nam.

---

## 5. Thẩm định Pháp lý Giấy phép `OpenRAIL`

1. **Bản chất của OpenRAIL (Open Responsible AI License):**
   - Đây là hệ thống giấy phép mở hiện đại ra đời nhằm thúc đẩy chia sẻ tự do trong cộng đồng AI nhưng vẫn kiểm soát đạo đức sử dụng AI (tương tự như các mô hình BigCode StarCoder hay Stable Diffusion).
   - **Quyền được cấp:** Cho phép sử dụng miễn phí, sao chép, sửa đổi, fine-tune và khai thác thương mại sản phẩm.
   - **Hạn chế bắt buộc (Annex A Use Restrictions):** Nghiêm cấm dùng dữ liệu/mô hình vào các mục đích: phân biệt đối xử, quân sự/vũ khí, chẩn đoán y tế tự động không có bác sĩ, giám sát sinh trắc học diện rộng trái phép, tạo nội dung độc hại.
2. **Đánh giá mức độ tuân thủ của Golden Time:**
   - Ứng dụng nhận diện và phân tích chất lượng trái cây trong thương mại điện tử hoàn toàn **nằm ngoài danh mục cấm** của Annex A.
   - Về mặt pháp lý, Golden Time được phép sử dụng dữ liệu này để huấn luyện mô hình.
   - Tuy nhiên, khi phân phối mô hình hoặc API ra cộng đồng, dự án cần kèm theo ghi chú bản quyền OpenRAIL trong tài liệu Model Card (`AI_MODEL_CARD.md`).
3. **Đánh giá nguồn gốc dữ liệu:**
   - Về mặt hình ảnh và cấu trúc thư mục, nhóm ảnh trái cây trong `Densu341` tương đồng với bộ dữ liệu Kaggle của Sriram Reddy Kalluri (CC0 Public Domain). Dù vậy, khi phân phối sản phẩm phái sinh hoặc công bố Model Card, dự án vẫn thực hiện ghi nhận Attribution đầy đủ theo cả OpenRAIL và CC0 để loại trừ mọi rủi ro pháp lý.

---

## 6. Đề xuất Phương án Xử lý & Tiếp nhận Dữ liệu Cục bộ

Để bảo vệ chất lượng mô hình và tuân thủ các quy tắc an toàn:

1. **Tuyệt đối không sao chép nguyên trạng 2.88 GB:**
   - Không đưa 6 loại rau củ (*khổ qua, ớt chuông, dưa chuột, đậu bắp, khoai tây, cà chua*) vào pipeline nhận diện trái cây của Golden Time.
   - Không sử dụng cấu trúc split có sẵn của `Test` để tránh lỗi 4,039 ảnh bị Data Leakage.
2. **Quy trình Trích xuất Đề xuất (Chờ Chủ dự án duyệt):**
   - Chỉ lọc lấy ảnh từ 6 thư mục: `freshapples`, `rottenapples`, `freshbanana`, `rottenbanana`, `freshoranges`, `rottenoranges`.
   - Chạy hàm băm SHA-256 để loại bỏ hoàn toàn các ảnh trùng lặp.
   - Lấy mẫu cân bằng chất lượng cao (ví dụ: lấy khoảng 300–400 ảnh tươi và 300–400 ảnh hỏng cho mỗi loại quả).
   - Resize chuẩn hóa về $224 \times 224$ pixels (JPEG 90).
   - Tự phân chia ngẫu nhiên lại theo đúng tỷ lệ độc lập: **70% Train / 15% Val / 15% Test** vào `golden-time-ai/datasets/fruits/`.
   - Dung lượng sau khi trích xuất và tối ưu sẽ chỉ tốn thêm **khoảng 35–45 MB**, giữ nguyên cam kết tối ưu hóa bộ nhớ dự án.

---

## 7. Các Điểm Cần Chủ Dự án Phê duyệt Tiếp theo

Kính đề nghị Chủ dự án xem xét và duyệt các bước thực hiện:

1. **Phê duyệt khai thác chọn lọc:** Cho phép trích xuất ảnh nhãn *Fresh* và *Rotten* của 3 loại quả (**Táo, Chuối, Cam**) từ `Downloads/dataset`, bỏ qua 6 loại rau củ.
2. **Phê duyệt tái phân chia split (Deduplicate & Resplit):** Cho phép thuật toán lọc bỏ 4,039 ảnh trùng lặp và tự chia lại tập Train/Val/Test để triệt tiêu nguy cơ Data Leakage.
3. **Nguồn cho 4 lớp quả còn lại:** Tiếp tục chờ Chủ dự án cung cấp thông tin/dataset cho 4 lớp quả đặc sản Việt Nam còn thiếu (*Sầu riêng, Mít, Nhãn, Mãng cầu*).
