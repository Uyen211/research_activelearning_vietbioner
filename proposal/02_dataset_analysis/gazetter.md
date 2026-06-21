# Tài liệu Đặc tả và Tiền xử lý Bộ Tri thức Gazetteer (Tài liệu 02 - Bổ sung)

Để hỗ trợ thuật toán Học chủ động giải quyết bài toán mất cân bằng lớp y sinh, đề tài tích hợp bộ tri thức Gazetteer lâm sàng tiếng Việt gồm 5 lớp thực thể mục tiêu. Tài liệu này mô tả chi tiết nguồn gốc, phương pháp tiền xử lý làm sạch rò rỉ dữ liệu (data leakage) và cơ chế vận hành của Gazetteer trong luồng thực nghiệm.

---

## 1. Nguồn gốc và Quy mô của 5 Gazetteer

Các tệp Gazetteer được xây dựng từ hai nguồn chính: trích xuất chuyên ngành từ y văn (đối với `Organisation` và `DiagnosticProcedure`) và sinh tự động bằng mô hình ngôn ngữ lớn (LLM) (đối với `DateTime`, `Location`, và `Symptom_and_Disease` dựa trên ngữ cảnh bệnh lao và y tế Việt Nam):

1.  **`symptom_and_disease.json` (SYM)**: Danh mục các triệu chứng lâm sàng (ho, ho đờm, sốt nhẹ về chiều...) và bệnh lý liên quan đến hệ hô hấp, bệnh lao. (Sinh từ LLM).
2.  **`diagnostic_procedures_tb.json` (DP)**: Danh mục các xét nghiệm vi sinh, kỹ thuật nhuộm soi, thủ thuật chẩn đoán hình ảnh chuyên sâu lao phổi. (Trích xuất từ y văn).
3.  **`location.json` (LOC)**: Danh mục các địa danh y tế, các khoa phòng lâm sàng (khoa lao, ICU, phòng cách ly...) và các địa phương Việt Nam. (Sinh từ LLM).
4.  **`datetime.json` (DATE)**: Các trạng từ, khoảng thời gian y khoa chỉ phác đồ điều trị, chu kỳ dùng thuốc (2 tháng tấn công, phác đồ 6 tháng...). (Sinh từ LLM).
5.  **`healthcare_organizations.json` (ORG)**: Tên các bệnh viện đa khoa, trung tâm y tế, bệnh viện chuyên khoa lao phổi, các tổ chức y tế (WHO, CDC...). (Trích xuất từ y văn).

---

## 2. Tiền xử lý Lọc Rò rỉ Dữ liệu (Zero-Leakage Filter)

### 2.1. Tại sao phải lọc rò rỉ?
Trong thực nghiệm Học chủ động (Active Learning), tập Validation (`dev.txt`) và Test (`test.txt`) phải được giữ cô lập hoàn toàn (cả về ngữ cảnh lẫn từ vựng thực thể) để đánh giá khách quan khả năng suy rộng của mô hình. 
Nếu một cụm từ trong Gazetteer xuất hiện trong tập Test, việc sử dụng Gazetteer này để tăng cường dữ liệu huấn luyện (Distant Supervision) sẽ tương đương với việc **gián tiếp nạp tri thức tập kiểm thử vào mô hình**, làm sai lệch kết quả thực nghiệm F1-score (Data Leakage).

### 2.2. Thuật toán lọc trùng tự động (`filter_gazetteers.py`)
Chúng tôi đã xây dựng và chạy script [filter_gazetteers.py](file:///C:/Users/Admin/.gemini/antigravity-ide/brain/1a923252-cf62-43f9-9226-5dcb22a6c310/scratch/filter_gazetteers.py) thực thi quy trình sau:
1.  Đọc tập `dev.txt` và `test.txt` của VietBioNER.
2.  Duyệt qua các dòng nhãn BIO, trích xuất tất cả các span thực thể hoàn chỉnh theo từng lớp.
3.  Chuẩn hóa các span này (chuyển chữ thường, cắt khoảng trắng dư thừa, và đồng bộ dấu gạch dưới thành dấu cách).
4.  Đối chiếu từng từ khóa trong 5 Gazetteer gốc: Nếu từ khóa (sau khi chuẩn hóa) tồn tại trong tập thực thể của dev/test cùng lớp, hệ thống **loại bỏ hoàn toàn** từ khóa đó khỏi Gazetteer tĩnh.

### 2.3. Kết quả lọc chi tiết

Bảng thống kê số lượng từ khóa của Gazetteer trước và sau khi làm sạch:

| Tệp Gazetteer | Loại thực thể | Số lượng gốc | Số bị loại bỏ (Trùng khớp Dev/Test) | Số lượng còn lại |
| :--- | :--- | :--- | :--- | :--- |
| **`symptom_and_disease.json`** | Symptom_and_Disease | 179 | **18** | **161** |
| **`diagnostic_procedures_tb.json`** | DiagnosticProcedure | 327 | **1** | **326** |
| **`location.json`** | Location | 180 | **5** | **175** |
| **`datetime.json`** | DateTime | 162 | **1** | **161** |
| **`healthcare_organizations.json`** | Organisation | 554 | **2** | **552** |

*Các thực thể cụ thể bị lọc bỏ bao gồm*:
- Đối với `symptom_and_disease.json`: các từ trùng như *"ho"*, *"tràn dịch màng phổi"*, *"lao phổi"*, *"lao kháng thuốc"*, *"đái tháo đường"*, *"hôn mê"*, *"ran phổi"*, v.v.
- Đối với `diagnostic_procedures_tb.json`: *"sinh thiết màng phổi"*.
- Đối với `location.json`: *"long an"*, *"cần thơ"*, *"quận gò vấp"*, *"việt nam"*, *"phòng xét nghiệm"*.
- Đối với `datetime.json`: *"3 tuần"*.
- Đối với `healthcare_organizations.json`: *"bệnh viện phạm ngọc thạch"*, *"tổ chức y tế thế giới"*.

---

## 3. Tiền xử lý Tách từ ghép Động (Dynamic Tokenization)

Khi chèn một thực thể từ Gazetteer vào câu gốc trong pha Tăng cường dữ liệu (Distant Supervision), để đảm bảo không phá vỡ cấu trúc ngữ liệu tiếng Việt đã được tách từ ghép:
*   Mã nguồn thực nghiệm gọi thư viện PyVi (`ViTokenizer.tokenize`) trên chuỗi thực thể mới lấy từ Gazetteer để tách từ ghép động (ví dụ: *"sinh thiết màng phổi"* $\rightarrow$ *"sinh_thiết màng_phổi"*).
*   Chia tách chuỗi kết quả theo khoảng trắng để tạo mảng token mới.
*   Gán nhãn `B-` cho token đầu tiên và nhãn `I-` cho các token còn lại.
*   Chèn mảng token và nhãn này vào vị trí thực thể cũ trong câu thô, đồng thời tự động tịnh tiến các chỉ mục nhãn BIO còn lại của câu (Index Shifting).

Quy trình khép kín này đảm bảo tính tương thích ranh giới 100% giữa dữ liệu gốc và dữ liệu tăng cường.