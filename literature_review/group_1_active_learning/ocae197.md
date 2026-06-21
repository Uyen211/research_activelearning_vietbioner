## PHÂN TÍCH HỌC THUẬT BÀI BÁO: "Utilizing active learning strategies in machine-assisted annotation for clinical named entity recognition: a comprehensive analysis considering annotation costs and target effectiveness"

---

### 1. TÓM TẮT BÀI BÁO

*   **Bài toán nghiên cứu là gì?**
    *   Tối ưu hóa các chiến lược học chủ động (Active Learning - AL) trong kịch bản gán nhãn có sự hỗ trợ của máy (machine-assisted annotation) cho tác vụ nhận diện thực thể lâm sàng (Clinical NER), nhằm đạt được hiệu năng mục tiêu cao với chi phí hiệu chỉnh của con người là thấp nhất.
*   **Động lực nghiên cứu xuất phát từ đâu?**
    *   **Chi phí nhân công cao:** Gán nhãn hồ sơ lâm sàng chất lượng cao đòi hỏi các chuyên gia y tế có kinh nghiệm, dẫn đến chi phí lao động cực lớn.
    *   **Sự phổ biến của pre-annotation:** Ngày nay, quy trình gán nhãn thường được hỗ trợ bằng cách cho mô hình AI dự đoán nhãn trước (pre-annotate), sau đó con người chỉ việc chỉnh sửa các nhãn sai/thiếu.
    *   **Hạn chế của các chiến lược AL tĩnh:** Các chiến lược AL truyền thống (chỉ dùng độ bất định hoặc tính đa dạng) giữ nguyên thuật toán lựa chọn trong suốt quá trình lặp. Tuy nhiên, tính chất học của mô hình thay đổi theo thời gian: giai đoạn đầu cần đa dạng hóa dữ liệu (diversity) để bao phủ khái niệm, giai đoạn sau cần tập trung vào các mẫu mập mờ (uncertainty) để tinh chỉnh biên thực thể.
*   **Khoảng trống nghiên cứu (research gap) mà tác giả muốn giải quyết là gì?**
    *   Thiếu một khung lấy mẫu động (dynamic sampling framework) có khả năng tự động chuyển đổi từ chiến lược dựa trên tính đa dạng sang chiến lược dựa trên độ bất định khi mô hình đạt trạng thái bão hòa kiến thức ban đầu.
    *   Thiếu việc đánh giá hiệu quả của học chủ động dựa trên độ đo chi phí thực tế trong kịch bản con người hiệu chỉnh pre-annotation (thay vì gán nhãn thủ công hoàn toàn).
*   **Những đóng góp chính của bài báo là gì?**
    *   Đề xuất 3 chiến lược AL mới:
        1.  **CLUSTER:** Chiến lược đa dạng hóa dựa trên Sentence-BERT và phân cụm K-Means.
        2.  **CLC:** Chiến lược động bắt đầu bằng CLUSTER và chuyển sang Least Confidence (LC) khi sự sụt giảm training loss giữa các vòng lặp $< 0.005$.
        3.  **CNBSE:** Chiến lược động bắt đầu bằng CLUSTER và chuyển sang N-best Sequence Entropy (NBSE) khi training loss sụt giảm $< 0.005$.
    *   Đề xuất độ đo chi phí gán nhãn thực tế: **Số lượng thao tác hiệu chỉnh (number of edits)**, đo bằng khoảng cách Levenshtein giữa nhãn AI dự đoán và nhãn chuẩn.
    *   Đánh giá thực nghiệm mô phỏng trên 3 bộ dữ liệu lâm sàng lớn về thông tin thuốc: i2b2 2009, n2c2 2018 (Track 2) và MADE 1.0, sử dụng backbone là BioClinicalBERT.
    *   Chứng minh chiến lược động **CNBSE** giúp tiết kiệm tới **20.4%** chi phí hiệu chỉnh so với chỉ dùng chiến lược độ bất định tĩnh.

---

### 2. Ý TƯỞNG CHÍNH

*   **Ý tưởng cốt lõi:**
    *   Khai thác thế mạnh của cả hai miền: Ở các vòng đầu, dùng mô hình mã hóa câu mạnh (Sentence-BERT) để gom nhóm dữ liệu chưa gán nhãn thành các cụm ngữ nghĩa đa dạng và lấy mẫu đại diện từ mỗi cụm. Khi mô hình bắt đầu học tốt và lượng loss giảm chậm lại, chuyển đổi sang chiến lược độ bất định (entropy chuỗi) để chọn các câu chứa các thực thể khó hoặc mập mờ để gán nhãn chi tiết.
*   **Đóng góp mới (novel contributions):**
    *   **Cơ chế chuyển đổi chiến lược động (Dynamic Switching Mechanism):** Giúp tối ưu hóa việc lấy mẫu ở từng giai đoạn học của mô hình y sinh.
    *   **Độ đo chi phí Levenshtein Edit Distance:** Phản ánh chính xác công sức chỉnh sửa nhãn của con người (bao gồm thêm nhãn, xóa nhãn sai, sửa nhãn lệch biên) trong các công cụ gán nhãn thông minh hiện nay.
*   **Những điểm khác biệt so với các nghiên cứu trước:**
    *   Khác với các nghiên cứu AL tĩnh hoặc các nghiên cứu lai hai mức (hybrid hai tầng như CARLS chạy song song), chiến lược động của tác giả thay đổi hoàn toàn hành vi chọn mẫu theo thời gian thực dựa trên tín hiệu huấn luyện của mô hình.

---

### 3. PHÂN TÍCH DATASET

#### Các dataset thử nghiệm
*   **i2b2 2009 (Medication Extraction):** 261 hồ sơ có nhãn vàng (6 loại thực thể: medication, dosage, mode, frequency, duration, reason).
*   **n2c2 2018 (Track 2 - ADE & Medication):** 505 bệnh án điện tử từ MIMIC-III (9 loại thực thể: drug, strength, form, dosage, frequency, route, duration, reason, ADE).
*   **MADE 1.0 (Medication & Adverse Event):** 1.089 ghi chú lâm sàng (9 loại thực thể: medication, indication, frequency, severity, dosage, duration, route, ADE, SSLIF).

---

### 4. PHÂN TÍCH PHƯƠNG PHÁP

#### Pipeline huấn luyện mô phỏng:
1.  **Khởi tạo:** Lấy ngẫu nhiên 1% số câu của dataset để làm dữ liệu gán nhãn ban đầu, huấn luyện mô hình BioClinicalBERT đầu tiên.
2.  **Lấy mẫu AL:** Áp dụng chiến lược lựa chọn để chấm điểm tập chưa gán nhãn, chọn ra 1% số lượng tokens để đưa cho oracle gán nhãn.
3.  **Huấn luyện lại:** Huấn luyện lại mô hình BioClinicalBERT từ đầu trên tập dữ liệu đã mở rộng.
4.  **Chuyển đổi thuật toán:** Nếu $\text{Loss}_{(L-1)} - \text{Loss}_{(L)} < 0.005$, chuyển thuật toán chọn mẫu từ CLUSTER sang LC/NBSE. Lặp lại tối đa 30 vòng.

#### 4.1 Định nghĩa chi phí hiệu chỉnh (Number of Edits)
*   Trong kịch bản có pre-annotation, annotator sẽ nhìn thấy các nhãn BIO dự đoán bởi mô hình. Chi phí hiệu chỉnh được tính bằng số lượng thao tác cần thiết để sửa chuỗi nhãn dự đoán thành chuỗi nhãn chuẩn (Levenshtein distance). Ví dụ, nếu mô hình bỏ sót nhãn, cần 1 thao tác chèn (insert); nếu mô hình gán sai nhãn, cần 1 thao tác thế (replace).

#### 4.2 Mô hình
*   **Backbone:** BioClinicalBERT (được pre-train trên kho dữ liệu khổng lồ MIMIC-III v1.4 chứa các ghi chú lâm sàng thực tế).
*   **Đầu ra:** Dự đoán nhãn BIO cấp độ token sử dụng lớp phân loại tuyến tính.
*   **Học chủ động:** Sử dụng chiến lược CNBSE, trong đó phần NBSE (N-best sequence entropy) tính toán entropy trên 3 chuỗi nhãn có xác suất cao nhất dự đoán bởi mô hình cho mỗi câu:
    $$\text{Entropy}(t) = -\sum_{i=1}^{N} P(y_i \mid t) \log P(y_i \mid t)$$

---

### 5. PHÂN TÍCH KẾT QUẢ THỰC NGHIỆM

*   **So sánh chi phí ở các mức F1 mục tiêu (Table 1):**
    *   **Đạt 98% hiệu năng mục tiêu:** Chiến lược **CLUSTER** đạt F1 nhanh nhất với số lượng chỉnh sửa ít nhất (chỉ cần chỉnh sửa trung bình **4.0%** tổng số nhãn của toàn bộ dataset).
    *   **Đạt 99% hiệu năng mục tiêu:** Chiến lược CLUSTER và RANDOM bị bão hòa và không thể đạt được mức F1 này trong vòng 30 iterations. Ngược lại, chiến lược động **CNBSE** đạt mục tiêu với số lượng chỉnh sửa ít nhất (giúp giảm **20.4%** số lượng chỉnh sửa so với chỉ dùng NBSE tĩnh). Chiến lược CLC giảm **16.9%** số lượng chỉnh sửa so với LC tĩnh.
*   **Hiệu quả trên các thực thể khó (Table 3):**
    *   Với các thực thể khó nhận diện như triệu chứng chỉ định (`Indication`, `Reason`) hay tác dụng phụ (`ADE`), CLUSTER và RANDOM thất bại trong việc đạt 93% F1 mục tiêu.
    *   Chiến lược **CNBSE** giúp giảm tới **22.5%** số lượng chỉnh sửa cần thiết so với NBSE để đạt 99% hiệu năng mục tiêu trên các nhãn khó này.

---

### 6. ĐÁNH GIÁ KHOA HỌC

*   **Điểm mạnh:**
    *   Thực tế hóa quy trình đánh giá AL thông qua kịch bản con người sửa lỗi pre-annotation và đo đạc bằng edit distance.
    *   Chứng minh được tính hiệu quả vượt trội của việc chuyển đổi chiến lược lấy mẫu động dựa trên training loss.
*   **Điểm yếu:**
    *   Ngưỡng sụt giảm training loss (0.005) và số lượng cụm K-Means ($K=100$) được chọn cố định dựa trên trực giác và thử nghiệm sơ bộ, chưa có cơ chế tối ưu hóa tự động các siêu tham số này cho từng dataset cụ thể.

---

### 7. BẢNG TỔNG KẾT

| Thành phần | Nội dung |
|------------|----------|
| **Bài toán** | Cost-effective Active Learning for Clinical Named Entity Recognition |
| **Dataset** | i2b2 2009, n2c2 2018 (Track 2), MADE 1.0 (các bộ dữ liệu đơn thuốc lâm sàng) |
| **Mô hình** | BioClinicalBERT |
| **Loss** | Cross-Entropy Loss |
| **Metric** | Micro-F1, Levenshtein edit distance (số lần hiệu chỉnh nhãn) |
| **Kết quả** | CNBSE đạt 99% F1 mục tiêu nhanh nhất, giảm 20.4% số lần sửa nhãn của con người so với NBSE tĩnh |
| **Đóng góp chính** | Đề xuất chiến lược AL động CLC và CNBSE; Độ đo chi phí hiệu chỉnh Levenshtein distance |
| **Hạn chế** | Các ngưỡng chuyển đổi và số lượng cụm còn mang tính thủ công, chưa tự động hóa tối ưu |
