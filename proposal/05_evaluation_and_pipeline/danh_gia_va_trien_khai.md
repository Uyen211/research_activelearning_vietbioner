# Phương Pháp Đánh Giá Hiệu Năng & Kế Hoạch Triển Khai Thực Nghiệm

---

## 1. Phương pháp đối chứng Trực diện để trả lời Câu hỏi Nghiên cứu

Để chứng minh tính thuyết phục học thuật của đề tài, chúng ta thiết lập hai kịch bản đối chứng chặt chẽ. Cả hai nhánh đều sử dụng chung mô hình nền tảng, định dạng đầu vào (Entity Descriptions) và phương pháp tăng cường dữ liệu (Distant Supervision) để đảm bảo so sánh công bằng trong cùng một môi trường:

```
                  ┌──> [Nhánh A: Active Learning (AL)] ──> Đường cong học tập AL (F1 vs Budget)
[Tập ban đầu L_0] ┤
                  └──> [Nhánh B: Random Sampling (RS)] ──> Đường cong học tập RS (F1 vs Budget)
```

### 1.1. Thí nghiệm So sánh Đường cong Học tập (Learning Curves Comparison)
Chúng ta vẽ biểu đồ biểu thị F1-score (trục Y) theo Ngân sách gán nhãn (trục X, đo bằng số lượng câu được gán nhãn hoặc số thao tác chỉnh sửa):
1.  **Đường cong AL (Nhánh A)**: F1-score đạt được qua các vòng lặp khi chọn mẫu bằng Active Learning đề xuất.
2.  **Đường cong RS (Nhánh B - Baseline)**: F1-score đạt được khi chọn mẫu ngẫu nhiên.
*   **Chứng minh sự thành công**: Nếu tại mọi điểm ngân sách (ví dụ: khi gán nhãn 10%, 30%, 50% dữ liệu), đường cong AL nằm trên đường cong RS với khoảng cách F1-score có ý nghĩa thống kê ($p < 0.05$ thông qua kiểm định Wilcoxon signed-rank test trên các bước hoặc t-test trên Diện tích dưới đường cong học tập - AULC qua 5 hạt giống khác nhau, kết hợp hiệu chỉnh đa thử nghiệm Bonferroni Correction), câu hỏi nghiên cứu thứ nhất được xác nhận là **Đúng**.

### 1.2. Định lượng Mức độ Tiết kiệm Chi phí (Quantifying Savings)
Chúng ta xác định một mức hiệu năng mục tiêu (Target Performance), ví dụ: **F1-score = 75.0%** (xấp xỉ 95% hiệu năng tối đa của mô hình khi huấn luyện trên 100% dữ liệu VietBioNER).

#### Chỉ số 1: Tỷ lệ tiết kiệm số lượng mẫu câu (Sentence Saving Ratio - SSR)
*   Gọi $N_{\text{AL}}$ là số câu cần gán nhãn trong Nhánh A để đạt F1 mục tiêu.
*   Gọi $N_{\text{RS}}$ là số câu cần gán nhãn trong Nhánh B để đạt F1 mục tiêu.
    $$\text{SSR (\%)} = \left(1 - \frac{N_{\text{AL}}}{N_{\text{RS}}}\right) \times 100\%$$
*   *Kỳ vọng nghiên cứu*: SSR đạt từ **30% đến 50%**, nghĩa là chỉ cần gán nhãn một nửa lượng dữ liệu là đạt hiệu năng tương đương.

#### Chỉ số 2: Tỷ lệ tiết kiệm thao tác hiệu chỉnh (Edit Saving Ratio - ESR)
*   Gọi $E_{\text{AL}}$ là tổng số lần sửa nhãn (edit distance) chuyên gia phải thực hiện trên các mẫu pre-annotation của Nhánh A để đạt F1 mục tiêu. $E_t$ được định nghĩa cụ thể là tổng tích lũy (tổng số) khoảng cách Levenshtein giữa các chuỗi nhãn gộp của mô hình và nhãn chuẩn của VietBioNER trên tất cả các token được chọn từ Vòng 0 đến Vòng $t$.
*   Gọi $E_{\text{RS}}$ là tổng số lần sửa nhãn trong Nhánh B để đạt F1 mục tiêu.
    $$\text{ESR (\%)} = \left(1 - \frac{E_{\text{AL}}}{E_{\text{RS}}}\right) \times 100\%$$
*   *Kỳ vọng nghiên cứu*: ESR đạt từ **20% đến 40%**. Chỉ số này chứng minh AL không chỉ chọn ít câu hơn, mà còn chọn các câu giúp mô hình học cách pre-annotate chính xác hơn ở các vòng sau, giảm trực tiếp số lỗi sai biên thực thể mà con người phải sửa.

### 1.3. Phân tích Ablation Study (Đóng góp của các thành phần)
Để đánh giá định lượng đóng góp của từng thành phần đề xuất, chúng ta thực hiện Ablation Study trên Nhánh A bằng cách chạy và so sánh các biến thể rút gọn sau:
1.  **Đầy đủ (Nhánh A đầy đủ)**: `ViPubmedDeBERTa + CRF Marginal Entropy + Distinct-K Filter + Entity Descriptions + Distant Supervision`.
2.  **Bỏ Distant Supervision**: `ViPubmedDeBERTa + CRF Marginal Entropy + Distinct-K Filter + Entity Descriptions` (Không áp dụng thế thực thể cho nhãn hiếm).
3.  **Bỏ bộ lọc Distinct-K Filter**: `ViPubmedDeBERTa + CRF Marginal Entropy + Entity Descriptions` (Chọn mẫu trực tiếp theo entropy biên cao nhất mà không lọc trùng ngữ nghĩa).
4.  **Bỏ Entity Descriptions**: `ViPubmedDeBERTa + CRF Marginal Entropy + Distinct-K Filter` (Đầu vào mô hình chỉ có câu văn bản thô $s$ thay vì ghép nối mô tả nhãn $d_c$, áp dụng cho cả hai nhánh đối chứng để đo lường mức độ cải thiện chung của biểu diễn nhãn).
*   Nghiên cứu này sẽ chỉ rõ sự kết hợp của các module nâng cao đóng góp bao nhiêu % vào tổng lượng F1-score cải thiện và mức độ tối ưu hóa chi phí.

### 1.4. Ý nghĩa khoa học và Phương pháp tính Levenshtein Edit Distance trong Thực nghiệm Mô phỏng

Mặc dù nghiên cứu này sử dụng phương pháp **Mô phỏng Học chủ động (Simulated Active Learning)** – tức là nhãn của tập chưa gán nhãn ($U$) được hé lộ tự động bằng chương trình dựa trên nhãn chuẩn (gold labels) có sẵn của `VietBioNER` mà không thuê chuyên gia y tế gán nhãn trực tiếp theo thời gian thực – việc đo lường khoảng cách hiệu chỉnh **Levenshtein Edit Distance** vẫn đóng vai trò là một **độ đo đánh giá gián tiếp (Proxy Metric) cực kỳ quan trọng và có giá trị học thuật cao** nhằm giải quyết trực tiếp mục tiêu nghiên cứu của đề tài:

#### A. Tại sao cần độ đo này trong nghiên cứu mô phỏng?
1. **Khắc phục hạn chế của độ đo số lượng câu truyền thống (Sentence Count - SSR)**:
   - Các nghiên cứu AL truyền thống thường chỉ đo lường chi phí bằng số lượng câu hoặc số lượng token được chọn. Điều này giả định ngầm rằng mọi câu đều có chi phí gán nhãn như nhau.
   - Trong thực tế y sinh, gán nhãn một câu ngắn chứa 1 thực thể đơn giản sẽ tốn ít công sức hơn rất nhiều so với một câu dài chứa nhiều thực thể quy trình chẩn đoán phức tạp. SSR không phản ánh được sự không đồng đều này.
2. **Mô phỏng quy trình hậu hiệu chỉnh thực tế (Post-editing Scenario)**:
   - Quy trình gán nhãn trong công nghiệp và nghiên cứu hiện đại luôn là **machine-assisted (có máy hỗ trợ)**: Mô hình hiện tại dự đoán trước nhãn (Pre-annotation), chuyên gia chỉ cần hiệu chỉnh (thêm thực thể bị thiếu, xóa nhãn sai, sửa ranh giới nhãn lệch).
   - Chi phí thực tế mà con người bỏ ra (thời gian, công sức) tỷ lệ thuận với **số lượng thao tác hiệu chỉnh (chèn, xóa, thay thế)** trên giao diện phần mềm. Khoảng cách Levenshtein giữa chuỗi nhãn dự đoán của mô hình và nhãn chuẩn của VietBioNER chính là đại lượng toán học mô phỏng hoàn hảo cho số lượng thao tác này của chuyên gia.
3. **Đo lường gián tiếp tốc độ hội tụ chất lượng Pre-annotation**:
   - Nếu thuật toán AL chọn mẫu thông minh, mô hình NER sẽ học nhanh hơn và dự đoán ngày càng chính xác hơn trên các tập dữ liệu mới. Điều này dẫn đến khoảng cách Levenshtein trên mỗi batch được chọn sẽ **giảm nhanh hơn** qua các vòng lặp so với Random Sampling.
   - Đo lường chỉ số này chứng minh được rằng: **Active Learning không chỉ giúp giảm số câu cần gán nhãn để đạt hiệu năng mong muốn, mà còn làm cho các gợi ý nhãn (pre-annotation) ở các vòng sau chính xác hơn, trực tiếp làm giảm "độ khó" và nỗ lực sửa lỗi trên từng câu cho chuyên gia**.

#### B. Phương pháp tính toán lập trình (Programmatic Computation)
Trong mã nguồn mô phỏng, khoảng cách Levenshtein được tính toán tự động sau mỗi vòng AL như sau:
*   Hệ thống chạy dự đoán trên 5 chuỗi mô tả nhãn của câu $s$, sau đó gộp nhãn và áp dụng cơ chế **giải quyết xung đột cấp độ thực thể (Span-level)** dựa trên trung bình xác suất biên từ lớp CRF để tạo ra chuỗi nhãn gộp đa phân lớp hợp lệ và sạch lỗi định dạng BIO làm chuỗi dự đoán cuối cùng $Y_{\text{pred}} = [y_1, y_2, ..., y_n]$ cho câu $s$. Chuỗi nhãn chuẩn là $Y_{\text{gold}} = [g_1, g_2, ..., g_n]$.
*   Khoảng cách Levenshtein $D_L(Y_{\text{pred}}, Y_{\text{gold}})$ được tính bằng số thao tác tối thiểu (chèn, xóa, thế) để biến đổi $Y_{\text{pred}}$ thành $Y_{\text{gold}}$ ở cấp độ token.
*   **Ví dụ minh họa**:
    - $Y_{\text{pred}} = [\text{O}, \text{B-DIS}, \text{O}]$ (Dự đoán thiếu một nhãn phụ thuộc)
    - $Y_{\text{gold}} = [\text{O}, \text{B-DIS}, \text{I-DIS}]$
    - $\rightarrow$ Khoảng cách Levenshtein là **1** (cần 1 thao tác chèn nhãn `I-DIS` vào vị trí cuối).
*   Bằng cách tích lũy khoảng cách Levenshtein qua các vòng lặp, chúng ta vẽ được đường cong học tập biểu diễn: **Hiệu năng F1-score (Trục Y) theo Tổng số thao tác hiệu chỉnh tích lũy (Trục X)**. Đây là bằng chứng định lượng thuyết phục nhất chứng minh Active Learning giúp chuyên gia gán nhãn nhanh hơn và nhàn hơn so với Random Sampling trong thực tiễn triển khai.

---

## 2. Kế hoạch Triển khai Thực nghiệm

### Giai đoạn 1: Thiết lập môi trường và Huấn luyện mô hình nền tảng
*   Cài đặt thư viện Python (PyTorch, Transformers, seqeval, PyVi).
*   Sử dụng mô hình nền tảng **ViPubmedDeBERTa-base** (86M tham số) làm backbone chính.
*   Thiết lập và kiểm tra mã nguồn huấn luyện NER cơ bản trên VietBioNER để ghi nhận baseline khi dùng 100% dữ liệu.

### Giai đoạn 2: Lập trình Mô phỏng Học chủ động (Simulated Active Learning)
*   Viết mã nguồn quản lý Unlabeled Pool $U$, Labeled Set $L$, và tập Validation/Test cố định của VietBioNER.
*   Triển khai hai bộ chọn mẫu chính: Chọn mẫu chủ động đề xuất (Nhánh A) và chọn mẫu ngẫu nhiên (Nhánh B).
*   Mô phỏng quy trình gán nhãn tự động bằng cách lấy nhãn chuẩn từ tập train gốc của VietBioNER để phản hồi cho mô hình sau mỗi vòng chọn mẫu. Ghi nhận số lượng câu gán nhãn.

### Giai đoạn 3: Tích hợp Module Nâng cao
*   Tích hợp kỹ thuật gán nhãn dựa trên mô tả thực thể (Entity Type Description) và Distant Supervision Augmentation.
*   Chạy thử nghiệm Ablation Study để đo lường tác động của từng module.
*   Mô phỏng quy trình tính khoảng cách Levenshtein edit distance giữa dự đoán của mô hình và nhãn chuẩn để đo lường chi phí thao tác của con người.

### Giai đoạn 4: Thu thập số liệu và Viết báo cáo
*   Vẽ biểu đồ so sánh F1-score, đường cong chi phí hiệu chỉnh nhãn giữa các phương pháp.
*   Tính toán các chỉ số tiết kiệm SSR và ESR.
*   Tổng hợp kết quả, viết báo cáo học thuật hoàn chỉnh cho đề tài.
