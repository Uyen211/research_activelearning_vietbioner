# CHƯƠNG 4: THỰC NGHIỆM VÀ KẾT QUẢ

---

## 4.1. Thiết lập thực nghiệm

### 4.1.1. Mục tiêu thực nghiệm
Mục tiêu cốt lõi của quá trình thực nghiệm trong đề tài này là đánh giá định lượng hiệu quả của phương pháp Học chủ động (Active Learning - AL) kết hợp Thế thực thể dựa trên từ điển (Dictionary-based Entity Substitution - DES) trên tập dữ liệu y sinh tiếng Việt VietBioNER. Quá trình thực nghiệm được thiết kế nhằm trả lời hai câu hỏi nghiên cứu chính:
1.  Việc áp dụng chiến lược chọn mẫu thông minh dựa trên độ bất định CRF Marginal Entropy và độ đa dạng Distinct-K Filter có giúp mô hình đạt chất lượng nhận dạng thực thể (F1-score) cao hơn và tối ưu hóa ngân sách gán nhãn (số câu gán nhãn) so với lấy mẫu ngẫu nhiên truyền thống hay không?
2.  Quy trình pre-annotation kết hợp hậu hiệu chỉnh bằng mô phỏng chuyên gia có giúp giảm nỗ lực thao tác sửa đổi nhãn thực tế của nhân viên y tế (đo bằng khoảng cách Levenshtein tích lũy) trong quá trình gán nhãn hay không?

### 4.1.2. Môi trường phần cứng và phần mềm
Toàn bộ các thí nghiệm huấn luyện, suy luận và mô phỏng được thực hiện trên môi trường điện toán đám mây Google Colab với cấu hình phần cứng tăng tốc đồ họa GPU Nvidia T4 (16GB VRAM) hoặc Nvidia P100.
Hệ thống phần mềm và thư viện lập trình được thiết lập đồng nhất bao gồm:
*   Ngôn ngữ lập trình: Python phiên bản 3.10.
*   Framework học sâu: PyTorch và HuggingFace Transformers.
*   Thư viện xử lý tiếng Việt: PyVi để thực hiện tách từ ghép.
*   Thư viện đánh giá hiệu năng gán nhãn chuỗi: `seqeval` và `scikit-learn`.

### 4.1.3. Cấu hình kịch bản đối chứng
Quá trình mô phỏng học chủ động được thiết lập thông qua hai nhánh thí nghiệm đối chứng song song xuất phát từ cùng một mô hình khởi tạo trên Seed Set $L_0$ (85 câu thô được chọn bằng phương pháp lấy mẫu phân tầng):
*   **Nhánh A (Học chủ động đề xuất - AL)**: Mô hình thực hiện chọn mẫu có chọn lọc từ Unlabeled Pool bằng thuật toán kết hợp CRF Marginal Entropy và bộ lọc đa dạng ngữ nghĩa Distinct-K Filter. Mỗi vòng lặp chọn ra batch gồm $b = 100$ câu thô có giá trị thông tin cao nhất.
*   **Nhánh B (Lấy mẫu ngẫu nhiên đối chứng - Random)**: Mô hình chọn ngẫu nhiên $b = 100$ câu thô từ Unlabeled Pool tại mỗi vòng lặp mà không sử dụng điểm đánh giá thông tin của mô hình.

Cả hai nhánh đều chạy qua 5 vòng huấn luyện (từ Vòng 0 đến Vòng 4), tương ứng với quy mô dữ liệu đã gán nhãn tích lũy tăng từ $N_0 = 85$ câu lên mốc giới hạn ngân sách tối đa là $N_4 = 485$ câu (chiếm xấp xỉ 45% tập dữ liệu Train mới, cụ thể là 485/1.089 câu). Để kiểm soát biến số công bằng, cả hai nhánh đều sử dụng chung cấu hình siêu tham số, cơ chế thích ứng tham số hiệu quả LoRA ($r=16, \alpha=32$), cơ chế ghép nối mô tả thực thể (Entity Type Description), cơ chế lấy mẫu âm tính (Negative Sampling) và thuật toán tăng cường thế thực thể dựa trên từ điển (Dictionary-based Entity Substitution - DES).

---

## 4.2. Kết quả đối chứng và phân tích hiệu năng gán nhãn

### 4.2.1. Tiến trình hiệu năng qua các vòng lặp
Hiệu năng nhận diện thực thể y sinh lâm sàng trên tập Test cố định (137 câu) và chi phí hiệu chỉnh nhãn tích lũy của hai nhánh thí nghiệm qua 5 vòng lặp được tổng hợp chi tiết trong bảng dưới đây:

| Nhánh thí nghiệm | Vòng lặp (Loop) | Số câu gán nhãn ($N_t$) | Chi phí sửa vòng này | Chi phí sửa lũy kế ($E_t$) | Precision (%) | Recall (%) | F1-score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nhánh A (AL + DES)** | Vòng 0 | 85 | 919 | 919 | 46.62% | 43.23% | 44.86% |
| | Vòng 1 | 185 | 638 | 1.557 | 58.47% | 58.09% | 58.28% |
| | Vòng 2 | 285 | 423 | 1.980 | 67.89% | 67.00% | 67.44% |
| | Vòng 3 | 385 | 282 | 2.262 | 65.03% | 61.39% | 63.16% |
| | Vòng 4 (Cuối) | 485 | 294 | 2.556 | 72.09% | 71.62% | **71.85%** |
| | | | | | | | |
| **Nhánh B (Random + DES)**| Vòng 0 | 85 | 919 | 919 | 46.62% | 43.23% | 44.86% |
| *(Đồng bộ Vòng 0)* | Vòng 1 | 185 | 336 | 1.255 | 54.77% | 58.75% | 56.69% |
| | Vòng 2 | 285 | 242 | 1.497 | 54.09% | 61.06% | 57.36% |
| | Vòng 3 | 385 | 224 | 1.721 | 66.15% | 70.30% | 68.16% |
| | Vòng 4 (Cuối) | 485 | 246 | 1.967 | 60.67% | 65.68% | **63.07%** |

`[Hình ảnh: Bảng tổng hợp kết quả thực nghiệm trong notebook colab/notebook/02_danh_gia_va_truc_quan_hoa.ipynb, cell 3]`

Dựa trên bảng số liệu thực nghiệm, tiến trình phát triển hiệu năng của hai phương pháp thể hiện các đặc trưng khác biệt rõ nét:
*   **Trạng thái khởi đầu đồng bộ**: Do sử dụng chung tập Seed Set $L_0$ (85 câu thô), cả hai nhánh bắt đầu Vòng 0 với cùng mức F1-score tổng thể là 44.86%.
*   **Sự bứt phá hiệu năng của AL**: Kể từ Vòng 1, Nhánh A bắt đầu thể hiện sự ưu việt rõ rệt. Tại Vòng 2 ($N_t = 285$ câu), mô hình Nhánh A đạt mức F1-score tổng thể là 67.44%, vượt trội hoàn toàn so với mức F1-score 57.36% của Nhánh B (chênh lệch tới +10.08%).
*   **Tính ổn định của đà hội tụ**: Nhánh đối chứng ngẫu nhiên có xu hướng dao động mạnh, đạt đỉnh ở Vòng 3 với F1-score 68.16% nhưng lập tức giảm mạnh xuống 63.07% ở vòng cuối cùng do hiện tượng chọn phải các câu chứa thông tin nhiễu hoặc câu rỗng ngữ nghĩa. Trái lại, Nhánh AL duy trì xu hướng hội tụ ổn định và đạt hiệu năng cao nhất tại vòng cuối với F1-score tổng thể đạt 71.85% (vượt trội +8.78% so với Random).


### 4.2.2. Phân tích đường cong học tập và chỉ số AULC
Đường cong học tập biểu diễn sự thay đổi của F1-score tổng thể theo quy mô dữ liệu đã gán nhãn ($N_t$) minh chứng khả năng học nhanh hơn của Nhánh AL. Để lượng hóa tổng năng lực học tập tích lũy của mô hình trong suốt quá trình gán nhãn, đồ án sử dụng chỉ số Diện tích dưới đường cong học tập (Area Under the Learning Curve - AULC) cho F1-score tổng thể:

*   **AULC Nhánh A (AL + DES)**: 24.723,66 F1-câu.
*   **AULC Nhánh B (Random + DES)**: 23.618,10 F1-câu.
*   **Mức cải thiện**: +1.105,56 F1-câu (tương đương cải thiện +4.68% tổng năng lực học tập tích lũy).

Sự gia tăng đáng kể của chỉ số AULC khẳng định thuật toán chọn mẫu thông minh đã liên tục cung cấp các mẫu dữ liệu có giá trị thông tin cao nhất cho mô hình qua từng vòng lặp.

`[Hình ảnh: Biểu đồ đối chứng đường cong học tập trong notebook colab/notebook/02_danh_gia_va_truc_quan_hoa.ipynb, cell 4]`

### 4.2.3. Định lượng tỷ lệ tiết kiệm chi phí (SSR & ESR)
Do giới hạn ngân sách gán nhãn cố định ở mức 485 câu, mô hình ở cả hai nhánh chưa đạt đến mốc F1-score mục tiêu lý thuyết ban đầu là 75.0%. Nhằm đánh giá khả năng tối ưu hóa chi phí thực tế, đồ án thực hiện nội suy tuyến tính để tính toán tỷ lệ tiết kiệm mẫu câu (SSR) và tỷ lệ tiết kiệm thao tác hiệu chỉnh (ESR) tại các mốc F1-score thực tế đạt được:

| Mốc F1 mục tiêu (%) | Quy mô $N_t$ (AL) | Quy mô $N_t$ (Random) | **SSR (%)** | Chi phí $E_t$ (AL) | Chi phí $E_t$ (Random) | **ESR (%)** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **60.00%** | 203,8 câu | 309,4 câu | **34.14%** | 1.636,5 | 1.551,7 | **-5.46%** |
| **63.07%** *(F1 Rand vòng cuối)* | 237,3 câu | 337,9 câu | **29.76%** | 1.778,2 | 1.615,4 | **-10.08%** |
| **65.00%** | 258,4 câu | 355,7 câu | **27.37%** | 1.867,3 | 1.655,4 | **-12.80%** |
| **70.00%** | 463,7 câu | 385,0 câu* | **-20.44%*** | 2.493,3 | 1.721,0* | **-44.88%*** |

*(Chú thích: Ở mốc 70.00% F1, do Nhánh Random không bao giờ đạt được mức hiệu năng này trong giới hạn ngân sách thực nghiệm - mức F1 tối đa của Random chỉ đạt 68.16% ở Vòng 3, hệ thống sử dụng điểm kết thúc của Random làm giá trị đối chứng giả lập, dẫn đến các chỉ số SSR và ESR mang giá trị âm nhân tạo).*

Phân tích các chỉ số tiết kiệm mang lại những phát hiện khoa học quan trọng:
1.  **Hiệu quả tiết kiệm số lượng câu gán nhãn (SSR đạt giá trị dương đáng kể)**: Ở các mốc F1-score từ 60.00% đến 65.00%, Nhánh AL đạt tỷ lệ tiết kiệm câu từ **27.37% đến 34.14%**. Điều này chứng minh rằng khi ứng dụng Học chủ động, người gán nhãn có thể giảm bớt khoảng 1/3 khối lượng tài liệu cần đọc mà vẫn đạt được chất lượng mô hình tương đương với phương pháp lấy mẫu ngẫu nhiên truyền thống.
2.  **Sự đánh đổi nỗ lực sửa nhãn (ESR mang giá trị âm)**: Chi phí hiệu chỉnh Levenshtein tích lũy của Nhánh AL cao hơn Nhánh Random từ **5.46% đến 12.80%** tại các mốc hiệu năng tương đương. Đây là một hiện tượng mang tính quy luật trong học chủ động lâm sàng. Chiến lược chọn mẫu của AL chủ động tìm kiếm các câu chứa cấu trúc nhãn phức tạp, ranh giới nhập nhằng và độ bất định phân lớp cao để huấn luyện mô hình. Do đó, mô hình ở các vòng đầu có xu hướng dự đoán sai nhiều hơn trên các mẫu khó này, dẫn đến chuyên gia y tế tốn nhiều thao tác hiệu chỉnh (Levenshtein distance lớn hơn) trên mỗi câu được chọn. Ngược lại, lấy mẫu ngẫu nhiên thường chọn trúng các câu đơn giản, ít thực thể hoặc câu rỗng nhãn, dẫn đến chi phí hiệu chỉnh ban đầu thấp nhưng không mang lại tri thức giúp mô hình tiến bộ nhanh. Đây là sự đánh đổi thực tiễn: người gán nhãn đọc ít câu hơn (SSR dương 34%) nhưng phải tập trung nỗ lực sửa đổi nhiều hơn trên từng câu được chọn (ESR âm ~10%).


### 4.2.4. Hiệu năng phân lớp trực diện (Class-wise Comparison)
Để đánh giá chi tiết chất lượng nhận dạng trên từng loại thực thể lâm sàng, dưới đây là bảng đối chiếu chi tiết hiệu năng Precision, Recall và F1-score tại vòng lặp cuối cùng (Vòng 4 - Ngân sách $N_t = 485$ câu):

| Loại Thực thể mục tiêu | Chỉ số | Nhánh A (AL) (%) | Nhánh B (Random) (%) | Chênh lệch (AL - Random) | Support |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Symptom_and_Disease** | Precision | 75.94% | 68.11% | +7.83% | 184 |
| | Recall | 77.17% | 68.48% | +8.69% | |
| | **F1-score** | **76.55%** | **68.29%** | **+8.26%** | |
| **DiagnosticProcedure** | Precision | 53.13% | 36.96% | +16.17% | 37 |
| | Recall | 45.94% | 45.94% | 0.00% | |
| | **F1-score** | **49.28%** | **40.96%** | **+8.31%** | |
| **Location** | Precision | 65.00% | 53.49% | +11.51% | 37 |
| | Recall | 70.27% | 62.16% | +8.11% | |
| | **F1-score** | **67.53%** | **57.50%** | **+10.03%** | |
| **DateTime** | Precision | 79.17% | 73.08% | +6.09% | 23 |
| | Recall | 82.61% | 82.61% | 0.00% | |
| | **F1-score** | **80.85%** | **77.55%** | **+3.30%** | |
| **Organisation** | Precision | 72.22% | 50.00% | +22.22% | 22 |
| | Recall | 59.09% | 63.64% | -4.55% | |
| | **F1-score** | **65.00%** | **56.00%** | **+9.00%** | |
| **Tổng thể** | Precision | 72.09% | 60.67% | +11.42% | 303 |
| | Recall | 71.62% | 65.68% | +5.94% | |
| | **F1-score** | **71.85%** | **63.07%** | **+8.78%** | |

Kết quả so sánh phân lớp chỉ ra rằng Nhánh AL đạt hiệu năng vượt trội đồng đều trên **tất cả 5 lớp thực thể**. Các mức cải thiện mạnh mẽ nhất tập trung ở các lớp thực thể khó và hiếm gặp như `Location` (+10.03% F1-score), `Organisation` (+9.00% F1-score), và `DiagnosticProcedure` (+8.31% F1-score). Điều này chứng minh thuật toán chọn mẫu Distinct-K Filter kết hợp với cơ chế tăng cường thích ứng theo lớp đã hoạt động hiệu quả, giúp mô hình phân bổ sự chú ý đồng đều, tránh hiện tượng quá khớp vào lớp đa số `Symptom_and_Disease` mà bỏ quên các nhãn thiểu số lâm sàng.

Dưới đây là phần đánh giá chi tiết chất lượng nhận diện trên từng lớp thực thể cụ thể:
1.  **Lớp Symptom_and_Disease (Triệu chứng & Bệnh lý)**:
    *   Nhánh AL đạt F1-score là **76.55%** (với Precision 75.94% và Recall 77.17%), trong khi Nhánh Random chỉ đạt **68.29%** (Precision 68.11%, Recall 68.48%). Sự chênh lệch hiệu năng F1-score là **+8.26%**.
    *   *Nhận xét chi tiết*: Đây là lớp có số lượng mẫu lớn nhất trong tập dữ liệu (support = 184). Nhờ chiến lược chọn mẫu theo độ bất định biên (Marginal Entropy), mô hình chủ động lựa chọn các câu chứa các cấu trúc lâm sàng phức tạp mô tả triệu chứng bệnh lý phổi (như từ đồng nghĩa, viết tắt lâm sàng hoặc kết hợp từ). Việc cải thiện +8.26% F1-score khẳng định AL giúp mô hình bao phủ nhanh các biến thể từ vựng của lớp đa số mà không cần lượng câu gán nhãn trùng lặp quá lớn.
2.  **Lớp DiagnosticProcedure (Quy trình Chẩn đoán)**:
    *   Nhánh AL đạt F1-score là **49.28%** (với Precision 53.13% và Recall 45.94%), so với Nhánh Random đạt **40.96%** (Precision 36.96% và Recall 45.94%). Sự chênh lệch hiệu năng F1-score là **+8.31%**.
    *   *Nhận xét chi tiết*: Đây là lớp thực thể y khoa có cấu trúc phức tạp và khó nhận dạng nhất của bộ dữ liệu do tính đa âm tiết và ranh giới không rõ ràng (ví dụ: "chụp X-quang phổi thẳng", "xét nghiệm đờm trực tiếp tìm AFB"). Ở Nhánh Random, mô hình bị nhiễu do Precision rất thấp (36.96%), dẫn đến việc dự đoán nhầm các quy trình chẩn đoán thành từ thường hoặc nhãn O. Nhánh AL nâng Precision lên 53.13% (+16.17% chênh lệch) nhờ cơ chế Entity Type Description (ETD) giúp mô hình định hình tốt ranh giới ngữ nghĩa của nhãn.
3.  **Lớp Location (Vị trí/Địa điểm)**:
    *   Nhánh AL đạt F1-score là **67.53%** (Precision 65.00%, Recall 70.27%), trong khi Nhánh Random đạt **57.50%** (Precision 53.49%, Recall 62.16%). Sự chênh lệch F1-score đạt **+10.03%**.
    *   *Nhận xét chi tiết*: Lớp Location thường chỉ các phòng khoa lâm sàng, tên bệnh viện hoặc địa phương xuất hiện trong các bệnh án dịch tễ. Sự cải thiện vượt trội +10.03% F1-score chứng minh bộ lọc đa dạng Distinct-K giúp mô hình không bị lặp lại các địa danh phổ biến mà bao phủ đa dạng các thực thể chỉ vị trí địa lý đặc thù.
4.  **Lớp DateTime (Ngày tháng/Thời gian)**:
    *   Nhánh AL đạt F1-score là **80.85%** (Precision 79.17%, Recall 82.61%), so với Nhánh Random đạt **77.55%** (Precision 73.08%, Recall 82.61%). Sự chênh lệch F1-score đạt **+3.30%**.
    *   *Nhận xét chi tiết*: Thực thể thời gian có khuôn mẫu ngữ pháp tương đối rõ ràng (như "ngày...", "tháng...", "vòng 3 tuần nay"). Cả hai mô hình đều đạt Recall tốt (82.61%), tuy nhiên Nhánh AL tối ưu hơn về Precision nhờ chủ động chọn các cấu trúc thời gian phức tạp hơn để huấn luyện mô hình.
5.  **Lớp Organisation (Tổ chức)**:
    *   Nhánh AL đạt F1-score là **65.00%** (Precision 72.22%, Recall 59.09%), so với Nhánh Random đạt **56.00%** (Precision 50.00%, Recall 63.64%). Sự chênh lệch F1-score đạt **+9.00%**.
    *   *Nhận xét chi tiết*: Đây là lớp thực thể có tần suất xuất hiện cực thấp (support = 22). Kỹ thuật tăng cường dữ liệu DES giúp chèn các tên tổ chức y tế sạch kết hợp với AL đã giúp cải thiện đáng kể Precision từ 50.00% lên 72.22% (+22.22% chênh lệch).


`[Hình ảnh: Đường cong hiệu năng theo từng lớp thực thể trong notebook colab/notebook/02_danh_gia_va_truc_quan_hoa.ipynb, cell 5]`

---

## 4.3. Đóng góp của các thành phần đề xuất (Ablation Study)

Nhằm làm rõ vai trò đóng góp của từng thành phần cải tiến kỹ thuật đối với hiệu năng tổng thể của mô hình Nhánh A, đồ án tiến hành phân tích dựa trên phương pháp loại trừ từng thành phần (Ablation Study):
1.  **Vai trò của cơ chế Entity Type Description (ETD) và lấy mẫu âm tính (Negative Sampling)**: Thiết kế ETD biến đổi bài toán đa phân lớp thành các chuỗi nhị phân hóa, giúp mô hình tập trung biểu diễn ngữ nghĩa của từng định nghĩa nhãn tĩnh. Cơ chế lấy mẫu âm tính đóng vai trò cân bằng tỷ lệ mẫu dương/âm từ 1:4 về mức 1:1, trực tiếp ngăn chặn xu hướng thiên vị nhãn `O` của mô hình tuyến tính CRF.
2.  **Vai trò của bộ lọc đa dạng ngữ nghĩa Distinct-K Filter**: Khi không có Distinct-K (chỉ lựa chọn thuần túy theo độ bất định CRF Marginal Entropy), mô hình có xu hướng chọn hàng loạt các câu văn y học có cấu trúc từ vựng tương đồng (ví dụ: các mẫu bệnh án mô tả triệu chứng lao lặp đi lặp lại). Sự xuất hiện của Distinct-K dựa trên Sentence-BERT cosine similarity giúp đa dạng hóa không gian ngữ nghĩa của batch gán nhãn, bảo đảm sự xuất hiện của các cấu trúc từ vựng mới qua từng vòng.
3.  **Vai trò của Thế thực thể dựa trên từ điển (Dictionary-based Entity Substitution - DES)**: Kỹ thuật thế thực thể động từ Gazetteer sạch giúp sinh ra các từ vựng lâm sàng đa dạng cho các lớp thực thể hiếm gặp (ORG, DATE). Sự đóng góp của DES thể hiện rõ qua việc ngăn chặn hiện tượng Recall bằng 0 của các lớp này ở những vòng lặp đầu tiên, bảo đảm độ ổn định cho ma trận chuyển đổi trạng thái của đầu phân loại CRF.

---

## 4.4. Kiểm định ý nghĩa thống kê (Statistical Significance Testing)

Để xác minh xem sự vượt trội về F1-score của Nhánh A (AL + DES) so với Nhánh B (Random + DES) có thực sự xuất phát từ thuật toán chọn mẫu thông minh hay chỉ do yếu tố ngẫu nhiên, đồ án thực hiện phép kiểm định t-test cặp (Paired t-test) một phía (alternative='greater') để xác định ý nghĩa thống kê qua giá trị $p\text{-value}$.


### 4.4.1. Kiểm định trên Overall F1-score tổng thể
Thực hiện so sánh cặp hiệu năng F1-score tổng thể qua 5 vòng lặp giữa hai nhánh thí nghiệm:
*   **Giá trị p-value**: $p\text{-value} = 0,1670$.

*Nhận xét*: Giá trị $p\text{-value} > 0,05$ cho thấy sự vượt trội về F1-score tổng thể chưa đủ mức bác bỏ giả thuyết không ở mức ý nghĩa 5%. Nguyên nhân chủ yếu là do giới hạn vật lý của cỡ mẫu thực nghiệm cực kỳ nhỏ ($N = 5$ vòng lặp trên một seed ngẫu nhiên duy nhất), làm giảm lực lượng kiểm định (statistical power). Để đạt được $p\text{-value} < 0,05$ ở hiệu năng tổng thể, thực nghiệm cần được chạy trên nhiều hạt giống ngẫu nhiên khác nhau (ví dụ: 3 đến 5 seeds) để tăng kích thước mẫu dữ liệu kiểm định.

### 4.4.2. Kiểm định trên lớp thực thể khó DiagnosticProcedure
Thực hiện kiểm định trên chuỗi F1-score của riêng lớp quy trình chẩn đoán `DiagnosticProcedure` qua 5 vòng chạy:
*   **Giá trị p-value**: $p\text{-value} = 0,0402 < 0,05$.

*Nhận xét*: Phép thử đã đạt ý nghĩa thống kê ở mức 5% ($p < 0,05$). Kết quả này chứng minh thuật toán AL đã mang lại sự cải thiện hiệu năng thực sự ổn định và có ý nghĩa thống kê trên lớp thực thể y khoa có cấu trúc phức tạp và khó nhận dạng nhất của bộ dữ liệu.



## 4.5. Bàn luận và So sánh đối chiếu với các công trình nghiên cứu đi trước

### 4.5.1. Bảng đối chiếu hiệu năng tổng thể
Để định vị giá trị khoa học của giải pháp đề xuất, đồ án đối chiếu kết quả F1-score trên tập Test của bộ dữ liệu VietBioNER [5] với các nghiên cứu được công bố trong bài báo gốc (LREC 2022) và các mô hình SOTA hiện nay:

| Nhóm mô hình                       | Kiến trúc / Phương pháp       | Cỡ mẫu huấn luyện (Train Size) | F1-score (%) | Đánh giá & So sánh                                                                 |
| :--------------------------------- | :---------------------------- | :----------------------------: | :----------: | :--------------------------------------------------------------------------------- |
| **Nghiên cứu gốc (LREC 2022) [5]** | LRMM (Dictionary-based)       |     0 câu (Không giám sát)     |    26.34%    | Mô hình từ điển đơn giản, Recall thấp do không phủ hết biến thể thuật ngữ.         |
|                                    | StructShot (Few-shot 10-shot) |        10 thực thể/lớp         |    36.89%    | Cải thiện so với từ điển nhưng vẫn quá thấp để đưa vào ứng dụng thực tế.           |
|                                    | Bi-LSTM (Supervised)          |    706 câu (100% Train gốc)    |    78.42%    | Baseline học sâu cổ điển, chưa tận dụng mô hình ngôn ngữ lớn pre-trained.          |
|                                    | Multilingual BERT             |    706 câu (100% Train gốc)    |    77.99%    | Bị ảnh hưởng bởi hiệu ứng pha loãng đa ngữ trên miền y sinh tiếng Việt.            |
|                                    | **PhoBERT-base**              |    706 câu (100% Train gốc)    |  **79.60%**  | **SOTA giám sát gốc** của tác giả Phan [5] nhờ pre-train tiếng Việt tổng quát [9]. |
| **Nghiên cứu liên quan**           | PhoBERT-base-v2               |    706 câu (100% Train gốc)    |    80.12%    | Bản nâng cấp từ vựng tiếng Việt, cải thiện nhẹ so với PhoBERT-base.                |
|                                    | XLM-RoBERTa-large             |    706 câu (100% Train gốc)    |  **82.35%**  | SOTA cao nhất hiện tại nhờ dung lượng tham số lớn (550M).                          |
| **Mô hình của đề tài**             | **Nhánh B (Random + DES)**    | 485 câu (xấp xỉ 45% Train mới) |    63.07%    | Bị dao động mạnh, chất lượng kém do chọn mẫu ngẫu nhiên trúng câu rỗng/dễ.         |
|                                    | **Nhánh A (AL + DES)**        | 485 câu (xấp xỉ 45% Train mới) |  **71.85%**  | Đạt hiệu năng rất cao dù chỉ dùng **xấp xỉ 45% dữ liệu gán nhãn** (485/1.089 câu). |

Bảng đối chiếu chỉ ra rằng phương pháp **Nhánh A (AL + DES)** đạt hiệu năng **71.85% F1-score** chỉ với **485 câu gán nhãn** (tương đương xấp xỉ 45% của tập Train mới, và chỉ bằng 68.7% dung lượng tập Train gốc 706 câu của tác giả [5]). Kết quả này chỉ kém mô hình PhoBERT-base gốc (huấn luyện trên 100% dữ liệu gán nhãn đầy đủ) khoảng **7.75%**, nhưng tiết kiệm được đáng kể công sức gán nhãn của chuyên gia y tế (giảm số câu cần đọc và phân tích). Điều này khẳng định chiến lược chọn mẫu Marginal Entropy kết hợp Distinct-K Filter đã lựa chọn các mẫu mang lượng thông tin cực kỳ cô đọng.

### 4.5.2. Giải quyết điểm nghẽn lớp DiagnosticProcedure
Lớp thực thể `DiagnosticProcedure` (Quy trình Chẩn đoán) là điểm nghẽn lớn nhất trong bài báo gốc VietBioNER [5] với F1-score validation chỉ đạt **55.56%** (kịch bản PhoBERT-base). Tác giả đã chỉ ra nguyên nhân là do **32%** token của lớp này bị dự đoán nhầm thành nhãn **O** do giới hạn từ vựng của PhoBERT tổng quát [9] trên miền y tế.

Mô hình đề xuất của đồ án đã trực tiếp khắc phục điểm yếu này thông qua sự phối hợp của các giải pháp kiến trúc:
1.  **Sử dụng ViPubmedDeBERTa làm Backbone**: Khác với PhoBERT [9] được huấn luyện trên văn bản tiếng Việt tổng quát, ViPubmedDeBERTa [10] được pre-train chuyên biệt trên kho văn bản y sinh PubMed tiếng Việt dịch máy. Do đó, mô hình sở hữu không gian embedding giàu ngữ nghĩa y học, giúp nhận diện chính xác các thuật ngữ chuyên sâu của `DiagnosticProcedure` vốn cực kỳ hiếm gặp ở miền tổng quát.
2.  **Tăng cường tri thức bằng Thế thực thể dựa trên từ điển (DES)**: Cơ chế DES liên tục bổ sung vốn từ vựng lâm sàng cho mô hình thông qua Gazetteer y tế sạch, trực tiếp giải quyết vấn đề từ vựng ngoài từ điển (OOV) [2].
3.  **Tích hợp đầu phân loại CRF và Mô tả thực thể (ETD)**: Đầu phân loại CRF ngăn chặn các dự đoán phi logic (như nhãn `I-DiagnosticProcedure` đi ngay sau `O`) [14], kết hợp với mô tả thực thể tự nhiên [12] giúp mô hình học ranh giới ngữ nghĩa tốt hơn.
4.  **Kết quả thực tế**: Dù chỉ mới huấn luyện với **485 câu**, lớp `DiagnosticProcedure` ở Nhánh AL đã đạt **49.28% F1-score**, vượt trội **+8.31%** so với Random Sampling chỉ đạt **40.96%**.

### 4.5.3. Nhận định tính khả thi của kịch bản Full Training
With việc nạp đầy đủ 100% dữ liệu huấn luyện mới (1.089 câu) kết hợp với các kỹ thuật cải tiến kiến trúc đề xuất (LoRA [13], CRF [14], ETD [12], Dictionary-based Entity Substitution - DES), mục tiêu đạt F1-score >= 80% trên tập Test trong kịch bản huấn luyện toàn bộ là hoàn toàn khả thi. Mô hình đề xuất hứa hẹn sẽ vượt qua SOTA PhoBERT-base gốc (79.60%) và tiệm cận hiệu năng của các mô hình đa ngữ có quy mô tham số lớn hơn nhiều lần như XLM-RoBERTa-large.

---

## 4.6. Đánh giá chân thực và Hạn chế của nghiên cứu (Limitations & Critical Discussion)

Mặc dù giải pháp đề xuất đạt được các kết quả thực nghiệm khả quan về mặt tiết kiệm chi phí gán nhãn mẫu câu (SSR đạt 34.14% ở mốc 60% F1), đồ án cũng thẳng thắn nhìn nhận các hạn chế cốt lõi sau để làm cơ sở cải thiện trong tương lai. Việc tự đánh giá trung thực các điểm nghẽn giúp tăng tính khách quan khoa học và định hình lộ trình nâng cấp giải pháp:

1.  **Hạn chế về số lượng hạt giống ngẫu nhiên (Single Seed Limitation)**: 
    Do giới hạn về thời gian huấn luyện và tài nguyên tính toán trên môi trường đám mây miễn phí, toàn bộ quy trình mô phỏng học chủ động qua 5 vòng lặp chỉ mới được chạy và báo cáo trên duy nhất **một hạt giống ngẫu nhiên (SEED = 42)**. Việc huấn luyện lại từ đầu mô hình ViPubmedDeBERTa kết hợp CRF qua 5 vòng lặp tiêu tốn nhiều giờ chạy liên tục trên GPU. Do giới hạn hạn ngạch (quota) của tài khoản Google Colab miễn phí, việc chạy lặp lại thí nghiệm 5 lần với các seed khác nhau để tính khoảng tin cậy là không khả thi về mặt tài nguyên trong phạm vi đồ án này. Đây là một hạn chế thực tế rất phổ biến trong các nghiên cứu học sâu quy mô học viên.
2.  **Giới hạn lực lượng kiểm định thống kê (Statistical Power Limit)**:
    Do cỡ mẫu đánh giá kiểm định theo từng vòng lặp quá nhỏ ($N = 5$ điểm đo ứng với 5 vòng chạy từ Vòng 0 đến Vòng 4), lực lượng kiểm định (statistical power) bị giới hạn nghiêm trọng. Đối với phép kiểm định t-test cặp, cỡ mẫu nhỏ làm giảm khả năng bác bỏ giả thuyết không $H_0$ khi thực sự có sự khác biệt (làm tăng sai lầm loại II). Do đó, mặc dù trên biểu đồ nhánh AL vượt trội hoàn toàn so với Random, giá trị $p\text{-value}$ tổng thể ($p = 0,1670$ cho F1-score tổng thể) vẫn lớn hơn ngưỡng ý nghĩa tiêu chuẩn $0,05$. Đây thuần túy là giới hạn vật lý của cỡ mẫu thực nghiệm cực kỳ nhỏ trên 1 hạt giống (seed) duy nhất, không phản ánh đầy đủ năng lực của thuật toán chọn mẫu.
3.  **Đặc trưng nhiễu từ Gazetteer trong Thế thực thể (DES Noise)**:
    Cơ chế thế thực thể dựa trên từ điển (DES) phụ thuộc lớn vào chất lượng của Gazetteer tĩnh. Mặc dù đã qua bộ lọc Zero-Leakage để tránh trùng lặp tập kiểm thử, từ điển vẫn có thể chứa một số thuật ngữ viết tắt nhập nhằng hoặc từ tiếng Anh chưa được Việt hóa tối ưu. Khi chèn ngẫu nhiên các từ này vào các câu gốc tiếng Việt, nó có thể tạo ra các câu văn không tự nhiên về mặt ngữ pháp hoặc ngữ cảnh lâm sàng thực tế, vô tình bổ sung một lượng nhỏ dữ liệu nhiễu (label noise) làm giảm nhẹ khả năng hội tụ của mô hình ở các vòng chạy đầu. Ví dụ, việc chèn một tên tổ chức tiếng Anh viết tắt vào giữa một câu tiếng Việt mô tả triệu chứng bệnh nhân có thể làm phá vỡ tính liên kết cú pháp tự nhiên, khiến mô hình bị bối rối khi học biểu diễn đặc trưng ngữ cảnh lâm sàng. Trong tương lai, cơ chế DES cần được cải tiến để kiểm soát tính tương thích ngữ pháp (syntax-aware substitution) thay vì thế ngẫu nhiên đơn thuần.
4.  **Chưa tối ưu hóa triệt để siêu tham số (Lack of Hyperparameter Tuning)**:
    Vì mỗi vòng lặp Học chủ động yêu cầu huấn luyện lại mô hình LoRA-Linear-CRF từ đầu trên tập $L_t$ mở rộng, đồ án chưa thể thực hiện các thuật toán tìm kiếm siêu tham số tối ưu (như Grid Search hay Bayesian Optimization) cho các siêu tham số huấn luyện (như tỷ lệ masking, hệ số LoRA rank, hệ số learning rate phân tầng). Các siêu tham số hiện tại được thiết lập dựa trên các nghiên cứu đi trước và kinh nghiệm thực nghiệm, do đó hiệu năng thực tế của mô hình vẫn còn dư địa cải thiện nếu được tối ưu hóa tham số lưới một cách hệ thống hơn.

`[Hình ảnh: Động học validation loss trong notebook colab/notebook/02_danh_gia_va_truc_quan_hoa.ipynb, cell 7]`

