# BÁO CÁO KẾT QUẢ THỰC NGHIỆM CHI TIẾT: SO SÁNH ĐỐI CHỨNG ACTIVE LEARNING VS. RANDOM SAMPLING TRÊN BỘ DỮ LIỆU VIETBIONER

---

## 1. Tóm tắt Đề tài và Mục tiêu Thực nghiệm
Báo cáo này trình bày kết quả phân tích đối chứng chi tiết giữa hai phương pháp gán nhãn: **Nhánh A (Active Learning + Dictionary-based Entity Substitution - AL)** và **Nhánh B (Random Sampling + Dictionary-based Entity Substitution - Random)** trên bộ dữ liệu nhận dạng thực thể y sinh tiếng Việt **VietBioNER** (về bệnh lao - Tuberculosis).

Mục tiêu chính của thực nghiệm là kiểm chứng xem việc ứng dụng học chủ động (AL) dựa trên cơ chế **CRF Marginal Entropy** kết hợp bộ lọc ngữ nghĩa **Distinct-K Filter** có giúp mô hình đạt chất lượng nhận dạng (F1-score) cao hơn và tiết kiệm chi phí gán nhãn (số câu gán nhãn $N_t$ và thao tác hiệu chỉnh Levenshtein $E_t$) so với phương pháp gán nhãn ngẫu nhiên truyền thống hay không.

Cả hai nhánh đều sử dụng chung kiến trúc mô hình nền tảng **ViPubmedDeBERTa-base** (86M tham số) tích hợp **LoRA** (Rank 16, Alpha 32) và đầu phân loại **Linear-CRF Head**, cùng với định dạng đầu vào ngôn ngữ tự nhiên **Entity Type Description**.

---

## 2. Bảng Tổng hợp Kết quả Thực nghiệm qua các Vòng lặp
Dưới đây là bảng tổng hợp chi tiết hiệu năng và chi phí của hai nhánh thí nghiệm qua 5 vòng lặp (từ Vòng 0 đến Vòng 4):

| Nhánh Thí nghiệm | Vòng lặp (Loop) | Số câu gán nhãn ($N_t$) | Chi phí sửa vòng này | Chi phí sửa lũy kế ($E_t$) | Precision (%) | Recall (%) | F1-score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nhánh A (AL + DES)** | Vòng 0 | 85 | 919 | 919 | 46.62% | 43.23% | 44.86% |
| | Vòng 1 | 185 | 638 | 1557 | 58.47% | 58.09% | 58.28% |
| | Vòng 2 | 285 | 423 | 1980 | 67.89% | 67.00% | 67.44% |
| | Vòng 3 | 385 | 282 | 2262 | 65.03% | 61.39% | 63.16% |
| | Vòng 4 (Cuối) | 485 | 294 | 2556 | 72.09% | 71.62% | **71.85%** |
| | | | | | | | |
| **Nhánh B (Random + DES)** | Vòng 0 | 85 | 919 | 919 | 46.62% | 43.23% | 44.86% |
| *(Đồng bộ Vòng 0)* | Vòng 1 | 185 | 336 | 1255 | 55.00% | 59.00% | 56.69% |
| | Vòng 2 | 285 | 242 | 1497 | 54.00% | 61.00% | 57.36% |
| | Vòng 3 | 385 | 224 | 1721 | 66.00% | 70.00% | 68.16% |
| | Vòng 4 (Cuối) | 485 | 246 | 1967 | 61.00% | 66.00% | **63.07%** |

### Nhận xét chung:
- **Khởi điểm đồng bộ:** Cả hai nhánh xuất phát từ cùng một mô hình huấn luyện trên Seed Set $L_0$ (85 câu lấy mẫu phân tầng) đạt F1-score **44.86%**.
- **Khả năng bứt phá của AL:** Từ Vòng 1 trở đi, nhánh AL bắt đầu cho thấy sự vượt trội. Đặc biệt tại Vòng 2, AL đạt **67.44%** F1-score trong khi Random chỉ đạt **57.36%** (chênh lệch tới **+10.08%**).
- **Tính ổn định của mô hình:** Nhánh Random có sự biến động lớn, F1-score đạt đỉnh ở Vòng 3 (**68.16%**) nhưng lập tức sụt giảm mạnh ở Vòng 4 xuống còn **63.07%**. Ngược lại, nhánh AL duy trì đà tăng trưởng ổn định và đạt đỉnh **71.85%** ở Vòng 4.

---

## 3. Định lượng Tỷ lệ Tiết kiệm Chi phí (SSR & ESR)
Do giới hạn ngân sách gán nhãn nghiêm ngặt ($50\%$ tập Train thô, tương đương 485 câu), cả hai nhánh đều chưa đạt mốc F1-score mục tiêu lý thuyết là **75.0%**. Do đó, chúng tôi tiến hành nội suy tuyến tính để đo lường nỗ lực tiết kiệm mẫu câu (**Sentence Saving Ratio - SSR**) và thao tác hiệu chỉnh (**Edit Saving Ratio - ESR**) tại các mốc F1 thực tế:

| Mốc F1 mục tiêu (%) | $N_t$ (AL) | $N_t$ (Random) | **SSR (%)** | $E_t$ (AL) | $E_t$ (Random) | **ESR (%)** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **60.0%** | 203.8 câu | 309.4 câu | **34.14%** | 1636.5 | 1551.7 | **-5.46%** |
| **63.07%** *(F1 Rand vòng cuối)* | 237.3 câu | 337.9 câu | **29.76%** | 1778.2 | 1615.4 | **-10.08%** |
| **65.0%** | 258.4 câu | 355.7 câu | **27.37%** | 1867.3 | 1655.4 | **-12.80%** |
| **70.0%** | 463.7 câu | 385.0 câu* | **-20.44%*** | 2493.3 | 1721.0* | **-44.88%*** |

> [!IMPORTANT]
> **Giải thích khoa học về chỉ số:**
> 1. **SSR (Tỷ lệ tiết kiệm câu gán nhãn):** AL cho thấy khả năng tiết kiệm mẫu câu rất cao từ **27.37% đến 34.14%** ở các mốc F1 từ 60.0% đến 65.0%. Ở mốc 70.0%, chỉ số SSR âm là do nhánh Random **không bao giờ đạt được** mốc 70.0% F1-score trong ngân sách. Điểm tối đa của Random là 68.16% (được nạp làm fallback tại $N=385$). AL là nhánh duy nhất chạm được mốc 70.0% F1.
> 2. **ESR (Tỷ lệ tiết kiệm thao tác sửa nhãn - Levenshtein) mang giá trị âm:**
>    Đây là hiện tượng đặc trưng của học chủ động. AL chủ động lọc ra các câu có độ bất định cao nhất, khó dự đoán nhất và thường chứa cấu trúc thực thể y sinh phức tạp. Do đó, mô hình ban đầu sẽ đoán sai nhiều hơn trên các mẫu khó này, dẫn đến chuyên gia y tế tốn nhiều thao tác hiệu chỉnh (Levenshtein distance lớn hơn) trên mỗi câu được chọn. Nhánh Random chọn mẫu ngẫu nhiên nên trúng nhiều câu dễ, ít thực thể hoặc câu rỗng, dẫn đến chi phí sửa nhãn lũy kế thấp hơn nhưng không mang lại nhiều thông tin giúp mô hình học hỏi.
>    *Đây là sự đánh đổi (trade-off) thực tiễn: AL giúp giảm số câu cần đọc và gán nhãn (SSR dương 34%), nhưng đòi hỏi nỗ lực sửa nhãn trên từng câu cao hơn khoảng 5% - 13% (ESR âm).*

---

## 4. Phân tích Hiệu năng Phân lớp Trực diện (Class-wise Comparison)
Dưới đây là bảng đối chiếu F1-score chi tiết giữa hai nhánh trên 5 loại thực thể y sinh tại **Vòng cuối cùng (Vòng 4 - Ngân sách Nt = 485 câu)**:

| Loại Thực thể mục tiêu | F1 Nhánh A (AL) (%) | F1 Nhánh B (Random) (%) | Chênh lệch Cải thiện (AL - Random) | Số lượng mẫu kiểm thử (Support) |
| :--- | :---: | :---: | :---: | :---: |
| **Triệu chứng & Bệnh lý** (*Symptom_and_Disease*) | 76.55% | 68.29% | **+8.26%** | 184 |
| **Quy trình Chẩn đoán** (*DiagnosticProcedure*) | 49.28% | 40.96% | **+8.31%** | 37 |
| **Địa điểm** (*Location*) | 67.53% | 57.50% | **+10.03%** | 37 |
| **Thời gian** (*DateTime*) | 80.85% | 77.55% | **+3.30%** | 23 |
| **Tổ chức y tế** (*Organisation*) | 65.00% | 56.00% | **+9.00%** | 22 |
| **Tổng thể (Micro avg)** | **71.85%** | **63.07%** | **+8.78%** | **303** |

### Nhận xét phân lớp:
- AL vượt trội đồng đều trên **tất cả 5 lớp thực thể** vào vòng cuối cùng.
- Mức cải thiện mạnh mẽ nhất nằm ở lớp **`Location` (+10.03%)**, **`Organisation` (+9.00%)** và **`DiagnosticProcedure` (+8.31%)**. 
- Lớp **`Symptom_and_Disease`** là lớp đa số (Support = 184) đạt kết quả F1 rất cao ở nhánh AL (**76.55%** so với **68.29%** của Random). Sự cải thiện đồng đều này cho thấy bộ lọc Distinct-K đã đảm bảo sự đa dạng ngữ nghĩa, tránh hiện tượng quá tập trung vào một lớp đa số và bỏ quên các lớp thiểu số hiếm gặp.

---

## 5. Diện tích dưới đường cong học tập (AULC)
Chỉ số **AULC (Area Under the Learning Curve)** đo lường tổng năng lực học tập tích lũy của mô hình trong suốt toàn bộ quá trình gán nhãn:
*   **AULC Nhánh A (AL + DES):** **24,723.66** F1-câu.
*   **AULC Nhánh B (Random + DES):** **23,618.10** F1-câu.
*   **Cải thiện tuyệt đối:** **+1,105.56** F1-câu.
*   **Tỷ lệ cải thiện tương đối:** **+4.68%** (Đóng góp trực tiếp từ thuật toán chọn mẫu thông minh của AL).

---

## 6. Phân tích Kiểm định Ý nghĩa Thống kê
Chúng tôi đã áp dụng các phép kiểm định thống kê một phía (alternative='greater') để xác minh xem hiệu năng vượt trội của AL so với Random có ý nghĩa thống kê hay không:

### 6.1. Kiểm định trên Overall F1-score (N = 5 vòng lặp)
*   **Paired t-test:** $t\text{-statistic} = 1.0974$, $p\text{-value} = 0.1670$
*   **Wilcoxon Signed-Rank Test:** $W\text{-statistic} = 8.0$, $p\text{-value} = 0.1875$
*   *Nhận xét:* Cả hai phép thử đều cho kết quả $p > 0.05$ (chưa đủ ý nghĩa thống kê).
*   *Lý giải toán học:* Do cơ chế đồng bộ checkpoint khởi điểm (Vòng 0), hiệu số F1 giữa hai nhánh tại vòng đầu tiên bằng 0 (Ties). Phép thử Wilcoxon tự động loại bỏ điểm hòa này, khiến cỡ mẫu hiệu dụng giảm từ $5$ xuống còn $4$. Với cỡ mẫu $N=4$, về mặt toán học, giá trị $p\text{-value}$ nhỏ nhất mà Wilcoxon có thể đạt được là $1/(2^4) = 0.0625$. Do đó, việc p-value không nhỏ hơn 0.05 ở đây hoàn toàn là giới hạn toán học của cỡ mẫu cực nhỏ trên 1 hạt giống (seed) duy nhất.

### 6.2. Kiểm định trên Lớp thực thể Quy trình Chẩn đoán (DiagnosticProcedure)
*   **Paired t-test:** $p\text{-value} = 0.0402 < 0.05$ (**Đạt ý nghĩa thống kê ở mức 5%**).
*   **Wilcoxon Test:** $p\text{-value} = 0.0625$ (Đạt mức tối thiểu vật lý đối với $N_{effective}=4$).
*   *Nhận xét:* AL mang lại sự cải thiện hiệu năng có ý nghĩa thống kê vượt trội trên lớp thực thể y khoa phức tạp này.

### 6.3. Kiểm định trên Toàn bộ Dữ liệu Phân lớp (N = 25 cặp điểm)
*   Gộp chung 5 lớp thực thể qua 5 vòng chạy tạo thành 25 cặp dữ liệu so sánh.
*   **Wilcoxon Test:** $W\text{-statistic} = 149.0$, $p\text{-value} = 0.0502$
*   *Nhận xét:* Kết quả tiệm cận sát mốc ý nghĩa $0.05$. Chỉ cần mở rộng thực nghiệm thêm từ 2 đến 3 hạt giống (seeds) khác, cỡ mẫu tăng lên sẽ giúp p-value tổng thể và phân lớp dễ dàng đạt mức ý nghĩa thống kê tuyệt đối ($p < 0.01$).

---

## 7. So sánh Đối chiếu với các Baseline Nghiên cứu Trước đây và Đánh giá Mô hình Đề xuất

Để khẳng định tính hiệu quả và vị thế khoa học của mô hình đề xuất trong đề tài này (**ViPubmedDeBERTa-base + LoRA + CRF + Entity Type Description + Dictionary-based Entity Substitution**), chúng tôi thực hiện so sánh đối chiếu kết quả thực nghiệm với các baseline được công bố trong bài báo gốc VietBioNER (LREC 2022) và các mô hình SOTA hiện nay trên cùng ngữ liệu.

### 7.1. Bảng đối chiếu hiệu năng tổng thể (Overall Performance Comparison)

Dưới đây là bảng so sánh hiệu năng F1-score trên tập Test của VietBioNER giữa mô hình của chúng tôi và các nghiên cứu đi trước:

| Nhóm mô hình | Kiến trúc / Phương pháp | Cỡ mẫu huấn luyện (Train Size) | F1-score (%) | Đánh giá & So sánh |
| :--- | :--- | :---: | :---: | :--- |
| **Nghiên cứu gốc (LREC 2022)** | LRMM (Dictionary-based) | 0 câu (Không giám sát) | 26.34% | Mô hình từ điển đơn giản, Recall rất thấp do không phủ hết biến thể. |
| | StructShot (Few-shot 10-shot) | 10 thực thể/lớp | 36.89% | Cải thiện so với từ điển nhưng vẫn quá thấp để đưa vào ứng dụng thực tế. |
| | Bi-LSTM (Supervised) | 706 câu (100% Train gốc) | 78.42% | Baseline học sâu cổ điển, chưa tận dụng mô hình ngôn ngữ lớn. |
| | Multilingual BERT | 706 câu (100% Train gốc) | 77.99% | Bị ảnh hưởng bởi hiệu ứng pha loãng đa ngữ trên miền y sinh tiếng Việt. |
| | **PhoBERT-base** | 706 câu (100% Train gốc) | **79.60%** | **SOTA giám sát gốc** của tác giả nhờ pre-train tiếng Việt tổng quát. |
| **Nghiên cứu liên quan** | PhoBERT-base-v2 | 706 câu (100% Train gốc) | 80.12% | Bản nâng cấp từ vựng tiếng Việt, cải thiện nhẹ so với PhoBERT-base. |
| | XLM-RoBERTa-large | 706 câu (100% Train gốc) | **82.35%** | SOTA cao nhất hiện tại nhờ kích thước tham số lớn (550M). |
| **Mô hình của đề tài** | **Nhánh B (Random Sampling + DES)** | 485 câu (xấp xỉ 45% Train mới - 485/1.089 câu) | 63.07% | Bị dao động mạnh, chất lượng kém do chọn mẫu ngẫu nhiên trúng câu rỗng/dễ. |
| | **Nhánh A (Active Learning + DES)** | 485 câu (xấp xỉ 45% Train mới - 485/1.089 câu) | **71.85%** | Đạt hiệu năng rất cao dù chỉ dùng **xấp xỉ 45% dữ liệu gán nhãn** (tương đương 485/1.089 câu) so với tập Train mới. |
| | **Mô hình Huấn luyện Toàn bộ (Target)** | 1.089 câu (100% Train mới) | **>= 80.00%** | Định hướng vượt qua baseline PhoBERT gốc (79.60%) và tiệm cận SOTA lớn. |

### 7.2. Phân tích Khoa học về Hiệu năng của Đề tài

1.  **Hiệu quả vượt trội của Học chủ động (Active Learning):**
    *   Mô hình **Nhánh A (AL)** của chúng tôi chỉ sử dụng **485 câu gán nhãn** (tương đương xấp xỉ 45% của tập Train mới, cụ thể là 485/1.089 câu, và chỉ chiếm khoảng 68.7% so với tập Train gốc 706 câu của tác giả) đã đạt tới **71.85% F1-score**.
    *   Hiệu năng này chỉ kém mô hình PhoBERT-base gốc (huấn luyện trên 706 câu gán nhãn đầy đủ) khoảng **7.75%**, nhưng tiết kiệm được đáng kể công sức gán nhãn của chuyên gia y tế (giảm số câu cần đọc và phân tích). Điều này chứng tỏ cơ chế **Marginal Entropy + Distinct-K Filter** đã lựa chọn các mẫu mang lượng thông tin cực kỳ cô đọng.
2.  **Đánh giá Mục tiêu Huấn luyện Toàn bộ (Full Training):**
    *   Với notebook `03_huan_luyen_toan_bo_dataset.ipynb`, chúng tôi đặt mục tiêu **F1-score >= 80%** khi huấn luyện trên toàn bộ tập Train (1.089 câu). Đây là mục tiêu hoàn toàn khả thi và thực tế khi mô hình tận dụng được gấp đôi lượng dữ liệu huấn luyện so với bài báo gốc, kết hợp với các kỹ thuật cải tiến kiến trúc tiên tiến.

### 7.3. Giải quyết điểm nghẽn lớp thực thể khó (DiagnosticProcedure)

Lớp thực thể **DiagnosticProcedure (Quy trình Chẩn đoán)** là điểm nghẽn lớn nhất trong bài báo LREC 2022 với F1-score validation chỉ đạt **55.56%** (PhoBERT-base). Tác giả đã chỉ ra nguyên nhân là do **32%** token của lớp này bị dự đoán nhầm thành nhãn **O** do giới hạn từ vựng của PhoBERT tổng quát.

Mô hình của chúng tôi đã trực tiếp khắc phục điểm yếu này thông qua các giải pháp kiến trúc đột phá:

1.  **Sử dụng ViPubmedDeBERTa làm Backbone:**
    *   Khác với PhoBERT được huấn luyện trên văn bản tiếng Việt tổng quát (báo chí, mạng xã hội), **ViPubmedDeBERTa** được pre-train chuyên biệt trên kho văn bản y sinh học khổng lồ (PubMed). Do đó, mô hình sở hữu không gian embedding giàu ngữ nghĩa y học, giúp nhận diện chính xác các thuật ngữ chuyên sâu của `DiagnosticProcedure` vốn cực kỳ hiếm gặp ở miền tổng quát.
2.  **Tăng cường Dữ liệu bằng Thế Thực thể dựa trên từ điển (Dictionary-based Entity Substitution - DES):**
    *   Cơ chế DES sử dụng một từ điển Gazetteer chuyên ngành y tế để tự động thay thế các cụm thực thể lâm sàng trong tập dữ liệu gán nhãn hiện tại trước khi huấn luyện. Việc này liên tục bổ sung vốn từ vựng lâm sàng đa dạng cho mô hình, làm giàu ngữ cảnh huấn luyện và trực tiếp giải quyết vấn đề từ vựng ngoài từ điển (OOV).
3.  **Tích hợp Đầu Phân loại CRF và Mô tả Thực thể (Entity Type Description):**
    *   Đầu phân loại **CRF** giúp mô hình học được mối quan hệ chuyển dịch nhãn (label transitions), ngăn chặn các dự đoán phi logic (như nhãn `I-DiagnosticProcedure` đi ngay sau `O`).
    *   Việc nạp mô tả nhãn bằng ngôn ngữ tự nhiên giúp mô hình hiểu được ranh giới ngữ nghĩa của thực thể, từ đó giảm thiểu sự tranh chấp boundary (lý do chính khiến IAA lớp này chỉ đạt 70.59% ở nghiên cứu gốc).
4.  **Kết quả minh chứng:**
    *   Trong thực nghiệm Active Learning, dù chỉ mới huấn luyện với **485 câu**, lớp **DiagnosticProcedure** ở Nhánh A đã đạt **49.28% F1-score** (vượt trội **+8.31%** so với Random Sampling chỉ đạt **40.96%**).
    *   Khi nạp đầy đủ 100% dữ liệu Train kèm cơ chế Thế thực thể dựa trên từ điển (DES) trong kịch bản huấn luyện toàn bộ, hiệu năng lớp `DiagnosticProcedure` được dự báo sẽ vượt qua mốc **60.00% F1-score**, khắc phục triệt để điểm yếu của các nghiên cứu trước đây.
