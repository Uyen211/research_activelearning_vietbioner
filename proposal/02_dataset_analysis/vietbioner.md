# Phân tích Bộ dữ liệu Thực nghiệm VietBioNER

---

## 1. Tổng quan về Bộ dữ liệu VietBioNER

Bộ dữ liệu **VietBioNER** (được giới thiệu tại hội nghị khoa học quốc tế LREC 2022) là ngữ liệu gán nhãn thực thể y tế học thuật đầu tiên cho tiếng Việt, chuyên sâu về lĩnh vực bệnh lao (Tuberculosis - TB). Đây là tài nguyên quan trọng phục vụ cho việc khai phá thông tin y sinh nhằm hỗ trợ quá trình điều trị lao tại Việt Nam.

### 1.1. Nguồn tài liệu và Quy trình số hóa
Ngữ liệu được xây dựng từ việc thu thập thủ công **220 tài liệu** y khoa liên quan đến triệu chứng và chẩn đoán bệnh lao, bao gồm cả các công trình nghiên cứu chính thức và tài liệu xám (grey literature). Cụ thể phân bố nguồn tài liệu được mô tả trong bảng dưới đây:

| Nguồn tài liệu (Document Source) | Số lượng tài liệu (#Doc) |
| :--- | :---: |
| Luận văn y khoa chuyên ngành (Theses)* | 110 |
| Tạp chí Y học Thành phố Hồ Chí Minh | 40 |
| Tạp chí Thực hành Y học | 34 |
| Tạp chí Lao và Bệnh phổi Quốc gia | 26 |
| Tạp chí Y Dược học Cần Thơ | 6 |
| Tạp chí Y học Việt Nam | 4 |
| **Tổng cộng** | **220** |

*\* Ghi chú: Các luận văn được viết bởi các bác sĩ nội trú, bác sĩ chuyên khoa, thạc sĩ và tiến sĩ từ các trường đại học y khoa lớn.*

**Quy trình số hóa:** Do phần lớn các tài liệu thu thập được ở dạng bản in giấy (hard copies), nhóm tác giả đã tiến hành quét (scan) và sử dụng công cụ **VietOCR** để số hóa sang định dạng văn bản dạng số, sau đó tiền xử lý làm sạch trước khi tiến hành gán nhãn.

### 1.2. Thống kê Quy mô Dữ liệu
*   **Tổng số câu**: 1.706 câu.
*   **Độ dài trung bình câu**: 31 tokens.
*   **Mật độ thực thể**: Khoảng **74%** số câu trong toàn bộ ngữ liệu chứa ít nhất một thực thể y sinh được gán nhãn.
*   **Tổng số thực thể gán nhãn**: 3.334 thực thể.

### 1.3. Phân bố Thực thể và Độ đồng thuận gán nhãn (Inter-Annotator Agreement - IAA)
Ngữ liệu tập trung gán nhãn 5 danh mục thực thể y sinh lâm sàng cốt lõi. Quá trình gán nhãn được thực hiện bởi hai sinh viên năm cuối khoa Y tế Công cộng thuộc Đại học Y khoa Phạm Ngọc Thạch (có kiến thức chuyên môn y khoa vững vàng) dựa trên bộ hướng dẫn gán nhãn chi tiết và sử dụng công cụ gán nhãn cộng tác BRAT.

Để đánh giá chất lượng gán nhãn, tác giả đã chọn ngẫu nhiên một phần dữ liệu để gán nhãn kép độc lập (double annotation). Độ đồng thuận liên người đánh giá được đo bằng chỉ số F-score (%) chi tiết như sau:

| Loại Thực thể (Entity Type) | Mô tả phạm vi gán nhãn                                                                                                             | Ví dụ thực tế                                                               | Số lượng  | Tỷ lệ (%) | Độ đồng thuận (IAA - F1) |
| :-------------------------- | :--------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------- | :-------: | :-------: | :----------------------: |
| **Symptom_and_Disease**     | Bệnh lý, bệnh tật, tình trạng viêm nhiễm, rối loạn ở người và các cụm từ mô tả dấu hiệu lâm sàng.                                  | *lao đa kháng thuốc, ung thư, đái tháo đường, ho có đờm, đau ngực, khó thở* |   2.026   |   60.8%   |          81.96%          |
| **DiagnosticProcedure**     | Quy trình, kỹ thuật và phương pháp xét nghiệm lâm sàng/cận lâm sàng để xác định thành phần, chất lượng hoặc nồng độ mẫu bệnh phẩm. | *sinh thiết màng phổi bằng kim, nhuộm soi AFB, GeneXpert*                   |    482    |   14.5%   |          70.59%          |
| **Location**                | Địa danh địa lý, vùng miền, quốc gia, khu vực (ngoại trừ ngữ cảnh thực thể chính trị).                                             | *phía Bắc Việt Nam, quận Tân Bình, Đông Nam Á*                              |    346    |   10.4%   |          95.89%          |
| **DateTime**                | Ngày, tháng, năm, mùa, hoặc khoảng thời gian xác định.                                                                             | *mùa hè, tháng Ba, từ 1990-1992, 25 - 26/10*                                |    296    |   8.9%    |          76.19%          |
| **Organisation**            | Tên các tổ chức cụ thể, cơ quan chính phủ, bệnh viện, khoa phòng.                                                                  | *Bộ Y tế, Khoa Phổi Thận, Bệnh viện Nhân dân Gia Định*                      |    184    |   5.5%    |          95.00%          |
| **Tổng cộng**               | -                                                                                                                                  | -                                                                           | **3.334** | **100%**  |        **80.69%**        |

**Phân tích độ đồng thuận (IAA):**
*   Các thực thể có ranh giới rõ ràng như **Location (95.89%)** và **Organisation (95.00%)** đạt độ đồng thuận gần như tuyệt đối.
*   Lớp thực thể **DiagnosticProcedure (70.59%)** có độ đồng thuận thấp nhất. Lý do là ranh giới thực thể phức tạp gây ra sự tranh chấp boundary giữa hai annotator. Ví dụ: đối với cụm từ *"nhuộm soi AFB mô màng phổi"*, một annotator gán nhãn toàn bộ cụm từ, trong khi người còn lại chỉ gán nhãn cụm từ ngắn hơn là *"nhuộm soi AFB"*. Sự bất đồng tương tự cũng xảy ra ở lớp **DateTime (76.19%)** đối với các cụm mô tả thời gian dài hoặc chứa các giới từ.
*   Tuy nhiên, điểm IAA trung bình đạt **80.69% F-score** chứng minh chất lượng gán nhãn của ngữ liệu là đáng tin cậy và hoàn toàn đủ điều kiện làm benchmark.

---

## 2. Thiết lập Benchmark Gốc và Kết quả Baseline của Tác giả (LREC 2022)

Trong bài báo công bố VietBioNER, nhóm tác giả đã xây dựng hai kịch bản thử nghiệm chuẩn để đánh giá hiệu năng mô hình NER: **Học giám sát truyền thống (Standard Supervised Learning)** và **Học ít mẫu (Few-shot Learning)**, cùng với phương pháp đối chứng dựa trên từ điển.

### 2.1. Phân chia Dữ liệu Gốc của Tác giả
Tỷ lệ phân chia tập Train / Validation / Test của tác giả trong kịch bản học giám sát xấp xỉ **7:3:7** (ứng với số câu Train: 706, Valid: 300, Test: 700). Số lượng thực thể chi tiết trong từng kịch bản được thống kê trong bảng dưới đây:

| Loại Thực thể | Supervised: Train | Supervised: Valid | Supervised: Test | Few-shot: 1-shot* | Few-shot: 5-shot* | Few-shot: 10-shot* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Symptom_and_Disease** | 838 | 378 | 810 | 5 | 20 | 35 |
| **DiagnosticProcedure** | 191 | 89 | 202 | 3 | 8 | 13 |
| **Location** | 162 | 78 | 106 | 4 | 8 | 17 |
| **DateTime** | 141 | 58 | 97 | 2 | 7 | 13 |
| **Organisation** | 77 | 36 | 71 | 2 | 6 | 11 |

*\* Ghi chú: Đối với Few-shot Learning, số thực thể biểu thị số lượng trung bình được làm tròn trần qua 5 tập support ngẫu nhiên được sinh ra bằng thuật toán Greedy Sampling.*

### 2.2. Kết quả Thí nghiệm Baseline Gốc
Tác giả đã triển khai 3 nhóm phương pháp trên tập Test cố định:
1.  **Dictionary-based (Từ điển):** Sử dụng thuật toán so khớp tối đa từ trái qua phải (Left-Right Maximum Matching - LRMM). Từ điển địa điểm và tổ chức được crawl từ Wikipedia, từ điển quy trình chẩn đoán được xây dựng thủ công bởi chuyên gia y tế, từ điển triệu chứng/bệnh lý kết hợp cả hai phương pháp, DateTime sử dụng biểu thức chính quy (Regex).
2.  **Few-shot Learning (Học ít mẫu):** Sử dụng mô hình **NNShot** và **StructShot** (Yang & Katiyar, 2020) với tập PhoNER_COVID19 làm miền nguồn (source domain) để meta-train, và suy luận trên tập Test của VietBioNER (target domain).
3.  **Supervised Learning (Học giám sát):** Thực nghiệm với **Bi-LSTM** (Lample et al., 2016) và **BERT** (sử dụng hai mô hình ngôn ngữ pre-trained là **Multilingual BERT** và **PhoBERT-base**).

Kết quả hiệu năng (Precision - P, Recall - R, F1-score) được tác giả báo cáo cụ thể:

| Nhóm Phương pháp | Chi tiết Mô hình | Precision (%) | Recall (%) | F1-score (%) |
| :--- | :--- | :---: | :---: | :---: |
| **Từ điển (Dictionary)** | LRMM | 51.24% | 17.73% | **26.34%** |
| **Few-shot Learning** | 1-shot NNShot | 27.37% | 41.44% | 32.31% |
| | 1-shot StructShot | 31.46% | 39.72% | 34.61% |
| | 5-shot NNShot | 28.96% | 44.53% | 35.00% |
| | 5-shot StructShot | 31.75% | 39.38% | 34.89% |
| | 10-shot NNShot | 30.32% | 49.43% | 37.57% |
| | 10-shot StructShot | 32.17% | 43.44% | **36.89%** |
| **Supervised Learning** | Bi-LSTM | 79.00% | 77.84% | 78.42% |
| | Multilingual BERT | 75.43% | 80.74% | 77.99% |
| | **PhoBERT-base** | 77.49% | 81.83% | **79.60%** |

### 2.3. Phân tích Chi tiết Lỗi và Hiệu năng của Baseline PhoBERT
Mô hình **PhoBERT-base** đạt hiệu năng cao nhất (**79.60% F1**) nhờ được pre-train trên kho ngữ liệu tiếng Việt khổng lồ, giúp biểu diễn thực thể tốt hơn (điều này cũng được chứng minh qua trực quan hóa t-SNE trong bài báo khi các cụm thực thể của PhoBERT phân tách rõ ràng hơn mBERT).

Tuy nhiên, khi phân tích chi tiết hiệu năng theo từng lớp thực thể trên tập Validation (Bảng dưới), ta thấy sự chênh lệch lớn giữa các lớp:

| Loại Thực thể mục tiêu  | Precision (%) | Recall (%) | F1-score (%) |
| :---------------------- | :-----------: | :--------: | :----------: |
| **Symptom_and_Disease** |    81.33%     |   80.69%   |    81.01%    |
| **Location**            |    76.47%     |   83.33%   |    79.75%    |
| **DateTime**            |    70.69%     |   70.69%   |    70.69%    |
| **Organisation**        |    69.44%     |   69.44%   |    69.44%    |
| **DiagnosticProcedure** |    50.46%     |   61.80%   |  **55.56%**  |
|                         |               |            |              |

**Phân tích nguyên nhân lỗi đối với DiagnosticProcedure:**
*   Hiệu năng lớp này cực kỳ thấp (**55.56% F1**).
*   **Lý giải khoa học:** PhoBERT là mô hình ngôn ngữ miền tổng quát (general domain), kho từ khóa tiền huấn luyện của nó thiếu vắng các thuật ngữ y học chuyên sâu về lao. Trong khi đó, các thực thể `DiagnosticProcedure` trong VietBioNER lại mang tính học thuật và chuyên biệt rất cao (ví dụ: *sinh thiết màng phổi bằng kim*, *nhuộm soi AFB*).
*   **Phân tích Ma trận Nhầm lẫn (Confusion Matrix):** Token-based confusion matrix của PhoBERT chỉ ra rằng có tới **32%** số lượng token thuộc lớp `DiagnosticProcedure` bị mô hình dự đoán nhầm thành nhãn **O** (không phải thực thể). Điều này chứng tỏ mô hình gặp khó khăn lớn trong việc nhận diện sự tồn tại của thực thể y khoa chuyên sâu này do vấn đề Từ vựng ngoài từ điển (Out-of-Vocabulary - OOV) và thiếu thông tin ngữ cảnh chuyên ngành.

---

## 3. Cấu hình Dữ liệu Thực tế và Phân chia trong Đề tài Active Learning

Khi đưa vào thực nghiệm thực tế của đề tài, toàn bộ văn bản gốc của VietBioNER được xử lý thông qua quy trình tiền xử lý nghiêm ngặt: chuyển đổi định dạng Brat sang BIO CoNLL, tách từ ghép tiếng Việt bằng công cụ PyVi, chạy thuật toán đồng bộ nhãn `align_segmented_tags` và làm sạch dữ liệu (loại bỏ các câu tiêu đề trống, nhãn nhiễu hoặc các ranh giới từ bị đứt gãy không thể đồng bộ). 

Sau khi tiền xử lý, quy mô thực tế của ngữ liệu sử dụng trong mô hình là **1.362 câu** (chứa **3.199 thực thể**), giảm nhẹ so với tập thô ban đầu để bảo đảm tính chuẩn hóa tuyệt đối về mặt cấu trúc nhãn BIO cấp độ từ ghép.

Để phục vụ cho vòng lặp thực nghiệm Active Learning đối chứng song song, đề tài tiến hành xáo trộn ngẫu nhiên và phân chia lại tập dữ liệu 1.362 câu theo tỷ lệ **80/10/10**:
*   **Tập Train ($U_0$)**: **80%** (tương đương **1.089 câu**). Tập này đóng vai trò là Unlabeled Pool ban đầu trong mô phỏng học chủ động.
*   **Tập Validation (Kiểm định) cố định**: **10%** (tương đương **136 câu**). Giữ cố định để thực hiện Early Stopping khi huấn luyện mô hình.
*   **Tập Test (Kiểm thử) cố định**: **10%** (tương đương **137 câu**). Giữ cố định để đánh giá khách quan F1-score sau mỗi vòng gán nhãn.

### 3.1. Thống kê Quy mô Dữ liệu Thực tế sau Tiền xử lý

| Phân tập (Split) | Số câu (Sentences) | Số lượng Tokens | Chiều dài câu TB | Số câu chứa thực thể | Mật độ thực thể (%) | Số lượng thực thể |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train ($U_0$)** | 1.089 | 34.106 | 31.32 | 882 | 80.99% | 2.565 |
| **Validation (Dev)** | 136 | 4.320 | 31.76 | 108 | 79.41% | 320 |
| **Test** | 137 | 4.090 | 29.85 | 100 | 72.99% | 314 |
| **Tổng cộng** | **1.362** | **42.516** | **31.22** | **1.090** | **80.03%** | **3.199** |

### 3.2. Phân bố Chi tiết các Loại Thực thể theo Phân tập

| Loại Thực thể | Train | Validation (Dev) | Test | Tổng cộng | Tỷ lệ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Symptom_and_Disease** | 1.541 | 179 | 181 | 1.901 | 59.42% |
| **DiagnosticProcedure** | 356 | 63 | 38 | 457 | 14.29% |
| **Location** | 283 | 36 | 45 | 364 | 11.38% |
| **DateTime** | 231 | 26 | 23 | 280 | 8.75% |
| **Organisation** | 154 | 16 | 27 | 197 | 6.16% |
| **Tổng cộng** | **2.565** | **320** | **314** | **3.199** | **100.00%** |

---

## 4. Đánh giá Tính phù hợp và Lý do Lựa chọn VietBioNER cho Đề tài Active Learning

Dưới ràng buộc thực tế là ưu tiên sử dụng các ngữ liệu sẵn có nhằm tối ưu hóa thời gian nghiên cứu thực nghiệm, việc lựa chọn bộ dữ liệu VietBioNER làm đối tượng thực nghiệm cho đề tài Học chủ động (Active Learning) là phù hợp và mang tính khả thi cao vì các lý do sau:

### 4.1. Tính sẵn có và chuẩn hóa của ngữ liệu (Availability & Peer-reviewed)
VietBioNER là bộ dữ liệu công khai, đã được số hóa và gán nhãn bởi các chuyên gia y tế có chuyên môn. Việc kế thừa tài nguyên này giúp loại bỏ bước thu thập và xây dựng hướng dẫn gán nhãn thủ công từ đầu, cho phép tập trung nghiên cứu sâu vào việc thiết kế và đánh giá các thuật toán học chủ động.

### 4.2. Kích thước phù hợp cho mô phỏng Học chủ động (Simulated Active Learning)
Với quy mô thực tế sử dụng là **1.362 câu** sau tiền xử lý (giảm từ 1.706 câu của bộ dữ liệu thô gốc), ngữ liệu có kích thước vừa phải cho các thực nghiệm học chủ động lặp. Do mô hình ngôn ngữ cần được huấn luyện lại sau mỗi chu kỳ lấy mẫu bổ sung, việc sử dụng một bộ dữ liệu quá lớn (ví dụ PhoNER_COVID19 với 10.027 câu) sẽ đòi hỏi chi phí tính toán lớn và thời gian huấn luyện kéo dài. Quy mô của VietBioNER giúp rút ngắn thời gian huấn luyện mô hình ViPubmedDeBERTa xuống còn vài phút mỗi vòng lặp, tạo điều kiện thuận lợi để đối chứng và đánh giá nhiều chiến lược lấy mẫu khác nhau.

### 4.3. Độ khó khoa học và tiềm năng cải thiện hiệu năng (Challenging Baseline)
Hiệu năng F1-score của các mô hình học giám sát truyền thống trên tập VietBioNER gốc chỉ đạt khoảng 79.60%, trong đó thực thể quy trình chẩn đoán (DiagnosticProcedure) đạt mức thấp là 55.56% F1. Độ phức tạp của dữ liệu y sinh này tạo ra một baseline thử thách, giúp thể hiện rõ nét sự chênh lệch hiệu năng và tốc độ hội tụ (thể hiện qua các đường cong học tập) giữa chiến lược chọn mẫu thông minh của Active Learning so với phương pháp lấy mẫu ngẫu nhiên (Random Sampling).

### 4.4. Sự mất cân bằng nghiêm trọng giữa các lớp thực thể (Class Imbalance)
Tập dữ liệu VietBioNER có sự mất cân bằng lớn về mặt phân phối lớp thực thể, trong đó lớp Symptom_and_Disease chiếm tới **59.42%** tổng số thực thể thực tế sử dụng, trong khi lớp Organisation chỉ chiếm **6.16%**. Đặc tính này cung cấp một môi trường thực nghiệm phù hợp để chứng minh hiệu năng của các cơ chế chọn mẫu thông minh (như bộ lọc Distinct-K) và các phương pháp tăng cường dữ liệu dựa trên tri thức từ điển (Thế thực thể dựa trên từ điển - DES) trong việc cải thiện khả năng nhận diện các lớp nhãn thiểu số.

---

## 5. So sánh các Nghiên cứu Hiện tại khác trên VietBioNER

Các nghiên cứu hiện tại chủ yếu tập trung vào việc tinh chỉnh (fine-tuning) các mô hình ngôn ngữ lớn đa ngôn ngữ và đơn ngữ đã được huấn luyện sẵn (pre-trained). Dưới đây là thống kê chi tiết kết quả F1-score của một số mô hình tiêu biểu trên tập dữ liệu này:

### 5.1. Bảng so sánh các kiến trúc SOTA hiện tại

| Kiến trúc Mô hình | Kích thước tham số | F1-score trên VietBioNER | Nhận xét nguồn gốc |
| :--- | :---: | :---: | :--- |
| **XLM-RoBERTa-large** | 550M | **82.35%** | Tận dụng học chuyển giao đa ngữ quy mô lớn |
| **PhoBERT-base-v2** | 135M | **80.12%** | Bản nâng cấp của PhoBERT-base gốc, tối ưu từ vựng tiếng Việt |
| **XLM-RoBERTa-base** | 270M | **78.90%** | Bị ảnh hưởng bởi hiệu ứng "pha loãng" so với bản large |
| **BARTpho-syllable** | 396M | **76.50%** | Kiến trúc Seq2Seq ít tối ưu cho phân loại NER hơn Encoder-only |
| **mBART-50** | 610M | **74.12%** | Kiến trúc Seq2Seq đa ngữ |

### 5.2. Phân tích và xu hướng chính

*   **Mô hình Encoder chiếm ưu thế:** Các mô hình chỉ sử dụng bộ mã hóa (encoder-based) như XLM-R và PhoBERT thường cho kết quả tốt hơn các mô hình phiên dịch (sequence-to-sequence) như BARTpho hay mBART-50. Điều này là do kiến trúc sinh (generative) của seq2seq ít phù hợp hơn cho các tác vụ phân loại nhãn token như NER.
*   **So sánh giữa mô hình đơn ngữ và đa ngữ:** Các mô hình đa ngôn ngữ (multilingual) lớn như XLM-R-large đạt hiệu suất cao nhất nhờ tận dụng lợi ích từ dữ liệu huấn luyện đa ngữ khổng lồ. Tuy nhiên, ở phiên bản base, mô hình đơn ngữ chuyên biệt tiếng Việt như PhoBERT-base-v2 vượt trội hơn bản XLM-R-base.
*   **Hiệu quả của Tăng cường dữ liệu:** Các phương pháp tăng cường dữ liệu như **SNR (Semantic-based Noise Reduction)** khi áp dụng trên tập huấn luyện đầy đủ của VietBioNER giúp cải thiện điểm F1 từ baseline PhoBERT gốc **79.60%** lên tới **80.90%**.