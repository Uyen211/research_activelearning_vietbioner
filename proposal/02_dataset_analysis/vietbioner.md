# Phân tích Bộ dữ liệu Thực nghiệm VietBioNER

---

## 1. Tổng quan về Bộ dữ liệu VietBioNER

Bộ dữ liệu **VietBioNER** (được giới thiệu tại hội nghị khoa học quốc tế LREC 2022) là ngữ liệu gán nhãn thực thể y tế học thuật đầu tiên cho tiếng Việt, chuyên sâu về lĩnh vực bệnh lao (Tuberculosis). Dữ liệu được thu thập từ các tạp chí y học uy tín tại Việt Nam (Tạp chí Y học TP.HCM, Tạp chí Thực hành Y học, Tạp chí Lao và Bệnh phổi Quốc gia...) và các luận văn y khoa chuyên ngành được quét bản cứng và số hóa qua OCR.

### 1.1. Thống kê Quy mô Dữ liệu
*   **Tổng số câu**: 1.706 câu.
*   **Độ dài trung bình câu**: 31 tokens.
*   **Tỷ lệ câu chứa thực thể**: ~74% (cho thấy mật độ thực thể y học dày đặc).
*   **Tổng số thực thể được gán nhãn**: 3.334 thực thể.

### 1.2. Phân bố 5 loại thực thể đích

Bộ dữ liệu gán nhãn 5 loại thực thể y tế lâm sàng quan trọng:

| Loại Thực thể | Mô tả | Ví dụ | Số lượng | Tỷ lệ (%) | Độ đồng thuận (IAA %) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Symptom_and_Disease** | Bệnh lý & Triệu chứng lâm sàng | *lao kháng thuốc, ho đờm, khó thở* | ~1.880 | 56.4% | 81.96% |
| **DiagnosticProcedure** | Quy trình/Thủ thuật chẩn đoán | *nhuộm soi AFB, GeneXpert, sinh thiết* | ~620 | 18.6% | 70.59% |
| **Location** | Khu vực địa lý, địa danh | *phía Bắc Việt Nam, quận Tân Bình* | ~400 | 12.0% | 95.89% |
| **DateTime** | Ngày tháng, khoảng thời gian | *từ 1990-1992, tháng Ba, mùa hè* | ~250 | 7.5% | 76.19% |
| **Organisation** | Tổ chức y tế, bệnh viện, bộ phận | *Bộ Y tế, Khoa Phổi Thận, Vinmec* | ~184 | 5.5% | 95.00% |

### 1.3. Cấu hình Phân chia Thực nghiệm (Experiment Split Configuration)
Để phục vụ cho vòng lặp thực nghiệm Active Learning đối chứng song song, thay vì sử dụng cấu hình phân tách mặc định của tác giả (Train 706 / Val 300 / Test 700), đề tài tiến hành xáo trộn ngẫu nhiên và phân chia lại toàn bộ 1.706 câu của corpus theo tỷ lệ đề xuất **80/10/10** để tối đa hóa không gian Unlabeled Pool ($U_0$) cho mô hình chủ động tìm kiếm mẫu khó:
*   **Tập Train gốc ($U_0$)**: **80%** (tương đương **1.365 câu**). Tập này sẽ đóng vai trò là Unlabeled Pool ban đầu.
*   **Tập Validation (Kiểm định) cố định**: **10%** (tương đương **170 câu**). Giữ cố định xuyên suốt các vòng lặp để làm mốc kích hoạt Early Stopping khi huấn luyện ở cả hai nhánh.
*   **Tập Test (Kiểm thử) cố định**: **10%** (tương đương **171 câu**). Giữ cố định xuyên suốt để đánh giá khách quan F1-score ở cuối mỗi vòng gán nhãn.

---

## 2. Đánh giá Tính phù hợp và Lý do Lựa chọn VietBioNER cho Đề tài Active Learning

Dưới ràng buộc thực tế là **ưu tiên các bộ dữ liệu có sẵn, không tự xây dựng dữ liệu mới để tiết kiệm thời gian**, việc lựa chọn bộ dữ liệu **VietBioNER** làm đối tượng thực nghiệm cho đề tài Active Learning là **lựa chọn tối ưu và hoàn toàn hợp lý** vì các lý do sau:

### 2.1. Tính sẵn có và Chuẩn hóa (Available & Peer-reviewed)
VietBioNER là bộ dữ liệu công khai, đã được số hóa sạch sẽ và gán nhãn bởi các chuyên gia có trình độ y tế cao (sinh viên năm cuối Đại học Y Phạm Ngọc Thạch). Bạn hoàn toàn có thể tải về và sử dụng ngay lập tức mà không tốn bất kỳ thời gian thu thập dữ liệu hay xây dựng guidelines gán nhãn.

### 2.2. Kích thước lý tưởng cho mô phỏng Học chủ động (Simulated Active Learning)
Với quy mô **1.706 câu**, VietBioNER có kích thước vừa phải. Trong thực nghiệm Active Learning, mô hình phải được huấn luyện lặp đi lặp lại nhiều vòng (sau mỗi batch dữ liệu mới được chọn).
Nếu bạn chọn một bộ dữ liệu quá lớn (ví dụ PhoNER_COVID19 có 10.027 câu), việc chạy vòng lặp AL sẽ tốn rất nhiều thời gian huấn luyện và đòi hỏi GPU cấu hình cao. Với VietBioNER, một vòng huấn luyện DeBERTa chỉ mất vài phút, giúp bạn hoàn thành các vòng lặp AL nhanh chóng, dễ dàng thử nghiệm nhiều chiến lược khác nhau.

### 2.3. Độ khó khoa học cao (Challenging Baseline)
Hiệu năng F1-score của các mô hình supervised trên VietBioNER hiện tại chỉ đạt khoảng **79.6%**, trong đó thực thể quy trình chẩn đoán (`DiagnosticProcedure`) chỉ đạt **55.56% F1** (do thuật ngữ hẹp và ranh giới không rõ, chỉ số đồng thuận IAA chỉ đạt 70.59%).
*Tại sao điều này quan trọng cho đề tài?* Nếu bạn chọn một bộ dữ liệu quá dễ (như PhoNER_COVID19 với F1 baseline đã đạt 94.5%), mô hình đã quá thông minh và sẽ có rất ít khoảng trống để Active Learning chứng minh sự khác biệt lớn so với Random Sampling. Với VietBioNER, độ khó của dữ liệu sẽ tạo ra sự chênh lệch rõ rệt (khoảng cách F1 rộng) giữa đường cong học tập của AL và Random, làm nổi bật giá trị học thuật của đề tài.

### 2.4. Phân bố thực thể mất cân bằng (Class Imbalance)
VietBioNER có tỷ lệ thực thể phân hóa mạnh (`Symptom_and_Disease` chiếm tới hơn 56%, `Organisation` chỉ chiếm 5.5%). Đây là môi trường hoàn hảo để chứng minh các chiến lược lọc thông minh của Active Learning (nhuy Distinct-K) và cơ chế bù đắp dữ liệu bằng thế thực thể (Distant Supervision) có thể giải quyết các lớp nhãn thiểu số tốt hơn gán nhãn ngẫu nhiên.

Dựa trên các kết quả tìm kiếm, tập dữ liệu VietBioNER được sử dụng chủ yếu cho bài toán Nhận dạng Thực thể Được đặt tên (NER) trong lĩnh vực y sinh học tiếng Việt. Các nghiên cứu hiện tại chủ yếu tập trung vào việc tinh chỉnh (fine-tuning) các mô hình ngôn ngữ lớn đa ngôn ngữ và đơn ngữ đã được huấn luyện sẵn (pre-trained).

Dưới đây là bảng thống kê chi tiết kết quả và độ chính xác (đo bằng chỉ số F1) của một số mô hình tiêu biểu trên tập dữ liệu này.

### 📊 Bảng tổng hợp kết quả trên VietBioNER

Bảng dưới đây tổng hợp kết quả từ một số nghiên cứu. Cần lưu ý rằng các con số được báo cáo trong các điều kiện thí nghiệm khác nhau (ví dụ: văn bản tham chiếu `reference text` và đầu ra nhận dạng giọng nói `ASR output`), do đó không có sự so sánh trực tiếp tuyệt đối.

| Mô hình (Model) | Kiểu văn bản đầu vào | Độ chính xác (Precision) | Độ phủ (Recall) | **F1-Score** | Loại mô hình / Ghi chú |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XLM-R_large** | Văn bản tham chiếu | 0.71 | 0.77 | **0.74** | Đa ngôn ngữ (Encoder-based)  |
| **PhoBERT_base-v2** | Văn bản tham chiếu | 0.68 | 0.79 | **0.74** | Đơn ngữ (Monolingual Encoder)  |
| **PhoBERT_large** | Văn bản tham chiếu | 0.69 | 0.77 | **0.73** | Đơn ngữ (Monolingual Encoder)  |
| **PhoBERT_base** | Văn bản tham chiếu | 0.67 | 0.78 | **0.72** | Đơn ngữ (Monolingual Encoder)  |
| **XLM-R_base** | Văn bản tham chiếu | 0.64 | 0.73 | **0.69** | Đa ngôn ngữ (Encoder-based)  |
| **BARTpho** | Văn bản tham chiếu | 0.64 | 0.73 | **0.68** | Đơn ngữ (Seq2Seq)  |
| **mBART-50** | Văn bản tham chiếu | 0.64 | 0.66 | **0.65** | Đa ngôn ngữ (Seq2Seq)  |
| **SNR (Data Aug.)** | Full Training Set | - | - | **80.90** | Phương pháp Tăng cường Dữ liệu  |
| **XLM-R_large** | ASR (w2v2-Viet) | 0.61 | 0.77 | **0.68** | Đa ngôn ngữ (Encoder) xử lý đầu ra ASR  |
| **PhoBERT_base-v2** | ASR (w2v2-Viet) | 0.56 | 0.77 | **0.65** | Đơn ngữ (Encoder) xử lý đầu ra ASR  |
| **BARTpho** | ASR (w2v2-Viet) | 0.60 | 0.73 | **0.66** | Đơn ngữ (Seq2Seq) xử lý đầu ra ASR  |

### 💡 Phân tích và xu hướng chính

Từ các kết quả trên, có thể thấy một số xu hướng và nhận xét chính:

*   **Mô hình Encoder chiếm ưu thế:** Các mô hình chỉ sử dụng bộ mã hóa (encoder-based) như XLM-R và PhoBERT thường cho kết quả tốt hơn các mô hình phiên dịch (sequence-to-sequence) như BARTpho hay mBART-50. Điều này có thể là do kiến trúc sinh (generative) của seq2seq ít phù hợp hơn cho các tác vụ phân loại như NER .
*   **So sánh giữa mô hình đơn ngữ và đa ngữ:** Các mô hình đa ngôn ngữ (multilingual) như XLM-R đạt hiệu suất cao nhất khi chúng có đủ dung lượng mô hình (ví dụ: XLM-R_large với 550M tham số) để tận dụng lợi ích từ dữ liệu huấn luyện đa ngữ khổng lồ. Mô hình đa ngữ nhỏ hơn (XLM-R_base) có thể bị ảnh hưởng bởi hiệu ứng "pha loãng" (transfer-dilution) và cho kết quả kém hơn so với mô hình đơn ngữ chuyên biệt như PhoBERT_base-v2 .
*   **Tác động của dữ liệu tiếng ồn (ASR):** Hiệu suất của tất cả các mô hình đều giảm đáng kể (từ mức F1 ~0.74 xuống còn ~0.58-0.68) khi xử lý đầu ra từ hệ thống Nhận dạng Giọng nói Tự động (ASR), minh chứng cho tính chất nhiễu của dữ liệu loại này .
*   **Tăng cường dữ liệu (Data Augmentation):** Các phương pháp tăng cường dữ liệu, đặc biệt là **SNR (Semantic-based Noise Reduction)**, đã được chứng minh là có hiệu quả trong việc cải thiện hiệu suất NER. Khi áp dụng SNR trên toàn bộ tập huấn luyện, điểm F1 có thể đạt tới **80.90**, cao hơn đáng kể so với đường cơ sở (baseline) là 79.60 .