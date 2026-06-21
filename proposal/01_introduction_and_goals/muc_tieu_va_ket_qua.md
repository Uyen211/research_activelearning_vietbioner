# Mục Tiêu Nghiên Cứu & Kết Quả Kỳ Vọng của Đề Tài

---

## 1. Mục Tiêu của Project (Research Objectives)

Đề tài được thiết lập nhằm giải quyết bài toán tối ưu hóa chi phí gán nhãn dữ liệu y sinh tiếng Việt dưới điều kiện giới hạn về thời gian và nhân lực chuyên gia y khoa. Cụ thể, các mục tiêu nghiên cứu được chia thành mục tiêu cốt lõi và các mục tiêu cụ thể bổ trợ như sau:

### 1.1. Mục tiêu Chính (Core Objective)
*   **Trả lời câu hỏi nghiên cứu chính**: *"Liệu với cùng một ngân sách gán nhãn (cùng số lượng mẫu được gán nhãn), việc ứng dụng Active Learning (Học chủ động) có giúp mô hình NER đạt chất lượng nhận diện thực thể cao hơn so với phương pháp gán nhãn ngẫu nhiên truyền thống hay không? Và mức độ tiết kiệm thực tế là bao nhiêu?"*
*   **Mục tiêu thực tiễn**: Chứng minh bằng thực nghiệm trên bộ dữ liệu `VietBioNER` rằng phương pháp học chủ động đề xuất giúp giảm thiểu đáng kể số lượng câu cần gán nhãn và nỗ lực sửa đổi nhãn của chuyên gia y tế mà không làm suy giảm hiệu năng của hệ thống nhận diện thực thể y sinh tiếng Việt.

### 1.2. Các Mục tiêu Phụ và Nhiệm vụ Thử nghiệm cụ thể (Sub-objectives)
Để trả lời trọn vẹn câu hỏi nghiên cứu chính, project thực hiện các nhiệm vụ nghiên cứu con bao gồm:

*   **Mục tiêu Thiết kế & Tích hợp Kiến trúc**:
    *   Xây dựng hệ thống nhận diện thực thể chuỗi trên nền tảng **ViPubmedDeBERTa-base** kết hợp cơ chế thích ứng **LoRA** và đầu phân loại **Linear + CRF Head**: Tích hợp thuật toán Forward-Backward trên tầng CRF để tính toán chính xác xác suất biên của nhãn tại từng vị trí từ, từ đó đo lường độ bất định toán học trực tiếp thông qua **CRF Marginal Entropy (Entropy Xác suất biên)** của các câu chưa gán nhãn mà không cần thêm tham số phụ.

*   **Mục tiêu Tối ưu hóa Batch gán nhãn (Query Selection)**:
    *   Triển khai bộ lọc **Distinct-K Filter** sử dụng độ tương đồng Cosine trên các vector nhúng **S-BERT pre-computed** nhằm giải quyết bài toán Tập độc lập lớn nhất trên đồ thị tương đồng ngữ nghĩa. Giải pháp này giúp loại bỏ hiện tượng bất đẳng hướng (anisotropy) của vector [CLS] thô của Transformer và tối ưu hóa tài nguyên tính toán của bộ lọc. Mục tiêu là chọn ra một batch mẫu gán nhãn ($b$ câu) có độ bất định cao nhất nhưng có tính đa dạng ngữ nghĩa lớn nhất, tránh lãng phí ngân sách vào các mẫu trùng lặp thông tin.
*   **Mục tiêu Giải quyết mất cân bằng nhãn**:
    *   Thử nghiệm kỹ thuật **Tăng cường Dữ liệu Giám sát từ xa (Distant Supervision Augmentation)** thông qua từ điển thuật ngữ y học (Gazetteer) để bổ trợ tri thức cho các lớp nhãn hiếm gặp (như `DiagnosticProcedure`, `Organisation`) trong tập huấn luyện của các vòng AL.
*   **Mục tiêu Thiết lập Đối chứng Nghiêm ngặt (Comparative Baselines)**:
    *   Thiết lập và chạy song song 2 nhánh thực nghiệm xuất phát từ cùng một tập dữ liệu khởi tạo ($L_0$), cả hai nhánh đều sử dụng chung cơ chế biểu diễn đầu vào **Entity Descriptions**:
        *   **Nhánh A (Đề xuất)**: CRF Marginal Entropy + Distinct-K Filter + Entity Descriptions + Distant Supervision.
        *   **Nhánh B (Random Baseline)**: Chọn mẫu ngẫu nhiên hoàn toàn (Random Sampling) + Entity Descriptions + Distant Supervision.
    *   *Mục tiêu phân tách*: Việc so sánh đối chứng trực diện giữa Nhánh A (học chủ động đề xuất) và Nhánh B (chọn mẫu ngẫu nhiên) sẽ giúp làm nổi bật và định lượng chính xác hiệu năng cải thiện cũng như mức độ tiết kiệm chi phí gán nhãn thực tế của các giải pháp đề xuất.
*   **Mục tiêu Lượng hóa nỗ lực gán nhãn (Annotation Effort Quantification)**:
    *   Tích hợp độ đo khoảng cách hiệu chỉnh **Levenshtein Edit Distance** vào kịch bản simulated oracle feedback. Mục tiêu là lượng hóa công sức chỉnh sửa nhãn máy gợi ý (Pre-annotation) của chuyên gia ở từng vòng lặp, phản ánh chính xác chi phí gán nhãn trong quy trình thực tế (AI-assisted annotation).

---

## 2. Kết Quả Mong Muốn Đạt Được (Expected Outcomes)

Thông qua việc thực hiện đề tài, nghiên cứu kỳ vọng đạt được các kết quả định lượng và định tính cụ thể như sau:

### 2.1. Chỉ số Hiệu năng & Tỷ lệ Tiết kiệm Chi phí (Quantitative Metrics)
*   **Hiệu năng F1-score mục tiêu**: Mô hình NER đạt giá trị F1-score tối thiểu **75.0%** trên tập Test của `VietBioNER` (đạt xấp xỉ 95% giới hạn hiệu năng của mô hình khi huấn luyện trên 100% dữ liệu gốc) khi chỉ sử dụng tối đa **50%** ngân sách gán nhãn.
*   **Tỷ lệ tiết kiệm mẫu câu (Sentence Saving Ratio - SSR)**: 
    *   Nhánh A đạt mức SSR từ **30% đến 50%** so với Nhánh B (Random). Nghĩa là, để đạt cùng một mức F1-score mục tiêu, Nhánh A cần gán nhãn ít hơn từ 30% đến 50% số câu so với việc chọn mẫu ngẫu nhiên.
*   **Tỷ lệ tiết kiệm thao tác hiệu chỉnh (Edit Saving Ratio - ESR)**:
    *   Nhánh A đạt mức ESR từ **20% đến 40%** so với Nhánh B. Chỉ số này chứng minh rằng Active Learning giúp mô hình dự đoán gợi ý nhãn (pre-annotation) tốt hơn, khiến tổng số ký tự/thẻ nhãn chuyên gia cần chỉnh sửa ít hơn đáng kể qua các vòng lặp.

### 2.2. Giá trị Học thuật và Đóng góp Lý thuyết (Qualitative Contributions)
*   **Framework Học chủ động cho Y sinh tiếng Việt**: Xây dựng thành công một quy trình (pipeline) học chủ động hoàn chỉnh, có kiểm chứng khoa học, áp dụng hiệu quả cho ngôn ngữ ít tài nguyên như tiếng Việt và miền chuyên sâu như y sinh.
*   **Phân tích đóng góp thành phần (Ablation Study)**: Đưa ra báo cáo phân tích chi tiết về mức độ đóng góp của từng kỹ thuật (Distant Supervision, Entity Descriptions, CRF Marginal Entropy, Distinct-K Filter) vào tổng hiệu năng cải thiện của hệ thống.
*   **Lập luận khoa học về chi phí hiệu chỉnh**: Đóng góp một góc nhìn đánh giá mới về chi phí gán nhãn dựa trên nỗ lực sửa đổi (edit distance) trong các nghiên cứu học chủ động mô phỏng tại Việt Nam.

### 2.3. Sản phẩm Bàn giao Vật lý (Deliverables)
*   **Mã nguồn thực nghiệm**: Bộ mã nguồn viết bằng Python sử dụng PyTorch và Transformers, được tổ chức modular hóa, dễ đọc và dễ tái tạo kết quả thực nghiệm.
*   **Lịch sử thực nghiệm (Logs)**: File log chi tiết ghi nhận các thông số F1, Precision, Recall, số câu gán nhãn, khoảng cách Levenshtein của cả 2 nhánh thí nghiệm qua từng vòng lặp $t$.
*   **Biểu đồ & Báo cáo trực quan**: Tập hợp các biểu đồ đường cong học tập F1 vs. Budget, F1 vs. Edit Distance phục vụ trực tiếp cho việc trình bày và viết luận văn/bài báo khoa học.
