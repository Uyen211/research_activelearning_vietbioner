# Bài học Kinh nghiệm Rút ra từ Thực nghiệm Học chủ động (VietBioNER)

Tài liệu này tổng hợp toàn bộ các bài học kinh nghiệm về mặt thuật toán NLP, thiết kế bài toán (task design) và tối ưu hóa tài nguyên phần cứng (Google Colab T4 GPU) thu được qua quá trình debug và tối ưu hóa hệ thống nhận diện thực thể y sinh tiếng Việt.

---

## 1. Bài học về Tokenization & Căn chỉnh Nhãn (Index Alignment)

### 1.1. Lỗi dịch chuyển chỉ mục (Index Shift) trong mô hình CRF
*   **Hiện tượng**: Khi huấn luyện chuỗi kết hợp DeBERTa + CRF, lớp CRF nhận đầu vào là các đặc trưng ẩn từ câu đã được tokenize. Khi giải mã Viterbi (`crf.decode`), chuỗi nhãn trả về có độ dài bằng số lượng token active thực tế (bao gồm `[CLS]` và các subtokens của câu gốc), chứ không bằng số từ gốc trong câu văn bản thô.
*   **Bài học rút ra**: 
    *   Tuyệt đối không sử dụng chỉ số vòng lặp từ gốc `t` để lập chỉ mục trực tiếp vào Viterbi path. Việc này sẽ gán nhãn của token `[CLS]` cho từ đầu tiên và gây dịch chuyển lệch pha toàn bộ các từ phía sau.
    *   **Giải pháp chuẩn**: Sử dụng `word_ids` thu được từ Hugging Face Tokenizer để ánh xạ. Với mỗi từ gốc index `w`, tìm subtoken active đầu tiên của nó trong chuỗi tokenized để lấy nhãn đại diện.

### 1.2. Mất nhãn `B-` khi gộp từ ghép tiếng Việt (PyVi Alignment)
*   **Hiện tượng**: Tiếng Việt sử dụng khoảng trắng làm ranh giới âm tiết. Khi dùng công cụ tách từ ghép (PyVi), các âm tiết đơn lẻ được nối lại thành một từ ghép (ví dụ: `triệu` [B], `chứng` [I], `lâm` [I], `sàng` [I] $\rightarrow$ `triệu_chứng_lâm_sàng`).
*   **Bài học rút ra**:
    *   Nếu sử dụng cơ chế đếm đa số phiếu (majority voting) để gán nhãn cho từ ghép mới, nhãn `I-` sẽ luôn áp đảo nhãn `B-` do một thực thể chỉ có duy nhất 1 nhãn `B-` và nhiều nhãn `I-`. Kết quả là từ ghép mới bị gán nhãn `I-`, vi phạm quy chuẩn BIO (chuỗi nhãn bắt đầu bằng `I-` mà không có `B-`).
    *   **Giải pháp chuẩn**: Khi căn chỉnh nhãn từ âm tiết sang từ ghép, phải ưu tiên tuyệt đối nhãn `B-`. Nếu từ ghép chứa bất kỳ âm tiết nào mang nhãn `B-`, từ ghép mới bắt buộc phải được gán nhãn `B-`.

---

## 2. Bài học về Biểu diễn Không gian Vector & Hiện tượng Anisotropy

### 2.1. Sự sụp đổ biểu diễn (Representation Collapse) của vector `[CLS]` thô
*   **Hiện tượng**: Các mô hình Transformer pre-trained (DeBERTa/BERT) khi chưa được tối ưu hóa đặc thù cho Sentence Similarity thường gặp hiện tượng **anisotropy** (bất đẳng hướng). Các vector biểu diễn câu bị co cụm trong một hình nón hẹp, khiến Cosine Similarity giữa các câu bất kỳ luôn rất cao (hầu hết > 0.90).
*   **Bài học rút ra**:
    *   Không nên sử dụng trực tiếp Cosine Similarity trên vector `[CLS]` của mô hình NER thô với các ngưỡng lọc đa dạng tiêu chuẩn (như `theta = 0.85`). Việc này sẽ lọc bỏ hầu như toàn bộ ứng viên và làm vô hiệu hóa bộ lọc đa dạng (Distinct-K Filter), đẩy thuật toán rơi vào trường hợp fallback (uncertainty thuần túy).
    *   **Giải pháp tối ưu (Phương án C)**: Sử dụng các mô hình Sentence-BERT (như `keepitreal/vietnamese-sbert`) đã được tinh chỉnh bằng hàm loss tương phản để sinh ra các vector biểu diễn câu có phân phối đều hơn.

### 2.2. Kỹ thuật Caching Embeddings để tối ưu tài nguyên
*   **Bài học rút ra**: 
    *   Việc chạy mô hình Sentence-BERT trong vòng lặp Active Learning ở mỗi vòng để mã hóa các ứng viên chưa gán nhãn gây tốn tài nguyên GPU và làm tăng thời gian chạy.
    *   **Giải pháp tối ưu**: Tính toán S-BERT embeddings cho toàn bộ tập Train **một lần duy nhất** ở Vòng 0 và lưu trữ dưới dạng tệp tin NumPy (`.npy`) trên Google Drive. Ở các vòng sau (hoặc khi Resume), ta chỉ cần nạp tệp này vào RAM CPU. Cách làm này vừa giữ được ngưỡng tương đồng nhạy bén `theta = 0.85` vừa giải phóng hoàn toàn GPU RAM.

---

## 3. Bài học về Huấn luyện CRF & Phân tầng Tốc độ học (Differential Learning Rates)

### 3.1. Sự hội tụ chậm của đầu CRF khởi tạo ngẫu nhiên
*   **Hiện tượng**: Mô hình NER sử dụng chung một tốc độ học nhỏ `2e-5` cho cả DeBERTa backbone và đầu Linear-CRF. Đầu Linear-CRF được khởi tạo ngẫu nhiên nên cần số lượng cập nhật lớn hơn để học ma trận chuyển trạng thái, trong khi DeBERTa đã có sẵn trọng số tốt chỉ cần điều chỉnh nhẹ (fine-tune).
*   **Bài học rút ra**:
    *   Nếu áp dụng tốc độ học quá thấp (`2e-5`) cho CRF trong điều kiện số epoch rất nhỏ (5-12 epoch) trên tập dữ liệu AL cực bé (85 câu), CRF sẽ không thể học được các luật chuyển đổi trạng thái hợp lệ, dẫn đến dự đoán ra các chuỗi nhãn BIO hỗn loạn.
    *   **Giải pháp chuẩn**: Áp dụng **tốc độ học phân tầng (differential learning rates)**. Giữ tốc độ học nhỏ cho backbone (`2e-5`) và đặt tốc độ học lớn hơn cho các lớp ngẫu nhiên (`5e-4` cho Linear head và `1e-3` cho CRF).

### 3.2. Quy chuẩn BIO trong Tăng cường Dữ liệu bằng Thế thực thể dựa trên từ điển (DES)
*   **Bài học rút ra**:
    *   Khi sinh câu tăng cường bằng thế thực thể y khoa (Entity Substitution), nếu gán cứng toàn bộ nhãn của thực thể mới là `I-`, mô hình sẽ bị dạy sai quy luật chuyển trạng thái (cho phép chuyển trực tiếp từ `O` sang `I-`).
    *   **Giải pháp chuẩn**: Luôn đảm bảo token đầu tiên của thực thể mới chèn vào mang nhãn `B-` và các token sau mang nhãn `I-`.

### 3.3. Thay thế Loss-Prediction Module bằng CRF Marginal Entropy để khắc phục quá khớp
*   **Hiện tượng**: Mạng MLP phụ (Loss-Prediction Module) học cách dự đoán loss từ các đặc trưng ẩn. Tuy nhiên, ở các vòng AL đầu tiên, kích thước tập huấn luyện quá nhỏ (chỉ 85 câu), khiến MLP dễ dàng bị quá khớp (overfitting) trên dữ liệu huấn luyện và đưa ra các dự đoán loss bị nhiễu cao trên tập chưa gán nhãn $U_t$.
*   **Bài học rút ra**:
    *   Với tập dữ liệu cực kỳ nhỏ ở giai đoạn đầu, các độ đo độ bất định có tham số học thêm rất không ổn định.
    *   **Giải pháp chuẩn**: Sử dụng **CRF Marginal Entropy (Entropy Xác suất biên)**. Thuật toán Forward-Backward trên CRF cho phép tính toán trực tiếp xác suất biên của từng nhãn tại mỗi vị trí token từ ma trận chuyển tiếp và phát xạ hiện tại của mô hình mà không cần thêm bất kỳ tham số học thêm nào. Độ đo này ổn định tuyệt đối và phản ánh chính xác 100% tri thức hiện tại của mô hình.

---

## 4. Bài học về Quản lý Tài nguyên Phần cứng trên Google Colab T4

### 4.1. Hiện tượng tích lũy bộ nhớ đồ họa (GPU Memory Leak)
*   **Hiện tượng**: Trong vòng lặp Active Learning, mô hình được khởi tạo lại ở mỗi vòng lặp. Nếu không thu hồi bộ nhớ, các đồ thị tính toán và trọng số của mô hình cũ vẫn bị giữ lại trong GPU RAM, gây tràn bộ nhớ (OOM) sau 3-4 vòng lặp.
*   **Bài học rút ra**:
    *   Sau mỗi vòng lặp AL, bắt buộc phải giải phóng bộ nhớ một cách tường minh: giải phóng tham chiếu của đối tượng `model`, chạy thu gom rác của Python (`gc.collect()`) và giải phóng cache của CUDA (`torch.cuda.empty_cache()`).

### 4.2. Khống chế hệ số tăng cường dữ liệu
*   **Bài học rút ra**:
    *   Hệ số tăng cường dữ liệu động $M$ (số lượng thực thể được thế từ Gazetteer cho mỗi câu) ảnh hưởng trực tiếp đến kích thước tập huấn luyện mở rộng $L_{t,\text{aug}}$. 
    *   Việc giảm hệ số từ `M_range = (3, 5)` xuống `M_range = (2, 3)` giúp tập dữ liệu huấn luyện nhẹ hơn, giảm thời gian huấn luyện mỗi epoch đáng kể để tránh nguy cơ bị ngắt kết nối (timeout) trên tài khoản Colab miễn phí.

---

## 5. Bài học rút ra từ Quá trình Đồng bộ hóa Notebook & Cơ chế Resume

### 5.1. Lỗi ghi chỉ mục gán nhãn trước khi cập nhật (Write-Before-Update Bug) trong Resume của Nhánh B
*   **Vấn đề**: Trong vòng lặp huấn luyện Nhánh B (Random), mã nguồn cũ ghi đè file `current_L_indices_random.json` lên đĩa *trước* khi thêm batch $b = 100$ câu mới gán nhãn. Khi tiến trình bị ngắt đột ngột (ví dụ: mất kết nối GPU trên Kaggle sau vài vòng lặp), việc resume sẽ nạp lại file JSON bị thiếu 100 câu, dẫn đến lệch pha số lượng câu gán nhãn thực tế ($L_t$ chỉ có 185 thay vì 285 ở Vòng 2).
*   **Bài học rút ra**: 
    *   Trong bất kỳ vòng lặp AL/Random nào có cơ chế lưu vết (checkpointing), quy trình ghi trạng thái lên đĩa bắt buộc phải tuân theo thứ tự: **Cập nhật danh sách chỉ mục trong bộ nhớ $ightarrow$ Ghi danh sách chỉ mục đã cập nhật lên tệp tin $ightarrow$ Thực hiện huấn luyện và lưu checkpoint**. Việc này đảm bảo tính toàn vẹn dữ liệu trong mọi trường hợp mất điện hay ngắt kết nối đột ngột.

### 5.2. Giải pháp Đồng bộ hóa Notebooks đa nền tảng (Colab vs Kaggle)
*   **Vấn đề**: Việc phát triển song song hai phiên bản notebook cho Google Colab và Kaggle dễ dẫn đến lệch pha logic (code drift) nếu có các thay đổi thuật toán.
*   **Bài học rút ra**:
    *   Thiết kế notebook thống nhất: Tách biệt hoàn toàn logic ML/AL cốt lõi khỏi logic thiết lập hệ thống.
    *   Sử dụng biến cờ toàn cục `IS_KAGGLE = True/False` trong class Config để tự động ánh xạ đường dẫn.
    *   Các cell setup phụ thuộc nền tảng (như `kaggle_secrets` hay `google.colab.drive`) phải được đóng gói gọn và đặt ở đầu notebook. Điều này giúp các cell logic ML/AL cốt lõi phía sau hoàn toàn giống nhau 100%, dễ dàng đồng bộ hóa bằng các công cụ so khớp tự động.

### 5.3. Vượt qua hạn chế hệ thống tệp Read-Only trên Kaggle
*   **Vấn đề**: Thư mục đầu vào của Kaggle (`/kaggle/input/...`) là Read-only. Mọi thao tác ghi tệp xử lý dữ liệu động của PyVi hay ghi đè checkpoints cũ của cơ chế Resume đều gây lỗi runtime crash.
*   **Bài học rút ra**:
    *   **Kaggle Resume Bridge**: Cần tạo một ô thiết lập chuyển tiếp ở đầu notebook Kaggle để tự động sao chép logs và checkpoint cũ từ Input sang thư mục làm việc ghi được `/kaggle/working/logs/`, và trỏ biến đường dẫn `LOGS_DIR` về thư mục này.
    *   **Caching Preprocessed Data**: Thực hiện PyVi tokenization tĩnh một lần duy nhất, lưu kết quả preprocessed vào dataset zip tải lên Kaggle. Ở pha chạy chính, mô hình chỉ việc đọc tệp preprocessed từ `INPUT_DIR` mà không gọi PyVi động hay cố gắng ghi đè lên thư mục Input.

---

## 6. Bảng So sánh Trước và Sau khi Sửa đổi Logic

| Thành phần | Trước khi Sửa đổi (Lỗi) | Sau khi Sửa đổi (Đúng) | Hậu quả / Kết quả |
| :--- | :--- | :--- | :--- |
| **Gộp nhãn từ ghép** | Bỏ phiếu đa số phi-'O' $\rightarrow$ Thực thể ghép mất nhãn `B-`, bắt đầu bằng `I-`. | Ưu tiên nhãn `B-` $\rightarrow$ Đảm bảo đúng định dạng BIO lâm sàng. | Trước: F1-score = 0% do `seqeval` loại bỏ nhãn lỗi.<br>Sau: F1-score hợp lệ. |
| **Giải quyết xung đột** | Lấy nhãn bằng index từ gốc `t` truy xuất vào Viterbi path (độ dài subtoken). | Dùng `word_ids` và `sequence_ids` để map subtoken đầu của từ gốc. | Trước: Bị lệch chỉ mục nghiêm trọng (Index Shift).<br>Sau: Gộp nhãn chính xác. |
| **Gán nhãn tăng cường** | Gán cứng toàn bộ nhãn thực thể chèn mới là `I-`. | Gán nhãn `B-` ở đầu thực thể mới, còn lại gán nhãn `I-`. | Trước: CRF học sai luật chuyển trạng thái.<br>Sau: Hợp lệ BIO. |
| **Tốc độ học (Optimizer)** | Dùng chung `2e-5` cho cả DeBERTa và đầu ngẫu nhiên Linear-CRF. | Phân tầng: Backbone `2e-5`, BiLSTM `5e-4`, CRF `1e-3`. | Trước: CRF không kịp hội tụ trong 5 epoch.<br>Sau: CRF hội tụ nhanh chóng. |
| **Đo lường độ bất định** | Dùng Loss-Prediction Module (đầu MLP phụ) huấn luyện bằng Pairwise Ranking Loss. | Dùng **CRF Marginal Entropy** tính trực tiếp bằng thuật toán Forward-Backward trên CRF. | Trước: MLP bị quá khớp nặng trên tập dữ liệu AL nhỏ gây nhiễu và sụt giảm F1.<br>Sau: Đo lường chính xác, không tham số học thêm, F1 tăng trưởng ổn định. |
| **Độ tương đồng ngữ nghĩa** | Tính Cosine trên vector `[CLS]` thô của DeBERTa (Ngưỡng `0.85`). | Tính Cosine trên vector S-BERT pre-computed lưu ở RAM CPU (Ngưỡng `0.85`). | Trước: Bị anisotropy vô hiệu hóa Distinct-K.<br>Sau: Lọc trùng ngữ nghĩa tối ưu, giải phóng GPU RAM. |
| **Quản lý GPU Colab** | Không xóa mô hình cũ $\rightarrow$ tích lũy RAM GPU qua các vòng AL. | Thực hiện `del model`, `gc.collect()`, `empty_cache()` cuối mỗi vòng AL. | Trước: Bị sập OOM sau vài vòng chạy.<br>Sau: Bộ nhớ GPU ổn định xuyên suốt. |
