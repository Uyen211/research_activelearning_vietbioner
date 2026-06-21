# Giải Thích Cơ Chế Mô Tả Loại Thực Thể (Entity Type Description)

---

Cơ chế **Entity Type Description (Mô tả loại thực thể)** kế thừa tri thức từ nghiên cứu **OPENBIONER** (NAACL 2025) nhằm chuyển đổi bài toán nhận diện thực thể có tên (NER) truyền thống sang bài toán **đọc hiểu văn bản và so khớp ngữ nghĩa (Reading Comprehension & Semantic Matching)**. 

Dưới đây là thuyết minh kỹ thuật chi tiết về cách thức triển khai cụ thể cơ chế này trong các pha huấn luyện, suy luận thực tế, cách thức gộp nhãn, lập luận toán học về không gian biểu diễn vector và cơ chế chống học vẹt (Entity Masking Regularizer).

---

## 1. Cách Thức Hoạt Động Trong Pha Huấn Luyện (Training Phase)

Trong bài toán NER y sinh chuyên sâu như `VietBioNER`, chúng ta cần trích xuất $K = 5$ loại thực thể: `Disease` (Bệnh), `Symptom` (Triệu chứng), `DiagnosticProcedure` (Quy trình chẩn đoán), `Treatment` (Phương pháp điều trị), và `Organisation` (Tổ chức y tế).

Khi huấn luyện mô hình, đối với mỗi câu gốc $s$ trong tập huấn luyện ($L_t$), hệ thống sẽ tự động nhân bản nó thành **$K = 5$ chuỗi đầu vào độc lập** ghép nối với mô tả của từng nhãn tương ứng. Đồng thời, nhãn đích (target labels) cũng được biến đổi loại trừ để mô hình chỉ học cách trích xuất duy nhất loại thực thể đang được truy vấn. Để đảm bảo tính đồng nhất về mặt token hóa và tránh hiện tượng lệch pha token hóa (tokenization mismatch) khi tính toán Cross-Attention, các đoạn mô tả tĩnh $d_c$ cũng được xử lý tách từ ghép bằng **PyVi** tương tự như câu gốc $s$ trước khi ghép nối.

### Ví dụ cụ thể:
*   **Câu gốc ($s$)**: *"Bệnh nhân nghi lao phổi được chỉ định làm phản ứng Mantoux."*
*   **Nhãn chuẩn gốc của câu**:
    *   `"lao phổi"` $\rightarrow$ **Disease**
    *   `"phản ứng Mantoux"` $\rightarrow$ **DiagnosticProcedure**
*   **Quy trình nhân bản và gán nhãn đích khi huấn luyện**:

| Chuỗi đầu vào nạp vào DeBERTa (`[CLS] s [SEP] d_c [SEP]`) | Loại thực thể truy vấn ($c$) | Nhãn đích chuyển đổi (Target BIO Labels) |
| :--- | :--- | :--- |
| `[CLS] Bệnh_nhân nghi lao phổi được chỉ_định làm phản_ứng Mantoux . [SEP] d_Disease [SEP]` | **Disease** | `O` `O` `B` `I` `O` `O` `O` `O` `O` `O` *(Chỉ giữ nhãn thực thể Bệnh; các thực thể khác chuyển thành O)* |
| `[CLS] Bệnh_nhân nghi lao phổi được chỉ_định làm phản_ứng Mantoux . [SEP] d_DiagnosticProcedure [SEP]` | **DiagnosticProcedure** | `O` `O` `O` `O` `O` `O` `O` `B` `I` `O` *(Chỉ giữ nhãn thực thể Chẩn đoán; các thực thể khác chuyển thành O)* |
| `[CLS] Bệnh_nhân nghi lao phổi được chỉ_định làm phản_ứng Mantoux . [SEP] d_Symptom [SEP]` | **Symptom** | `O` `O` `O` `O` `O` `O` `O` `O` `O` `O` *(Không có thực thể Triệu chứng trong câu $\rightarrow$ Toàn nhãn O)* |
| `[CLS] Bệnh_nhân nghi lao phổi được chỉ_định làm phản_ứng Mantoux . [SEP] d_Treatment [SEP]` | **Treatment** | `O` `O` `O` `O` `O` `O` `O` `O` `O` `O` *(Không có thực thể Điều trị trong câu $\rightarrow$ Toàn nhãn O)* |
| `[CLS] Bệnh_nhân nghi lao phổi được chỉ_định làm phản_ứng Mantoux . [SEP] d_Organisation [SEP]` | **Organisation** | `O` `O` `O` `O` `O` `O` `O` `O` `O` `O` *(Không có thực thể Tổ chức trong câu $\rightarrow$ Toàn nhãn O)* |

> [!NOTE]
> Nhờ cách biến đổi này, không gian nhãn đầu ra của mô hình được đơn giản hóa từ đa phân lớp dạng `[B-Disease, I-Disease, B-DiagnosticProcedure...]` về hệ nhãn nhị phân động dạng `[B, I, O]`. Mô hình học được cách đối chiếu ngữ nghĩa: *"Khi đọc mô tả nào ở đuôi, tôi chỉ kích hoạt nhãn B/I cho các từ tương thích với mô tả đó trong câu đầu vào"*.

---

## 2. Cách Thức Hoạt Động Trong Pha Suy Luận Thực Tế (Inference/Deployment Phase)

Trong thực tế ngoài đời (hoặc khi đánh giá trên tập Test của VietBioNER), bác sĩ chỉ cung cấp một câu văn bản thô $s$. Quy trình dự đoán nhãn được phần mềm thực hiện tự động qua các bước sau:

```
[Câu mới s] 
    │
    ├──> Ghép mô tả 1 (Disease) ──────────> [Mô hình] ──> Dự đoán 1 (BIO) ──┐
    ├──> Ghép mô tả 2 (DiagProcedure) ────> [Mô hình] ──> Dự đoán 2 (BIO) ──┼──> [Bộ gộp nhãn & Giải quyết xung đột] ──> Kết quả NER cuối cùng
    ├─...                                                                   │
    └──> Ghép mô tả 5 (Organisation) ─────> [Mô hình] ──> Dự đoán 5 (BIO) ──┘
```

1.  **Lưu trữ Tĩnh**: Đoạn mô tả của 5 loại thực thể được viết sẵn và lưu trữ tĩnh trong file cấu hình hệ thống (không cần sinh lại cho mỗi câu).
2.  **Nhân bản tự động**: Hệ thống nhân bản câu $s$ thành 5 chuỗi độc lập tương ứng với 5 mô tả tĩnh, nạp qua mô hình để lấy về 5 chuỗi nhãn dự đoán độc lập.
3.  **Bộ gộp nhãn (Merging Mechanism)**: 
    *   Hệ thống quét qua từng token từ trái sang phải.
    *   Nếu tại vị trí token $t_i$, lần chạy thứ $c$ dự đoán nhãn là `B` hoặc `I`, hệ thống sẽ gắn nhãn thực tế cho token đó dạng `B-[Tên nhãn c]` (ví dụ: `B-Disease`).
4.  **Giải quyết xung đột (Label Conflict Resolution)**:
    *   *Tình huống xung đột*: Trong trường hợp rất hiếm gặp, cùng một cụm từ trong câu (ví dụ `"Mantoux"`) bị mô hình nhận diện nhãn `B` ở cả lần chạy 1 (Disease) và lần chạy 2 (DiagnosticProcedure).
    *   *Cách giải quyết*: Thay vì sử dụng lớp Softmax độc lập cho từng token (vốn không tương thích trực tiếp với thuật toán tìm đường đi tối ưu Viterbi của CRF), hệ thống sử dụng thuật toán **Forward-Backward** trên mô hình CRF để tính toán **xác suất biên (marginal probability)** $P(y_i = c | X)$ cho từng token bị xung đột. Nhãn thực thể nào có xác suất biên cao nhất sẽ được giữ lại, nhãn có xác suất thấp hơn sẽ được chuyển thành nhãn phụ thuộc hoặc nhãn `O`.

---

## 3. Lý Luận Khoa Học Về Sự Nhất Quán Trong Không Gian Vector (Representation Space)

### Tại sao không thể "nửa vời" (chỉ ghép mô tả cho nhãn hiếm, nhãn thường thì dùng câu thô)?

Nếu thiết kế bất đối xứng (ví dụ: nhãn hiếm `DiagnosticProcedure` thì ghép mô tả, nhãn phổ biến `Disease` thì chỉ truyền câu thô), mô hình sẽ gặp lỗi **bất đồng nhất không gian vector biểu diễn đặc trưng ẩn (Representation Discrepancy)**:

1.  **Cơ chế Cross-Attention**: Khi câu văn bản $s$ đi qua **ViPubmedDeBERTa-base**, các token trong câu $s$ sẽ tính toán tương tác attention chéo với toàn bộ các token đi kèm. 
    *   Nếu câu có ghép mô tả: Vector biểu diễn $v_{\text{word}}$ của từ trong câu sẽ **bị thay đổi mạnh mẽ (conditioned)** do hấp thụ thông tin ngữ nghĩa từ các từ khóa trong đoạn mô tả.
    *   Nếu câu không ghép mô tả: Vector biểu diễn $v_{\text{word}}$ của từ trong câu chỉ chứa ngữ cảnh thô của câu.
2.  **Sự bối rối của đầu phân loại Linear-CRF**: 
    *   Đầu phân loại Linear-CRF được đặt phía trên backbone để học cách chuyển đổi vector ẩn thành điểm số nhãn. Nếu không gian vector đầu vào bị phân tách thành 2 phân phối hoàn toàn khác biệt (một loại có nhúng mô tả, một loại không), mạng Linear-CRF sẽ phải học hai chiến lược phân tách ranh giới độc lập trên cùng một tham số.
    *   Điều này gây nhiễu gradient khi lan truyền ngược, khiến mô hình NER khó hội tụ, mất đi sự mượt mà trong không gian biểu diễn đặc trưng ẩn và làm giảm nghiêm trọng F1-score tổng thể.
3.  **Quyết định**: Để bảo đảm tính đồng nhất và tối ưu hóa không gian vector biểu diễn, định dạng đầu vào ghép nối mô tả thực thể bắt buộc phải được áp dụng đồng bộ và nhất quán ở cả hai nhánh thí nghiệm A và B cho toàn bộ các mẫu dữ liệu.

---

## 4. Cơ Chế Chống Học Vẹt (Entity Masking Regularizer)

Do đặc thù đoạn mô tả nhãn $d_c$ được giữ cố định và lặp đi lặp lại trên nhiều câu khác nhau, mô hình NER có nguy cơ bị **học vẹt các từ khóa cố định** (ví dụ: chỉ cần thấy mô tả có chứa từ *"xét nghiệm"* là mô hình tự động gán nhãn bừa bãi cho bất kỳ cụm từ nào đi sau từ *"được chỉ định"* mà không thực sự học cấu trúc ngữ pháp thực tế).

Để giải quyết triệt để vấn đề này, dự án đề xuất tích hợp thêm kỹ thuật **Entity Masking Regularizer** kế thừa từ OPENBIONER vào quá trình huấn luyện:

*   **Cơ chế hoạt động**: Khi huấn luyện mô hình ở Bước 5 (ở cả hai nhánh), hệ thống sẽ che giấu ngẫu nhiên (thay thế bằng token `[MASK]`) các thực thể mục tiêu trong câu đầu vào với xác suất từ **$15\%$ đến $20\%$** (giảm tỷ lệ so với mức 30%-50% ban đầu để tránh làm suy yếu đặc trưng từ vựng lâm sàng quan trọng).
*   **Ví dụ**:
    *   *Câu gốc chưa che*: `[CLS] Bệnh_nhân ho nhiều được chỉ_định chụp X-quang phổi . [SEP] d_Diag [SEP]`
    *   *Câu sau khi áp dụng Masking (xác suất p = 0.2)*: `[CLS] Bệnh_nhân ho nhiều được chỉ_định [MASK] [MASK] [MASK] . [SEP] d_Diag [SEP]`
*   **Tác dụng**: Vì từ khóa thực tế `"chụp X-quang phổi"` đã bị ẩn đi một cách ngẫu nhiên, mô hình không thể dựa vào việc ghi nhớ từ vựng để gán nhãn. Mô hình bắt buộc phải học cách phối hợp ngữ cảnh xung quanh (từ khóa gợi ý `"được chỉ chỉ định..."`) và ngữ nghĩa của đoạn mô tả $d_{\text{Diag}}$ ở đuôi để suy luận ra vị trí bị che là quy trình chẩn đoán.
*   **Phạm vi áp dụng**: Cơ chế che giấu này **chỉ được áp dụng trong pha huấn luyện mô hình** (Pha Train). Khi chạy suy luận để đánh giá trên tập Validation/Test (Pha Test) hoặc khi chạy suy luận trên tập chưa gán nhãn $U_t$ để xếp hạng chọn mẫu (Pha Query Selection), hệ thống **tuyệt đối không áp dụng che giấu** để tránh hiện tượng bất đồng nhất dữ liệu (distribution shift) khi triển khai thực tế.
*   **Giá trị học thuật**: Tích hợp cơ chế này giúp nâng cao đáng kể năng lực khái quát hóa (generalization), ngăn ngừa overfitting trên tập dữ liệu nhỏ `VietBioNER`, và tối ưu hóa khả năng nhận diện các thực thể ngoài từ điển (OOV) ngoài đời thực.

---

## 5. Thảo Luận Học Thuật (FAQ)

### 1. Tại sao không thể chỉ ghép mô tả cho nhãn hiếm gặp?
**Trả lời**: Việc này vi phạm nguyên tắc đồng nhất biểu diễn token (Representation Consistency). Khi DeBERTa xử lý một câu đầu vào ghép mô tả, Cross-Attention sẽ làm biến đổi sâu sắc vector ẩn của câu. Nếu nhãn hiếm có mô tả mà nhãn thường không có, đầu phân loại chuỗi (Linear-CRF) sẽ nhận được các vector ẩn có phân phối không gian hoàn toàn khác nhau cho cùng một từ. Điều này làm rối loạn gradient, gây khó khăn cho việc tối ưu hóa trọng số của đầu phân loại. Việc đồng nhất đầu vào `[CLS] s [SEP] d_c [SEP]` là bắt buộc để mô hình học hiệu quả.

### 2. Liệu mô hình có bị phụ thuộc vào sự tồn tại của mô tả không?
**Trả lời**: Có, mô hình Cross-Encoder **bắt buộc phải nhận mô tả ở đầu vào để hoạt động**. Đây không phải là điểm yếu mà là **thiết kế giao thức (Interface)** của mô hình. Trong thực tế, các đoạn mô tả của các nhãn đích đã được chuẩn bị sẵn và lưu tĩnh trong file cấu hình, hệ thống sẽ tự động ghép nối và chạy suy luận mà không cần người dùng hay bác sĩ phải viết mô tả cho mỗi câu.

### 3. Entity Masking Regularizer giải quyết triệt để học vẹt như thế nào?
**Trả lời**: Bằng cách che giấu ngẫu nhiên thực thể mục tiêu khi huấn luyện, mô hình không thể dựa vào sự trùng khớp ký tự thô giữa câu và mô tả (ví dụ từ `"chụp X-quang"` xuất hiện ở cả hai phần). Mô hình bắt buộc phải học cách ánh xạ ngữ cảnh ngữ pháp lâm sàng và biểu diễn trừu tượng của từ để đưa ra quyết định gán nhãn, từ đó đạt được khả năng suy luận ngữ nghĩa thực sự thay vì học vẹt.