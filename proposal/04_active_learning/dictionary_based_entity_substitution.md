# Giải Thích Cơ Chế Tăng Cường Dữ Liệu Bằng Thế Thực Thể Dựa Trên Từ Điển (Dictionary-based Entity Substitution)

---

Cơ chế **Dictionary-based Entity Substitution (DES - Tăng cường Dữ liệu bằng Thế Thực Thể Dựa Trên Từ Điển)** kế thừa và phát triển từ các nghiên cứu về tăng cường dữ liệu dựa trên từ điển chuyên ngành trong NLP [2]. Trong nghiên cứu này, DES đóng vai trò là giải pháp cốt lõi để khắc phục hiện tượng mất cân bằng lớp nhãn nghiêm trọng của bộ dữ liệu `VietBioNER` dưới kịch bản dữ liệu gán nhãn ban đầu cực kỳ khan hiếm.

Dưới đây là thuyết minh kỹ thuật chi tiết về bối cảnh, quy trình thế thực thể y khoa (Entity Substitution) và cách thức tích hợp hiệu quả của DES vào vòng lặp Active Learning.

---

## 1. Bối Cảnh: Thách Thức Mất Cân Bằng Nhãn Lâm Sàng

Bộ dữ liệu y học tiếng Việt `VietBioNER` [5] (chuyên sâu về bệnh lao) chứa 5 loại thực thể y khoa phức tạp. Tần suất xuất hiện của các thực thể này trong tập huấn luyện gốc bị lệch lệch cực kỳ lớn:
*   Các thực thể phổ biến: `Disease` (Bệnh), `Symptom` (Triệu chứng) xuất hiện dày đặc.
*   Các thực thể hiếm gặp: `DiagnosticProcedure` (Quy trình chẩn đoán), `Organisation` (Tổ chức y tế) xuất hiện với tần suất rất thấp.

Khi chạy Active Learning, ở các vòng lặp đầu tiên ($t = 0, 1, 2$), tập dữ liệu được gán nhãn ($L_t$) có quy mô cực kỳ nhỏ (ví dụ Seed Set $L_0$ chỉ chiếm xấp xỉ 8% tập huấn luyện mới, tương đương khoảng 85 câu). Số lượng câu chứa thực thể hiếm trong $L_t$ có thể chỉ đếm trên đầu ngón tay (từ 1 đến 3 câu). 

### Hậu quả:
Nếu huấn luyện trực tiếp mô hình học sâu trên tập dữ liệu quá nhỏ và mất cân bằng như vậy, mô hình sẽ phớt lờ hoàn toàn các lớp nhãn hiếm (đưa F1-score của các lớp này về xấp xỉ 0%). Điều này làm giảm tính toàn diện của hệ thống NER lâm sàng.

---

## 2. Quy Trình 3 Bước Triển Khai Thực Tế

Cơ chế DES sử dụng phương pháp **Thế thực thể (Entity Substitution)** để tự động nhân bản dữ liệu chất lượng cao từ các ngữ cảnh chứa nhãn hiếm có sẵn mà không cần tốn thêm chi phí gán nhãn của chuyên gia y tế.

```
       [Từ điển Gazetteer tĩnh]
                  │ (Lấy ngẫu nhiên thực thể cùng loại khác)
                  ▼
[Câu gốc chứa thực thể hiếm trong L_t] ──> [Thế thực thể] ──> [Các câu augmented mới s_aug] ──> [Đưa vào L_t,aug]
```

### Bước A: Chuẩn bị Cơ sở Tri thức Thuật ngữ Y khoa (Gazetteer / Knowledge Base) & Lọc Rò rỉ Dữ liệu
Hệ thống xây dựng một từ điển Gazetteer chứa danh sách các từ vựng thuộc loại thực thể hiếm gặp. Đối với `DiagnosticProcedure`, danh sách Gazetteer bao gồm các quy trình, xét nghiệm y khoa:
*   *Gazetteer mẫu*: `{"chụp X-quang phổi", "nội soi phế quản", "xét nghiệm đờm", "làm phản ứng Mantoux", "siêu âm màng phổi", "chụp CT lồng ngực", "sinh thiết phổi"}`.
*   **Nguyên tắc phòng chống Rò rỉ Dữ liệu (Data Leakage Prevention)**: Để đảm bảo tính khách quan của thực nghiệm và tuân thủ chặt chẽ nguyên tắc khoa học, trước khi đưa từ điển Gazetteer vào sử dụng, hệ thống bắt buộc phải quét và **loại bỏ vô điều kiện tất cả các cụm thực thể xuất hiện trong tập Validation và tập Test cố định** ra khỏi Gazetteer tĩnh (bất kể thực thể đó có nằm trong tập Train gốc hay không). Việc này ngăn chặn triệt để mô hình "học trước" các từ khóa kiểm thử thông qua các câu tăng cường trong tập Train.

### Bước B: Quét và Lọc câu ứng viên trong L_t
Ở mỗi vòng lặp AL $t$, trước khi đưa dữ liệu vào huấn luyện, hệ thống thực hiện quét qua toàn bộ tập dữ liệu đã gán nhãn $L_t$ hiện tại để tìm ra tất cả các câu có chứa thực thể hiếm cần tăng cường.
*   *Ví dụ tìm được câu gốc trong $L_t$*: 
    `"Bệnh nhân nghi lao phổi được chỉ định làm phản ứng Mantoux [DiagnosticProcedure] ."`

### Bước C: Thế thực thể để sinh dữ liệu mới và Tự động điều chỉnh nhãn (Giới hạn số lượng tăng cường)
Hệ thống giữ nguyên toàn bộ cấu trúc ngữ cảnh của câu gốc, thay thế cụm từ thực thể hiếm gốc (`"phản ứng Mantoux"`) bằng các từ vựng khác lấy ngẫu nhiên từ Gazetteer ở Bước A.
*   **Cơ chế giới hạn số lượng (Augmentation Constraint)**: Để tránh hiện tượng đảo ngược lớp nhãn (Class Inversion - biến lớp hiếm thành lớp đa số áp đảo) và tránh hiện tượng mô hình học vẹt mẫu câu (Template Overfitting), chúng ta giới hạn số câu tăng cường được sinh ra bằng cách **chỉ lấy ngẫu nhiên một số lượng giới hạn $M$ thực thể** (với $M = 2$ đến $3$) từ Gazetteer để thay thế cho mỗi câu gốc.
*   **Hệ thống tự động điều chỉnh**: Tự động tính toán lại độ dài token của thực thể mới để gán nhãn BIO nhị phân chuẩn xác cho mỗi câu biến thể (thuật toán tịnh tiến nhãn BIO khi số lượng từ thay đổi).

*   **Câu augmented 1 (với $M=3$)**: `"Bệnh nhân nghi lao phổi được chỉ định làm chụp X-quang phổi [DiagnosticProcedure] ."`
    *   *BIO Target tự động*: `chụp (B)` `X-quang (I)` `phổi (I)`
*   **Câu augmented 2**: `"Bệnh nhân nghi lao phổi được chỉ định làm nội soi phế quản [DiagnosticProcedure] ."`
    *   *BIO Target tự động*: `nội (B)` `soi (I)` `phế (I)` `quản (I)`
*   **Câu augmented 3**: `"Bệnh nhân nghi lao phổi được chỉ định làm xét nghiệm đờm [DiagnosticProcedure] ."`
    *   *BIO Target tự động*: `xét (B)` `nghiệm (I)` `đờm (I)`

Tất cả các câu mới sinh ra tự động này được gộp vào tập gán nhãn hiện tại $L_t$ để tạo thành tập huấn luyện mở rộng $L_{t,\text{aug}}$ để nạp vào huấn luyện mô hình ở Bước 5.

---

## 3. Lý Luận Khoa Học Về Sự Phối Hợp Giữa Active Learning và Thế Thực Thể Dựa Trên Từ Điển

Sự phối hợp giữa hai kỹ thuật này tạo ra hiệu ứng cộng hưởng mạnh mẽ (Symbiotic Relationship):

1.  **Active Learning định hướng ngữ cảnh**: Active Learning (thông qua CRF Marginal Entropy và Distinct-K) chịu trách nhiệm lọc chọn ra các câu "đắt giá" nhất trong không gian chưa gán nhãn $U_t$ chứa các thực thể hiếm hoặc ranh giới khó. Chuyên gia y tế chỉ cần gán nhãn 1 câu duy nhất này.
2.  **DES phóng đại ngữ cảnh**: Sau khi câu đắt giá đó được gán nhãn và đưa vào $L_t$, cơ chế thế thực thể dựa trên từ điển (DES) lập tức thực hiện ghép nối, sử dụng Gazetteer để nhân bản và phóng đại ngữ cảnh đó lên gấp $N$ lần.
3.  **Học ngữ cảnh sâu thay vì học từ vựng**: Bằng cách thay thế nhiều từ vựng khác nhau vào cùng một vị trí ngữ cảnh (ví dụ: *"được chỉ định làm [DiagnosticProcedure]"*), mô hình được ép buộc phải học cấu trúc ngữ pháp và đặc trưng phân loại của ngữ cảnh xung quanh thực thể, thay vì chỉ ghi nhớ máy móc các từ vựng cụ thể. Điều này giúp nâng cao đáng kể khả năng nhận diện các thực thể hiếm ở các câu y khoa khác ngoài đời thực.
