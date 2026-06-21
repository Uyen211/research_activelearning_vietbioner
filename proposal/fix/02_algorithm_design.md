# Đánh giá Thiết kế Thuật toán và Lỗi Kiến trúc BiLSTM (Tài liệu 02)

Tài liệu này tập trung phân tích thiết kế của lớp mô hình `MultiTaskBiLSTMCRF` và chỉ ra lỗi kiến trúc nghiêm trọng trong cách kết hợp BiLSTM với cơ chế Ghép nối Mô tả Thực thể (Entity Type Description) và xử lý padding.

---

## 1. Cơ chế Entity Type Description (Mô tả Thực thể)

Cơ chế này (được lấy cảm hứng từ mô hình *OPENBIONER* [12]) thiết kế đầu vào dưới dạng chuỗi ghép nối:
$$\text{Input} = \text{[CLS]} + s + \text{[SEP]} + d_c + \text{[SEP]} + \text{[PAD]}...$$
Trong đó:
*   $s$ là câu gốc (ví dụ: *"Các kỹ_thuật nhuộm..."*).
*   $d_c$ là câu mô tả tự nhiên của thực thể truy vấn $c$ (ví dụ: *"quy trình chẩn đoán xét nghiệm..."*).

Mục tiêu là thông qua cơ chế Self-Attention của DeBERTa, các token trong câu $s$ sẽ được nhúng ngữ cảnh và **conditioned** (phụ thuộc vào) mô tả thực thể $d_c$. Điều này giúp mô hình nhận diện động xem token đó có thuộc thực thể $c$ hay không.

---

## 2. Lỗi Kiến trúc Nghiêm trọng của BiLSTM Layer

Mô hình hiện tại đang xử lý đầu ra của DeBERTa bằng cách truyền trực tiếp tensor `last_hidden_state` có kích thước cố định là `(batch_size, 256, hidden_size)` qua một lớp `BiLSTM` mà **không đóng gói (pack) các chuỗi có độ dài thực tế**:

```python
lstm_out, _ = self.bilstm(last_hidden)  # last_hidden.shape = (batch_size, 256, 1024)
```

Điều này dẫn đến sự triệt tiêu thông tin mô tả truy vấn ở cả hai hướng di chuyển của BiLSTM đối với các token trong câu $s$:

### 2.1. Hướng Forward LSTM (Từ trái qua phải)
*   Forward LSTM bắt đầu từ `[CLS]`, đi qua các token của câu gốc $s$, rồi mới tới `[SEP]` và $d_c$.
*   Tại các vị trí token thuộc câu $s$ (nơi cần đưa ra dự đoán thực thể), Forward LSTM **chưa bao giờ đọc tới** mô tả thực thể $d_c$ (vì $d_c$ nằm ở cuối chuỗi).
*   Do đó, đặc trưng Forward của LSTM tại câu $s$ hoàn toàn không mang thông tin gì về loại thực thể đang được truy vấn.

### 2.2. Hướng Backward LSTM (Từ phải qua trái)
*   Backward LSTM bắt đầu từ vị trí index 255 (cuối chuỗi padding), đi ngược qua hàng trăm token `[PAD]`, sau đó qua `[SEP]`, qua mô tả $d_c$, rồi mới đến các token của câu $s$.
*   Với độ dài câu trung bình là 31 tokens và `MAX_LEN=256`, số lượng token `[PAD]` trung bình là **hơn 200 tokens**.
*   **Hiện tượng phai nhạt thông tin**: Khi đi qua hơn 200 token `[PAD]` vô nghĩa, các trạng thái ẩn (hidden states) và ô nhớ (cell states) của Backward LSTM bị bão hòa bởi thông tin padding. Khi đi ngược qua đoạn mô tả ngắn $d_c$ (chỉ 6-8 tokens), thông tin này bị pha loãng cực kỳ mạnh và bị lãng quên khi Backward LSTM chạm tới câu $s$.
*   Do đó, đặc trưng Backward của LSTM tại câu $s$ cũng bị mất đi thông tin conditioning về loại thực thể.

---

## 3. Hậu quả đối với Hiệu năng Mô hình

Vì cả hai hướng Forward và Backward của BiLSTM đều không thể truyền tải hiệu quả thông tin của mô tả thực thể $d_c$ đến các token trong câu $s$:
1.  **Vô hiệu hóa cơ chế mô tả thực thể**: Đầu ra của lớp BiLSTM tại các token câu gốc gần như giống hệt nhau đối với cả 5 loại thực thể khác nhau.
2.  **Mâu thuẫn dữ liệu huấn luyện**: Mô hình bị ép buộc phải dự đoán các nhãn khác nhau (ví dụ: một truy vấn là `O`, một truy vấn là `I-Organisation`) cho các đặc trưng đầu vào gần như trùng khớp.
3.  **Hội tụ về nhãn đa số**: Mô hình không thể phân biệt các truy vấn nhãn, dẫn đến việc chọn giải pháp an toàn nhất là dự đoán tất cả thành `O` (majority class), khiến F1-score của các lớp thiểu số rơi về **0.00** và mô hình bị underfit trầm trọng.
