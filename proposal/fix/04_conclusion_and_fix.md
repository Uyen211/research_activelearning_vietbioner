# Kết luận và Kế hoạch Nâng cao Hiệu năng Mô hình (Tài liệu 04)

Tài liệu này tổng hợp các quyết định kỹ thuật cốt lõi đã thực hiện (bao gồm việc loại bỏ lớp BiLSTM) và đề xuất các giải pháp nâng cao nhằm tối ưu hóa khả năng hội tụ và F1-score của mô hình trong điều kiện giới hạn ngân sách gán nhãn của Học chủ động (Active Learning).

---

## 1. Quyết định Kỹ thuật Cốt lõi: Loại bỏ lớp BiLSTM (Đã duyệt và áp dụng)

Qua thực nghiệm và phân tích lý thuyết tại [Tài liệu 02](file:///d:/Study/TLU/Nlp/active-learning-VietBioNER/proposal/fix/02_algorithm_design.md), chúng tôi chính thức **duyệt phương án loại bỏ lớp BiLSTM** khỏi kiến trúc mô hình `MultiTaskBiLSTMCRF`. 

### Lý do phê duyệt:
1.  **Giải quyết lỗi pha loãng thông tin (Padding Dilution)**: DeBERTa tự động tích hợp thông tin mô tả truy vấn ($d_c$) vào câu gốc ($s$) nhờ cơ chế Self-Attention đa đầu. Việc đưa thêm lớp BiLSTM chạy qua chuỗi padding cố định (`MAX_LEN=256`) sẽ triệt tiêu hoàn toàn thông tin $d_c$ ở hướng di chuyển ngược (Backward LSTM) do phải đi qua hơn 200 token `[PAD]` vô nghĩa.
2.  **Tăng tốc độ hội tụ và giảm tài nguyên**: Loại bỏ BiLSTM giúp giảm số lượng tham số cần tối ưu hóa, giảm thời gian huấn luyện mỗi epoch trên GPU/CPU xuống hơn 40%, đồng thời tránh được các lỗi tính toán ranh giới khi không đóng gói (pack) chuỗi tuần tự.
3.  **Tương thích hoàn hảo với CRF**: Đầu ra `last_hidden_state` của DeBERTa sau khi đi qua lớp Linear (`self.fc`) tạo ra các điểm số phát xạ (emissions) cực kỳ mạnh mẽ cho lớp CRF giải mã.

---

## 2. Quyết định Chốt Ngưỡng dữ liệu (Budget Limit) là 50%

Qua phân tích thực nghiệm và phản hồi ý kiến chuyên môn, chúng tôi quyết định **chốt giới hạn ngân sách gán nhãn ở mức 50% tập huấn luyện** (tương đương tối đa 682 câu thô trên tổng số 1.365 câu của tập Train gốc):

### Lý do chốt ngưỡng 50% (thay vì 30%):
*   **Đảm bảo số lượng mẫu của các lớp thiểu số**: Với quy mô nhỏ của VietBioNER (1.365 câu Train), việc giới hạn ở 30% (408 câu) sẽ khiến nhãn `Organisation` (chỉ có 168 thực thể trong toàn bộ Train set) xuất hiện dưới 50 lần, còn nhãn `Location` dưới 80 lần. Số lượng này quá ít để mô hình ViPubmedDeBERTa-base có thể khái quát hóa các ranh giới thực thể phức tạp. Tăng lên 50% sẽ nâng số mẫu thực tế của các lớp thiểu số này lên mức tối thiểu cần thiết để mô hình hội tụ tốt.
*   **Bảo đảm tính thực tiễn**: Tiết kiệm được **50% công sức gán nhãn** vẫn là một con số vô cùng ý nghĩa trong miền y sinh (giúp giảm tải một nửa thời gian làm việc của các chuyên gia y tế), trong khi vẫn bảo đảm F1-score của mô hình tiệm cận tối đa mức huấn luyện toàn phần.

### Cơ chế dừng thông minh bổ trợ:
*   **Cơ chế Ngắt Động (Adaptive Stopping Criterion)**: Bên cạnh ngưỡng cứng 50%, hệ thống tích hợp bộ giám sát F1-score trên tập Validation. Nếu chỉ số F1-score không tăng quá `0.005` (0.5%) trong 3 vòng liên tiếp, hệ thống tự động ngắt vòng lặp AL sớm để tránh lãng phí thêm tài nguyên gán nhãn vô ích.
*   **Cơ chế cảnh báo**: Nếu đạt giới hạn 50% mà F1-score trên tập Test vẫn dưới 75.0%, hệ thống tự động ghi nhận cảnh báo để nhà nghiên cứu phân tích chất lượng mẫu và độ khó của thực thể.

---

## 3. Chọn lọc Giải pháp Thực tế Chống Overfitting & Tăng tốc Hội tụ

### 3.1. Phân tích bối cảnh thực nghiệm hiện tại
*   **Môi trường chạy**: Notebook chạy đa nền tảng (Local CPU/GPU, Colab, Kaggle).
*   **Hạn chế của một số giải pháp**:
    *   *LoRA/PEFT*: Dễ gây ra lỗi xung đột phiên bản thư viện (`peft` và `transformers`) khi cài đặt trên môi trường Kaggle/Colab, gây khó khăn cho việc chạy tự động.
    *   *Context Perturbation (Dịch ngược / Thay từ đồng nghĩa)*: Yêu cầu cài đặt thêm các mô hình dịch hoặc API từ điển bên ngoài, làm chậm tốc độ của luồng xử lý dữ liệu và tăng độ phức tạp của code một cách không cần thiết.
*   **Giải pháp được lựa chọn**: Ưu tiên các phương pháp **không phụ thuộc thư viện ngoài (zero-dependency)**, trực tiếp can thiệp vào luồng dữ liệu (Data Pipeline) và có hiệu quả cao.

---

### 3.2. Chi tiết Giải pháp được Phê duyệt Triển khai

#### Giải pháp A: Negative Query Downsampling (Giảm mẫu truy vấn âm tính trong Train)
*   **Vấn đề**: Việc nhân bản câu $5\times$ để ghép mô tả nhãn dẫn đến việc lặp lại ngữ cảnh y hệt nhau 15-20 lần (sau khi gộp Distant Supervision), tạo ra lượng nhãn `O` áp đảo (>98%) và gây ra overfitting học thuộc lòng ngữ cảnh.
*   **Cách triển khai**: Trong tập dữ liệu huấn luyện, thay vì ghép mỗi câu với cả 5 nhãn mô tả (sinh ra 4 truy vấn âm tính), chúng ta:
    *   **Giữ lại 100% Truy vấn dương tính** (nhãn thực thể thực tế có trong câu).
    *   **Chỉ lấy ngẫu nhiên 1 truy vấn âm tính** (nơi thực thể không xuất hiện trong câu).
*   **Hiệu quả**: 
    *   Giảm ngay lập tức **60% số lượng mẫu trùng lặp** trong batch huấn luyện, tăng tốc độ chạy mỗi epoch lên gấp đôi.
    *   Giảm đáng kể hiện tượng "Context Memorization" và cân bằng lại phân bố nhãn.

#### Giải pháp B: Contextual Word Masking (Masking ngẫu nhiên cả từ ngữ cảnh)
*   **Vấn đề**: Các từ ngữ cảnh xung quanh thực thể bị lặp lại quá nhiều lần khiến DeBERTa học thuộc lòng vị trí thay vì học ngữ nghĩa thực thể.
*   **Cách triển khai**: Cập nhật `EntityMaskingCollator` trong pha Train: Ngoài việc mask ngẫu nhiên 18% token thực thể, ta thực hiện **mask thêm 12% - 15% các token thông thường (nhãn O)**.
*   **Hiệu quả**: Phá vỡ cấu trúc template hoàn hảo của các câu trùng lặp, buộc mô hình phải dựa vào cú pháp và ngữ cảnh rộng hơn để suy luận thực thể, cải thiện F1-score đáng kể trên tập Test sạch.

#### Giải pháp C: Tối ưu hóa Loss và Tốc độ Học (Weighted Loss & LLRD)
*   **Cách triển khai**:
    *   *Weighted CRF Loss*: Gán trọng số phạt lỗi lớn hơn cho nhãn thực thể `B` (hệ số 2.0) và `I` (hệ số 1.5) so với nhãn `O` (hệ số 1.0) khi tính log-likelihood.
    *   *Layer-wise Learning Rate Decay (LLRD)*: Giữ nguyên LR của DeBERTa backbone ở mức cực thấp (`2e-5`) để bảo toàn tri thức pre-trained, và đặt LR của FC/CRF ở mức cao (`5e-4` hoặc `1e-3`) để các lớp phân loại phía sau hội tụ nhanh hơn.
*   **Hiệu quả**: Mô hình học nhanh các thực thể hiếm ngay từ các vòng AL đầu tiên mà không làm biến dạng các trọng số ngôn ngữ của mô hình gốc.

