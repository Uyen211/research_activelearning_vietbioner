# Phân tích Đặc tính Bộ dữ liệu và Yêu cầu Mô hình hóa (Tài liệu 01)

Tài liệu này tập trung làm rõ đặc tính của bộ dữ liệu `VietBioNER` sau khi tiền xử lý và chỉ ra các yếu tố phân phối dữ liệu ảnh hưởng trực tiếp đến hiệu năng của mô hình.

---

## 1. Điểm Bất thường Cốt lõi: Sự Thiếu vắng Hoàn toàn của Nhãn B-

Qua rà soát chi tiết tệp dữ liệu thô gốc tại [train.txt](file:///d:/Study/TLU/Nlp/active-learning-VietBioNER/colab/dataset/vietbioner/train.txt) và tệp sau tiền xử lý [train_segmented.txt](file:///d:/Study/TLU/Nlp/active-learning-VietBioNER/colab/dataset/preprocessed/train_segmented.txt), chúng tôi phát hiện một đặc điểm bất thường vô cùng nghiêm trọng của bộ dữ liệu này:

*   **Không tồn tại bất kỳ nhãn `B-` nào**: Trong toàn bộ tập dữ liệu (Train, Dev, Test), tất cả các thực thể y sinh đều được gán nhãn bắt đầu bằng tiền tố `I-` (ví dụ: `I-Organisation`, `I-Location`, `I-Symptom_and_Disease`), ngay cả đối với từ/syllable đầu tiên của thực thể.
*   **Ví dụ thực tế**:
    *   Cụm từ *"Tổ chức Y tế thế giới"* được gán nhãn là:
        ```text
        Tổ_chức I-Organisation
        Y_tế I-Organisation
        thế_giới I-Organisation
        ```
        (Trong khi chuẩn BIO đúng phải là: `Tổ_chức B-Organisation`, các từ tiếp theo là `I-Organisation`).
    *   Cụm từ đơn lẻ *"2010"* được gán nhãn là:
        ```text
        2010 I-DateTime
        ```
        (Đúng chuẩn phải là: `2010 B-DateTime`).

### Ảnh hưởng trực tiếp đến việc mô hình hóa:
1.  **CRF Head bị lãng phí trạng thái**: CRF Head được khởi tạo với 3 nhãn đầu ra: `O` (0), `B` (1), và `I` (2). Tuy nhiên, do dữ liệu huấn luyện đầu vào chỉ chứa nhãn `O` và `I`, mô hình **không bao giờ được tối ưu hóa cho nhãn `B`**.
2.  **Học chuyển tiếp (Transitions) bị lệch**: Mô hình CRF phải học cách chuyển trạng thái trực tiếp từ `O -> I` thay vì `O -> B -> I` theo thiết kế lý thuyết chuẩn. Mặc dù CRF có khả năng học được điều này, cấu hình 3 nhãn làm loãng xác suất phân phối emissions của mô hình.

---

## 2. Mất cân bằng Lớp nghiêm trọng và Khởi động lạnh của Active Learning

Bộ dữ liệu VietBioNER có mức độ mất cân bằng lớp cực kỳ cao:
*   `Symptom_and_Disease` chiếm **56.4%** tổng số thực thể.
*   `Organisation` chỉ chiếm **5.5%** (184 thực thể trên toàn bộ 1.706 câu của corpus).
*   `DateTime` chiếm **7.5%** (250 thực thể).

Trong vòng lặp khởi tạo ban đầu (Vòng 0), hệ thống chọn ra **Seed Set $L_0$ gồm 85 câu** (5% dữ liệu) bằng phân cụm K-Means ngữ nghĩa:
*   **Tần suất thực tế của nhãn thiểu số cực kỳ thấp**: Với 85 câu, số thực thể `Organisation` trung bình chỉ xuất hiện khoảng **8 - 9 lần**, và `Location` chỉ xuất hiện khoảng **20 lần** trong toàn bộ tập huấn luyện ban đầu.
*   **Hậu quả**:
    *   Mô hình học sâu (ViPubmedDeBERTa-BiLSTM-CRF) với 86M tham số dễ dàng rơi vào trạng thái dự đoán toàn bộ các nhãn thiểu số thành `O` để giảm thiểu tối đa hàm loss một cách dễ dàng (bởi vì số lượng token `O` chiếm ưu thế tuyệt đối ~95% tổng số token).
    *   Hiện tượng này được thể hiện rõ ràng qua kết quả F1-score của `Location` và `Organisation` đều bằng **0.00** ở Vòng 0, 1 và 2.

---

## 3. Các Nghiên cứu Trước đây Đã Làm Thế Nào?

Theo các tài liệu tham chiếu tại [vietbioner.md](file:///d:/Study/TLU/Nlp/active-learning-VietBioNER/proposal/02_dataset_analysis/vietbioner.md):
*   Các baseline như **PhoBERT-base** hoặc **XLM-R-base** đạt F1-score khoảng **0.72 - 0.74** khi được huấn luyện **Supervised trên toàn bộ 100% tập dữ liệu** (không phải lặp AL trên tập nhỏ).
*   Khi huấn luyện supervised đầy đủ, mô hình có cơ hội tiếp cận toàn bộ 184 thực thể `Organisation` và 400 thực thể `Location`, giúp nó vượt qua được ngưỡng hội tụ tối thiểu.
*   Tuy nhiên, trong kịch bản Active Learning bắt đầu từ 85 câu ($L_0$), nếu kiến trúc mô hình hoặc luồng dữ liệu gặp lỗi, mô hình sẽ hoàn toàn không thể học được các lớp nhãn thiểu số này từ tập dữ liệu siêu nhỏ, dẫn đến underfitting kéo dài.
