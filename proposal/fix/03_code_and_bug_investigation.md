# Rà soát Lỗi Code và Logic Dữ liệu Đầu vào (Tài liệu 03)

Tài liệu này trình bày kết quả rà soát chi tiết mã nguồn, logic gán nhãn và làm rõ các hiểu lầm trong quá trình giám sát log debug đầu vào.

---

## 1. Mismatch trạng thái của CRF Head

Trong lớp `EntityMaskingCollator` ([cell_07.py](file:///C:/Users/Admin/.gemini/antigravity-ide/brain/1a923252-cf62-43f9-9226-5dcb22a6c310/scratch/cell_07.py)), tập nhãn đích được ánh xạ như sau:
```python
self.tag2id = {'O': 0, 'B': 1, 'I': 2}
```
Tuy nhiên, như đã chỉ ra ở [Tài liệu 01](file:///d:/Study/TLU/Nlp/active-learning-VietBioNER/proposal/fix/01_dataset_analysis.md), bộ dữ liệu VietBioNER chỉ chứa nhãn `I-` và `O`. Do đó:
*   Mảng nhãn chuẩn `tags` đưa vào huấn luyện chỉ chứa các giá trị `0` (O) và `2` (I). Giá trị `1` (B) hoàn toàn không xuất hiện.
*   **Lãng phí và nhiễu CRF**: CRF Head vẫn được cấu hình với 3 trạng thái. Khi huấn luyện, các tham số chuyển tiếp liên quan đến trạng thái `1` (B) không được cập nhật hoặc cập nhật sai lệch, làm giảm tính tối ưu của thuật toán Viterbi và Forward-Backward.

---

## 2. Giải thích Hiện tượng Debug Log chỉ hiển thị Seq IDs = 0

Trong cell huấn luyện Nhánh A, log in debug có dạng:
```text
=== LOG DEBUG TOKENIZATION (KIỂM TRA TẬP ĐẦU VÀO) ===
Câu gốc: ['Các', 'kỹ_thuật', 'nhuộm', 'đặc_biệt', 'còn', 'chưa', 'phổ_biến', 'và', 'không', 'phải', 'bất_cứ', 'bệnh_viện', 'nào', 'cũng', 'làm', 'được', '.']
Mô tả truy vấn: triệu_chứng lâm_sàng bệnh_lý dấu_hiệu bệnh lao_phổi
Word IDs: [None, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
Seq IDs: [None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Valid Mask: [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
```

### Phân tích:
*   **Không phải lỗi Tokenizer**: Lỗi ngộ nhận cho rằng tokenizer bỏ qua chuỗi mô tả truy vấn thứ hai ($d_c$) là **không chính xác**.
*   **Nguyên nhân cắt lát in**: Lệnh in debug trong [cell_14.py](file:///C:/Users/Admin/.gemini/antigravity-ide/brain/1a923252-cf62-43f9-9226-5dcb22a6c310/scratch/cell_14.py) chỉ in 15 phần tử đầu tiên (`[:15]`):
    ```python
    print(f"Word IDs: {sample_word_ids[:15]}")
    print(f"Seq IDs: {sample_seq_ids[:15]}")
    ```
*   Vì câu gốc có 17 từ (sau khi tách subwords sẽ nhiều hơn 17 tokens), phần tử thứ 15 của chuỗi token vẫn nằm trong câu gốc $s$ (thuộc `Seq ID == 0`). Phần chuỗi mô tả truy vấn thực chất nằm ở các index phía sau (bắt đầu từ index 21 trở đi, có `Seq ID == 1`) nên đã bị cắt mất khi in ra màn hình.
*   **Kết luận**: Luồng xử lý tokenizer của Hugging Face hoạt động hoàn toàn chính xác đối với đầu vào ghép nối khi được cấu hình `is_split_into_words=True`.

---

## 3. Thuật toán Giải quyết Xung đột và Gộp nhãn

Thuật toán `merge_and_resolve_conflicts` trong [cell_09.py](file:///C:/Users/Admin/.gemini/antigravity-ide/brain/1a923252-cf62-43f9-9226-5dcb22a6c310/scratch/cell_09.py) thực hiện việc gộp 5 chuỗi nhãn nhị phân động về 1 chuỗi nhãn đa lớp BIO:
*   Nếu có sự chồng lấn ranh giới thực thể, thuật toán sử dụng điểm xác suất biên trung bình (Average Marginal Probability) được tính bằng thuật toán Forward-Backward trên CRF làm tiêu chí xếp hạng.
*   **Cơ chế giải quyết hòa (Tie-breaker)**: Ưu tiên thực thể ngắn hơn để đảm bảo tính thận trọng trong y sinh, và ưu tiên nhãn xuất hiện trước theo thứ tự tĩnh.
*   **Nhận xét**: Logic gộp nhãn hoàn toàn đúng đắn. Lý do F1-score thấp hoặc bằng 0.0 không nằm ở hàm gộp nhãn, mà vì mô hình underfit nặng nên các chuỗi Viterbi giải mã hầu như chỉ chứa toàn nhãn `O`, không tạo ra bất kỳ `candidate_spans` nào để gộp.
