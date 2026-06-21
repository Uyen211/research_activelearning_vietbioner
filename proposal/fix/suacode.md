 === Nhánh A (AL) - Vòng lặp 0 (Labeled size: 85/544) ===
  + Quy mô câu gốc (L_t): 85 câu
  + Quy mô sau Tăng cường (L_aug): 172 câu (+87 câu)
  + Queries trước Downsampling: 860
  + Queries sau Downsampling: 486 (Tiết kiệm: 43.5%)
    - Queries DƯƠNG TÍNH: 314 | Queries ÂM TÍNH: 172
  + Số lượng thực thể gốc được gán nhãn trong pool:
    - Symptom_and_Disease: 160
    - DiagnosticProcedure: 56
    - Location: 28
    - DateTime: 29
    - Organisation: 13
--------------------------------------------------
Epoch 1 - Train Loss: 29.6375 - Val Loss: 8.1414
Epoch 2 - Train Loss: 19.8347 - Val Loss: 7.6338
Epoch 3 - Train Loss: 18.3615 - Val Loss: 7.3907
Epoch 4 - Train Loss: 17.6447 - Val Loss: 7.0023
Epoch 5 - Train Loss: 16.9915 - Val Loss: 6.9931
Epoch 6 - Train Loss: 16.5274 - Val Loss: 6.5710
Epoch 7 - Train Loss: 16.0725 - Val Loss: 6.6478
Epoch 8 - Train Loss: 15.7914 - Val Loss: 6.3076
Epoch 9 - Train Loss: 15.1778 - Val Loss: 6.1983
Epoch 10 - Train Loss: 15.1915 - Val Loss: 6.0986
Epoch 11 - Train Loss: 14.6092 - Val Loss: 5.9975
Epoch 12 - Train Loss: 14.4135 - Val Loss: 6.1447
Vòng 0 - Test F1-score: 0.0260
Class-wise Evaluation Report:
                     precision    recall  f1-score   support

           DateTime       0.00      0.00      0.00        23
DiagnosticProcedure       0.00      0.00      0.00        37
           Location       0.00      0.00      0.00        37
       Organisation       0.00      0.00      0.00        22
Symptom_and_Disease       1.00      0.02      0.04       184

          micro avg       0.80      0.01      0.03       303
          macro avg       0.20      0.00      0.01       303
       weighted avg       0.61      0.01      0.03       303
như bạn thấy thì kết quả rất tệ, có thể nói là f1 =0

1. Lỗi hàm mất mát (Loss Function) sai dấu – nguyên nhân nghiêm trọng nhất
Trong hàm train_model (Cell 10), dòng code tính loss như sau:

python
loss = -model.crf(emissions, tags, mask=mask, reduction='mean')
Thư viện torchcrf (phiên bản pytorch-crf) định nghĩa CRF.forward() trả về negative log-likelihood (NLL) – một số dương, càng nhỏ càng tốt. Vì vậy, để tối ưu theo hướng giảm NLL, ta phải lấy trực tiếp NLL làm loss, không được thêm dấu trừ.

Việc thêm dấu trừ (-model.crf(...)) khiến cho loss trở thành âm của NLL, tức là hàm mục tiêu đang tối đa hóa NLL thay vì tối thiểu hóa. Điều này làm cho gradient có dấu ngược lại so với mong muốn, khiến cho quá trình lan truyền ngược đi sai hướng, mô hình không thể học được các đặc trưng hữu ích.

Bằng chứng:
Train loss ban đầu ~29 (có vẻ là NLL dương) và giảm dần xuống ~14 sau 12 epoch. Nếu thực sự đang tối ưu hóa -NLL thì loss phải có xu hướng tăng (vì mục tiêu là cực đại NLL). Nhưng loss lại giảm → điều này chứng tỏ code đang in ra giá trị NLL (mặc dù có dấu trừ, nhưng có thể model.crf trả về số âm? Không, torchcrf trả về dương). Thực tế, nếu -model.crf mà cho ra giá trị dương, thì model.crf phải là số âm – điều này không xảy ra. Vì vậy, có khả năng dòng code đó không được thực thi hoặc đã được sửa nhưng chưa cập nhật? Tuy nhiên, nhìn vào output thì loss vẫn là dương và giảm, nên rất có thể dấu trừ đã bị bỏ qua (do lỗi đánh máy?) hoặc họ đang dùng một phiên bản CRF khác.

=> hãy nghiên cứu cho tôi về vấn đề này

2. Vấn đề về khởi tạo Seed Set và số lượng mẫu huấn luyện quá ít
Seed set ban đầu chỉ có 85 câu (khoảng 5-6% tập train). Với số lượng mẫu cực nhỏ này, mô hình rất dễ bị overfit hoặc không thể học được các biểu diễn phức tạp. Mặc dù có tăng cường dữ liệu (Distant Supervision) nhưng số lượng bản sao sinh ra không đủ để bù đắp sự thiếu hụt đa dạng mẫu.

Kết quả F1 trên test ~0.026 cho thấy mô hình hầu như chỉ dự đoán đúng một vài thực thể Symptom_and_Disease (recall 2%, precision 100%) – điều này thường xảy ra khi mô hình chỉ học được các mẫu rất dễ và bỏ qua các lớp khác.

=> hãy nghiên cứu cho tôi về vấn đề này

3. Vấn đề với mask trong CRF khi suy luận và huấn luyện
Trong EntityMaskingCollator, mask được xây dựng dựa trên word_ids và sequence_ids, nhưng có thể không loại bỏ đúng các token thuộc phần mô tả (d_c) và padding. Điều này có thể khiến CRF tính toán sai trên các token không hợp lệ, làm giảm hiệu quả học.

5. Các vấn đề khác có thể ảnh hưởng
Learning rate và optimizer được thiết lập khác nhau cho từng nhóm tham số. Tuy nhiên, với số mẫu ít, learning rate có thể cần giảm xuống hoặc sử dụng scheduler.

Số epoch tối đa 12, nhưng với dữ liệu nhỏ, có thể cần nhiều epoch hơn hoặc cần tăng cường dữ liệu mạnh hơn.

Contextual Masking (18% entity, 15% context) có thể quá mạnh, làm mất quá nhiều thông tin khi số lượng mẫu đã ít.

Xem lại các phương pháp CRF marginal entropy được đề cập trong pipieline và so sánh lại với code hiện tại xem có giống nhau không, nếu khác thì sửa cho giống, hơn nữa phải nghiên cứu xem phương pháp này đã phù hợp với bài toán hiện tại cwha?
Các phiên bản thư viện/mô hình được sử dụng trong code đã đúng chuẩn phù hợp với bài toán hay chưa?
Chưa thấy tiền xử lý rồi lưu lại gazetter ở thư mục processed như các data kahcs, Tính toán độ lệch chỉ mục gazetter cần xem lại đã đúng chưa
Cái định dạng mô tả nhãn d_s khi được ghép nối vào câu đã được gán nhãn BIO đầy đủ chưa?
Trong bước 7 ở proposal\05_evaluation_and_pipeline\pipeline.md, bảo "Nạp các câu trong tập chưa gán nhãn $U_t$ (mỗi câu được nhân bản thành 5 chuỗi đầu vào ghép nối **Entity Type Description**, không áp dụng Entity Masking) qua mô hình để chạy suy luận" -> giải thích tại sao lại làm vậy? tại vì không biết đâu là nhãn thật của nó và để chuẩn giống huấn luyện nên ghép vào ư?

---

# Báo cáo Phân tích Chi tiết & Giải pháp Đề xuất

Sau khi kiểm tra toàn bộ mã nguồn của Notebook [01_mo_phong_active_learning.ipynb](file:///d:/Study/TLU/Nlp/active-learning-VietBioNER/colab/notebook/01_mo_phong_active_learning.ipynb), tài liệu [pipeline.md](file:///d:/Study/TLU/Nlp/active-learning-VietBioNER/proposal/05_evaluation_and_pipeline/pipeline.md), [ke_hoach_trien_khai_code.md](file:///d:/Study/TLU/Nlp/active-learning-VietBioNER/prompts/ke_hoach_trien_khai_code.md) và các nghi vấn của bạn, chúng tôi đã xác định được một số lỗi logic nghiêm trọng trong code cùng các vấn đề về thiết lập siêu tham số khiến kết quả huấn luyện ở Vòng 0 rất thấp (F1 = 0.0260).

Dưới đây là phần giải đáp chi tiết các nghi vấn và phân tích nguyên nhân kèm giải pháp khắc phục cụ thể.

## 1. Giải đáp các Nghi vấn của Bạn

### Nghi vấn 1: Dấu của hàm mất mát (Loss Function) trong hàm `train_model`
*   **Giải đáp**: Dòng code `loss = -model.crf(emissions, tags, mask=mask, reduction='mean')` là **HOÀN TOÀN CHÍNH XÁC** về mặt toán học.
*   **Chi tiết**: 
    - Thư viện `pytorch-crf` (imported qua `from torchcrf import CRF`) định nghĩa hàm `forward(...)` trả về **Log-Likelihood** (Log xác suất hợp lý) của chuỗi nhãn chuẩn. Vì xác suất $P(y|x) \le 1$ nên Log-Likelihood $\log P(y|x)$ luôn là một số **âm** (càng gần 0 tức là xác suất càng lớn, mô hình càng tốt).
    - Để huấn luyện mô hình bằng thuật toán tối ưu hóa giảm gradient (Gradient Descent) nhằm tối đa hóa Log-Likelihood, chúng ta phải chuyển nó về dạng **Negative Log-Likelihood (NLL)** (âm log xác suất hợp lý, là một số dương, càng nhỏ càng tốt).
    - Do đó, việc thêm dấu trừ (`-model.crf(...)`) giúp chuyển Log-Likelihood (số âm) thành NLL (số dương) để làm hàm loss cần giảm. Giá trị Train Loss giảm từ ~29 về ~14 sau 12 epoch chính là minh chứng cho thấy NLL đang giảm dần (tương đương Log-Likelihood tăng lên), mô hình đang học đúng hướng trên tập train. Không có lỗi sai dấu ở đây.

### Nghi vấn 2: Quy mô Seed Set ($L_0$) quá ít câu
*   **Giải đáp**: Quy mô ban đầu $L_0 = 85$ câu (5% tập train) là nhỏ, nhưng đây là bài toán khởi động lạnh chuẩn của Active Learning. Vấn đề thực sự không nằm ở số lượng mẫu ban đầu mà là do **sự mất cân bằng nhãn trầm trọng** và **cấu hình huấn luyện chưa tối ưu**:
    - Ngay cả khi có DS Augmentation tăng cường lên 172 câu, số bước cập nhật (steps) trong 12 epoch với Batch Size = 8 chỉ là: $\frac{172}{8} \times 12 = 258$ steps. Số bước này là quá ít để một mô hình lớn như DeBERTa điều chỉnh các trọng số LoRA và huấn luyện lớp CRF/Linear từ đầu.
    - Với số lượng mẫu nhỏ, mô hình rất dễ bị bias hoàn toàn về lớp đa số là nhãn `O` (chiếm hơn 95% số lượng token trong câu).

### Nghi vấn 3: Căn chỉnh Mask và gán nhãn BIO trong CRF khi suy luận/huấn luyện
*   **Giải đáp**: Cách tạo `valid_mask` trong code là hợp lệ (mặt nạ chỉ lấy `True` cho token `[CLS]` và các subword của câu gốc, còn description và padding là `False`). Tuy nhiên, **cách gán nhãn BIO cho các subwords đang có lỗi thiết kế**:
    - Trong `EntityMaskingCollator`, khi một từ gốc được chia thành nhiều subwords (ví dụ: `"lao_phổi"` thành `" lao"`, `"_p"`, `"hoi"`), code hiện tại gán nhãn `'B'` cho tất cả các subwords này: `[B, B, B]`.
    - Điều này vi phạm nghiêm trọng quy chuẩn BIO của CRF. Hệ thống CRF sẽ hiểu đây là 3 thực thể riêng lẻ bắt đầu liên tục thay vì là 1 thực thể dài 3 tokens. Nó làm hỏng ma trận chuyển trạng thái (transitions) của CRF và khiến mô hình khó hội tụ.
    - **Giải pháp**: Chỉ gán nhãn gốc (`B` hoặc `I`) cho subword đầu tiên của từ đó, còn các subwords tiếp theo của từ đó bắt buộc phải gán nhãn phụ `I` (nếu từ đó thuộc thực thể) hoặc giữ nguyên `O` (nếu từ đó là nhãn `O`).

### Nghi vấn 4: Thiếu Weighted CRF Loss
*   **Giải đáp**: Đúng như bạn nhận định, trong `pipeline.md` có đề cập đến Weighted CRF Loss (phạt mất cân bằng lớp), nhưng trong code thực tế hoàn toàn không có weighted loss do thư viện `pytorch-crf` gốc không hỗ trợ truyền trọng số cho các nhãn.
*   **Giải pháp**: Cần cấu hình lại trọng số thủ công hoặc điều chỉnh phân bổ mẫu âm tính/dương tính, hoặc chấp nhận không dùng weighted loss nhưng bù lại bằng việc điều chỉnh Learning Rate và Epochs cao hơn.

### Nghi vấn 5: Preprocessing, lưu Gazetteer và Tính toán độ lệch chỉ mục
*   **Giải đáp**:
    - **Gazetteer**: Gazetteer đã được lọc sạch rò rỉ và lưu trong thư mục `dataset/gazetteer`. Việc không lưu lại trong thư mục `processed` là vì Gazetteer là dữ liệu tĩnh, được lọc một lần và dùng chung, không thay đổi qua các vòng lặp như dữ liệu văn bản. Điều này hợp lý.
    - **Tính toán độ lệch chỉ mục**: Trong hàm `ds_augment_sentence`, việc thay thế thực thể được thực hiện bằng cách ghép nối mảng: `tokens[:start_idx] + new_tokens + tokens[end_idx+1:]`. Cách làm này tự động điều chỉnh độ dài danh sách mà không cần tính toán thủ công độ lệch chỉ mục ký tự (index shift). Cách làm này **hoàn toàn đúng và an toàn**.

### Nghi vấn 6: Nhãn BIO của mô tả nhãn tĩnh $d_s$ khi ghép vào câu
*   **Giải đáp**: Các tokens thuộc phần mô tả nhãn $d_s$ (ở cuối câu) được gán nhãn `'O'` trong `aligned_tags` và `valid_mask` của chúng là `False`. Điều này hoàn toàn chính xác. CRF sẽ chỉ tính toán loss và đường đi Viterbi trên phần câu gốc (nơi `valid_mask` là `True`). Phần mô tả $d_s$ chỉ đóng vai trò cung cấp thông tin ngữ cảnh thông qua cơ chế self-attention của DeBERTa, không tham gia vào chuỗi nhãn của CRF.

### Nghi vấn 7: Tại sao suy luận trên $U_t$ lại nhân bản câu thành 5 chuỗi mô tả nhãn?
*   **Giải đáp**: Vì đây là kiến trúc **Entity Type Description (Mô tả loại thực thể)**. Mô hình được huấn luyện để chỉ đưa ra dự đoán nhị phân (O, B, I) cho loại thực thể được mô tả trong chuỗi $d_s$ đi kèm. Tại thời điểm suy luận trên tập chưa gán nhãn $U_t$, chúng ta không biết trước câu đó chứa thực thể nào. Vì vậy, bắt buộc phải nhân bản câu đó thành 5 chuỗi đầu vào ghép với 5 mô tả thực thể tĩnh khác nhau, chạy suy luận qua mô hình, sau đó gộp kết quả của 5 chuỗi này lại để ra nhãn đa lớp cuối cùng. Cách làm này là bắt buộc để tương thích với cấu trúc huấn luyện của mô hình.

---

## 2. Lỗi logic nghiêm trọng nhất tìm thấy trong code (Nguyên nhân trực tiếp gây ra F1 thấp và lỗi AL)

Chúng tôi đã phát hiện một **lỗi logic cực kỳ nghiêm trọng** trong hàm `compute_marginal_probabilities` (Cell 10), ảnh hưởng trực tiếp đến bước đánh giá kiểm thử (Test Evaluation) và bước tính toán Entropy chọn mẫu của Active Learning:

### Lỗi ghi đè tham số `beta` trong Backward Pass
Trong pha Backward Pass của thuật toán Forward-Backward để tính xác suất biên:
```python
        beta = torch.full((batch_size, seq_len, num_tags), -1e9, device=device)
        seq_lens = mask.sum(dim=1).long()
        beta[torch.arange(batch_size, device=device), seq_lens - 1, :] = end_transitions.unsqueeze(0)

        for t in range(seq_len - 2, -1, -1):
            next_beta = beta[:, t+1, :].unsqueeze(1) # (B, 1, K)
            trans = transitions.unsqueeze(0) # (1, K, K)
            emit = emissions[:, t+1, :].unsqueeze(1)
            prev_beta = torch.logsumexp(next_beta + trans + emit, dim=2)
            m = mask[:, t+1].unsqueeze(1)
            beta[:, t, :] = torch.where(m, prev_beta, beta[:, t+1, :])  # <--- LỖI TẠI ĐÂY!
```

*   **Phân tích lỗi**:
    - `beta` tại vị trí cuối cùng của chuỗi valid (`seq_lens - 1`) được khởi tạo bằng `end_transitions` (một giá trị hợp lệ). Phần đệm padding phía sau (từ `seq_lens` đến `seq_len - 1`) có giá trị `-1e9`.
    - Khi vòng lặp chạy lùi từ `seq_len - 2` về `0`. Tại bước `t = seq_lens - 1` (tương ứng với token cuối cùng của câu), ta có `t+1 = seq_lens` (token padding đầu tiên).
    - Do `mask[:, seq_lens]` là `False`, biến `m` sẽ mang giá trị `False`.
    - Lệnh `torch.where(m, prev_beta, beta[:, t+1, :])` khi gặp `m = False` sẽ chọn giá trị `beta[:, t+1, :]` (tức là `beta[:, seq_lens, :]`, vốn mang giá trị khởi tạo `-1e9`).
    - Kết quả là: **Giá trị khởi tạo `end_transitions` tại `beta[:, seq_lens - 1, :]` bị ghi đè hoàn toàn bằng `-1e9`!**
    - Ở các bước lùi tiếp theo (`t < seq_lens - 1`), do `m` là `True`, code tính toán `prev_beta` dựa trên `beta[:, t+1, :]` (vốn đã bị ghi đè thành `-1e9`).
    - Do đó, toàn bộ bảng `beta` cho các token hợp lệ đều bị kéo về giá trị cực kỳ âm (`-1e9`).
*   **Hậu quả**:
    - `log_p = alpha + beta` bị sai lệch nghiêm trọng (mang giá trị ~ `-1e9` ở mọi vị trí).
    - Khi qua hàm `softmax`, xác suất biên `marginals` bị suy biến thành phân phối đều `[0.333, 0.333, 0.333]` (hoặc NaN).
    - Trong bước đánh giá (`merge_and_resolve_conflicts`), độ tin cậy `prob_entity` của tất cả các thực thể đều bằng `0.333`, làm hỏng hoàn toàn việc giải quyết xung đột nhãn (luôn ưu tiên các thực thể có độ dài ngắn nhất thay vì thực thể có độ tự tin cao nhất).
    - Trong bước Active Learning, điểm Entropy của mọi câu ứng viên đều bằng nhau ($\approx 1.1$), dẫn đến việc chọn mẫu AL bị ngẫu nhiên hóa hoặc mất tác dụng.

*   **Cách khắc phục**:
    Thay thế dòng lỗi bằng:
    ```python
    beta[:, t, :] = torch.where(m, prev_beta, beta[:, t, :])
    ```
    *Giải thích*: Khi `m` là `False` (tức là token `t+1` là padding, nghĩa là ta đang ở vị trí kết thúc chuỗi `t = seq_lens - 1` hoặc vùng padding `t >= seq_lens`), ta chỉ cần **giữ nguyên** giá trị hiện tại của `beta[:, t, :]` (đã được khởi tạo là `end_transitions` tại `seq_lens - 1` hoặc `-1e9` tại vùng padding) chứ không được ghi đè bằng giá trị đệm phía sau.

---

## 3. Các đề xuất cấu hình tối ưu để tăng F1-score ở Vòng 0

Ngoài việc sửa lỗi code trên, để mô hình có thể học tốt trên tập dữ liệu ban đầu cực kỳ nhỏ (85 câu), chúng tôi đề xuất điều chỉnh các siêu tham số trong lớp `Config` như sau:

1.  **Tăng tốc độ học của LoRA (LoRA Learning Rate)**:
    - Hiện tại: `LEARNING_RATE = 2e-5`.
    - Đề xuất: Tăng lên `1e-4` hoặc `2e-4`. Do DeBERTa được đóng băng gần hết và chỉ tinh chỉnh LoRA, tốc độ học `2e-5` là quá chậm khiến LoRA hầu như không thay đổi gì sau vài trăm steps. Tăng lên `1e-4` giúp mô hình hội tụ nhanh hơn trên tập dữ liệu nhỏ.
2.  **Tăng số Epochs tối đa của mỗi vòng lặp AL**:
    - Hiện tại: `AL_EPOCHS = 12`.
    - Đề xuất: Tăng lên `25` hoặc `30` epoch. Với dữ liệu nhỏ, mô hình cần nhiều lượt quét hơn để tìm ra các tín hiệu thực thể thưa thớt.
3.  **Tích hợp Learning Rate Scheduler**:
    - Đề xuất: Sử dụng `get_linear_schedule_with_warmup` với tỷ lệ warmup là 10% tổng số steps để ổn định hóa quá trình huấn luyện đầu phân loại CRF khi mới bắt đầu.
4.  **Cân bằng gán nhãn Subword**:
    - Đề xuất: Sửa đổi logic căn chỉnh nhãn trong `EntityMaskingCollator` và test loop. Đối với một từ mang nhãn `B`, chỉ subword đầu tiên nhận nhãn `B`, các subword tiếp theo của từ đó nhận nhãn `I`.
5.  **Giảm tỉ lệ Contextual Masking**:
    - Hiện tại: 18% cho thực thể, 15% cho từ ngữ cảnh.
    - Đề xuất: Giảm xuống 10% cho thực thể và 5% cho ngữ cảnh (hoặc tắt hoàn toàn Contextual Masking ở Vòng 0-2) vì lượng thông tin trong 85 câu là quá ít, việc che giấu quá nhiều sẽ làm mất đi ngữ cảnh học tập quan trọng của mô hình.

---

# Phân tích Bổ sung & Các Đề xuất Cải tiến Kỹ thuật sâu hơn

Dưới đây là các phân tích chi tiết bổ sung liên quan đến các ý kiến đóng góp của bạn về Gazetteer, hiện tượng giảm loss ảo, cơ chế Context Masking, hoạt động của CRF Mask, chất lượng tăng cường của Distant Supervision và logic gộp nhãn.

## 4. Tiền xử lý và Lưu trữ dữ liệu Gazetteer tĩnh
*   **Vấn đề hiện tại**: Hàm `ds_augment_sentence` trong [01_mo_phong_active_learning.ipynb](file:///d:/Study/TLU/Nlp/active-learning-VietBioNER/colab/notebook/01_mo_phong_active_learning.ipynb) gọi `ViTokenizer.tokenize(new_entity_str)` một cách động cho mỗi mẫu được chọn thay thế tại mỗi bước huấn luyện.
*   **Nhận xét**: Điều này gây dư thừa về mặt tính toán. Việc chạy PyVi động trên CPU trong vòng lặp huấn luyện làm giảm tốc độ huấn luyện trên Colab. Hơn nữa, Gazetteer là dữ liệu tĩnh, việc xử lý động lặp đi lặp lại một lượng từ điển cố định là không tối ưu.
*   **Giải pháp đề xuất**:
    - Xây dựng một script tiền xử lý tĩnh (ví dụ: `preprocess_gazetteers.py`) để duyệt qua cả 5 file JSON trong `colab/dataset/gazetteer`, áp dụng `ViTokenizer.tokenize()` cho toàn bộ các thực thể, sau đó lưu lại dưới dạng `[name]_segmented.json` trong thư mục `PREPROCESSED_DIR` (`dataset/preprocessed/`).
    - Trong luồng chạy chính của Notebook, ta chỉ cần nạp các tệp Gazetteer đã được phân đoạn sẵn này vào bộ nhớ. Khi thực hiện thay thế thực thể, ta chỉ việc lấy chuỗi đã phân đoạn sẵn và gọi `.split()` để lấy các tokens, loại bỏ hoàn toàn việc gọi PyVi trong vòng lặp chính. Điều này giúp tăng tốc độ huấn luyện đáng kể.

## 5. Phân tích hiện tượng "Loss giảm nhưng F1 không tăng"
Hiện tượng Loss trên tập Train giảm mạnh (từ ~29 về ~14) và Loss trên tập Validation cũng giảm (từ 8.14 về 6.14) nhưng F1-score trên tập Test lại cực thấp (~0.0260) là một biểu hiện rất rõ ràng của:

1.  **Overfitting cực độ trên tập dữ liệu quá nhỏ (85 câu)**:
    - Mô hình nhanh chóng học thuộc lòng các mẫu đặc trưng nông của 85 câu huấn luyện thay vì học cách nhận diện thực thể tổng quát. Do đó, Train loss giảm rất đẹp nhưng không có khả năng tổng quát hóa trên tập kiểm thử độc lập.
2.  **Mất cân bằng nhãn và Đánh giá lệch pha trên tập Validation**:
    - Trong tập Train, chúng ta sử dụng **Negative Query Downsampling** để giữ tỷ lệ cân bằng 50/50 giữa truy vấn dương (chứa thực thể) và truy vấn âm (không chứa thực thể).
    - Nhưng trong tập Validation và Test, chúng ta giữ nguyên toàn bộ 5 truy vấn mô tả nhãn tĩnh cho mỗi câu mà không hề downsampling. Điều này dẫn đến việc tập Validation có tới hơn 80% là các truy vấn âm tính (chỉ chứa toàn nhãn `O`).
    - Mô hình chỉ cần có xu hướng thiên lệch mạnh về việc dự đoán nhãn `O` là đã có thể dễ dàng giảm thiểu loss trên tập Validation một cách giả tạo (Val Loss giảm ảo từ 8.14 xuống 6.14). Vì vậy, Validation Loss ở đây là một độ đo gây nhiễu, khiến cơ chế Early Stopping bị kích hoạt sai thời điểm hoặc bị đánh lừa là mô hình đang học tốt.
3.  **Giải pháp đề xuất**:
    - Thay vì sử dụng Validation Loss để làm tiêu chí cho Early Stopping, bắt buộc phải sử dụng **Validation F1-score** (hoặc Macro F1-score trên 5 lớp thực thể) làm tiêu chí chọn checkpoint tốt nhất.
    - Cân bằng lại số lượng truy vấn âm tính trong tập Validation để tránh hiện tượng giảm loss ảo do thiên lệch nhãn `O`.

## 6. Phân tích Context Masking trước khi Token hóa
*   **Vấn đề hiện tại**: Trong `EntityMaskingCollator`, việc che giấu (masking) được áp dụng trực tiếp trên list từ gốc (`tokens[i] = "[MASK]"`). Sau đó, list tokens này mới được truyền vào tokenizer để mã hóa.
*   **Phân tích ảnh hưởng**:
    - Việc thay thế cả từ ghép gốc bằng `[MASK]` trước khi tokenize có nghĩa là toàn bộ từ ghép (ví dụ: `"lao_phổi"`) sẽ biến thành một token `[MASK]` duy nhất sau khi đi qua tokenizer. Điều này làm thay đổi độ dài chuỗi token thực tế được đưa vào mô hình so với khi không mask.
    - Việc che giấu toàn bộ từ ghép (Whole Word Masking) về mặt lý thuyết giúp mô hình không thể đoán từ thông qua các subword còn lại, nhưng việc áp dụng tỉ lệ mask quá cao (18% thực thể, 15% ngữ cảnh) trên tập dữ liệu siêu nhỏ (85 câu) làm mất đi hầu hết các thông tin cú pháp và ngữ cảnh y khoa quan trọng.
    - Đặc biệt, do tập Validation và Test **hoàn toàn không có mask**, sự lệch pha phân phối (train-test mismatch) này khiến mô hình không thể nhận diện được các thực thể khi chúng xuất hiện đầy đủ trong ngữ cảnh sạch ở pha kiểm thử.
*   **Giải pháp đề xuất**:
    - Tắt hoàn toàn cơ chế Context Masking ở các vòng lặp đầu tiên (Vòng 0 đến Vòng 2) để mô hình tập trung học các đặc trưng cú pháp cơ bả## 10. Giải pháp nâng cao chất lượng Seed Set Selection
*   **Ý kiến đóng góp**: Gazetteer có thể không đủ bao phủ hoặc không đủ chính xác, vì vậy K-Means có thể vẫn bỏ sót thực thể.
*   **Giải pháp bổ sung (Chốt)**: Thay vì sử dụng K-Means thuần túy hay Gazetteer-guided, ta sẽ sử dụng phương pháp **Stratified Sampling (Lấy mẫu phân tầng) dựa trên nhãn chuẩn (gold labels)** của tập Train để xây dựng Seed Set $L_0$:
    - Vì đây là pha mô phỏng (simulation benchmark), chúng ta đã có sẵn nhãn chuẩn của tập Train.
    - Chúng ta phân nhóm các câu trong tập Train thành các tầng lớp (strata) dựa trên các thực thể xuất hiện trong đó, ưu tiên từ lớp hiếm nhất đến phổ biến nhất:
      1. Tầng 1: Các câu chứa `Organisation` (ORG) - lớp hiếm nhất.
      2. Tầng 2: Các câu chứa `DateTime` (DATE) - lớp rất hiếm.
      3. Tầng 3: Các câu chứa `Location` (LOC) - lớp ít.
      4. Tầng 4: Các câu chứa `DiagnosticProcedure` (DP) - lớp trung bình.
      5. Tầng 5: Các câu chứa `Symptom_and_Disease` (SYM) - lớp phổ biến nhất.
      6. Tầng 6: Các câu chỉ chứa nhãn `O` (không có thực thể).
    - Để tạo ra $L_0$ gồm 85 câu, chúng ta sẽ lần lượt rút mẫu từ Tầng 1 đến Tầng 5 (ví dụ: lấy 15 câu chứa ORG, 15 câu chứa DATE, 15 câu chứa LOC, 15 câu chứa DP, 15 câu chứa SYM, và 10 câu còn lại từ các câu ngẫu nhiên hoặc chỉ chứa O).
    - Phương pháp này giải quyết triệt để vấn đề "khởi động lạnh" (cold start), đảm bảo mô hình có đủ mẫu đại diện cho tất cả các lớp ngay từ Vòng 0 để học ma trận transitions và đặc trưng thực thể hiếm, đồng thời duy trì tính khoa học của một thực nghiệm AL đối chứng.

## 11. Giải pháp tích hợp Weighted CRF Loss trong pytorch-crf
*   **Ý kiến đóng góp**: Việc tăng trọng số đồng đều 2x ở cấp độ câu (query-level) vẫn không giải quyết được vấn đề mất cân bằng ở cấp độ token (nhãn O vẫn chiếm >95% trong câu dương tính), dễ gây overfit cho câu dương tính và không giải quyết được mất cân bằng giữa các lớp thực thể (Organisation hiếm hơn Symptom_and_Disease).
*   **Giải pháp bổ sung (Chốt) - Class-aware Positive Query Weighting**:
    - Để giải quyết triệt để cả sự mất cân bằng token-level và class-level mà không vi phạm cấu trúc của thư viện `pytorch-crf`, chúng ta chốt giải pháp **Class-aware Positive Query Weighting (Trọng số truy vấn dương tính phân biệt theo lớp)**:
    - Trong hàm `train_model`, ta sử dụng `reduction='none'` trong `model.crf(...)` để trả về vector loss của từng câu trong batch.
    - Sử dụng bảng trọng số lớp có tính đến độ hiếm của thực thể:
      ```python
      class_weights = {
          'Organisation': 5.0,        # Cực kỳ thiểu số
          'DateTime': 4.0,            # Rất ít
          'Location': 3.0,            # Ít
          'DiagnosticProcedure': 2.0,   # Trung bình
          'Symptom_and_Disease': 1.0    # Đa số
      }
      ```
    - Duyệt qua từng câu truy vấn trong batch để tính trọng số động (`weights`):
      *   Nếu câu truy vấn là **Dương tính** (chứa thực thể nhãn B hoặc I của lớp đang truy vấn): `weight = class_weights[C]` (với C là lớp thực thể tương ứng với `query_label_idx` của câu đó. Ví dụ: câu dương tính chứa thực thể Organisation sẽ nhận trọng số `5.0`).
      *   Nếu câu truy vấn là **Âm tính** (chỉ chứa toàn nhãn O): `weight = 1.0` (giữ nguyên trọng số mặc định để tránh phóng đại loss của nhãn O, giải quyết triệt để sự áp đảo của nhãn O ở cấp độ token).
    - Công thức tính loss có trọng số:
      ```python
      loss_vector = -model.crf(emissions, tags, mask=mask, reduction='none')
      
      # Xác định câu chứa thực thể (Positive Query)
      query_has_entity = (tags == 1).any(dim=1) | (tags == 2).any(dim=1)
      
      # Tạo vector trọng số dựa trên lớp đang truy vấn (chỉ áp dụng cho Positive queries)
      batch_weights = []
      for idx, item_has_entity in enumerate(query_has_entity):
          if item_has_entity:
              # Lấy nhãn lớp đang truy vấn từ batch
              q_label_idx = batch["query_label_idx"][idx].item()
              q_label_name = label_list[q_label_idx]
              batch_weights.append(class_weights[q_label_name])
          else:
              batch_weights.append(1.0)
      
      weights_tensor = torch.tensor(batch_weights, dtype=torch.float, device=device)
      loss = (loss_vector * weights_tensor).mean()
      ```
    - **Ưu điểm**:
      1.  **Cân bằng Token-level**: Chỉ nhân trọng số cao cho các câu thực sự có thực thể. Loss của các câu không có thực thể (100% nhãn O) được giữ ở mức 1.0, ngăn chặn hiệu quả việc loss của nhãn O áp đảo loss của thực thể.
      2.  **Cân bằng Class-level**: Lớp càng hiếm (như ORG) khi xuất hiện trong câu dương tính sẽ nhận trọng số phạt lỗi càng cao (5.0), giúp mô hình tập trung tối ưu hóa các lớp thiểu số.
      3.  **Bảo vệ CRF Transitions**: Trọng số tối đa là 5.0 (không quá lớn) và chỉ áp dụng cho loss tổng thể của câu, giúp định hướng gradient của mô hình học ranh giới thực thể mà không làm méo mó các phân phối chuyển đổi trạng thái của CRF.
hể phá vỡ cấu trúc cú pháp tự nhiên của câu.
    - Tuy nhiên, trong văn bản lâm sàng tiếng Việt, ngữ cảnh xung quanh tổ chức thường khá cố định (ví dụ: *"nhập viện tại [ORG]"*, *"chuyển đến [ORG]"*). Do đó, mức độ nhiễu ngữ pháp là chấp nhận được.
*   **Giải pháp đề xuất**:
    - Tiếp tục giữ nguyên cơ chế DS thế thực thể cho ORG với $M=3$ để bù đắp sự thiếu hụt nghiêm trọng của lớp thiểu số này.
    - Cần kiểm soát chặt chẽ việc lọc rò rỉ dữ liệu (đã hoàn thành tốt thông qua việc loại bỏ các thực thể xuất hiện trong tập Test/Val khỏi Gazetteer).

## 9. Đánh giá logic gộp nhãn và vai trò giải quyết xung đột bằng marginal probabilities
*   **Mô tả hoạt động**:
    - Hàm `merge_and_resolve_conflicts` sử dụng Viterbi decoding để tìm đường đi nhãn tối ưu cho từng truy vấn nhãn đơn lẻ, sau đó sử dụng xác suất biên từ CRF (Marginal Probability) để tính độ tin cậy trung bình của các span thực thể nhằm phân xử khi có sự chồng lấn ranh giới.
*   **Phân tích sự ảnh hưởng của lỗi `beta`**:
    - Như đã chỉ ra ở Mục 2, lỗi ghi đè `beta` bằng `-1e9` khiến xác suất biên `marginals` bị suy biến thành `0.333` ở mọi token.
    - Do đó, phần tính toán độ tin cậy trung bình (`avg_prob`) của mọi span ứng viên đều bằng `0.333`.
    - Khi giải quyết xung đột nhãn bằng cách sắp xếp: `sorted(candidate_spans, key=lambda x: (-x[3], x[1] - x[0], x[2]))`. Do điểm tự tin `x[3] = 0.333` bằng nhau ở mọi span, bộ gộp nhãn bị buộc phải phân xử hoàn toàn dựa vào độ dài span ngắn hơn (`x[1] - x[0]`) và thứ tự nhãn (`x[2]`).
    - Điều này làm vô hiệu hóa hoàn toàn cơ chế phân xử thông minh bằng xác suất biên toán học của CRF, dẫn đến việc giải quyết xung đột bị sai lệch và làm giảm F1-score của các thực thể dài hoặc thực thể xuất hiện sau trong danh sách nhãn.
*   **Kết luận**: Logic thuật toán gộp nhãn là đúng, nhưng hiệu quả của nó bị phá hủy bởi lỗi tính toán `beta` trong hàm Forward-Backward. Sau khi sửa lỗi `beta` ở Mục 2, cơ chế giải quyết xung đột nhãn bằng xác suất biên sẽ hoạt động chính xác trở lại và cải thiện rõ rệt F1-score trên tập Test.

## 10. Giải pháp nâng cao chất lượng Seed Set Selection
*   **Ý kiến**: K-Means với S-BERT mặc dù chọn các câu đại diện rất tốt nhưng không đảm bảo bao phủ đủ các lớp thiểu số (như `Organisation` chỉ có 13 thực thể trong pool).
*   **Phân tích**: Điều này hoàn toàn chính xác. Trong thực tế, K-Means phân cụm theo đặc trưng ngữ nghĩa tổng quát của câu, không ưu tiên các từ khóa thực thể hiếm. Điều này dẫn đến nguy cơ lớp `Organisation` (ORG) bị bỏ sót hoàn toàn hoặc có số lượng cực nhỏ trong Seed Set $L_0$, gây ra lỗi "khởi động lạnh" (cold start).
*   **Giải pháp đề xuất**:
    - Để giải quyết vấn đề này mà vẫn đảm bảo tính đa dạng của K-Means, chúng ta sẽ áp dụng cơ chế **Constrained Seed Selection (Lọc Seed Set có ràng buộc)**.
    - Cụ thể: Sau khi phân cụm K-Means ($K=85$), thay vì chọn câu gần tâm cụm nhất một cách mù quáng, chúng ta sẽ duyệt qua các câu ứng viên trong cụm đó (sử dụng Gazetteer để đếm số lượng thực thể khớp hoặc dựa vào nhãn chuẩn trong mô phỏng) và ưu tiên chọn câu chứa các thực thể thiểu số (ORG, DATE, LOC) trước, miễn là câu đó vẫn nằm trong phạm vi tương đồng chấp nhận được của cụm.
    - Đảm bảo trong 85 câu được chọn vào Seed Set $L_0$ phải có sự xuất hiện của ít nhất 5-10 thực thể cho mỗi lớp thực thể thiểu số như `Organisation` và `DateTime`.

## 11. Giải pháp tích hợp Weighted CRF Loss trong pytorch-crf
*   **Ý kiến**: Thư viện `pytorch-crf` không hỗ trợ weighted loss trực tiếp ở cấp độ token.
*   **Phân tích**: Hàm `forward()` của `pytorch-crf` chỉ thực hiện cộng tổng log-likelihood trên các nhãn và trả về một số thực duy nhất. Nó không hỗ trợ tham số `weight` như `CrossEntropyLoss` để phạt lỗi trên các nhãn thực thể (`B`, `I`) nặng hơn nhãn ngữ cảnh (`O`).
*   **Giải pháp đề xuất**:
    - Giải pháp tối ưu và sạch nhất là sử dụng **Query-level Weighted Loss (Trọng số cấp độ câu truy vấn)**.
    - Thư viện `pytorch-crf` hỗ trợ tham số `reduction='none'` để trả về một vector chứa NLL loss của từng câu truy vấn trong batch (thay vì tự động lấy trung bình).
    - Chúng ta sẽ tận dụng điều này để tính toán trọng số động trong hàm `train_model`:
      ```python
      # emissions: (B, seq_len, 3), tags: (B, seq_len), mask: (B, seq_len)
      loss_vector = -model.crf(emissions, tags, mask=mask, reduction='none')
      
      # Xác định câu nào chứa thực thể mục tiêu (truy vấn dương tính: có nhãn B hoặc I)
      query_has_entity = (tags == 1).any(dim=1) | (tags == 2).any(dim=1)
      
      # Tạo vector trọng số: nhân 2.0 cho câu chứa thực thể, giữ 1.0 cho câu không chứa
      weights = torch.where(query_has_entity, torch.tensor(2.0, device=device), torch.tensor(1.0, device=device))
      
      # Tính loss trung bình có trọng số
      loss = (loss_vector * weights).mean()
      ```
    - Phương pháp này giúp mô hình tập trung tối ưu hóa các câu truy vấn dương tính (nơi thực thể thực sự xuất hiện), hạn chế việc mô hình bị áp đảo bởi các truy vấn âm tính tràn ngập nhãn `O` mà không vi phạm cấu trúc thư viện `pytorch-crf`.


