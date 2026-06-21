# Cơ chế Đo Độ Bất Định Bằng CRF Marginal Entropy (Entropy Xác Suất Biên)

Tài liệu này giải thích chi tiết cơ sở toán học và cơ chế đo lường độ bất định (Uncertainty) bằng phương pháp **CRF Marginal Entropy (Entropy Xác suất Biên)**. Kỹ thuật này thay thế cho mô hình dự đoán loss phụ trợ (Loss-Prediction Module) để khắc phục lỗi quá khớp (overfitting) trên tập dữ liệu mẫu nhỏ và cải thiện độ tin cậy trong quá trình Học chủ động (Active Learning).

---

## 1. Cơ sở Toán học của Mô hình CRF (Conditional Random Field)

Trong bài toán nhận diện thực thể y sinh (NER) dạng phân loại chuỗi, với câu đầu vào $x$ có chiều dài $T$ và một chuỗi nhãn mục tiêu tương ứng $y = (y_1, y_2, \dots, y_T)$, xác suất điều kiện của chuỗi nhãn $y$ cho bởi công thức:

$$P(y|x) = \frac{\exp(\text{Score}(x, y))}{\sum_{y'} \exp(\text{Score}(x, y'))}$$

Trong đó, điểm số tương thích $\text{Score}(x, y)$ được định nghĩa là tổng điểm phát xạ (emission) và điểm chuyển trạng thái (transition):

$$\text{Score}(x, y) = \sum_{t=1}^T E(x, t, y_t) + \sum_{t=1}^{T-1} T(y_t, y_{t+1}) + T_{start}(y_1) + T_{end}(y_T)$$

Ở đây:
*   $E(x, t, y_t)$ là điểm phát xạ (được trích xuất từ lớp đầu ra tuyến tính Linear) cho nhãn $y_t$ tại vị trí $t$.
*   $T(y_t, y_{t+1})$ là điểm chuyển tiếp từ nhãn $y_t$ sang nhãn $y_{t+1}$ (được mô hình tự học trong ma trận chuyển trạng thái).
*   Mẫu số $Z(x) = \sum_{y'} \exp(\text{Score}(x, y'))$ là hàm phân hoạch (partition function), đại diện cho tổng điểm số của tất cả các chuỗi nhãn khả dĩ.

---

## 2. Tính toán Xác suất Biên bằng Thuật toán Forward-Backward

Để đo lường độ bất định của mô hình tại từng vị trí từ tố $t$, ta không thể chỉ sử dụng điểm số Viterbi toàn cục. Thay vào đó, ta cần biết **Xác suất biên (Marginal Probability)** của từng nhãn $l \in \{O, B, I\}$ tại vị trí $t$, ký hiệu là $P(y_t = l | x)$:

$$P(y_t = l | x) = \sum_{y: y_t = l} P(y|x)$$

Xác suất này được tính toán một cách hiệu quả thông qua thuật toán **Forward-Backward** trên đồ thị lưới (lattice):

### 2.1. Biến Forward ($\alpha$)
Biến forward $\alpha_t(l)$ là log-sum-exp của tất cả các chuỗi tiền tố kết thúc bằng nhãn $l$ tại bước $t$:

$$\alpha_t(l) = E(x, t, l) + \log \sum_{j} \exp\left(\alpha_{t-1}(j) + T(j, l)\right)$$

Với điều kiện biên: $\alpha_1(l) = T_{start}(l) + E(x, 1, l)$.

### 2.2. Biến Backward ($\beta$)
Biến backward $\beta_t(l)$ là log-sum-exp của tất cả các chuỗi hậu tố bắt đầu từ nhãn $l$ tại bước $t$:

$$\beta_t(l) = \log \sum_{j} \exp\left(\beta_{t+1}(j) + T(l, j) + E(x, t+1, j)\right)$$

Với điều kiện biên tại bước cuối cùng $T$: $\beta_T(l) = T_{end}(l)$.

### 2.3. Xác định Xác suất Biên
Sau khi tính được $\alpha$ và $\beta$, điểm số marginal chưa chuẩn hóa cho nhãn $l$ tại bước $t$ được tính bằng:

$$\log \tilde{P}(y_t = l | x) = \alpha_t(l) + \beta_t(l)$$

Áp dụng hàm Softmax trên toàn bộ không gian nhãn $\mathcal{Y} = \{O, B, I\}$ tại bước $t$, ta thu được xác suất biên chuẩn hóa:

$$P(y_t = l | x) = \frac{\exp\left(\alpha_t(l) + \beta_t(l)\right)}{\sum_{j \in \mathcal{Y}} \exp\left(\alpha_t(j) + \beta_t(j)\right)}$$

---

## 3. Công thức tính Shannon Entropy cho Câu

Độ bất định (Uncertainty) của mô hình NER tại vị trí từ tố $t$ chính là độ hỗn loạn thông tin (Shannon Entropy) của phân phối xác suất biên:

$$H_t(x) = - \sum_{l \in \{O, B, I\}} P(y_t = l | x) \log \left( P(y_t = l | x) + \epsilon \right)$$

Trong đó $\epsilon = 10^{-9}$ là hằng số mịn tránh lỗi chia cho $0$ khi tính toán logarit.

Độ bất định tổng thể của câu văn bản $x$ được tính bằng trung bình cộng giá trị entropy của các vị trí từ tố hợp lệ (active tokens, nằm trong `mask` của câu gốc, loại trừ phần padding và mô tả nhãn tĩnh):

$$\text{Uncertainty}(x) = \frac{1}{|V|} \sum_{t \in V} H_t(x)$$

Với $V$ là tập hợp các chỉ số từ tố thuộc câu gốc $s$ (ở dạng word-level). Câu nào có $\text{Uncertainty}(x)$ càng cao chứng tỏ mô hình càng không chắc chắn về cấu trúc thực thể của câu đó.

---

## 4. Tại sao CRF Marginal Entropy vượt trội hơn Loss-Prediction Module?

Trong thực nghiệm Active Learning VietBioNER, việc chuyển đổi từ cơ chế học dự đoán loss (LPM) sang tính toán trực tiếp CRF Marginal Entropy mang lại 3 ưu thế vượt trội:

1.  **Triệt tiêu hoàn toàn hiện tượng quá khớp (Overfitting)**:
    *   *Loss-Prediction*: MLP head là một mạng nơ-ron hồi quy phụ. Khi dữ liệu đã gán nhãn ở các vòng AL đầu rất nhỏ (ví dụ 85 câu), MLP dễ bị quá khớp cực kỳ nặng, dự đoán điểm loss $\hat{l}$ trên tập $U_t$ bị lệch lạc lớn (nhiễu).
    *   *Marginal Entropy*: Không có bất kỳ tham số học thêm nào. Entropy được tính toán trực tiếp từ trọng số hiện tại của mô hình qua thuật toán Forward-Backward, phản ánh chính xác 100% độ bất định thực tế của mô hình tại vòng lặp đó.
2.  **Ràng buộc cấu trúc chuỗi toàn cục (Global Constraints)**:
    *   Khác với Entropy mức token của đầu Softmax độc lập, CRF Marginal Entropy tính toán xác suất biên dựa trên cả ma trận chuyển trạng thái $T$. Nghĩa là mô hình không chỉ đo độ bất định của bản thân từ đó, mà còn đo độ bất định trong mối quan hệ chuyển đổi nhãn với các từ xung quanh (ví dụ: mô hình phân vân không biết ranh giới thực thể nên kéo dài đến từ tiếp theo hay dừng lại).
3.  **Tối ưu tài nguyên huấn luyện**:
    *   Bỏ đầu MLP và hàm loss phụ (Pairwise Ranking Loss) giúp cấu trúc mô hình tinh gọn, giảm tải tính toán đồ thị đạo hàm ngược (backward propagation), chạy nhanh hơn và kiểm soát rủi ro tràn RAM GPU (OOM) hoàn hảo trên Colab T4.
