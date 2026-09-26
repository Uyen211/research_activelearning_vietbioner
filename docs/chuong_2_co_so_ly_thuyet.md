# CHƯƠNG 2: CƠ SỞ LÝ THUYẾT VÀ CÁC CÔNG TRÌNH NGHIÊN CỨU LIÊN QUAN

## 2.1. Các khái niệm nền tảng

### 2.1.1. Bài toán Nhận dạng Thực thể Y sinh (BioNER) và Sequence Labeling
Nhận dạng thực thể có tên (Named Entity Recognition - NER) là một nhiệm vụ nền tảng trong xử lý ngôn ngữ tự nhiên, được phát biểu dưới dạng bài toán gán nhãn chuỗi (Sequence Labeling). Cho một chuỗi văn bản đầu vào $S = (w_1, w_2, \dots, w_n)$ gồm $n$ từ (tokens), mục tiêu của mô hình là tìm ra chuỗi nhãn tương ứng $Y = (y_1, y_2, \dots, y_n)$ sao cho mỗi nhãn $y_i$ thuộc về một tập nhãn định trước $\mathcal{T}$. Trong nghiên cứu này, tập nhãn $\mathcal{T}$ tuân thủ định dạng BIO (Beginning, Inside, Outside). Cụ thể, nhãn `B-Type` biểu diễn từ bắt đầu của một thực thể thuộc loại `Type`, nhãn `I-Type` biểu diễn các từ tiếp theo nằm trong thực thể đó, và nhãn `O` biểu diễn các từ không thuộc bất kỳ thực thể nào.

Nhận dạng thực thể y sinh (BioNER) có những đặc thù riêng biệt so với bài toán NER trong miền tổng quát. Các văn bản lâm sàng thường chứa nhiều thuật ngữ y khoa phức tạp, đa âm tiết và có tính nhập nhằng ranh giới cao. Ranh giới của các thực thể như triệu chứng (`Symptom_and_Disease`) và phương pháp chẩn đoán (`DiagnosticProcedure`) dễ bị chồng lấn và phụ thuộc chặt chẽ vào ngữ cảnh y học xung quanh. Do đó, mô hình nhận dạng cần khả năng biểu diễn ngữ nghĩa chuyên sâu để xác định chính xác biên thực thể.

### 2.1.2. Học chủ động (Active Learning) và Thế thực thể dựa trên từ điển (Dictionary-based Entity Substitution - DES)
Học chủ động (Active Learning - AL) là phương pháp học máy hướng tới việc tối ưu hóa hiệu năng mô hình dưới một ngân sách gán nhãn hạn chế. Thay vì huấn luyện trên toàn bộ tập dữ liệu được gán nhãn ngẫu nhiên, quy trình AL tiến hành lựa chọn có chọn lọc các mẫu chưa gán nhãn có giá trị thông tin cao nhất để chuyển cho chuyên gia hiệu chỉnh. Về mặt toán học, quy trình AL hoạt động theo cơ chế lặp: tại mỗi vòng lặp $t$, mô hình $\mathcal{M}_t$ được huấn luyện trên tập dữ liệu đã gán nhãn $L_t$. Sau đó, mô hình này sẽ dự đoán trên tập dữ liệu chưa gán nhãn $U_t$. Một hàm chọn mẫu (Query Strategy) $\mathcal{Q}$ sẽ tính toán và lựa chọn một batch $b$ gồm các mẫu tối ưu từ $U_t$ dựa trên các tiêu chí cụ thể để chuyên gia gán nhãn chuẩn. Tập dữ liệu sau đó được cập nhật theo công thức:
$$L_{t+1} = L_t \cup \{b\}$$
$$U_{t+1} = U_t \setminus \{b\}$$

Các chiến lược chọn mẫu chính trong AL bao gồm:
*   **Chọn mẫu dựa trên độ bất định (Uncertainty-based Sampling)**: Ưu tiên lựa chọn các mẫu dữ liệu mà mô hình hiện tại phân vân nhất (độ tin cậy dự đoán thấp nhất hoặc entropy phân phối nhãn cao nhất).
*   **Chọn mẫu dựa trên độ đa dạng (Diversity-based Sampling)**: Đảm bảo các mẫu được chọn phân bổ đều trong không gian ngữ nghĩa, tránh chọn các câu có nội dung trùng lặp.

Thế thực thể dựa trên từ điển (Dictionary-based Entity Substitution - DES) là một kỹ thuật tăng cường dữ liệu chuyên biệt cho bài toán NER. Trong BioNER, phương pháp DES thế thực thể (Entity Substitution) giúp sinh ra các câu huấn luyện mới bằng cách thay thế các thực thể y khoa trong câu gốc bằng các thực thể cùng nhóm từ Gazetteer (từ điển chuyên ngành) đã được làm sạch rò rỉ dữ liệu. Điều này giúp làm phong phú vốn từ vựng lâm sàng cho mô hình, đặc biệt hữu ích đối với các lớp thực thể hiếm gặp mà không làm phát sinh thêm chi phí gán nhãn thủ công.

### 2.1.3. Kiến trúc DeBERTa và Mô hình ViPubmedDeBERTa-base
Kiến trúc DeBERTa (Decoding-enhanced BERT with Disentangled Attention) đại diện cho một bước cải tiến ý nghĩa so với các mô hình tiền nhiệm thuộc họ BERT như BERT và RoBERTa. Ưu thế công nghệ của DeBERTa so với BERT và RoBERTa xuất phát từ hai đặc tính thiết kế chính: cơ chế chú ý phân tách (Disentangled Attention) và bộ giải mã mặt nạ tăng cường (Enhanced Mask Decoder). 

Trong các kiến trúc BERT và RoBERTa truyền thống, thông tin nội dung và vị trí tuyệt đối của mỗi token được cộng trực tiếp với nhau tạo thành một vector biểu diễn duy nhất trước khi nạp vào các lớp chú ý. Thiết kế này vô tình làm mờ đi các đặc tính vị trí tương đối giữa các từ trong câu. Trái lại, DeBERTa phân tách hai đặc trưng nội dung và vị trí tương đối thành hai ma trận nhúng độc lập. Điểm số chú ý (attention score) giữa token thứ $i$ và token thứ $j$ được tính toán thông qua tích vô hướng phân rã giữa nội dung ($c$) và vị trí tương đối tương ứng với khoảng cách $\delta(i, j)$:
$$A_{i,j} = c_i c_j^T + c_i P^T_{\delta(i,j)} + P_{\delta(i,j)} c_j^T$$
Trong đó $\delta(i,j)$ biểu diễn khoảng cách tương đối giữa token $i$ và $j$, và $P$ là ma trận nhúng vị trí tương đối. Việc loại bỏ thành phần vị trí - vị trí ($P_{\delta(i,j)}P^T_{\delta(i,j)}$) giúp mô hình tập trung vào sự tương tác chéo giữa nội dung và khoảng cách tương đối của từ ngữ cảnh, tối ưu hóa năng lực học cấu trúc cú pháp và ranh giới thực thể trong câu văn.

Trên cơ sở kiến trúc DeBERTa, mô hình ViPubmedDeBERTa-base được phát triển bằng cách kế thừa mô hình nền tảng tiếng Việt tổng quát ViDeBERTa (vốn xây dựng trên cấu trúc tối ưu DeBERTaV3) và tiếp tục thực hiện huấn luyện chuyên sâu (continual pre-training) trên tập dữ liệu gồm 20 triệu bản tóm tắt y văn PubMed đã được dịch thuật sang tiếng Việt. Sự kế thừa này mang lại lợi thế biểu diễn ngôn ngữ lâm sàng chuyên ngành vượt trội so với PhoBERT. Mô hình PhoBERT vốn được xây dựng trên nền tảng RoBERTa (một nhánh phát triển trực tiếp của BERT), do đó chịu giới hạn bởi cơ chế nhúng vị trí tuyệt đối và khó nắm bắt chính xác ranh giới của các thực thể y sinh đa từ, đa âm tiết phức tạp. Do đó, việc sử dụng ViPubmedDeBERTa-base như mô hình ngôn ngữ xương sống mang lại hiệu năng nhận diện thực thể lâm sàng cao hơn đáng kể.

### 2.1.4. Cơ chế thích ứng hiệu quả tham số LoRA (Low-Rank Adaptation)
Huấn luyện toàn bộ tham số (Full Fine-tuning) của các mô hình ngôn ngữ lớn trên tập dữ liệu AL nhỏ dễ dẫn đến hiện tượng quá khớp (overfitting) và đòi hỏi tài nguyên tính toán lớn. Phương pháp thích ứng hiệu quả tham số LoRA (Low-Rank Adaptation) giải quyết vấn đề này bằng cách đóng băng các trọng số ban đầu của mô hình $W_0 \in \mathbb{R}^{d \times k}$ và chỉ cập nhật thông qua ma trận gia số $\Delta W$. 

Ma trận $\Delta W$ được phân rã thành tích của hai ma trận có hạng thấp (low-rank):
$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$
Trong đó $B \in \mathbb{R}^{d \times r}$ và $A \in \mathbb{R}^{r \times k}$, với hạng $r \ll \min(d, k)$ và $\alpha$ là hằng số tỷ lệ. Trong quá trình huấn luyện, chỉ có ma trận $A$ (khởi tạo theo phân phối chuẩn) và ma trận $B$ (khởi tạo bằng 0) là có thể cập nhật tham số. Cơ chế này giúp giảm số lượng tham số cần huấn luyện xuống dưới 1%, bảo vệ tri thức gốc của mô hình pre-trained và ngăn chặn hiện tượng mất mát tri thức y khoa.

### 2.1.5. Lớp phân loại chuỗi CRF (Conditional Random Fields) Head
Trong các bài toán gán nhãn chuỗi, các nhãn y khoa có sự phụ thuộc lẫn nhau một cách chặt chẽ (ví dụ: nhãn `I-Disease` chỉ có thể xuất hiện sau nhãn `B-Disease` hoặc `I-Disease`, không thể xuất hiện sau nhãn `O`). Đầu phân loại tuyến tính độc lập (Softmax Head) bỏ qua các ràng buộc này, dễ dẫn đến các chuỗi nhãn không hợp lệ. Lớp CRF (Conditional Random Fields) giải quyết vấn đề này bằng cách tối ưu hóa xác suất đồng thời của toàn bộ chuỗi nhãn.

Đối với chuỗi đầu vào $S$ và chuỗi nhãn dự đoán $Y = (y_1, y_2, \dots, y_n)$, xác suất có điều kiện của $Y$ được định nghĩa:
$$P(Y|S) = \frac{1}{Z(S)} \exp \left( \sum_{i=1}^n \left( P_{i, y_i} + T_{y_{i-1}, y_i} \right) \right)$$
Trong đó $P_{i, y_i}$ là điểm phát xạ (emission score) của nhãn $y_i$ tại vị trí $i$ do đầu Linear dự đoán, $T_{y_{i-1}, y_i}$ là điểm dịch chuyển (transition score) biểu diễn xác suất chuyển đổi từ nhãn $y_{i-1}$ sang $y_i$, và $Z(S)$ là hàm chuẩn hóa tích lũy trên toàn bộ các chuỗi nhãn khả dĩ. Đồ án loại bỏ lớp BiLSTM trung gian trước CRF để tránh hiện tượng triệt tiêu thông tin ngữ cảnh tĩnh từ mô tả thực thể do cơ chế padding khi huấn luyện các batch có độ dài chênh lệch.

## 2.2. Tổng quan các phương pháp và công trình liên quan (Literature Review)

Các công trình nghiên cứu liên quan đến đề tài được phân tích có hệ thống theo ba nhóm chủ đề chính dưới đây:

### 2.2.1. Nhóm 1: Học chủ động và các chiến lược gán nhãn tối ưu chi phí
Nghiên cứu về giảm thiểu chi phí gán nhãn trong bài toán NER lâm sàng đã đạt được nhiều tiến bộ quan trọng. Liu và Wong [1] đề xuất cơ chế chuyển đổi chiến lược động (Dynamic Switching) trong Active Learning. Cụ thể, hệ thống bắt đầu bằng việc chọn mẫu dựa trên độ đa dạng ngữ nghĩa (Cluster-based) để bao phủ tối đa không gian khái niệm y học, sau đó tự động chuyển sang chọn mẫu dựa trên độ bất định (Least Confidence) khi tốc độ hội tụ loss giảm xuống dưới một ngưỡng xác định. Nghiên cứu này cũng tiên phong sử dụng khoảng cách hiệu chỉnh Levenshtein để đo lường thực tế nỗ lực sửa nhãn của chuyên gia thay vì chỉ đếm số câu tĩnh.

Để giải quyết sự mất cân bằng nhãn nghiêm trọng trong y văn, Silvestri và các cộng sự [2] giới thiệu quy trình lai kết hợp AL và Thế thực thể dựa trên từ điển (DES) để gán nhãn bệnh án điện tử tiếng Ý. Pha DES tiến hành trích xuất câu chứa thực thể hiếm và thực hiện thế từ vựng tự động từ cơ sở tri thức chuyên ngành, giúp giảm thời gian gán nhãn của bác sĩ xuống còn 1/5 so với quy trình thủ công. 

Tiếp tục tối ưu hóa batch chọn mẫu, Zhuang và các cộng sự [3] đề xuất mô hình MedNER cho y khoa tiếng Trung. MedNER tích hợp mô-đun dự đoán loss trung gian để đo độ bất định và sử dụng bộ lọc Distinct-K Filter nhằm giải quyết bài toán Tập độc lập lớn nhất (MIS) trên đồ thị tương đồng. Giải pháp này giúp loại bỏ tối đa tính trùng lặp thông tin trong batch được chọn, nâng cao hiệu suất học của mô hình qua từng vòng lặp.

### 2.2.2. Nhóm 2: Xây dựng tập dữ liệu y tế và benchmark tiếng Việt
Sự phát triển của BioNER tiếng Việt gắn liền với việc xây dựng các bộ dữ liệu chuẩn hóa. Truong và các cộng sự [4] công bố bộ dữ liệu PhoNER_COVID19 gồm hơn 10.000 câu về dịch tễ học và thiết lập benchmark với mô hình PhoBERT. Tuy nhiên, tập dữ liệu này giới hạn trong ngữ cảnh dịch bệnh COVID-19, gây khó khăn khi áp dụng vào các bệnh lý lâm sàng thông thường.

Để cung cấp tài nguyên cho các bệnh lý phổ biến, Phan và các cộng sự [5] xây dựng bộ dữ liệu VietBioNER tập trung vào bệnh lao phổi (Tuberculosis) từ các tài liệu học thuật. Nhóm tác giả chỉ ra rằng nhãn `DiagnosticProcedure` (thủ thuật chẩn đoán) là loại thực thể khó nhận diện nhất do cấu trúc đa âm tiết phức tạp. Nhằm mở rộng phạm vi ra đa bệnh lâm sàng, Duong và các cộng sự [6] phát triển bộ dữ liệu ViMedNER thu thập từ các cổng tư vấn sức khỏe trực tuyến tại Việt Nam. Nghiên cứu thực nghiệm trên ViMedNER chứng minh mô hình đa ngữ XLM-R_large đạt kết quả vượt trội nhờ dung lượng tham số lớn.

Ở các hướng tiếp cận gần đây, Le-Duc và các cộng sự [7] xây dựng bộ dữ liệu spoken BioNER mang tên VietMed-NER từ hội thoại ghi âm giữa bác sĩ và bệnh nhân, chỉ ra lỗi nhận dạng giọng nói (ASR) làm suy giảm 16% hiệu năng NER. Ngoài ra, Tran và các cộng sự [8] công bố bộ dữ liệu hỏi đáp y khoa ViMedAQA để đánh giá năng lực suy luận của các mô hình ngôn ngữ lớn (LLMs), cho thấy các mô hình LLM tiếng Việt có xu hướng hoạt động tốt hơn khi được hướng dẫn bằng các cấu trúc gợi ý tiếng Anh.

### 2.2.3. Nhóm 3: Các mô hình ngôn ngữ pre-trained tiếng Việt và tăng cường tri thức
Về mặt kiến trúc biểu diễn, các mô hình ngôn ngữ tiền huấn luyện chuyên biệt cho tiếng Việt đóng vai trò quan trọng trong việc nâng cao độ chính xác của các tác vụ downstream. Tran và các cộng sự [9] giới thiệu mô hình ViDeBERTa xây dựng trên kiến trúc nâng cấp DeBERTaV3 tiền huấn luyện trên 138GB dữ liệu tiếng Việt sạch. Nhóm tác giả chứng minh rằng ViDeBERTa-base đạt hiệu năng tương đương hoặc vượt trội mô hình PhoBERT-large mặc dù kích thước tham số chỉ bằng 23%. Sự vượt trội này có nguyên nhân từ việc DeBERTaV3 sử dụng cơ chế chú ý phân tách tương tác nội dung - vị trí và kỹ thuật học đối kháng thay thế token (Replaced Token Detection), vượt qua các giới hạn của cơ chế mã hóa dựa trên RoBERTa (vốn là nền tảng của PhoBERT).

Để mở rộng khả năng xử lý sang miền y sinh lâm sàng, Tran-Tien và các cộng sự [10] xây dựng mô hình ViPubmedDeBERTa-base bằng cách tiếp tục thực hiện tiền huấn luyện mô hình ViDeBERTa trên 20 triệu bản tóm tắt y văn PubMed được dịch máy sang tiếng Việt. Nghiên cứu thực nghiệm của nhóm tác giả khẳng định việc tận dụng kho tàng y văn dịch máy là giải pháp chuyển giao tri thức hiệu quả cho các ngôn ngữ ít tài nguyên như tiếng Việt, giúp ViPubmedDeBERTa-base tối ưu hóa khả năng hiểu ngữ cảnh y học hơn so với mô hình PhoBERT miền tổng quát.

Trong các phương pháp chuyển giao tri thức, Yin và các cộng sự [11] đề xuất phương pháp GERBERA huấn luyện đa nhiệm đồng thời miền tổng quát và miền y sinh để giảm lỗi ranh giới thực thể. Đối với kịch bản nhận diện thực thể mở (zero-shot BioNER), Cocchieri và các cộng sự [12] đề xuất kiến trúc OPENBIONER sử dụng Cross-Encoder kết hợp mô tả ngôn ngữ tự nhiên của nhãn (Entity Type Description), giúp mô hình nhận diện tốt các nhãn chưa từng xuất hiện khi huấn luyện.

### 2.2.4. Xác định khoảng trống nghiên cứu (Research Gap)
Qua tổng hợp các công trình trên, nghiên cứu này xác định hai khoảng trống nghiên cứu lớn:
1.  **Thiếu quy trình tối ưu tích hợp AL và DES cho tiếng Việt**: Chưa có công trình nào nghiên cứu sự kết hợp giữa AL chọn mẫu tối ưu (bất định và đa dạng Distinct-K) với DES thế thực thể thích ứng theo lớp có cơ chế lọc rò rỉ dữ liệu (Zero Leakage Filter) cho bài toán BioNER tiếng Việt.
2.  **Thiếu thước đo chi phí hiệu chỉnh thực tế**: Các nghiên cứu AL tiếng Việt hiện nay hầu như chỉ đánh giá hiệu năng dựa trên số lượng câu gán nhãn tĩnh ($N$), bỏ qua việc đo lường nỗ lực hiệu chỉnh thực tế của chuyên gia y tế (ESR) thông qua khoảng cách Levenshtein khi hậu hiệu chỉnh gợi ý của máy.

---

## 2.3. Các phương pháp đánh giá (Metrics)

### 2.3.1. Các chỉ số đánh giá hiệu năng mô hình NER
Hiệu năng nhận diện thực thể được đánh giá dựa trên phương pháp đối khớp chính xác (Exact Match) ở cấp độ thực thể, yêu cầu mô hình phải xác định chính xác cả ranh giới (boundary) và loại nhãn (type) của thực thể lâm sàng.

Các chỉ số cơ bản gồm:
*   **Precision (Độ chính xác - $P$)**: Tỷ lệ thực thể mô hình dự đoán chính xác trên tổng số thực thể được mô hình tìm ra:
    $$P = \frac{TP}{TP + FP}$$
*   **Recall (Độ bao phủ - $R$)**: Tỷ lệ thực thể mô hình dự đoán chính xác trên tổng số thực thể thực tế có trong dữ liệu:
    $$R = \frac{TP}{TP + FN}$$
*   **F1-score ($F1$)**: Trung bình điều hòa giữa Precision và Recall:
    $$F1 = 2 \times \frac{P \times R}{P + R}$$
Trong đó $TP$ (True Positive) là số lượng thực thể khớp hoàn toàn; $FP$ (False Positive) là số lượng thực thể mô hình dự đoán sai hoặc thừa; $FN$ (False Negative) là số lượng thực thể bị mô hình bỏ sót.

Các chỉ số Precision, Recall và F1-score nêu trên được tính toán chi tiết cho từng lớp thực thể y khoa cụ thể bao gồm: `Symptom_and_Disease`, `DiagnosticProcedure`, `Location`, `DateTime` và `Organisation`. Hiệu năng tổng thể của mô hình được đánh giá thông qua giá trị F1-score tổng thể (Overall F1-score) đại diện cho khả năng nhận diện thực thể trên toàn bộ tập dữ liệu kiểm thử.


### 2.3.2. Các chỉ số đánh giá nỗ lực gán nhãn trong Học chủ động
Để đo lường hiệu quả kinh tế và mức độ tiết kiệm chi phí của phương pháp học chủ động so với phương pháp gán nhãn ngẫu nhiên truyền thống, đồ án định nghĩa hai chỉ số cốt lõi:

*   **Tỷ lệ tiết kiệm mẫu câu (Sentence Saving Ratio - SSR)**:
    $$\text{SSR (\%)} = \left( 1 - \frac{N_{\text{AL}}}{N_{\text{RS}}} \right) \times 100\%$$
    Trong đó $N_{\text{AL}}$ và $N_{\text{RS}}$ lần lượt là quy mô mẫu câu đã gán nhãn cần thiết ở Nhánh A (Active Learning) và Nhánh B (Random Sampling) để cùng đạt được một mức F1-score mục tiêu (ví dụ 60.0%).
*   **Tỷ lệ tiết kiệm thao tác hiệu chỉnh (Edit Saving Ratio - ESR)**:
    $$\text{ESR (\%)} = \left( 1 - \frac{E_{\text{AL}}}{E_{\text{RS}}} \right) \times 100\%$$
    Trong đó $E_{\text{AL}}$ và $E_{\text{RS}}$ lần lượt là tổng khoảng cách hiệu chỉnh Levenshtein tích lũy từ Vòng 0 đến Vòng hiện tại ở hai nhánh. Khoảng cách Levenshtein được tính toán ở cấp độ token giữa chuỗi nhãn BIO gộp do mô hình dự đoán trước (Pre-annotation) và chuỗi nhãn BIO chuẩn (Gold labels) của chuyên gia. Chỉ số này mô phỏng chân thực công sức sửa nhãn (thêm, xóa, sửa nhãn) của chuyên gia y tế trong quy trình gán nhãn có sự hỗ trợ của máy tính.

### 2.3.3. Kiểm định ý nghĩa thống kê qua giá trị p-value (Statistical Significance Testing)

Nhằm xác định xem sự khác biệt về hiệu năng (F1-score) giữa phương pháp gán nhãn đề xuất (Học chủ động kết hợp thế thực thể DES) và phương pháp đối chứng (Lấy mẫu ngẫu nhiên) có thực sự mang ý nghĩa khoa học hay chỉ do các biến số ngẫu nhiên khi phân chia dữ liệu, đồ án thực hiện phép kiểm định t-test cặp (Paired t-test) một phía (alternative='greater') để tính toán giá trị $p\text{-value}$.

Cách thức tính toán giá trị $p\text{-value}$ được thực hiện qua các bước toán học như sau:
1.  **Xác định cặp quan sát**: Gọi $X_i$ và $Y_i$ lần lượt là giá trị F1-score của Nhánh A (AL) và Nhánh B (Random) tại vòng lặp thứ $i$ ($i = 1, \dots, N$, với $N$ là tổng số vòng chạy).
2.  **Tính toán độ chênh lệch**: Đối với mỗi cặp vòng chạy, tính độ lệch hiệu năng $d_i = X_i - Y_i$.
3.  **Tính giá trị trung bình và độ lệch chuẩn của hiệu số**:
    *   Giá trị trung bình của sự khác biệt hiệu năng:
        $$\bar{d} = \frac{1}{N} \sum_{i=1}^N d_i$$
    *   Độ lệch chuẩn hiệu chỉnh của sự khác biệt:
        $$s_d = \sqrt{\frac{1}{N-1} \sum_{i=1}^N (d_i - \bar{d})^2}$$
4.  **Tính sai số chuẩn (Standard Error - SE)**:
    $$SE = \frac{s_d}{\sqrt{N}}$$
5.  **Tính chỉ số kiểm định $t$-statistic**:
    $$t = \frac{\bar{d}}{SE}$$
6.  **Tính toán giá trị $p\text{-value}$**:
    Dưới giả thuyết không $H_0$ ($\bar{d} \le 0$ - tức phương pháp AL không làm cải thiện hiệu năng so với Random), chỉ số $t$ tuân theo phân phối Student với bậc tự do $df = N - 1$. Giá trị $p\text{-value}$ cho phép kiểm định một phía bên phải (AL tốt hơn Random) được tính bằng diện tích dưới đường cong mật độ xác suất của phân phối Student từ điểm $t$ đến vô cùng:
    $$p\text{-value} = P(T \ge t | H_0)$$
    Trong đó $T$ là biến ngẫu nhiên tuân theo phân phối Student với $df = N - 1$ bậc tự do. Ngưỡng ý nghĩa tiêu chuẩn được thiết lập ở mức $\alpha = 0,05$. Nếu $p\text{-value} < 0,05$, ta bác bỏ giả thuyết $H_0$ và kết luận hiệu năng vượt trội của AL có ý nghĩa thống kê.
