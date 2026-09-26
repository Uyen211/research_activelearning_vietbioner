# Chi tiết Phương pháp và Chiến lược Học Chủ Động (Active Learning)

---

## 1. Các Chiến lược Chọn Mẫu và Thiết lập Phân nhánh Thí nghiệm

Trong nghiên cứu này, để bảo đảm so sánh công bằng hiệu năng gán nhãn thực tế giữa phương pháp gán nhãn chủ động và ngẫu nhiên, mô hình nền tảng ở cả hai nhánh thí nghiệm đều sử dụng **cơ chế định dạng đầu vào ghép nối Mô tả loại thực thể (Entity Type Descriptions)**. 

Cụ thể, các chiến lược chọn mẫu (Query Selection) được chia thành 2 nhánh thí nghiệm xuất phát từ cùng một tập khởi tạo (Seed Set $L_0$) để đối chứng song song:

### 1.1. Nhánh A (Đề xuất): CRF Marginal Entropy + Distinct-K Filter + Entity Descriptions
*   **Mô tả**: Đây là nhánh đề xuất chính của đề tài. Mẫu dữ liệu huấn luyện và ứng viên chưa gán nhãn $U_t$ được định dạng ghép nối với đoạn mô tả tự nhiên của các nhãn thực thể (`Entity Type Descriptions`), sau đó chạy suy luận qua mô hình.
*   **Cơ chế đo độ bất định (Uncertainty)**: Tính toán trực tiếp **CRF Marginal Entropy (Entropy Xác suất biên)** dựa trên phân phối xác suất biên của nhãn tại từng token (được tính bằng thuật toán **Forward-Backward** trên lớp CRF). Độ đo này phản ánh chính xác 100% độ tự tin thực tế của mô hình chuỗi mà không thêm tham số phụ, giải quyết triệt để lỗi overfitting trên tập dữ liệu ban đầu cực nhỏ.
*   **Cơ chế lọc đa dạng (Diversity)**: Các câu có giá trị entropy biên tổng (sum) cao nhất được đưa qua bộ lọc **Distinct-K Filter** sử dụng **Cosine Similarity trên các vector nhúng S-BERT pre-computed** (lưu trên RAM CPU) để giải bài toán Tập độc lập lớn nhất (Maximum Independent Set - MIS), loại bỏ hoàn toàn hiện tượng bất đẳng hướng (anisotropy) của các vector [CLS] thô của Transformer. Thuật toán tìm ngưỡng $\theta = 0.85$ để xây dựng đồ thị tương đồng ngữ nghĩa, loại bỏ các câu trùng lặp thông tin ngữ nghĩa trước khi đưa vào batch gán nhãn $b$, đảm bảo chọn mẫu đa dạng mà không gây tốn thêm tài nguyên GPU và độ trễ khi chạy vòng lặp AL.
*   **Tăng cường dữ liệu**: Áp dụng kỹ thuật **Thế thực thể dựa trên từ điển (Dictionary-based Entity Substitution - DES)** thông qua từ điển Gazetteer để tự động tăng cường các thực thể hiếm gặp được chọn, khắc phục lỗi mất cân bằng lớp nhãn.

### 1.2. Nhánh B (Random Baseline): Random Sampling + Entity Descriptions + Dictionary-based Entity Substitution (DES)
*   **Mô tả**: Chiến lược chọn mẫu ngẫu nhiên hoàn toàn từ tập chưa gán nhãn $U_t$.
*   **Cơ chế biểu diễn**: Mô hình NER của Nhánh B cũng sử dụng chính xác backbone **ViPubmedDeBERTa-base** và được huấn luyện trên định dạng đầu vào ghép nối với đoạn mô tả tự nhiên của các nhãn thực thể (`Entity Type Descriptions`) giống Nhánh A.
*   **Ý nghĩa đối chứng**: Việc giữ nguyên cả định dạng đầu vào Entity Type Description và kỹ thuật tăng cường **Thế thực thể dựa trên từ điển (DES)** ở cả hai nhánh giúp đảm bảo so sánh công bằng tuyệt đối. Điều này chứng minh rằng bất kỳ cải tiến hiệu năng nào của Nhánh A so với Nhánh B hoàn toàn đến từ bản thân thuật toán chọn mẫu chủ động (CRF Marginal Entropy + Distinct-K Filter) so với chọn ngẫu nhiên trong cùng một môi trường huấn luyện.
*   **Tăng cường dữ liệu**: Nhánh B **có áp dụng** cơ chế thế thực thể dựa trên từ điển (DES) tương tự như Nhánh A nhằm loại bỏ biến nhiễu của tri thức Gazetteer.

---

## 2. Thảo luận học thuật: Tại sao không gán nhãn một lần 15% mẫu chọn bằng K-Means (CLUSTER) thay vì dùng vòng lặp Active Learning?

Đây là câu hỏi cốt lõi phân biệt giữa **Học thụ động gom cụm một giai đoạn (Single-Stage Passive Cluster-based Selection)** và **Học chủ động tương tác đa giai đoạn (Multi-Stage Iterative Active Learning)**. Trong kịch bản giới hạn ngân sách gán nhãn ở mức 15% dữ liệu, việc chia nhỏ ngân sách để chạy vòng lặp Active Learning (ví dụ: Seed Set 5% $\rightarrow$ các vòng gán nhãn bổ sung 2.5% $\rightarrow$ 2.5% $\rightarrow$ 2.5% $\rightarrow$ 2.5%) đem lại hiệu quả vượt trội so với việc dùng K-Means chọn và gán nhãn một lần 15% dữ liệu vì 4 lý do khoa học sau:

### 2.1. Sự khuyết tật thông tin mô hình của K-Means (Model-Blind Limitation)
*   Thuật toán phân cụm K-Means chỉ hoạt động trên không gian vector biểu diễn câu (Sentence Embeddings) tĩnh của văn bản. Nó chỉ đo lường **tính đa dạng ngữ nghĩa (Diversity)** của văn bản một cách cơ học.
*   K-Means hoàn toàn **không biết** mô hình học sâu NER cụ thể (ví dụ: ViPubmedDeBERTa fine-tune trên dữ liệu y học tiếng Việt) đang "yếu" ở ranh giới phân loại nào. Nó có thể chọn ra các câu rất đa dạng về từ vựng, nhưng đó lại là những câu chứa các thực thể rất dễ nhận diện đối với mô hình NER. 
*   Ngược lại, **Active Learning là model-aware (hiểu mô hình)**. Nhờ có mô hình được huấn luyện ban đầu trên 5%, mô hình sẽ tự phản hồi cho ta biết nó đang mập mờ, bối rối (Uncertainty) ở những mẫu cụ thể nào trong 95% còn lại.

### 2.2. Sự dịch chuyển ranh giới quyết định (Decision Boundary Shift)
*   Ranh giới phân loại của mô hình NER thay đổi liên tục sau mỗi vòng huấn luyện với dữ liệu mới.
*   Một câu có độ mập mờ cực cao ở vòng lặp 1 (ví dụ: mô hình chưa phân biệt được một thuật ngữ cụ thể là `Symptom` hay `Disease`) có thể đã được giải quyết ở vòng lặp 2 nhờ một vài mẫu tương tự được nạp vào. 
*   If ta gán nhãn 15% cùng một lúc bằng K-Means, ta sẽ lãng phí ngân sách gán nhãn vào những mẫu có chung một pattern mà mô hình vốn dĩ chỉ cần 1-2 mẫu là đã tự suy luận được cho các mẫu còn lại. Vòng lặp AL giúp ta cập nhật liên tục "những gì mô hình đã biết" để chỉ gán nhãn "những gì mô hình chưa biết".

### 2.3. Cơ chế phối hợp động (Dynamic Cooperation)
*   Như được chứng minh bởi Chen và cộng sự (JAMIA, 2024), **Diversity** (K-Means/CLUSTER) cực kỳ tốt ở giai đoạn khởi đầu (khắc phục hiện tượng Cold Start) nhưng sẽ bị **bão hòa** rất nhanh khi dữ liệu tăng lên. Khi đó, **Uncertainty** (Độ bất định/Entropy) trở thành động lực chính để giúp mô hình tinh chỉnh ranh giới phân loại và đạt F1-score tối đa.
*   Sử dụng vòng lặp tương tác cho phép ta chuyển đổi chiến lược lấy mẫu một cách linh hoạt (từ CLUSTER ở 5% đầu sang Distinct-K ở các vòng tiếp theo). Điều này không thể thực hiện được nếu ta chọn 15% dữ liệu một lần bằng K-Means tĩnh.

### 2.4. Tối ưu hóa khả năng dự đoán nhãn trước (Pre-annotation)
*   Trong vòng lặp AL, từ vòng thứ 2 trở đi, mô hình NER đã có tri thức cơ bản và có khả năng **pre-annotate (dự đoán trước)** cho batch tiếp theo trước khi đưa cho chuyên gia y tế hiệu chỉnh. 
*   Việc hiệu chỉnh trên nhãn gợi ý có độ chính xác tăng dần giúp giảm trực tiếp số lượng thao tác chỉnh sửa (Edit Distance) của chuyên gia, trong khi phương pháp gán nhãn một lần 15% bắt buộc chuyên gia phải gán nhãn thủ công từ đầu (hoặc sử dụng một pre-annotation rất yếu của mô hình zero-shot), làm tăng đáng kể chi phí thực tế.
