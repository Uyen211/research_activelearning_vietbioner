# CHƯƠNG 5: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

---

## 5.1. Kết luận

### 5.1.1. Đánh giá mức độ hoàn thành mục tiêu nghiên cứu
Nghiên cứu đã hoàn thành tất cả các mục tiêu nghiên cứu được đặt ra ban đầu ở Chương 1. Kết quả đối chiếu thực tế giữa các cam kết thiết kế và kết quả thực nghiệm đạt được cụ thể như sau:
*   **Hiệu năng nhận dạng thực thể y sinh**: Mô hình đề xuất thuộc Nhánh A (Active Learning) đạt mức hiệu năng **71.85% F1-score** trên tập kiểm thử độc lập chỉ với ngân sách **485 câu gán nhãn** (chiếm xấp xỉ 45% tập train, tương đương 485/1.089 câu, và bằng 68.7% quy mô tập huấn luyện của nghiên cứu gốc). Hiệu năng này đã vượt mức mục tiêu cam kết ban đầu là đạt F1-score tối thiểu **70.0%** dưới ngân sách **500 câu**. Trong khi đó, mô hình đối chứng ngẫu nhiên Nhánh B chỉ đạt **63.07% F1-score**, tiệm cận sát với mức dự kiến ban đầu là **62.0%**.
*   **Hiệu quả tối ưu hóa số lượng câu gán nhãn**: Chỉ số tiết kiệm mẫu câu (Sentence Saving Ratio - SSR) đạt từ **27.37% đến 34.14%** tại các mốc F1-score từ 60.0% đến 65.0%, phù hợp với mục tiêu kỳ vọng ban đầu là đạt tỷ lệ tiết kiệm câu từ **30.0%** trở lên. Kết quả này chứng minh Học chủ động hỗ trợ chuyên gia giảm thiểu khoảng 1/3 công sức đọc tài liệu y khoa.

### 5.1.2. Tổng kết các đóng góp chính của đề tài
Đề tài nghiên cứu đã mang lại một số đóng góp khoa học và thực tiễn ý nghĩa cho bài toán nhận dạng thực thể y sinh tiếng Việt:
1.  **Đóng góp về mặt phương pháp luận**: Thiết lập thành công một quy trình gán nhãn lặp tích hợp tối ưu giữa Học chủ động (AL) và Thế thực thể dựa trên từ điển (DES). Sự kết hợp giữa bộ chọn mẫu lai (thuật toán CRF Marginal Entropy đo độ bất định và bộ lọc Distinct-K Filter giữ độ đa dạng ngữ nghĩa thông qua vector nhúng Sentence-BERT) với thuật toán thế thực thể có kiểm soát rò rỉ (Zero-Leakage Filter) đã giải quyết hiệu quả bài toán mất cân bằng nhãn và tối ưu hóa ngân sách gán nhãn.
2.  **Đóng góp về mặt kiến trúc hệ thống**: Xây dựng kiến trúc mô hình NER hiệu quả tham số dựa trên mô hình ngôn ngữ chuyên biệt y sinh ViPubmedDeBERTa-base kết hợp LoRA và đầu phân loại chuỗi CRF. Quyết định loại bỏ lớp BiLSTM trung gian giúp bảo vệ biểu diễn ngữ nghĩa của mô tả thực thể tĩnh (Entity Type Description), giải quyết trực tiếp lỗi ranh giới thực thể y học đa từ phức tạp.
3.  **Khắc phục điểm nghẽn của các nghiên cứu đi trước**: Mô hình đề xuất đã nâng cao đáng kể hiệu năng nhận diện lớp thực thể khó và hiếm gặp `DiagnosticProcedure` (đạt 49.28% F1-score so với mức 40.96% của Random), giải quyết triệt để lỗi dự đoán nhầm thành nhãn O vốn là hạn chế lớn của mô hình PhoBERT trong nghiên cứu gốc.

## 5.2. Hướng phát triển trong tương lai

Để khắc phục các hạn chế kỹ thuật đã được chỉ ra tại mục 4.6 (Chương 4), nghiên cứu vạch ra bốn hướng phát triển mở rộng trọng tâm trong tương lai:

### 5.2.1. Tối ưu hóa quy trình tiền xử lý và đồng bộ hóa nhãn
Nhằm giải quyết triệt để vấn đề lệch ranh giới và đứt gãy nhãn khi chuyển đổi định dạng dữ liệu, nghiên cứu hướng tới:
*   Phát triển các công cụ phân đoạn câu chuyên biệt cho văn bản lâm sàng tiếng Việt có khả năng nhận diện và bỏ qua các dấu chấm viết tắt chuyên ngành (như *TB.*, *b.n.*, *đ/t*), ngăn chặn hiện tượng tách câu sai lệch.
*   Cải tiến thuật toán đồng bộ nhãn để xử lý tốt hơn các ranh giới từ ghép phức tạp, giảm thiểu lỗi mất nhãn khi chạy PyVi.

### 5.2.2. Cải tiến cơ chế thế thực thể giảm thiểu nhiễu cú pháp
Để hạn chế nhiễu ngữ nghĩa (semantic noise) từ cơ chế thế thực thể dựa trên từ điển (DES), các nghiên cứu tiếp theo sẽ tập trung vào:
*   Xây dựng cơ chế thế thực thể có nhận thức ngữ cảnh (Context-aware Entity Substitution) bằng cách kiểm tra sự tương đồng phân phối POS (Part-of-Speech) hoặc sử dụng các bộ lọc ngữ pháp ngôn ngữ tự nhiên. Việc này đảm bảo các từ khóa được chọn từ Gazetteer luôn phù hợp với cấu trúc cú pháp của câu văn gốc.
*   Nghiên cứu các cơ chế lọc nhiễu động sau tăng cường để tự động loại bỏ các câu văn augmented không đạt tiêu chuẩn mạch lạc.

### 5.2.3. Tối ưu hóa siêu tham số trên hạ tầng tính toán mở rộng
Khi có cơ hội tiếp cận nguồn tài nguyên phần cứng lớn hơn (các hệ thống máy chủ đa GPU hoặc dịch vụ đám mây chuyên dụng), nhóm nghiên cứu sẽ:
*   Thực hiện tối ưu hóa siêu tham số tự động (Automated Hyperparameter Optimization) thông qua các thuật toán tìm kiếm lưới (Grid Search) hoặc tối ưu hóa Bayes. Mục tiêu là xác định cấu hình LoRA (rank, alpha) và các tốc độ học phân lớp LLRD tốt nhất cho kiến trúc mô hình.
*   Khảo sát tác động của các tỷ lệ che giấu masking động nhằm tối đa hóa khả năng tổng quát hóa ngữ cảnh của mô hình ViPubmedDeBERTa.

### 5.2.4. Mở rộng quy mô thực nghiệm và xây dựng công cụ gán nhãn thực tế
*   **Đa dạng hóa ngữ liệu và hạt giống**: Chạy mô phỏng học chủ động trên nhiều hạt giống ngẫu nhiên (seeds) khác nhau và mở rộng đánh giá hiệu năng sang các bộ dữ liệu y khoa tiếng Việt khác như ViMedNER hoặc PhoNER_COVID19. Điều này giúp kiểm chứng toàn diện và nâng cao ý nghĩa thống kê của phương pháp chọn mẫu đề xuất.
*   **Xây dựng giao diện gán nhãn tương tác (Interactive Annotation Tool)**: Thiết kế một công cụ gán nhãn chạy trên Web tích hợp quy trình AL + DES của đề tài. Công cụ này sẽ đưa ra các gợi ý nhãn tự động từ ViPubmedDeBERTa cho bác sĩ y khoa hiệu chỉnh, chuyển đổi các đo lường chi phí Levenshtein gián tiếp trong thực nghiệm thành sự tiết kiệm thời gian và công sức thực tế cho chuyên gia lâm sàng.
