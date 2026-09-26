# CHƯƠNG 1: MỞ ĐẦU

## 1.1. Lý do chọn đề tài

Trong tiến trình chuyển đổi số ngành y tế, việc tự động hóa khai thác và xử lý dữ liệu từ bệnh án lâm sàng đóng vai trò thiết thực cho công tác quản lý và hỗ trợ chẩn đoán. Là bước đi nền tảng cho các nhiệm vụ trích xuất thông tin hạ nguồn, nhiệm vụ Nhận dạng Thực thể Y sinh (Biomedical Named Entity Recognition - BioNER) giúp định vị các thực thể lâm sàng cốt lõi như tên bệnh, triệu chứng và các phương pháp chẩn đoán hay điều trị, như được thảo luận trong các nghiên cứu của Liu và Wong [1] cũng như Silvestri và các cộng sự [2]. Tuy nhiên, các mô hình học sâu hiện đại đòi hỏi một lượng lớn dữ liệu gán nhãn chất lượng cao để đạt hiệu năng ổn định. Đối với ngôn ngữ tiếng Việt, việc xây dựng ngữ liệu BioNER chuẩn hóa đối mặt với rào cản lớn về tài nguyên dữ liệu y sinh, vốn hạn chế và rời rạc như ghi nhận trong các công bố về bộ dữ liệu VietBioNER của Phan và các cộng sự [5] hay ViMedNER của Pham và các cộng sự [6]. Việc gán nhãn dữ liệu này đòi hỏi chuyên môn lâm sàng cao từ các y bác sĩ, dẫn đến chi phí nhân lực thực tế lớn và tiêu tốn nhiều thời gian.

Bên cạnh chi phí, sự phân phối bất cân xứng giữa các loại thực thể y khoa trong thực tế cũng gây khó khăn cho quá trình huấn luyện mô hình. Trong khi các mô tả triệu chứng và bệnh chiếm tỷ lệ đa số, các thông tin lâm sàng quan trọng khác như ngày tháng hay tổ chức y tế lại xuất hiện với tần suất thưa thớt [5]. Do đó, phương pháp gán nhãn ngẫu nhiên truyền thống (Random Sampling - RS) không tối ưu trong trường hợp này do xu hướng lựa chọn trùng lặp các câu chứa ngữ nghĩa tương tự, dẫn đến hiện tượng quá khớp (overfitting) trên các mẫu phổ biến và bỏ sót các thực thể hiếm. Điều này trực tiếp gây lãng phí ngân sách gán nhãn mà không giúp cải thiện đáng kể hiệu năng tổng thể của mô hình.

Nhằm tối ưu hóa nguồn lực gán nhãn hạn chế, phương pháp Học chủ động (Active Learning - AL) được nghiên cứu như một giải pháp hiệu quả giúp lựa chọn các mẫu dữ liệu có độ bất định cao và đa dạng ngữ nghĩa nhất để ưu tiên gán nhãn trước [1], [3]. Bên cạnh đó, việc tích hợp Tăng cường dữ liệu bằng Thế thực thể dựa trên từ điển (Dictionary-based Entity Substitution - DES) thông qua việc thay thế thực thể từ các bộ từ điển chuyên ngành (Gazetteer) sạch sẽ hỗ trợ mở rộng vốn từ lâm sàng cho mô hình mà không làm phát sinh chi phí gán nhãn thủ công [2]. Sự kết hợp giữa thuật toán AL thông minh và cơ chế DES thích ứng, hoạt động trên mô hình ngôn ngữ lớn chuyên biệt miền y sinh ViPubmedDeBERTa-base [10] với LoRA và đầu phân loại Linear-CRF [12], hứa hẹn sẽ mang lại sự cân bằng giữa chất lượng nhận diện và chi phí gán nhãn.

Xuất phát từ nhu cầu thực tiễn nhằm giảm thiểu công sức gán nhãn thủ công cho đội ngũ y bác sĩ, đồng thời tối ưu hóa khả năng trích xuất thực thể lâm sàng tiếng Việt, đề tài *'Nghiên cứu ứng dụng Active Learning để giảm chi phí gán nhãn dữ liệu trong bài toán Nhận dạng Thực thể Có tên cho văn bản y sinh học tiếng Việt'* được thực hiện nhằm xây dựng, thực nghiệm và đánh giá một giải pháp toàn diện và tối ưu cho bài toán này.

## 1.2. Mục tiêu nghiên cứu

Đề tài hướng tới việc xây dựng, thực nghiệm và kiểm chứng một quy trình gán nhãn lặp tối ưu dựa trên học chủ động cho dữ liệu y sinh tiếng Việt. Để đạt được mục tiêu chung đó, nghiên cứu tập trung giải quyết các mục tiêu cụ thể dưới đây:

*   **Xây dựng cơ chế chọn mẫu dựa trên độ bất định**: Thiết lập thuật toán CRF Marginal Entropy bằng phương pháp Forward-Backward trên lớp CRF. Thuật toán này đo lường độ bất định trực tiếp trên phân phối xác suất biên của nhãn mà không phụ thuộc vào các tham số học thêm, giúp loại bỏ hiện tượng quá khớp trên tập dữ liệu nhỏ.
*   **Phát triển cơ chế lọc đa dạng ngữ nghĩa**: Thiết kế bộ lọc Distinct-K Filter sử dụng Cosine Similarity trên các vector nhúng Sentence-BERT đã được tính toán sẵn. Mục tiêu là loại bỏ các mẫu câu trùng lặp ngữ nghĩa trong tập ứng viên trước khi gán nhãn.
*   **Tích hợp cơ chế tăng cường dữ liệu thích ứng**: Xây dựng thuật toán Thế thực thể dựa trên từ điển (DES) sử dụng Gazetteer chuyên ngành, áp dụng tỷ lệ tăng cường thích ứng động dựa trên tần suất phân bố thực tế của từng loại thực thể để chống hiện tượng mất cân bằng nhãn.
*   **Cấu hình mô hình nhận dạng hiệu quả tham số**: Huấn luyện bộ thích ứng LoRA kết hợp đầu phân loại Linear-CRF trên mô hình ngôn ngữ ViPubmedDeBERTa-base [10], áp dụng cơ chế Entity Type Description [12] để chuyển đổi bài toán nhận dạng thực thể đa phân lớp thành phân loại nhị phân động.
*   **Đánh giá định lượng hiệu quả tiết kiệm chi phí**: Đo lường hai chỉ số Sentence Saving Ratio (SSR) và Edit Saving Ratio (ESR) dựa trên khoảng cách Levenshtein để chứng minh mức độ giảm tải công sức gán nhãn thực tế so với phương pháp gán nhãn ngẫu nhiên truyền thống.

## 1.3. Đối tượng và phạm vi nghiên cứu

### Đối tượng nghiên cứu
*   Các chiến lược Học chủ động trong bài toán gán nhãn chuỗi (Sequence Labeling).
*   Thuật toán đo độ bất định CRF Marginal Entropy và cơ chế lọc đa dạng ngữ nghĩa Distinct-K Filter.
*   Phương pháp tăng cường thế thực thể lâm sàng dựa trên từ điển chuyên ngành (Gazetteer).
*   Kiến trúc mô hình ViPubmedDeBERTa-base tích hợp adapter LoRA, lớp Linear-CRF Head và phương thức định dạng Entity Type Description.
*   Các thước đo hiệu quả chi phí gán nhãn gồm SSR và ESR.

### Phạm vi nghiên cứu
*   **Về dữ liệu**: Nghiên cứu giới hạn thực nghiệm trên bộ dữ liệu VietBioNER [5] (ngữ liệu lâm sàng về bệnh lao phổi chứa 1.362 câu thực tế sử dụng sau tiền xử lý và tách từ ghép) và 5 bộ Gazetteer y khoa chuyên ngành được xử lý lọc rò rỉ dữ liệu để đảm bảo tính khách quan cho tập kiểm thử.
*   **Về công nghệ và mô hình**: Sử dụng mô hình tiền huấn luyện ViPubmedDeBERTa-base [10] với cấu hình LoRA ($r=16, \alpha=32$), kết hợp thư viện PyTorch, thư viện PyVi để phân đoạn từ tiếng Việt, và Sentence-BERT để tạo các vector nhúng ngữ nghĩa.
*   **Về môi trường thực nghiệm**: Nghiên cứu mô phỏng phản hồi của chuyên gia y tế (Simulated Oracle) bằng cách truy xuất nhãn chuẩn có sẵn của bộ dữ liệu VietBioNER trong môi trường thực nghiệm Google Colab và Kaggle.

## 1.4. Kết quả dự kiến đạt được

Nghiên cứu hướng tới việc bàn giao các kết quả học thuật và sản phẩm thực nghiệm có thể kiểm chứng sau:

*   **Hiệu năng nhận diện của mô hình**: Mô hình học chủ động đề xuất (Nhánh A) đạt F1-score tối thiểu **70.0%** trên tập kiểm thử độc lập khi quy mô dữ liệu gán nhãn không vượt quá **500 câu** (tương đương xấp xỉ 45% tập huấn luyện ban đầu, cụ thể là 485/1.089 câu). Trong khi đó, mô hình đối chứng sử dụng chọn mẫu ngẫu nhiên (Nhánh B) dự kiến chỉ đạt F1-score khoảng **62.0%** dưới cùng một giới hạn ngân sách gán nhãn.
*   **Hiệu quả giảm thiểu chi phí gán nhãn**: 
    *   Chỉ số tiết kiệm mẫu câu (SSR) đạt tối thiểu **30.0%** so với phương pháp gán nhãn ngẫu nhiên khi hướng tới cùng một hiệu năng đích của mô hình.
    *   Chỉ số tiết kiệm thao tác hiệu chỉnh (ESR) đạt tối thiểu **20.0%**, thể hiện sự giảm thiểu đáng kể số lượng lỗi nhãn cần sửa của chuyên gia y tế nhờ sự hỗ trợ từ các gợi ý nhãn chất lượng của mô hình (Pre-annotation).
*   **Sản phẩm đóng góp**: 
    *   Mã nguồn thực nghiệm hoàn chỉnh chạy trên Google Colab và Kaggle, bảo đảm tính tái lập kết quả cao.
    *   Báo cáo phân tích chi tiết về đường cong học tập và các đánh giá ý nghĩa thống kê của phương pháp học chủ động so với phương pháp ngẫu nhiên.

## 1.5. Cấu trúc của đồ án

Báo cáo đồ án được tổ chức thành 6 chương chính như sau:

*   **Chương 1: Mở đầu**: Trình bày bối cảnh nghiên cứu, tính cấp thiết của đề tài, các mục tiêu cụ thể, đối tượng, phạm vi nghiên cứu và kết quả dự kiến đạt được.
*   **Chương 2: Cơ sở lý thuyết và các công trình liên quan**: Tổng quan bài toán nhận dạng thực thể có tên, cơ chế thích ứng LoRA, lớp Linear-CRF, phương pháp Entity Type Description và các nghiên cứu đi trước về học chủ động và tăng cường dữ liệu y sinh.
*   **Chương 3: Giải pháp và phương pháp thực hiện**: Mô tả chi tiết sơ đồ luồng xử lý của hệ thống, thuật toán tính toán độ bất định CRF Marginal Entropy, bộ lọc Distinct-K Filter, cơ chế tăng cường thế thực thể lâm sàng và thiết kế hai nhánh đối chứng song song.
*   **Chương 4: Thực nghiệm và kết quả**: Trình bày chi tiết thiết lập siêu tham số huấn luyện mô hình, kết quả F1-score của các vòng lặp, biểu đồ so sánh SSR/ESR và đánh giá ý nghĩa thống kê.
*   **Chương 5: Kết luận và hướng phát triển**: Tổng kết các đóng góp chính của đề tài, phân tích các hạn chế còn tồn tại và vạch ra các hướng nghiên cứu mở rộng trong tương lai.
