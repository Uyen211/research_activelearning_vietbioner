# Động lực Nghiên cứu & Câu hỏi Nghiên cứu Cốt lõi

---

## 1. Đặt vấn đề và Động lực Nghiên cứu

Trong lĩnh vực xử lý ngôn ngữ tự nhiên y sinh (Biomedical NLP) tiếng Việt, việc tạo dựng các bộ dữ liệu gán nhãn chất lượng cao là một thách thức lớn. Dữ liệu y sinh đòi hỏi người gán nhãn phải là các chuyên gia y tế (bác sĩ, dược sĩ) có chuyên môn sâu, dẫn đến chi phí nhân công và thời gian cực kỳ đắt đỏ. Bộ dữ liệu **VietBioNER** (văn bản học thuật chuyên sâu về bệnh lao) có quy mô tương đối nhỏ (1.706 câu) nhưng chứa các thực thể có độ phức tạp cao, đặc biệt là thực thể quy trình chẩn đoán (`DiagnosticProcedure`) với F1-score của các mô hình supervised hiện tại chỉ đạt khoảng 55.56%. 

Đề tài **"Nghiên cứu ứng dụng Active Learning để giảm chi phí gán nhãn dữ liệu trong bài toán Nhận dạng Thực thể Có tên cho văn bản y sinh học tiếng Việt"** được thực hiện nhằm giải quyết nút thắt này. 

Bằng cách huấn luyện mô hình lặp đi lặp lại và sử dụng các tiêu chí toán học để truy vấn các mẫu có giá trị cao nhất từ tập chưa gán nhãn, Học chủ động (Active Learning) giúp chúng ta đạt được hiệu năng tối đa của mô hình với số lượng mẫu gán nhãn ít nhất, từ đó giảm thiểu tối đa sự tham gia thủ công của các chuyên gia y khoa.

---

## 2. Câu hỏi Nghiên cứu Cốt lõi

Đề tài này tập trung giải quyết và trả lời hai câu hỏi nghiên cứu thực tiễn sau:

> [!IMPORTANT]
> **Câu hỏi nghiên cứu cốt lõi**:
> 1. *Liệu với cùng một ngân sách gán nhãn (cùng số lượng câu được gán nhãn), Active Learning có giúp đạt được chất lượng mô hình cao hơn so với gán nhãn ngẫu nhiên (Random Sampling) hay không?*
> 2. *Và mức độ tiết kiệm chi phí gán nhãn (số lượng câu cần gán nhãn và số thao tác hiệu chỉnh nhãn của con người) để đạt cùng một mức hiệu năng mục tiêu là bao nhiêu?*
