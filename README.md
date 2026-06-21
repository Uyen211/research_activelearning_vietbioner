# Đề Tài: Nghiên Cứu Ứng Dụng Active Learning Giảm Chi Phí Gán Nhãn Dữ Liệu Y Sinh Tiếng Việt

---

## 1. Giới thiệu Đề tài Nghiên cứu

*   **Tên Đề tài**: *"Nghiên cứu ứng dụng Active Learning để giảm chi phí gán nhãn dữ liệu trong bài toán Nhận dạng Thực thể Có tên cho văn bản y sinh học tiếng Việt"*
*   **Câu hỏi Nghiên cứu Chính**: *"Liệu với cùng một ngân sách gán nhãn (cùng số lượng mẫu được gán nhãn), việc ứng dụng Active Learning (Học chủ động) có giúp mô hình đạt chất lượng nhận dạng thực thể cao hơn so với gán nhãn ngẫu nhiên truyền thống hay không? Và mức độ tiết kiệm thực tế là bao nhiêu?"*
*   **Bộ dữ liệu**: [VietBioNER](file:///d:/Study/TLU/Nlp/active-learning-VietBioNER/proposal/02_dataset_analysis/phan_tich_dataset.md) – Corpus học thuật tiếng Việt đầu tiên về bệnh lao (Tuberculosis) dùng cho y học lâm sàng, chứa 5 loại thực thể y khoa phức tạp.
*   **Mô hình Nền tảng (Backbone)**: [ViPubmedDeBERTa-base](file:///d:/Study/TLU/Nlp/active-learning-VietBioNER/proposal/03_model_architecture/kien_truc_mo_hinh.md) (86M tham số) với kiến trúc disentangled attention vượt trội, kết hợp với đầu phân loại chuỗi **BiLSTM-CRF** và cơ chế đo độ bất định toán học toàn cục, không tham số học thêm **CRF Marginal Entropy**.

---

## 2. Cấu trúc Thư mục Dự án (Repository Structure)

Dự án đã được quy hoạch và cấu trúc lại một cách khoa học để phân định rõ ràng giữa tài liệu nghiên cứu lý thuyết, đề xuất kỹ thuật chi tiết, các bản nháp và tài liệu lập trình phụ trợ:

```
active-learning-VietBioNER/
│
├── proposal/                                 # Chi tiết đề cương giải pháp và thiết kế kỹ thuật
│   ├── 01_introduction_and_goals/
│   │   ├── muc_tieu_va_ket_qua.md            # [Mục tiêu nghiên cứu & kết quả kỳ vọng]
│   │   └── dong_luc_nghien_cuu.md            # [Động lực nghiên cứu & câu hỏi cốt lõi]
│   ├── 02_dataset_analysis/
│   │   └── phan_tich_dataset.md              # [Phân tích phân phối bộ dữ liệu VietBioNER]
│   ├── 03_model_architecture/
│   │   └── kien_truc_mo_hinh.md              # [Thiết kế mô hình ViPubmedDeBERTa-base + BiLSTM-CRF]
│   ├── 04_active_learning/
│   │   ├── active_learning_chi_tiet.md       # [Toán học AL, Distinct-K Filter & Baseline]
│   │   └── giai_thich_crf_marginal_entropy.md # [Giải thích lý thuyết toán học CRF Marginal Entropy]
│   ├── 05_evaluation_and_pipeline/
│   │   ├── pipeline.md                       # [Quy trình 10 bước và thiết lập đối chứng song song]
│   │   └── danh_gia_va_trien_khai.md         # [Chỉ số SSR, ESR, Levenshtein & kế hoạch thực nghiệm]
│   └── 06_research_rationale/
│       └── chuoi_suy_luan_thiet_ke.md        # [Lý luận khoa học & chuỗi suy luận thiết kế luồng]
│
├── literature_review/                        # Tài liệu tổng hợp và phân tích 12 bài báo nền tảng
│   ├── summary_synthesis.md                  # [Báo cáo tổng hợp tri thức & phân nhóm 3 group]
│   ├── group_1_active_learning/              # [Group 1: Các chiến lược AL tối ưu chi phí]
│   │   ├── ocae197.md                        # [Tóm tắt chi tiết ocae197]
│   │   ├── applsci-12-05775.md               # [Tóm tắt chi tiết applsci-12-05775]
│   │   └── 3678178.md                        # [Tóm tắt chi tiết 3678178 - MedNER]
│   ├── group_2_datasets/                     # [Group 2: Xây dựng dữ liệu & Benchmark tiếng Việt]
│   │   ├── 2021.naacl-main.173.md            # [Tóm tắt chi tiết PhoNER_COVID19]
│   │   ├── 2022.lrec-1.385.md                # [Tóm tắt chi tiết VietBioNER]
│   │   ├── 5221-INIS.md                      # [Tóm tắt chi tiết ViMedNER]
│   │   └── 2406.13337v3.md                   # [Tóm tắt chi tiết VietMed-NER]
│   ├── group_3_models/                       # [Group 3: Các mô hình tiền huấn luyện y khoa nâng cao]
│   │   ├── 2023.findings-eacl.79.md          # [Tóm tắt chi tiết ViDeBERTa]
│   │   ├── 2023.paclic-1.83.md               # [Tóm tắt chi tiết ViPubmedDeBERTa]
│   │   ├── 2024.acl-srw.31.md                # [Tóm tắt chi tiết ViMedAQA]
│   │   ├── 2406.10671v4.md                   # [Tóm tắt chi tiết GERBERA]
│   │   └── 2025.findings-naacl.47.md         # [Tóm tắt chi tiết OPENBIONER]
│   └── raw_summaries/                        # [Thư mục lưu các bản dịch/tóm tắt thô các bài báo gốc]
│
├── colab/                                    # Mã nguồn chạy thực nghiệm trên Google Colab
│   ├── dataset/                              # Bộ dữ liệu VietBioNER gốc
│   └── notebook/                             # Các notebook Jupyter
│       ├── 01_mo_phong_active_learning.ipynb # [Notebook chạy chính 2 nhánh thí nghiệm]
│       └── 02_danh_gia_va_truc_quan_hoa.ipynb # [Notebook vẽ biểu đồ so sánh SSR/ESR]
│
├── prompts/                                  # Tài liệu prompt hỗ trợ định hướng mô hình AI
│   └── research_paper_prompt.md              # [Prompt định nghĩa và tóm tắt paper gốc]
│
├── lessons_learned.md                        # Bài học kinh nghiệm đúc kết từ thực nghiệm
└── README.md                                 # Hướng dẫn chung và tổng quan dự án (File hiện tại)
```

---

## 3. Các Nhánh Thí nghiệm Đối chứng (Comparative Evaluation)

Dự án thiết lập **2 nhánh thí nghiệm song song** xuất phát từ cùng một Seed Set $L_0$ (5% dữ liệu) để trả lời khách quan các câu hỏi khoa học:

*   **Nhánh A (Đề xuất)**: 
    *   *Mô hình*: ViPubmedDeBERTa-base + BiLSTM-CRF + Entity Type Descriptions.
    *   *Chọn mẫu*: CRF Marginal Entropy + Distinct-K Filter.
    *   *Tăng cường*: Distant Supervision Augmentation (DS) thế thực thể hiếm.
*   **Nhánh B (Random Baseline)**:
    *   *Mô hình*: ViPubmedDeBERTa-base + BiLSTM-CRF + Entity Type Descriptions.
    *   *Chọn mẫu*: Ngẫu nhiên hoàn toàn (Random Sampling).
    *   *Tăng cường*: Distant Supervision Augmentation (DS) thế thực thể hiếm (áp dụng đồng bộ để đảm bảo so sánh công bằng).

---

## 4. Phương pháp luận Đánh giá Chi phí Gán nhãn

Thay vì chỉ sử dụng độ đo số câu gán nhãn đơn giản như các nghiên cứu AL truyền thống, đề tài áp dụng hai độ đo định lượng chi phí:
1.  **Tỷ lệ tiết kiệm mẫu câu (Sentence Saving Ratio - SSR)**: Đo lường phần trăm số câu giảm được để đạt F1 mục tiêu.
2.  **Tỷ lệ tiết kiệm thao tác hiệu chỉnh (Edit Saving Ratio - ESR)**: Đo lường phần trăm số thao tác chèn, xóa, sửa đổi thẻ nhãn giảm được đối với chuyên gia con người (mô phỏng thông qua khoảng cách **Levenshtein Edit Distance** giữa gợi ý nhãn của máy và nhãn chuẩn của VietBioNER). ESR giúp lượng hóa công sức thực tế trong quy trình hậu hiệu chỉnh (Post-editing) lâm sàng.

---

## 5. Lộ trình Triển khai Đề tài (Roadmap)

- [x] **Giai đoạn 1**: Nghiên cứu tài liệu nền tảng và thiết lập Đề cương chi tiết (Literature Review & Proposal Layout).
- [x] **Giai đoạn 2**: Chuẩn hóa cấu trúc thư mục dự án, cập nhật luồng Dual-branch đối chứng, xây dựng [Chuỗi suy luận thiết kế](file:///d:/Study/TLU/Nlp/active-learning-VietBioNER/proposal/06_research_rationale/chuoi_suy_luan_thiet_ke.md).
- [x] **Giai đoạn 3**: Thiết lập môi trường Python, tiền xử lý và tách từ bộ dữ liệu `VietBioNER` (sử dụng *PyVi*).
- [x] **Giai đoạn 4**: Lập trình kiến trúc mô hình chính *ViPubmedDeBERTa-base + BiLSTM-CRF* và cơ chế đo độ bất định *CRF Marginal Entropy*.
- [x] **Giai đoạn 5**: Triển khai mã nguồn mô phỏng vòng lặp AL và bộ lọc *Distinct-K Filter* cho Nhánh A.
- [x] **Giai đoạn 6**: Chạy thí nghiệm hai nhánh, ghi nhật ký (logging) kết quả đánh giá (F1, SSR, ESR) qua từng vòng lặp.
- [x] **Giai đoạn 7**: Trực quan hóa dữ liệu (vẽ đường cong học tập) và đồng bộ hóa báo cáo khoa học/đề cương giải pháp.
