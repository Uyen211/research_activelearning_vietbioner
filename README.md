# Nghiên Cứu Ứng Dụng Active Learning Giảm Chi Phí Gán Nhãn Dữ Liệu Y Sinh Tiếng Việt (VietBioNER)

---

## 1. Giới thiệu Đề tài Nghiên cứu

*   **Tên Đề tài**: *"Nghiên cứu ứng dụng Active Learning để giảm chi phí gán nhãn dữ liệu trong bài toán Nhận dạng Thực thể Có tên cho văn bản y sinh học tiếng Việt"*
*   **Câu hỏi Nghiên cứu Chính**: *"Liệu với cùng một ngân sách gán nhãn (cùng số lượng mẫu được gán nhãn), việc ứng dụng Active Learning (Học chủ động) có giúp mô hình NER đạt chất lượng nhận dạng thực thể cao hơn so với gán nhãn ngẫu nhiên truyền thống hay không? Và mức độ tiết kiệm thực tế là bao nhiêu?"*
*   **Bộ dữ liệu**: [VietBioNER](file:///d:/Study/TLU/Nlp/active-learning-VietBioNER/proposal/02_dataset_analysis/vietbioner.md) – Ngữ liệu lâm sàng tiếng Việt đầu tiên về bệnh lao (Tuberculosis), chứa 5 loại thực thể y khoa phức tạp.
*   **Mô hình Nền tảng (Backbone)**: [ViPubmedDeBERTa-base](file:///d:/Study/TLU/Nlp/active-learning-VietBioNER/proposal/03_model_architecture/kien_truc_mo_hinh.md) (86M tham số) với cơ chế disentangled attention vượt trội chuyên biệt y sinh học, tích hợp cấu hình thích ứng tham số hiệu quả **LoRA** (Rank $r=16$, Alpha $=32$).
*   **Đầu Phân loại Chuỗi**: Lớp **Linear-CRF Head** (loại bỏ lớp BiLSTM để tránh triệt tiêu thông tin truy vấn tĩnh và tăng tốc độ hội tụ).
*   **Cơ chế đo độ bất định**: **CRF Marginal Entropy (Entropy Xác suất biên)** sử dụng thuật toán **Forward-Backward** trên lớp CRF để tính trực tiếp xác suất biên mà không cần thêm tham số học thêm nào, giúp loại bỏ hoàn toàn overfitting trên tập dữ liệu nhỏ.
*   **Cơ chế lọc đa dạng**: **Distinct-K Filter** sử dụng Cosine Similarity trên các vector nhúng **Sentence-BERT pre-computed** nhằm giải quyết bài toán Tập độc lập lớn nhất (MIS) trên đồ thị tương đồng ngữ nghĩa.

---

## 2. Cấu trúc Thư mục Dự án (Repository Structure)

Dự án được quy hoạch khoa học để phân định rõ ràng giữa tài liệu lý thuyết, đề xuất thiết kế, các bài báo tham chiếu và mã nguồn chạy thực nghiệm:

```text
active-learning-VietBioNER/
│
├── proposal/                                 # Đề cương giải pháp và thiết kế kỹ thuật chi tiết
│   ├── 01_introduction_and_goals/
│   │   ├── muc_tieu_va_ket_qua.md            # Mục tiêu nghiên cứu & kết quả kỳ vọng
│   │   └── dong_luc_nghien_cuu.md            # Động lực nghiên cứu & câu hỏi cốt lõi
│   ├── 02_dataset_analysis/
│   │   ├── vietbioner.md                     # Phân tích phân phối bộ dữ liệu VietBioNER
│   │   └── gazetter.md                       # Phân tích các bộ Gazetteer y tế
│   ├── 03_model_architecture/
│   │   └── kien_truc_mo_hinh.md              # Thiết kế mô hình ViPubmedDeBERTa-base + LoRA + Linear-CRF
│   ├── 04_active_learning/
│   │   ├── active_learning_chi_tiet.md       # Toán học AL, Distinct-K Filter & Baseline
│   │   ├── Entity_type_description.md        # Giải thích cơ chế Mô tả Loại Thực thể và Masking
│   │   ├── dictionary_based_entity_substitution.md # Tăng cường dữ liệu bằng thế thực thể dựa trên từ điển (DES)
│   │   └── giai_thich_crf_marginal_entropy.md # Giải thích lý thuyết toán học CRF Marginal Entropy
│   ├── 05_evaluation_and_pipeline/
│   │   ├── pipeline.md                       # Quy trình 10 bước và thiết lập đối chứng song song
│   │   └── danh_gia_va_trien_khai.md         # Chỉ số SSR, ESR, Levenshtein & kế hoạch thực nghiệm
│   └── 06_research_rationale/
│       └── chuoi_suy_luan_thiet_ke.md        # Lý luận khoa học & chuỗi suy luận thiết kế luồng
│
├── literature_review/                        # Tài liệu tổng hợp và phân tích 12 bài báo nền tảng
│   ├── summary_synthesis.md                  # Báo cáo tổng hợp tri thức & phân nhóm 3 nhóm bài báo
│   ├── group_1_active_learning/              # Nhóm 1: Các chiến lược AL tối ưu chi phí (ocae197, MedNER...)
│   │   ├── ocae197.md
│   │   ├── applsci-12-05775.md
│   │   └── 3678178.md
│   ├── group_2_datasets/                     # Nhóm 2: Xây dựng dữ liệu & Benchmark tiếng Việt
│   │   ├── 2021.naacl-main.173.md
│   │   ├── 2022.lrec-1.385.md
│   │   ├── 5221-INIS.md
│   │   └── 2406.13337v3.md
│   ├── group_3_models/                       # Nhóm 3: Các mô hình tiền huấn luyện y khoa nâng cao
│   │   ├── 2023.findings-eacl.79.md
│   │   ├── 2023.paclic-1.83.md
│   │   ├── 2024.acl-srw.31.md
│   │   ├── 2406.10671v4.md
│   │   └── 2025.findings-naacl.47.md
│   └── raw_summaries/                        # Thư mục lưu các bản dịch/tóm tắt thô các bài báo gốc
│
├── colab/                                    # Mã nguồn chạy thực nghiệm trên Google Colab / Kaggle
│   ├── dataset/                              # Bộ dữ liệu VietBioNER gốc & Gazetteer
│   └── notebook/                             # Các notebook Jupyter chạy chính
│       ├── 01_mo_phong_active_learning.ipynb # Notebook chạy chính 2 nhánh thí nghiệm (Colab)
│       ├── 01_mo_phong_active_learning_kaggle.ipynb # Notebook chạy chính 2 nhánh thí nghiệm (Kaggle)
│       └── 02_danh_gia_va_truc_quan_hoa.ipynb # Notebook vẽ biểu đồ so sánh SSR/ESR
│
├── prompts/                                  # Tài liệu prompt hỗ trợ định hướng mô hình AI
│   └── research_paper_prompt.md
│
├── lessons_learned.md                        # Bài học kinh nghiệm đúc kết từ thực nghiệm
└── README.md                                 # Hướng dẫn chung và tổng quan dự án (File hiện tại)
```

---

## 3. Các Nhánh Thí nghiệm Đối chứng (Comparative Evaluation)

Dự án thiết lập **2 nhánh thí nghiệm song song** xuất phát từ cùng một Seed Set $L_0$ để trả lời khách quan các câu hỏi khoa học:

*   **Nhánh A (Đề xuất)**:
    *   *Mô hình*: ViPubmedDeBERTa-base + LoRA + Linear-CRF + Entity Type Descriptions.
    *   *Chọn mẫu*: CRF Marginal Entropy + Distinct-K Filter ($b = 100$, $\theta = 0.85$, fallback $\theta \in [0.90, 0.95]$).
    *   *Tăng cường*: Thế thực thể dựa trên từ điển (Dictionary-based Entity Substitution - DES) thế thực thể hiếm gặp từ Gazetteer sạch (đã lọc rò rỉ dữ liệu).
*   **Nhánh B (Random Baseline)**:
    *   *Mô hình*: ViPubmedDeBERTa-base + LoRA + Linear-CRF + Entity Type Descriptions.
    *   *Chọn mẫu*: Chọn mẫu ngẫu nhiên hoàn toàn (Random Sampling) với kích thước $b = 100$ câu/vòng.
    *   *Tăng cường*: Thế thực thể dựa trên từ điển (Dictionary-based Entity Substitution - DES) thế thực thể hiếm gặp tương tự Nhánh A (áp dụng đồng bộ để bảo đảm so sánh công bằng).

### Cơ chế đồng bộ hóa Baseline Vòng 0
Để triệt tiêu các sai số ngẫu nhiên do khởi tạo trọng số ngẫu nhiên (initial weights) và shuffling dữ liệu ở vòng lặp đầu tiên, hệ thống áp dụng **cơ chế đồng bộ hóa checkpoint Vòng 0**:
1. Nhánh A chạy huấn luyện Vòng 0 trên Seed Set $L_0$, lưu kết quả và checkpoint tốt nhất `best_model_AL_0.pt`.
2. Nhánh B khi khởi động sẽ tự động nạp checkpoint này làm baseline xuất phát điểm và chuyển thẳng sang Vòng 1. Điều này đảm bảo so sánh công bằng tuyệt đối 100% từ cùng một xuất phát điểm.

---

## 4. Các Siêu tham số Kỹ thuật (Hyperparameters)

Dưới đây là bảng cấu hình các tham số hệ thống được sử dụng đồng nhất trong mã nguồn và tài liệu đề cương:

| Tham số | Ý nghĩa | Giá trị trong Code |
| :--- | :--- | :--- |
| **MODEL_CHECKPOINT** | Mô hình nền tảng | `"manhtt-079/vipubmed-deberta-base"` |
| **LORA_R** | Rank của bộ điều hợp LoRA | `16` (Nâng cao sức học adapter) |
| **LORA_ALPHA** | Alpha của bộ điều hợp LoRA | `32` (Tương ứng 2x Rank) |
| **MAX_LEN** | Chiều dài câu tối đa đầu vào | `256` |
| **BATCH_SIZE** | Quy mô lô huấn luyện | `16` (Colab) / `8` (Kaggle - OOM prevention) |
| **LEARNING_RATE** | Tốc độ học của LoRA | `2e-4` (Tăng tốc hội tụ thích ứng) |
| **Linear LR** | Tốc độ học lớp Linear Head | `5e-4` (Phân tầng LLRD) |
| **CRF LR** | Tốc độ học lớp CRF Head | `1e-3` (Phân tầng LLRD học ma trạng trạng thái) |
| **AL_EPOCHS** | Số epoch tối đa mỗi vòng AL | `25` |
| **PATIENCE** | Kiên nhẫn dừng sớm (Early Stopping) | `5` |
| **MASK_ENTITY** | Tỷ lệ che giấu thực thể mục tiêu | `0.18` (Áp dụng từ **Vòng 2 trở đi**) |
| **MASK_CONTEXT** | Tỷ lệ che giấu từ ngữ cảnh nhãn O | `0.15` (Áp dụng từ **Vòng 2 trở đi**) |
| **BUDGET_LIMIT** | Giới hạn ngân sách gán nhãn tối đa | `0.50` (50% tập Train thô, tương đương 682 câu) |
| **BATCH_SELECT** | Số câu chọn thêm mỗi vòng lặp ($b$) | `100` câu |
| **THETA** | Ngưỡng tương đồng Cosine Distinct-K | `0.85` (Fallback động lên `0.90` và `0.95`) |
| **SEED** | Hạt giống ngẫu nhiên | `42` |
| **Class Loss Weights** | Trọng số phạt CRF Loss dương tính | `Organisation`: 5.0, `DateTime`: 4.0, `Location`: 3.0, `DiagnosticProcedure`: 2.0, `Symptom_and_Disease`: 1.0 |

---

## 5. Phương pháp luận Đánh giá Chi phí Gán nhãn

Dự án áp dụng hai thước đo định lượng cốt lõi để chứng minh mức độ tiết kiệm chi phí của Active Learning:

1.  **Tỷ lệ tiết kiệm mẫu câu (Sentence Saving Ratio - SSR)**:
    $$	ext{SSR (\%)} = \left(1 - rac{N_{	ext{AL}}}{N_{	ext{RS}}}ight) 	imes 100\%$$
    Trong đó $N$ là số mẫu câu cần thiết để đạt mức F1-score mục tiêu (75.0%).
2.  **Tỷ lệ tiết kiệm thao tác hiệu chỉnh (Edit Saving Ratio - ESR)**:
    $$	ext{ESR (\%)} = \left(1 - rac{E_{	ext{AL}}}{E_{	ext{RS}}}ight) 	imes 100\%$$
    Trong đó $E$ là tổng tích lũy khoảng cách **Levenshtein Edit Distance** cấp độ token giữa gợi ý nhãn (Pre-annotation) của mô hình và nhãn chuẩn của dữ liệu. Chỉ số này mô phỏng nỗ lực thực tế (chèn, xóa, sửa nhãn) của chuyên gia y tế khi hậu hiệu chỉnh nhãn máy gợi ý (Post-editing) trong quy trình thực tiễn.

---

## 6. Hướng dẫn Thực thi Thực nghiệm (Execution Guide)

### 6.1. Chạy trên Google Colab
1.  Tải toàn bộ thư mục dự án lên Google Drive cá nhân của bạn tại đường dẫn gốc: `/content/drive/MyDrive/active-learning-VietBioNER/`.
2.  Mở tệp [01_mo_phong_active_learning.ipynb](file:///d:/Study/TLU/Nlp/active-learning-VietBioNER/colab/notebook/01_mo_phong_active_learning.ipynb) bằng Google Colab.
3.  Cấu hình môi trường sử dụng **GPU T4** (miễn phí) hoặc tốt hơn.
4.  Chạy **Cell 1**: Cell này sẽ thiết lập môi trường, tự động cài đặt các phiên bản thư viện tương thích Python 3.12+ (bao gồm bản vá monkeypatch `collections.Iterable` cho `torchcrf`) và tự động khởi động lại Runtime.
5.  Chạy tuần tự tất cả các ô tiếp theo. Sau khi hoàn thành, logs và checkpoints sẽ được tự động lưu trữ tại `/content/drive/MyDrive/active-learning-VietBioNER/logs/seed_42/`.

### 6.2. Chạy và Resume trên Kaggle
Kaggle cung cấp GPU T4 x2 miễn phí, tuy nhiên thư mục Input là Read-only nên cần cấu hình đặc thù để chạy cơ chế Resume:
1.  **Tải Dataset lên Kaggle**: Nén thư mục dữ liệu y sinh thành file `.zip` theo cấu trúc sau và tải lên Kaggle Dataset với tên `active-learning-vietbioner3`:
    ```text
    active-learning-vietbioner3/
    ├── dataset/
    │   ├── vietbioner/ (train.txt, dev.txt, test.txt)
    │   ├── gazetteer/ (datetime.json, location.json...)
    │   └── preprocessed/ (các tệp đã chạy phân đoạn PyVi tĩnh)
    └── logs/ (seed_indices.json, sbert_embeddings.npy, current_L_indices.json...)
    ```
2.  **Khởi tạo Notebook**: Tạo một Notebook mới trên Kaggle, nạp nội dung của tệp [01_mo_phong_active_learning_kaggle.ipynb](file:///d:/Study/TLU/Nlp/active-learning-VietBioNER/colab/notebook/01_mo_phong_active_learning_kaggle.ipynb) và mount Dataset `active-learning-vietbioner3` vừa tạo.
3.  **Kaggle Resume Bridge (Cell 1)**: Đoạn mã này sẽ tự động sao chép toàn bộ logs và checkpoints từ Input (Read-only) sang thư mục làm việc ghi được `/kaggle/working/logs/`.
4.  **Chạy Resume**: Khi bấm Run All, `Cell 11a` sẽ tự động nhận diện chỉ mục đã gán nhãn cũ từ `/kaggle/working/logs/current_L_indices.json` và nạp mô hình đã lưu để tiếp tục vòng AL tiếp theo (ví dụ Vòng 2) mà không cần chạy lại từ Vòng 0.
5.  **Tránh tràn bộ nhớ (OOM)**: Notebook trên Kaggle tự động ghi đè tham số `Config.BATCH_SIZE = 8` để hoạt động an toàn trên RAM GPU T4 của Kaggle.

---

## 7. Tài liệu Tham chiếu Chính (References)

*   **[1] ocae197**: Chen, Y., et al. *"Utilizing active learning strategies in machine-assisted annotation for clinical named entity recognition."* Journal of Biomedical Informatics (2019).
*   **[2] applsci-12-05775**: *"Iterative Annotation of Biomedical NER Corpora with Deep Neural Networks and Knowledge Bases."* Applied Sciences (2022).
*   **[3] 3678178 (MedNER)**: *"MedNER: Enhanced Named Entity Recognition in Medical Corpus via Optimized Balanced and Deep Active Learning."* ACM Transactions on Intelligent Systems and Technology (2024).
*   **[5] VietBioNER**: *"A Named Entity Recognition Corpus for Vietnamese Biomedical Texts to Support Tuberculosis Treatment."* LREC (2022).
*   **[9] ViPubmedDeBERTa**: *"ViPubmedDeBERTa: A Pre-trained Model for Vietnamese Biomedical Text."* PACLIC (2023).
*   **[12] OPENBIONER**: *"Lightweight Open-Domain Biomedical Named Entity Recognition Through Entity Type Description."* NAACL (2025).
