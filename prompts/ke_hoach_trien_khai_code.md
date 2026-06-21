# Kế Hoạch Triển Khai Lập Trình Trên Google Colab (Google Colab Notebook Plan)

Tài liệu này đặc tả chi tiết lộ trình lập trình dự án nghiên cứu Active Learning trên bộ dữ liệu `VietBioNER` sử dụng mô hình nền tảng `ViPubmedDeBERTa-base` dưới dạng các khối mã lệnh (Code Cells) được phân bổ hợp lý trong **2 tệp Google Colab Notebook (`.ipynb`)** lưu trữ trên Google Drive.

---

## 1. Môi trường và Cấu trúc Thư mục trên Google Drive

Dự án được tổ chức thành 2 tệp notebook chạy độc lập và tuần tự. Dữ liệu trung gian giữa hai tệp notebook được truyền thông qua các tệp dữ liệu lưu trữ trực tiếp trên Google Drive tại đường dẫn: `/content/drive/MyDrive/active-learning-VietBioNER/`.

### 1.1. Sơ đồ thư mục dự án trên Drive

```text
/content/drive/MyDrive/active-learning-VietBioNER/
│
├── dataset/
│   ├── vietbioner/             # Tập dữ liệu VietBioNER gốc (syllable-level)
│   │   ├── train.txt           # 1.365 câu thô
│   │   ├── dev.txt             # 170 câu thô
│   │   └── test.txt            # 171 câu thô
│   │
│   ├── preprocessed/           # THƯ MỤC MỚI: Dữ liệu sau khi đã tách từ ghép (word-level)
│   │   ├── train_segmented.txt
│   │   ├── dev_segmented.txt
│   │   ├── test_segmented.txt
│   │   └── entity_descriptions_segmented.json # 5 câu mô tả nhãn tĩnh đã tách từ ghép
│   │
│   └── gazetteer/              # Bộ tri thức thuật ngữ y khoa (ĐÃ LỌC CÁCH 1)
│       ├── diagnostic_procedures_tb.json   # 327 thực thể quy trình chẩn đoán sạch
│       └── healthcare_organizations.json   # 554 thực thể tổ chức y tế sạch
│
├── notebooks/
│   ├── 01_mo_phong_active_learning.ipynb   # Tiền xử lý, định nghĩa mô hình, DS & vòng lặp AL chính
│   └── 02_danh_gia_va_truc_quan_hoa.ipynb  # Đọc file logs để so sánh đối chứng & kiểm định thống kê
│
└── logs/
    ├── logs_al.json            # Nhật ký lịch sử F1, Nt, Et của Nhánh A (AL)
    └── logs_random.json        # Nhật ký lịch sử F1, Nt, Et của Nhánh B (Random)
```
