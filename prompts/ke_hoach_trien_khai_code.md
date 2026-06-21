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
│       ├── diagnostic_procedures_tb.json   
│       └── healthcare_organizations.json   
│       └── .... (và các gazetteer cho nhãn khác)   
│
├── notebooks/
│   ├── 01_mo_phong_active_learning.ipynb   # Tiền xử lý, định nghĩa mô hình, DS & vòng lặp AL chính
│   └── 02_danh_gia_va_truc_quan_hoa.ipynb  # Đọc file logs để so sánh đối chứng & kiểm định thống kê
│
└── logs/
    ├── logs_al.json            # Nhật ký lịch sử F1, Nt, Et của Nhánh A (AL)
    └── logs_random.json        # Nhật ký lịch sử F1, Nt, Et của Nhánh B (Random)
```
# Phân tích và Đánh giá Mã nguồn Thực nghiệm Active Learning

## 1. Mô tả chi tiết từng bộ phận trong Code

### 1.1. Cell 1: Thiết lập Môi trường & Cài đặt Thư viện

**Đầu vào**: Môi trường Python 3.12+ với các thư viện cần thiết

**Xử lý**: 
- Hàm `check_and_install_libs()` kiểm tra sự tồn tại của các thư viện quan trọng: `peft`, `seqeval`, `pyvi`, `sentence_transformers`, `torchcrf`, `Levenshtein`, `bitsandbytes`, `triton`
- Nếu thiếu, tiến hành gỡ bỏ các phiên bản cũ và cài đặt phiên bản tương thích với CUDA 12.x và Python 3.12+ chuẩn bao gồm:
  - `transformers==4.44.2`
  - `peft==0.12.0`
  - `accelerate==0.34.2`
  - `bitsandbytes==0.43.3`
  - `triton==3.0.0`
  - `seqeval==1.2.2`
  - `pyvi==0.1.1`
  - `pytorch-crf==0.7.2` (Module import: `torchcrf`)
  - `sentence-transformers==2.7.0`
  - `Levenshtein==0.25.1`
- Tự động khởi động lại Runtime nếu cần

**Đầu ra**: Môi trường được cấu hình với đúng phiên bản thư viện

**Công cụ sử dụng**: `pip`, `subprocess`, `sys`

---

### 1.1b. Cell 1.1: Kiểm tra phiên bản thư viện và kiểm tra tương thích (MỚI)

**Đầu vào**: Môi trường Python và các thư viện đã cài đặt

**Xử lý**:
- In ra phiên bản chi tiết của Python, PyTorch, Transformers, PEFT, Seqeval, PyVi, Levenshtein, và Sentence-Transformers.
- **Bản vá tương thích (Monkeypatch)**: Khắc phục lỗi tương thích của thư viện `pytorch-crf==0.7.2` trên Python 3.10+ (do hàm `collections.Iterable` đã bị loại bỏ ở các phiên bản Python mới, gây crash khi import module `torchcrf`). Bản vá được khai báo trước khi nạp `torchcrf`:
  ```python
  import collections
  try:
      collections.Iterable = collections.abc.Iterable
  except AttributeError:
      pass
  ```
- Khởi tạo lớp `CRF` từ `torchcrf` để xác thực hoạt động.
- Kiểm tra tính sẵn sàng của GPU CUDA và cấu hình `PYTORCH_CUDA_ALLOC_CONF`.

**Đầu ra**: Báo cáo kiểm định môi trường và xác thực tương thích thư viện thành công

**Công cụ sử dụng**: `sys`, `os`, `torch`, `collections`, `torchcrf`

---

### 1.2. Cell 2: Khai báo Thư viện & Cấu hình Hệ thống

**Đầu vào**: Cấu hình đường dẫn, tham số mô hình

**Xử lý**:

**Lớp `Config`** lưu trữ toàn bộ tham số cấu hình:

1. **Đường dẫn dữ liệu**:
   - Tự động phát hiện môi trường (Kaggle vs. Colab/Local)
   - `DATASET_DIR`: thư mục chứa dữ liệu VietBioNER (train/dev/test.txt)
   - `PREPROCESSED_DIR`: thư mục lưu dữ liệu đã tách từ ghép
   - `GAZETTEER_DIR`: thư mục chứa từ điển chuyên ngành (5 file JSON)
   - `LOGS_DIR`: thư mục lưu log và checkpoint

2. **Siêu tham số mô hình**:
   - `MODEL_CHECKPOINT = "manhtt-079/vipubmed-deberta-base"`: mô hình DeBERTa tiền huấn luyện trên PubMed tiếng Việt
   - `MAX_LEN = 256`: độ dài tối đa của chuỗi đầu vào
   - `BATCH_SIZE = 16` (sau đó được cập nhật xuống 8 thông qua ô trung gian `Config.BATCH_SIZE = 8` để tránh tràn bộ nhớ GPU)
   - `LEARNING_RATE = 1e-4`: tốc độ học cho LoRA (tăng để LoRA thích ứng tốt hơn trên tập L_t nhỏ)
   - `AL_EPOCHS = 30`: số epoch tối đa cho mỗi vòng lặp
   - `PATIENCE = 15`: số epoch chờ đợi trước khi Early Stopping (tránh dừng sớm ảo do bias nhãn O)
   - `MASK_ENTITY = 0.18` và `MASK_CONTEXT = 0.15`: tỷ lệ che giấu mặc định cho thực thể và ngữ cảnh (được cập nhật động thành 0.0 trong vòng lặp 0, 1, 2)

3. **Tham số Học chủ động**:
   - `BUDGET_LIMIT = 0.50`: giới hạn ngân sách tối đa 50% tập huấn luyện
   - `BATCH_SELECT = 50`: mỗi vòng chọn 50 câu
   - `THETA = 0.85`: ngưỡng Cosine Similarity cho bộ lọc Distinct-K
   - `SEED = 42`: hạt giống ngẫu nhiên cho tái lập thực nghiệm

4. **Danh sách nhãn**: 5 loại thực thể trong VietBioNER

**Bản vá tương thích (Monkeypatch)**: Tương tự Cell 1.1, khai báo vá `collections.Iterable` trước khi import `CRF` từ `torchcrf` để đảm bảo hệ thống không bị crash khi nạp cấu hình.

**Hàm `set_seed()`**:
- Thiết lập seed cho `random`, `numpy`, `torch` để đảm bảo tính tái lập
- Hỗ trợ cả CPU và GPU

**Đầu ra**: 
- Các thư mục được tạo (preprocessed, logs)
- In ra đường dẫn hiện tại

---

### 1.3. Cell 1.5: Thiết lập Hugging Face Token

**Đầu vào**: Token từ Kaggle Secrets

**Xử lý**:
- Sử dụng `UserSecretsClient` để lấy `HF_TOKEN` từ môi trường Kaggle
- Thiết lập biến môi trường `HF_TOKEN` để xác thực khi tải mô hình từ Hugging Face

**Đầu ra**: Token được thiết lập

**Công cụ sử dụng**: `kaggle_secrets`

---

### 1.4. Cell 3: Tiền xử lý & Tách từ ghép bằng PyVi

**Đầu vào**: Các file `.txt` định dạng CoNLL (train/dev/test)

**Xử lý**:

1. **`clean_and_normalize_vietnamese(text)`**: Chuẩn hóa văn bản (loại bỏ khoảng trắng thừa)

2. **`align_segmented_tags(orig_tokens, orig_tags, segmented_tokens)`**: 
   - **Vấn đề**: Khi PyVi tách từ ghép, số lượng token thay đổi (ví dụ: "chẩn_đoán" từ 2 token thành 1)
   - **Giải pháp**: Thuật toán ánh xạ nhãn từ token gốc sang token đã tách:
     - Duyệt từng token đã tách (`segmented_tokens`)
     - Gom các token gốc tương ứng bằng cách so khớp chuỗi chữ thường
     - Với mỗi token đã tách, xác định nhãn:
       - Nếu có token gốc nào trong nhóm mang nhãn `B-` → gán `B-`
       - Ngược lại, lấy nhãn phổ biến nhất (ưu tiên nhãn khác 'O')
     - Trả về danh sách nhãn mới với độ dài khớp với `segmented_tokens`

3. **`process_conll_file(file_path, output_path)`**:
   - Đọc file CoNLL, tách câu bằng dòng trống
   - Với mỗi câu: 
     - Nối các token thành câu văn bản
     - Dùng `ViTokenizer.tokenize()` để tách từ ghép
     - Gọi `align_segmented_tags()` để ánh xạ nhãn
   - Ghi kết quả ra file mới

4. **Định nghĩa mô tả thực thể**:
   - 5 đoạn văn bản mô tả ngắn cho 5 loại thực thể
   - Được tách từ ghép bằng PyVi và lưu thành file JSON

**Đầu ra**:
- Các file `train_segmented.txt`, `dev_segmented.txt`, `test_segmented.txt`
- File `entity_descriptions_segmented.json`
- File `metadata.json`
- 5 tệp Gazetteer tĩnh đã phân đoạn ranh giới từ ghép bằng PyVi lưu trong thư mục `PREPROCESSED_DIR`

**Công cụ sử dụng**: `PyVi` (ViTokenizer.tokenize)

---

### 1.5. Cell 4: Thiết lập Lớp Dataset và DataCollator

**Đầu vào**: Dữ liệu đã tách từ ghép, mô tả thực thể, tokenizer

**Xử lý**:

#### Lớp `VietBioNERDataset`:

1. **Khởi tạo**:
   - Đọc dữ liệu từ file hoặc nhận trực tiếp danh sách câu
   - Lưu các tham số: `descriptions`, `tokenizer`, `max_len`, `label_list`
   - Xây dựng danh sách `queries`:

2. **Xây dựng `queries`**:
   - Với mỗi câu, xác định các nhãn xuất hiện (`present_labels`)
   - **Queries dương tính (100%)**: Với mỗi nhãn có trong câu, tạo 1 query `(sent_idx, label_idx)`
   - **Queries âm tính (Downsampling)**:
     - Nếu là tập Train: lấy **ngẫu nhiên 1 nhãn không xuất hiện** trong câu
     - Nếu là tập Dev/Test: lấy **toàn bộ các nhãn không xuất hiện**

3. **`__getitem__`**:
   - Lấy câu và nhãn tương ứng với query
   - Tạo nhãn nhị phân cho loại thực thể được truy vấn:
     - Nếu tag khớp với `query_label` → giữ nguyên `B` hoặc `I`
     - Ngược lại → gán `O`
   - Trả về: `tokens`, `tags` (nhị phân), `query_desc`, `query_label_idx`

#### Lớp `EntityMaskingCollator`:

1. **Contextual Word Masking (Chỉ trong Train)**:
   - Sử dụng các tỷ lệ che giấu động từ `Config`:
     - **Thực thể**: thay thế bằng `[MASK]` với xác suất `Config.MASK_ENTITY` (mặc định 18%)
     - **Ngữ cảnh (nhãn O)**: thay thế bằng `[MASK]` với xác suất `Config.MASK_CONTEXT` (mặc định 15%)

2. **Mã hóa chuỗi ghép nối**:
   - Định dạng: `[CLS] s [SEP] d_c [SEP]`
   - Sử dụng tokenizer của DeBERTa với `is_split_into_words=True`

3. **Căn chỉnh nhãn và BIO cho Subwords (Sửa đổi logic)**:
   - Dùng `word_ids()` và `sequence_ids()` để xác định token nào thuộc câu gốc
   - `[CLS]` (t_idx=0) được gán nhãn `O`
   - Các token thuộc `seq_id=0` (câu gốc) và có `word_id` không None:
     - Subword đầu tiên của từ nhận nhãn gốc của từ đó (`B`, `I` hoặc `O`).
     - Các subwords tiếp theo của từ đó nhận nhãn `I` nếu từ gốc có nhãn thực thể (`B`/`I`), hoặc nhận nhãn `O` nếu từ gốc có nhãn `O`. Điều này sửa lỗi chuyển đổi chuỗi nhãn `B, B, B` làm hỏng ma trận transition của CRF.
   - Các token khác (mô tả, padding) → gán nhãn `O` và `valid_mask=False`

4. **Đầu ra batch**:
   - `input_ids`, `attention_mask`, `token_type_ids`
   - `tags`: nhãn nhị phân (0=O, 1=B, 2=I)
   - `mask`: mask cho các token hợp lệ (câu gốc)
   - `query_label_idx`: chỉ số nhãn đang truy vấn

#### Hàm `create_dataloader_from_raw`:
- Tạo Dataset từ danh sách câu
- Tạo DataLoader với batch size từ Config

**Đầu ra**: DataLoader sẵn sàng cho huấn luyện

---

### 1.6. Cell 5: Thiết lập Kiến trúc Mô hình MultiTaskBiLSTMCRF + LoRA

**Đầu vào**: `model_checkpoint` (đường dẫn đến ViPubmedDeBERTa)

**Xử lý**:

#### Lớp `MultiTaskBiLSTMCRF`:

1. **Tải backbone DeBERTa**:
   - `AutoModel.from_pretrained(model_checkpoint)`

2. **Tích hợp LoRA**:
   - Tạo `LoraConfig` với:
     - `r=8`: rank của ma trận thích ứng
     - `lora_alpha=16`: hệ số scale
     - `lora_dropout=0.1`: dropout
     - `target_modules=["query_proj", "value_proj"]`: chỉ áp dụng cho query và value projections
   - Sử dụng `get_peft_model()` để wrap backbone
   - In ra số lượng tham số trainable

3. **Đầu phân loại**:
   - `self.fc = nn.Linear(hidden_size, 3)`: ánh xạ hidden states sang 3 nhãn (O, B, I)
   - `self.crf = CRF(num_tags=3, batch_first=True)`: CRF để mô hình hóa phụ thuộc chuỗi

4. **Forward pass**:
   - Đưa input qua DeBERTa → hidden states
   - `emissions = self.fc(last_hidden)`: điểm phát xạ cho từng token
   - `cls_embedding = last_hidden[:, 0, :]`: vector embedding của token `[CLS]`
   - Trả về `{"emissions": emissions, "cls_embedding": cls_embedding}`

**Đầu ra**: Mô hình có thể huấn luyện với LoRA

**Công cụ sử dụng**: `transformers`, `peft`, `torchcrf`

---

### 1.7. Cell 6: Thuật toán CRF Forward-Backward và Xác suất biên

**Đầu vào**: Emissions, transitions, masks

**Xử lý**:

#### `compute_marginal_probabilities(emissions, transitions, start_transitions, end_transitions, mask)`:

1. **Forward Pass**:
   - Khởi tạo `alpha` với `-1e9`
   - `alpha[:, 0, :] = start_transitions + emissions[:, 0, :]`
   - Với mỗi bước t (1 → seq_len-1):
     - `prev_alpha = alpha[:, t-1, :].unsqueeze(2)` → (B, K, 1)
     - `trans = transitions.unsqueeze(0)` → (1, K, K)
     - `next_alpha = logsumexp(prev_alpha + trans, dim=1) + emissions[:, t, :]`
     - Nếu mask = True → cập nhật, ngược lại → giữ nguyên

2. **Backward Pass (Sửa lỗi ghi đè beta)**:
   - Khởi tạo `beta` với `-1e9`
   - `beta[seq_lens - 1, :] = end_transitions`
   - Với mỗi bước t từ seq_len-2 → 0:
     - `next_beta = beta[:, t+1, :].unsqueeze(1)` → (B, 1, K)
     - `trans = transitions.unsqueeze(0)` → (1, K, K)
     - `emit = emissions[:, t+1, :].unsqueeze(1)` → (B, 1, K)
     - `prev_beta = logsumexp(next_beta + trans + emit, dim=2)`
     - `beta[:, t, :] = torch.where(m, prev_beta, beta[:, t, :])`: Chỉ cập nhật nếu token tiếp theo hợp lệ (`m` là `True`). Nếu `m` là `False` (bước chuyển từ padding sang token cuối cùng hoặc trong vùng padding), giữ nguyên giá trị hiện tại của `beta[:, t, :]` thay vì ghi đè bằng giá trị `beta[:, t+1, :]` (vốn chứa `-1e9`).

3. **Tính xác suất biên**:
   - `log_p = alpha + beta` → log xác suất chưa chuẩn hóa
   - `marginals = softmax(log_p, dim=-1)` → xác suất biên

#### `compute_sentence_entropy(all_emissions, all_masks, transitions, start_transitions, end_transitions)`:

1. Gộp emissions và masks của 5 nhãn thành tensor shape (5, seq_len, 3) và (5, seq_len)
2. Tính xác suất biên cho toàn bộ 5 chuỗi
3. Tính Shannon Entropy: `-sum(marginals * log(marginals + 1e-9))`
4. Với mỗi nhãn, loại bỏ token `[CLS]` và token padding, tính trung bình entropy
5. Trả về trung bình entropy của 5 nhãn

#### `merge_and_resolve_conflicts(sentence_tokens, all_emissions, all_masks, crf_model, tokenizer, descriptions)`:

1. **Tính xác suất biên và giải mã Viterbi** cho 5 chuỗi
2. **Trích xuất spans** từ mỗi chuỗi dự đoán:
   - Dùng `word_ids` và `sequence_ids` để ánh xạ token dự đoán sang token câu gốc
   - Xác định các span thực thể (B → I → O)
   - Tính điểm tự tin trung bình cho mỗi span

3. **Giải quyết xung đột**:
   - Sắp xếp spans theo: điểm tự tin giảm dần → độ dài tăng dần → thứ tự nhãn
   - Duyệt spans, chỉ giữ spans không bị chồng lấn
   - Gán nhãn `B-` cho token đầu tiên, `I-` cho các token tiếp theo

4. **Trả về**: Chuỗi nhãn BIO đa phân lớp

**Đầu ra**: Xác suất biên, entropy, chuỗi nhãn gộp

---

### 1.8. Cell 7: Distant Supervision Augmentation

**Đầu vào**: 
- `tokens`, `tags`: câu và nhãn gốc
- 5 từ điển Gazetteer: `gaz_proc`, `gaz_org`, `gaz_date`, `gaz_loc`, `gaz_sym`

**Xử lý**:

#### `ds_augment_sentence(tokens, tags, gaz_proc, gaz_org, gaz_date, gaz_loc, gaz_sym)`:

1. **Trích xuất spans**:
   - Duyệt qua `tags`, xác định các span thực thể (B → I → ... → O)
   - Lưu: `(start_idx, end_idx, etype)`

2. **Chọn span ngẫu nhiên** để thay thế

3. **Xác định tham số augmentation theo loại thực thể**:
   - `Symptom_and_Disease`: M=1, p_activation=0.50
   - `DiagnosticProcedure`: M=2, p_activation=1.0
   - `Location`: M=2, p_activation=1.0
   - `DateTime`: M=2, p_activation=1.0
   - `Organisation`: M=3, p_activation=1.0

4. **Kiểm tra kích hoạt**:
   - Nếu `random.random() > p_activation` → không tạo bản sao

5. **Tạo bản sao**:
   - Chọn ngẫu nhiên M thực thể từ Gazetteer
   - Với mỗi thực thể mới:
     - Tách chuỗi bằng `.split()` trực tiếp để lấy các tokens (do Gazetteer đã được tiền xử lý và tách từ ghép sẵn ở Cell 3, loại bỏ việc gọi PyVi động).
     - Tạo nhãn BIO: `[B-etype] + [I-etype] * (len(new_tokens) - 1)`
     - Thay thế span cũ bằng span mới trong câu
     - Thêm câu mới vào danh sách

6. **Trả về**: Danh sách các câu đã augmented

#### `augment_dataset_with_ds(sentences, gaz_proc, gaz_org, gaz_date, gaz_loc, gaz_sym)`:
- Với mỗi câu trong `sentences`:
  - Thêm câu gốc vào pool
  - Gọi `ds_augment_sentence()` để tạo các bản sao
  - Thêm các bản sao vào pool
- Trả về: Pool dữ liệu đã augmented

**Đầu ra**: Dữ liệu đã tăng cường (gốc + bản sao)

**Công cụ sử dụng**: `PyVi`, `random`

---

### 1.9. Cell 8: Bộ chọn mẫu Distinct-K Filter

**Đầu vào**: 
- `candidates`: danh sách các chỉ số câu ứng viên
- `predicted_losses`: điểm entropy của từng ứng viên
- `cls_embeddings`: vector embedding của từng câu (từ S-BERT)
- `b`: số lượng câu cần chọn
- `theta`: ngưỡng cosine similarity

**Xử lý**:

#### `select_distinct_k(candidates, predicted_losses, cls_embeddings, b, theta)`:

1. **Tính Cosine Similarity**:
   - Chuẩn hóa `cls_embeddings` theo norm
   - `cos_sim = dot(normed_emb, normed_emb.T)`

2. **Sắp xếp theo entropy giảm dần**:
   - `sorted_indices = argsort(predicted_losses)[::-1]`

3. **Lọc Distinct-K**:
   - Duyệt `sorted_indices`
   - Nếu `idx` đã bị đánh dấu `ignored` → bỏ qua
   - Thêm `idx` vào `selected_indices`
   - Tìm tất cả các `neighbor` có `cos_sim[idx] > theta`
   - Đánh dấu tất cả `neighbor` là `ignored`

4. **Fallback khi thiếu mẫu**:
   - Nếu `len(selected_candidates) < b`:
     - Thử tăng theta lên 0.90 và 0.95
     - Nếu vẫn thiếu → bù bằng các câu có entropy cao nhất

5. **Trả về**: `b` câu được chọn

**Đầu ra**: Danh sách các chỉ số câu được chọn

---

### 1.10. Cell 9: Simulated Oracle & Levenshtein Distance

**Đầu vào**: `pred_tags`, `gold_tags` (hai chuỗi nhãn)

**Xử lý**:

#### `compute_levenshtein_distance(pred_tags, gold_tags)`:

1. **Ánh xạ nhãn sang ký tự**:
   - Duyệt qua `pred_tags`: với mỗi tag mới, gán một ký tự duy nhất
   - Làm tương tự với `gold_tags`
   - Đảm bảo cùng một tag được ánh xạ sang cùng một ký tự

2. **Tạo chuỗi ký tự**:
   - `pred_str = "".join(char_list_pred)`
   - `gold_str = "".join(char_list_gold)`

3. **Tính Levenshtein Distance**:
   - Sử dụng thư viện `Levenshtein.distance()`

4. **Trả về**: Khoảng cách chỉnh sửa tối thiểu

**Đầu ra**: Số nguyên - số lần chỉnh sửa cần thiết

**Công cụ sử dụng**: `Levenshtein`

---

### 1.11. Cell 10: Trình quản lý Huấn luyện & Early Stopping

**Đầu vào**: `model`, `train_loader`, `val_loader`, `epochs`, `lr`, `patience`

**Xử lý**:

#### `train_model(model, train_loader, val_loader, epochs, lr, patience, device, is_al_branch)`:

1. **Tối ưu hóa tham số có phân biệt (LLRD)**:
   - Lọc các tham số có `requires_grad=True`
   - Nhóm theo thành phần với learning rate khác nhau:
     - **DeBERTa + LoRA**: `lr = 2e-5`
     - **Linear Layer**: `lr = 5e-4`
     - **CRF**: `lr = 1e-3`

2. **Vòng lặp huấn luyện**:
   - Với mỗi epoch (tối đa `epochs`):
     - **Train**: 
       - Lấy batch từ `train_loader`
       - Forward → Tính CRF loss với `reduction='none'` để lấy loss riêng của từng truy vấn.
       - Xác định các Positive Queries trong batch (truy vấn thực sự chứa thực thể nhãn B hoặc I).
       - Tính trọng số động cho từng câu truy vấn: đối với Positive queries, trọng số bằng `class_weights[C]` (`Organisation`: 5.0, `DateTime`: 4.0, `Location`: 3.0, `DiagnosticProcedure`: 2.0, `Symptom_and_Disease`: 1.0); đối với các queries âm tính còn lại, trọng số là `1.0`.
       - Nhân loss từng câu với trọng số động này, tính trung bình cộng toàn batch → backward → optimizer step.
       - Ghi lại train loss.
     - **Validation**:
       - Lấy batch từ `val_loader`
       - Forward → Tính CRF loss có áp dụng trọng số động Class-aware tương tự như pha Train để đồng bộ hóa và phản ánh chính xác hiệu năng thực thể hiếm.
       - Ghi lại validation loss.

3. **Early Stopping**:
   - Nếu `val_loss < best_val_loss`:
     - Cập nhật `best_val_loss`
     - Lưu checkpoint
     - Reset `patience_counter = 0`
   - Ngược lại:
     - `patience_counter += 1`
     - Nếu `patience_counter >= patience` → dừng

4. **Khôi phục best checkpoint**:
   - Sau khi dừng, load model từ checkpoint tốt nhất

5. **Dọn dẹp**:
   - Xóa checkpoint tạm
   - Gọi `gc.collect()` và `torch.cuda.empty_cache()`

**Đầu ra**: 
- `history`: lịch sử train/val loss
- `best_val_loss`: loss validation tốt nhất
- `epoch_stopped`: số epoch đã huấn luyện
- `early_stopped`: True nếu dừng sớm

---

### 1.12. Cell 11: Nhánh A - Active Learning Loop (Được chia làm 3 ô nhỏ độc lập chạy tuần tự để tránh NameError)

**Đầu vào**: Dữ liệu train/dev/test đã tách từ ghép, các file tri thức Gazetteer tĩnh đã phân đoạn ranh giới từ ghép, mô tả nhãn.

---

#### **Ô 1 (Cell 11a): B. Hàm load_raw_conll và C. Cơ chế Resume**

**Xử lý**:
1. **Thiết lập và nạp các tệp tĩnh**:
   - Thiết lập device (ưu tiên CUDA GPU), AutoTokenizer nạp checkpoint `"manhtt-079/vipubmed-deberta-base"`.
   - Nạp 5 tệp Gazetteer tĩnh đã tách ranh giới từ ghép lưu trong `PREPROCESSED_DIR`.
   - Nạp mô tả nhãn tĩnh từ `entity_descriptions_segmented.json`.
2. **Hàm `load_raw_conll(file_path)`**:
   - Đọc dữ liệu từ file định dạng CoNLL, tách các câu bằng dòng trống, trả về danh sách câu ở dạng `[(tokens, tags), ...]`.
   - Nạp các file tập dữ liệu huấn luyện, kiểm định, đánh giá: `train_segmented`, `dev_segmented`, `test_segmented`.
3. **Cơ chế khôi phục Resume**:
   - Kiểm tra và đọc tệp trạng thái đã gán nhãn `current_L_indices.json` từ `Config.LOGS_DIR` nếu có để tải lại trạng thái huấn luyện vòng trước (L_indices, cumulative_edit_distance).
   - Tải file nhật ký `logs_al.json` để xác định vòng lặp khởi đầu `start_loop` và điểm `f1` tốt nhất đã đạt được trước đó.

**Đầu ra**: Dữ liệu huấn luyện thô, trạng thái khôi phục index và lịch sử vòng lặp.

---

#### **Ô 2 (Cell 11b): A. Khởi tạo Seed Set ($L_0$) bằng Stratified Sampling**

**Xử lý**:
1. **Mã hóa ngữ nghĩa (S-BERT)**:
   - Nếu tồn tại `sbert_embeddings.npy` → nạp trực tiếp.
   - Nếu không → sử dụng mô hình `'keepitreal/vietnamese-sbert'` mã hóa toàn bộ tập câu huấn luyện thành vector embeddings và lưu lại để tái sử dụng.
2. **Khởi tạo tập mẫu hạt giống bằng Stratified Sampling**:
   - Phân tầng tập câu huấn luyện thành 6 tầng (strata) dựa theo thứ tự độ hiếm thực thể: `Organisation` (ORG), `DateTime` (DATE), `Location` (LOC), `DiagnosticProcedure` (DP), `Symptom_and_Disease` (SYM), và tầng `O` (chỉ chứa nhãn O).
   - Rút trích ngẫu nhiên có kiểm soát từ mỗi tầng: **15 câu mỗi tầng từ 1 đến 5, và 10 câu từ tầng 6**, tạo lập đúng 85 câu cho Seed Set $L_0$ để đảm bảo phủ rộng đầy đủ thực thể hiếm ngay từ vòng 0.
   - Lưu indices vào `seed_indices.json` để đồng bộ baseline đối chứng.
   - Gán `current_L_indices = set(seed_indices)` nếu là phiên huấn luyện mới hoàn toàn.

**Đầu ra**: Tập embeddings của tập Train và tập chỉ mục hạt giống L_0.

---

#### **Ô 3 (Cell 11c): Vòng lặp Active Learning (AL Loop)**

**Xử lý**:
- Chạy vòng lặp AL từ `loop = start_loop` đến khi kích thước tập labeled đạt `max_L_size` (50% train):
  1. **Điều chỉnh động Contextual Masking**: Set `MASK_ENTITY` và `MASK_CONTEXT` bằng `0.0` cho 3 vòng lặp đầu (vòng 0, 1, 2) để ổn định học ranh giới thực thể, sau đó kích hoạt lại ở các vòng sau.
  2. **Tăng cường dữ liệu (Distant Supervision)**: Dùng 5 từ điển chuyên ngành để nhân bản và thay thế ngẫu nhiên thực thể trong các câu thuộc `current_L_indices`, thu được tập dữ liệu huấn luyện mở rộng `L_aug`.
  3. **Huấn luyện mô hình**: Huấn luyện `MultiTaskBiLSTMCRF` tích hợp LoRA trên `L_aug` với val_loader từ `dev_segmented` sử dụng AdamW, LLRD và Weighted CRF Loss (tính toán loss riêng cho từng query bằng `reduction='none'`, gán trọng số 1.0 -> 5.0 tùy độ hiếm thực thể đối với positive queries, gán 1.0 cho negative queries). Áp dụng Early Stopping (patience=15) và khôi phục checkpoint tốt nhất.
  4. **Đánh giá trên Test**: Suy luận trên tập `test_segmented` bằng mô hình, gộp ranh giới nhãn bằng thuật toán giải quyết xung đột marginals + Viterbi, tính toán F1-score tổng thể và class-wise.
  5. **Tính toán Entropy trên tập chưa gán nhãn U_t**: Dự đoán xác suất biên và tính Shannon Entropy cho các câu thuộc `current_U_indices`.
  6. **Lọc distinct-k**: Gom cụm embeddings và lọc chọn ra 50 câu có entropy cao nhất đồng thời thỏa điều kiện độ tương đồng cosine < `0.85`.
  7. **Simulated Oracle Feedback**: Đo chi phí chỉnh sửa nhãn bằng khoảng cách chỉnh sửa Levenshtein cấp độ token giữa dự đoán của mô hình và nhãn chuẩn, cộng dồn vào chi phí lũy kế `cumulative_edit_distance`.
  8. **Ghi log & Lưu checkpoint**: Lưu mô hình tốt nhất `best_model_AL.pt`, logs nhật ký `logs_al.json`, trạng thái gán nhãn `current_L_indices.json`. Cập nhật `current_L_indices` bằng các câu mới được gán nhãn.

**Đầu ra**:
- `logs_al.json`: nhật ký tiến trình học tập của nhánh AL.
- `best_model_AL.pt`: checkpoint mô hình tối ưu.
- `current_L_indices.json`: file trạng thái index gán nhãn.

---

### 1.13. Cell 12: Nhánh B - Random Sampling Baseline

**Đầu vào**: Tương tự Nhánh A

**Xử lý**: 

Gần giống Nhánh A, chỉ khác ở **Bước chọn mẫu**:

- **Thay vì**: `batch_selected = select_distinct_k(candidates, entropies, cls_embeddings, Config.BATCH_SELECT, 0.85)`
- **Sử dụng**: `batch_selected = random.sample(candidate_indices, min(Config.BATCH_SELECT, len(candidate_indices)))`

**Mục đích**: Tạo baseline để so sánh hiệu quả của Active Learning

**Đầu ra**:
- `logs_random.json`: lịch sử học tập
- `best_model_Random.pt`: mô hình tốt nhất
- `current_L_indices_random.json`: trạng thái hiện tại

---
