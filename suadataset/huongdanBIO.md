Để convert dữ liệu từ định dạng BRAT (`.txt` và `.ann`) sang định dạng BIO (`.txt` theo chuẩn CoNLL) bằng công cụ `Brat2BIO`, bạn có thể thực hiện theo các bước chi tiết dưới đây.

Quy trình này sẽ giúp bạn khôi phục lại đầy đủ các nhãn BIO chuẩn (như `B-Symptom_and_Disease`, `I-Symptom_and_Disease`, `O`...) của tập dữ liệu VietBioNER.

---

## Bước 1: Chuẩn bị môi trường và tải mã nguồn



1. **Cài đặt thư viện cần thiết:**
Công cụ này sử dụng thư viện `nltk` để tách từ/tách câu. Bạn cần cài đặt nó (nếu chưa có):
```bash
pip install nltk

```



---

## Bước 2: Thu thập và cấu trúc thư mục dữ liệu

Bạn cần tải thư mục dữ liệu BRAT từ repository VietBioNER và sắp xếp vào thư mục `Brat2BIO` vừa clone về.

1. Vào repo VietBioNER, tải toàn bộ file trong thư mục `data_brat` (bao gồm các cặp file trùng tên nhau nhưng khác đuôi, ví dụ: `sample1.txt` và `sample1.ann`).
2. Trong thư mục `Brat2BIO` trên máy bạn, hãy tạo một thư mục mới tên là `input_data` và bỏ tất cả các file `.txt` và `.ann` của VietBioNER vào đó.


## Bước 3: Chỉnh sửa Code để phù hợp với Tiếng Việt (Quan trọng)

Mặc định, `Brat2BIO` sử dụng bộ tách từ (`word_tokenize`) của NLTK vốn được tối ưu cho tiếng Anh (phân tách bằng khoảng trắng). Tiếng Việt của chúng ta có các từ ghép chứa khoảng trắng (ví dụ: *"đau đầu"* là 1 từ nhưng có 1 khoảng trắng).

Để dữ liệu không bị lỗi vị trí (offset) khi gán nhãn, bạn cần kiểm tra và điều chỉnh file `brat2bio.py`:

1. Mở file `brat2bio.py` bằng VS Code hoặc một trình soạn thảo bất kỳ.
2. Tìm đến hàm xử lý việc tách từ (thường sử dụng `nltk.word_tokenize` hoặc `split()`).
3. *Lưu ý an toàn:* Vì VietBioNER ở định dạng thô thường đã được tokenize sẵn bằng dấu gạch dưới (ví dụ: `đau_đầu`, `bệnh_nhân`), bạn hãy đảm bảo script chia tách từ theo khoảng trắng đơn thuần bằng phương thức `.split()` thay vì dùng bộ tokenize tiếng Anh của NLTK để tránh làm mất cấu trúc từ ghép tiếng Việt.

---

## Bước 4: Chạy lệnh Convert

Công cụ này chạy qua giao diện dòng lệnh (CLI). Bạn mở terminal tại thư mục `Brat2BIO` và thực hiện lệnh sau:

```bash
python brat2bio.py --input_dir ./input_data --output_dir ./output_data

```

**Giải thích tham số:**

* `--input_dir`: Đường dẫn đến thư mục chứa các file BRAT (`.txt` và `.ann`) của VietBioNER mà bạn đã chuẩn bị ở Bước 2.
* `--output_dir`: Thư mục mà script sẽ tự động tạo ra để lưu các file kết quả sau khi đã chuyển sang định dạng BIO.

---

## Bước 5: Kiểm tra kết quả

Sau khi script chạy xong, bạn vào thư mục `output_data`. Bạn sẽ thấy các file mới có định dạng `.txt` (hoặc `.bio` tùy phiên bản script). Khi mở file lên, dữ liệu sẽ được tổ chức theo dạng mỗi dòng một từ đi kèm nhãn BIO của nó, cách nhau bởi một khoảng trắng hoặc dấu tab:

```text
Bệnh_nhân O
xuất_hiện O
triệu_chứng O
đau_đầu B-Symptom_and_Disease
chóng_mặt I-Symptom_and_Disease
và O
sốt B-Symptom_and_Disease
nhẹ I-Symptom_and_Disease
. O

```

Bây giờ bạn đã có một dataset VietBioNER chuẩn định dạng BIO để đưa vào huấn luyện các mô hình Sequence Labeling như PhoBERT-NER rồi!

Dựa trên cấu trúc thư mục của repo `VietBioNER` được hiển thị trong đoạn mã nguồn bạn cung cấp, **cách sử dụng công cụ `Brat2BIO` hoàn toàn có thể áp dụng được**, nhưng bạn **bắt buộc phải thay đổi phương pháp gom dữ liệu** trước khi chạy lệnh script.

Lý do và các điểm cần chỉnh sửa chi tiết như sau:

### 1. Phân tích cấu trúc `data_brat` của VietBioNER

Trong repo VietBioNER, thư mục `data_brat` không chứa trực tiếp các file `.txt` và `.ann`. Thay vào đó, dữ liệu gán nhãn BRAT được chia nhỏ vào 2 thư mục con theo người gán nhãn (Annotators):

* `data_brat/Annotator_A/` (chứa từ file `1.txt`/`1.ann` đến `28.txt`/`28.ann` cùng các file `dup_x`)


* `data_brat/Annotator_B/` (chứa cấu trúc tương tự để đánh giá độ tương đồng giữa 2 người gán nhãn)



Công cụ `Brat2BIO` yêu cầu một thư mục phẳng (chứa trực tiếp các file `.txt` và `.ann` ở tầng cao nhất của thư mục đầu vào). Nếu bạn chỉ trỏ `--input_dir` vào `data_brat`, script sẽ báo lỗi hoặc không tìm thấy file nào để convert.

---

### 2. Các bước chỉnh sửa phương pháp thực hiện

#### Bước 1: Lựa chọn hoặc gộp Tập dữ liệu (Quan trọng nhất)

Vì có dữ liệu từ 2 người gán nhãn độc lập (A và B), bạn cần quyết định:

* **Nếu muốn huấn luyện trên tập của một người cụ thể (Ví dụ: Người A):** Chỉ copy toàn bộ file trong thư mục `Annotator_A` bỏ vào thư mục `input_data` của `Brat2BIO`.
* **Nếu muốn gộp chung (Không khuyến khích nếu tên file trùng nhau):** Vì tên các file ở hai thư mục giống hệt nhau (`1.txt`, `2.txt`...), nếu copy chung vào một chỗ chúng sẽ ghi đè lên nhau. Bạn nên tạo 2 thư mục riêng là `input_data_A` và `input_data_B` rồi chạy convert 2 lần độc lập.



#### Bước 2: Chỉnh sửa cơ chế tách từ Tiếng Việt (Tokenization)

Khi xem xét các file text thô của VietBioNER (như trong tập `data_supervised_learning` đi kèm), văn bản y tế Tiếng Việt ở đây thường đã được tiền xử lý tokenize dưới dạng **nối các từ ghép bằng dấu gạch dưới** (Ví dụ: `Bệnh_nhân`, `xuất_hiện`, `triệu_chứng_đau_đầu`).

Mặc định file `brat2bio.py` của công cụ `Brat2BIO` gọi hàm tách từ của tiếng Anh:

```python
# Trong file brat2bio.py gốc
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)

```

Hàm này sẽ tách dấu gạch dưới `_` ra khỏi chữ hoặc hiểu sai ranh giới từ tiếng Việt, dẫn đến việc tính toán vị trí ký tự (character offsets) của thực thể từ file `.ann` bị lệch hoàn toàn so với file `.txt`.

**Cách sửa code `brat2bio.py` bắt buộc:**
Bạn hãy mở file `brat2bio.py` lên, tìm đến dòng có sử dụng `word_tokenize` hoặc `nltk` để chia từ, sửa nó thành phương thức chia tách theo khoảng trắng đơn thuần của Python:

```python
# Thay vì dùng word_tokenize(text)
tokens = text.split() 

```

Do dữ liệu gốc VietBioNER đã được phân tách các từ rất rõ ràng bằng khoảng trắng (và các từ ghép dính nhau bằng dấu `_`), việc dùng `.split()` sẽ giữ nguyên cấu trúc `đau_đầu` hay `bệnh_nhân` như mong muốn, đảm bảo nhãn BIO xuất ra khớp 100% với từng từ.

#### Bước 3: Thực hiện chạy lệnh tương ứng

Sau khi đã copy các file từ thư mục `Annotator_A` vào `input_data` và sửa file script, bạn chạy lệnh terminal bình thường:

```bash
python brat2bio.py --input_dir ./input_data --output_dir ./output_data

```

Kết quả trong thư mục `output_data` lúc này sẽ là các file định dạng CoNLL chứa đầy đủ tập nhãn BIO y tế chuẩn của VietBioNER.