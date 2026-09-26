import os
import re
import random
import nltk
from nltk.tokenize import PunktSentenceTokenizer

# Đảm bảo tải đủ tài nguyên NLTK
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Hạt giống ngẫu nhiên để tái lập kết quả phân chia
random.seed(42)

# Thiết lập đường dẫn tương đối để chạy được trên mọi môi trường
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_BRAT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "VietBioNER", "data_brat"))
OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "colab", "dataset", "vietbioner"))

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Khởi tạo Sentence Tokenizer của NLTK với các từ viết tắt tiếng Việt để tránh chia sai câu
sent_tokenizer = PunktSentenceTokenizer()
extra_abbrevs = {
    'tp', 'bn', 'pưtcl', 'afb', 'hiv', 'tb', 'pgs', 'gs', 'ts', 'bs', 'ths', 'dr', 'co', 'tỷ', 'tỳ',
    'bđ', 'đt', 'đ/v', 'ksđ', 'kttđth', 'nv', 'tphcm', 'tphồ', 'tp.hồ', 'đmp', 'dmp', 'xn', 'ct', 
    'tpct', 'tcyttg', 'lpm', 'lptp'
}
for abbrev in extra_abbrevs:
    sent_tokenizer._params.abbrev_types.add(abbrev)

# Regex matching words (bao gồm cả từ ghép nối bằng dấu gạch dưới) hoặc các ký tự đặc biệt/dấu câu
token_pattern = re.compile(r'\w+|[^\w\s]')

def parse_ann_file(ann_path):
    """Đọc tệp .ann của Brat và trích xuất các thực thể gán nhãn."""
    annotations = {}
    if not os.path.exists(ann_path):
        return annotations
    
    with open(ann_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith('T'):
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                ann_id = parts[0] # e.g., T1
                label_info = parts[1].split()
                label = label_info[0] # e.g., Symptom_and_Disease
                
                # Xử lý trường hợp có nhiều span cách nhau bằng dấu chấm phẩy
                span_str = " ".join(label_info[1:])
                spans = []
                for sp in span_str.split(';'):
                    sub_parts = sp.split()
                    if len(sub_parts) == 2:
                        try:
                            spans.append((int(sub_parts[0]), int(sub_parts[1])))
                        except ValueError:
                            pass
                
                annotations[ann_id] = {
                    "label": label,
                    "spans": spans,
                    "text": parts[2]
                }
    # Chỉ giữ các annotation có span hợp lệ (bỏ qua AnnotatorNotes, v.v.)
    annotations = {k: v for k, v in annotations.items() if v["spans"]}
    return annotations

def process_file_pair(txt_path, ann_path):
    """Chuyển đổi một cặp file (.txt và .ann) thành danh sách các câu gán nhãn BIO."""
    annotations = parse_ann_file(ann_path)
    
    # Đọc text theo 2 chế độ newline để chọn chế độ khớp nhất với annotation
    text_lf = ""
    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            text_lf = f.read()
            
    text_raw = ""
    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8', newline='') as f:
            text_raw = f.read()
            
    # Đánh giá xem chế độ nào khớp hơn
    lf_matches = 0
    raw_matches = 0
    for ann_id, ann in annotations.items():
        expected_text = ann["text"]
        
        # Thử trích xuất theo LF
        extracted_lf = " ".join(text_lf[s:e] for s, e in ann["spans"])
        if extracted_lf == expected_text:
            lf_matches += 1
            
        # Thử trích xuất theo Raw
        extracted_raw = " ".join(text_raw[s:e] for s, e in ann["spans"])
        if extracted_raw == expected_text:
            raw_matches += 1
            
    # Chọn văn bản phù hợp nhất
    if lf_matches >= raw_matches:
        text = text_lf
    else:
        text = text_raw
        
    # Cảnh báo nếu có thực thể không khớp hoàn toàn
    for ann_id, ann in annotations.items():
        expected_text = ann["text"]
        extracted_text = " ".join(text[s:e] for s, e in ann["spans"])
        if extracted_text != expected_text:
            print(f"[Warning] Mismatch in {os.path.basename(txt_path)} for {ann_id}: expected {repr(expected_text)}, got {repr(extracted_text)}")
            
    sentence_spans = list(sent_tokenizer.span_tokenize(text))
    processed_sentences = []
    
    for sent_start, sent_end in sentence_spans:
        sent_text = text[sent_start:sent_end]
        if not sent_text.strip():
            continue
            
        # Tìm các annotation nằm hoàn toàn trong câu này
        sent_annotations = []
        for ann_id, ann in annotations.items():
            ann_min = min(s for s, e in ann["spans"])
            ann_max = max(e for s, e in ann["spans"])
            if sent_start <= ann_min and ann_max <= sent_end:
                for s_start, s_end in ann["spans"]:
                    sent_annotations.append({
                        "start": s_start,
                        "end": s_end,
                        "label": ann["label"],
                        "ann_id": ann_id
                    })
                    
        # Sắp xếp các span theo thứ tự tăng dần của offset bắt đầu
        sent_annotations.sort(key=lambda x: x["start"])
        
        sent_conll = []
        last_ann_id = None
        curr_pos = sent_start
        
        for ann in sent_annotations:
            ann_start = ann["start"]
            ann_end = ann["end"]
            ann_label = ann["label"]
            ann_id = ann["ann_id"]
            
            # Xử lý trường hợp chồng lấn ngoài ý muốn do dữ liệu lỗi
            if ann_start < curr_pos:
                if ann_end > curr_pos:
                    ann_start = curr_pos
                else:
                    continue
            
            # Nếu có phần văn bản không phải thực thể trước span hiện tại
            if ann_start > curr_pos:
                non_entity_text = text[curr_pos:ann_start]
                for match in token_pattern.finditer(non_entity_text):
                    t_text = match.group()
                    sent_conll.append((t_text, "O"))
                last_ann_id = None
                
            # Xử lý phần thực thể
            entity_text = text[ann_start:ann_end]
            entity_tokens = list(token_pattern.finditer(entity_text))
            
            for i, match in enumerate(entity_tokens):
                t_text = match.group()
                if i == 0 and last_ann_id != ann_id:
                    tag = f"B-{ann_label}"
                else:
                    tag = f"I-{ann_label}"
                sent_conll.append((t_text, tag))
                last_ann_id = ann_id
                
            curr_pos = ann_end
            
        # Xử lý phần văn bản còn lại sau thực thể cuối cùng của câu
        if sent_end > curr_pos:
            rest_text = text[curr_pos:sent_end]
            for match in token_pattern.finditer(rest_text):
                t_text = match.group()
                sent_conll.append((t_text, "O"))
                
        if sent_conll:
            processed_sentences.append(sent_conll)
            
    return processed_sentences

def post_process_conll_sentence(sent):
    """Chuẩn hóa nhãn và khắc phục lỗi lệch/thiếu nhãn trong câu."""
    mutable_sent = [list(item) for item in sent]
    words_lower = [item[0].lower() for item in mutable_sent]
    n = len(mutable_sent)
    
    rules = [
        # Location
        ( ['tp', '.', 'hồ', 'chí', 'minh'], ['B-Location', 'I-Location', 'I-Location', 'I-Location', 'I-Location'] ),
        ( ['tp', 'hồ', 'chí', 'minh'], ['B-Location', 'I-Location', 'I-Location', 'I-Location'] ),
        ( ['thành', 'phố', 'hồ', 'chí', 'minh'], ['B-Location', 'I-Location', 'I-Location', 'I-Location', 'I-Location'] ),
        ( ['tp', '.', 'hcm'], ['B-Location', 'I-Location', 'I-Location'] ),
        ( ['tp', 'hcm'], ['B-Location', 'I-Location'] ),
        ( ['tphcm'], ['B-Location'] ),
        ( ['tỉnh', 'bà', 'rịa', 'vũng', 'tàu'], ['B-Location', 'I-Location', 'I-Location', 'I-Location', 'I-Location'] ),
        ( ['thành', 'phố', 'vũng', 'tàu'], ['B-Location', 'I-Location', 'I-Location', 'I-Location'] ),
        ( ['thành', 'phó', 'vũng', 'tàu'], ['B-Location', 'I-Location', 'I-Location', 'I-Location'] ),
        ( ['nước', 'ta'], ['B-Location', 'I-Location'] ),
        ( ['cần', 'thơ'], ['B-Location', 'I-Location'] ),
        ( ['bắc', 'giang'], ['B-Location', 'I-Location'] ),
        ( ['phú', 'nhuận'], ['B-Location', 'I-Location'] ),
        ( ['hà', 'nội'], ['B-Location', 'I-Location'] ),
        ( ['hn'], ['B-Location'] ),
        ( ['hải', 'phòng'], ['B-Location', 'I-Location'] ),
        ( ['đà', 'nẵng'], ['B-Location', 'I-Location'] ),
        ( ['long', 'an'], ['B-Location', 'I-Location'] ),
        ( ['vũng', 'tàu'], ['B-Location', 'I-Location'] ),
        ( ['ninh', 'thuận'], ['B-Location', 'I-Location'] ),
        ( ['việt', 'nam'], ['B-Location', 'I-Location'] ),
        ( ['quận', 'gò', 'vấp'], ['B-Location', 'I-Location', 'I-Location'] ),
        ( ['quận', 'gò', 'vắp'], ['B-Location', 'I-Location', 'I-Location'] ),
        ( ['gò', 'vấp'], ['B-Location', 'I-Location'] ),
        ( ['gò', 'vắp'], ['B-Location', 'I-Location'] ),
        
        # Organisation
        ( ['bệnh', 'viện', 'lao', 'và', 'bệnh', 'phổi', 'tỉnh', 'bắc', 'giang'], ['B-Organisation', 'I-Organisation', 'I-Organisation', 'I-Organisation', 'I-Organisation', 'I-Organisation', 'B-Location', 'I-Location', 'I-Location'] ),
        ( ['bệnh', 'viện', 'lao', 'và', 'bệnh', 'phỗi', 'tỉnh', 'bắc', 'giang'], ['B-Organisation', 'I-Organisation', 'I-Organisation', 'I-Organisation', 'I-Organisation', 'I-Organisation', 'B-Location', 'I-Location', 'I-Location'] ),
        ( ['bệnh', 'viện', 'lao', 'và', 'bệnh', 'phổi', 'bắc', 'giang'], ['B-Organisation', 'I-Organisation', 'I-Organisation', 'I-Organisation', 'I-Organisation', 'I-Organisation', 'B-Location', 'I-Location'] ),
        ( ['bệnh', 'viện', 'lao', 'và', 'bệnh', 'phổi'], ['B-Organisation', 'I-Organisation', 'I-Organisation', 'I-Organisation', 'I-Organisation', 'I-Organisation'] ),
        ( ['bệnh', 'viện', 'lao', 'và', 'bệnh', 'phỗi'], ['B-Organisation', 'I-Organisation', 'I-Organisation', 'I-Organisation', 'I-Organisation', 'I-Organisation'] ),
        ( ['viện', 'lao', 'và', 'bệnh', 'phổi'], ['B-Organisation', 'I-Organisation', 'I-Organisation', 'I-Organisation', 'I-Organisation'] ),
        ( ['viện', 'lao', 'và', 'bệnh', 'phỗi'], ['B-Organisation', 'I-Organisation', 'I-Organisation', 'I-Organisation', 'I-Organisation'] ),
        ( ['bệnh', 'viện', 'phạm', 'ngọc', 'thạch'], ['B-Organisation', 'I-Organisation', 'I-Organisation', 'I-Organisation', 'I-Organisation'] ),
        ( ['bv', 'phạm', 'ngọc', 'thạch'], ['B-Organisation', 'I-Organisation', 'I-Organisation', 'I-Organisation'] ),
        ( ['bệnh', 'viện', 'bạch', 'mai'], ['B-Organisation', 'I-Organisation', 'I-Organisation', 'I-Organisation'] ),
        ( ['bv', 'bạch', 'mai'], ['B-Organisation', 'I-Organisation', 'I-Organisation'] ),
        ( ['bệnh', 'viện', 'chợ', 'rẫy'], ['B-Organisation', 'I-Organisation', 'I-Organisation', 'I-Organisation'] ),
        ( ['bv', 'chợ', 'rẫy'], ['B-Organisation', 'I-Organisation', 'I-Organisation'] ),
        ( ['tổ', 'chức', 'y', 'tế', 'thế', 'giới'], ['B-Organisation', 'I-Organisation', 'I-Organisation', 'I-Organisation', 'I-Organisation', 'I-Organisation'] ),
        ( ['chương', 'trình', 'chống', 'lao', 'quốc', 'gia'], ['B-Organisation', 'I-Organisation', 'I-Organisation', 'I-Organisation', 'I-Organisation', 'I-Organisation'] ),
        ( ['bộ', 'y', 'tế'], ['B-Organisation', 'I-Organisation', 'I-Organisation'] ),
        ( ['phòng', 'khám', 'ngoại', 'trú'], ['B-Organisation', 'I-Organisation', 'I-Organisation', 'I-Organisation'] ),
        ( ['viện', 'pasteur'], ['B-Organisation', 'I-Organisation'] ),
        ( ['who'], ['B-Organisation'] ),
        ( ['tcyttg'], ['B-Organisation'] ),
        ( ['ctclqg'], ['B-Organisation'] ),
        ( ['cdc'], ['B-Organisation'] ),
        
        # DiagnosticProcedure
        ( ['nhuộm', 'lam', 'tìm', 'afb'], ['B-DiagnosticProcedure', 'I-DiagnosticProcedure', 'I-DiagnosticProcedure', 'I-DiagnosticProcedure'] ),
        ( ['cấy', 'đờm', 'tìm', 'afb'], ['B-DiagnosticProcedure', 'I-DiagnosticProcedure', 'I-DiagnosticProcedure', 'I-DiagnosticProcedure'] ),
        ( ['nhuộm', 'soi', 'afb'], ['B-DiagnosticProcedure', 'I-DiagnosticProcedure', 'I-DiagnosticProcedure'] ),
        ( ['xét', 'nghiệm', 'afb'], ['B-DiagnosticProcedure', 'I-DiagnosticProcedure', 'I-DiagnosticProcedure'] ),
        ( ['soi', 'afb'], ['B-DiagnosticProcedure', 'I-DiagnosticProcedure'] ),
        ( ['ziehl', '-', 'neelsen'], ['B-DiagnosticProcedure', 'I-DiagnosticProcedure', 'I-DiagnosticProcedure'] ),
        ( ['x', '-', 'quang'], ['B-DiagnosticProcedure', 'I-DiagnosticProcedure', 'I-DiagnosticProcedure'] ),
        ( ['ziehl-neelsen'], ['B-DiagnosticProcedure'] ),
        ( ['x-quang'], ['B-DiagnosticProcedure'] ),
        ( ['xq'], ['B-DiagnosticProcedure'] ),
        ( ['clvt'], ['B-DiagnosticProcedure'] ),
        ( ['khv'], ['B-DiagnosticProcedure'] ),

        # General corrections
        ( ['kiểm', 'soát', 'lao'], ['O', 'O', 'B-Symptom_and_Disease'] ),
    ]
    
    # 1. Áp dụng quy tắc chuỗi cố định
    for pattern, tags in rules:
        p_len = len(pattern)
        i = 0
        while i <= n - p_len:
            if words_lower[i : i + p_len] == pattern:
                for offset in range(p_len):
                    mutable_sent[i + offset][1] = tags[offset]
                i += p_len
            else:
                i += 1
                
    # 2. Áp dụng quy tắc DateTime động
    i = 0
    while i < n:
        words_slice = words_lower[i : min(i + 15, n)]
        
        # Kiểm tra 'từ' <digits> '/' <digits> 'đến' <digits> '/' <digits>
        if len(words_slice) >= 8 and words_slice[0] == 'từ' and words_slice[1].isdigit() and words_slice[2] == '/' and words_slice[3].isdigit() and words_slice[4] == 'đến' and words_slice[5].isdigit() and words_slice[6] == '/' and words_slice[7].isdigit():
            tags = ['B-DateTime'] + ['I-DateTime'] * 7
            for offset in range(8):
                mutable_sent[i + offset][1] = tags[offset]
            i += 8
            continue
            
        # Kiểm tra 'tháng' <digits> '/' <digits>
        if len(words_slice) >= 4 and words_slice[0] == 'tháng' and words_slice[1].isdigit() and words_slice[2] == '/' and words_slice[3].isdigit():
            tags = ['B-DateTime'] + ['I-DateTime'] * 3
            for offset in range(4):
                mutable_sent[i + offset][1] = tags[offset]
            i += 4
            continue
            
        # Kiểm tra 'năm' <digits> '-' <digits>
        if len(words_slice) >= 4 and words_slice[0] == 'năm' and words_slice[1].isdigit() and words_slice[2] == '-' and words_slice[3].isdigit():
            tags = ['B-DateTime'] + ['I-DateTime'] * 3
            for offset in range(4):
                mutable_sent[i + offset][1] = tags[offset]
            i += 4
            continue
            
        # Kiểm tra 'năm' <digits>-<digits> e.g. 'năm' '2000-2005'
        if len(words_slice) >= 2 and words_slice[0] == 'năm':
            parts = words_slice[1].split('-')
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                tags = ['B-DateTime', 'I-DateTime']
                for offset in range(2):
                    mutable_sent[i + offset][1] = tags[offset]
                i += 2
                continue
                
        # Kiểm tra 'năm' <digits> <digits> (ví dụ split year: năm 201 1)
        if len(words_slice) >= 3 and words_slice[0] == 'năm' and words_slice[1].isdigit() and words_slice[2].isdigit():
            combined_digits = words_slice[1] + words_slice[2]
            if len(combined_digits) == 4:
                tags = ['B-DateTime', 'I-DateTime', 'I-DateTime']
                for offset in range(3):
                    mutable_sent[i + offset][1] = tags[offset]
                i += 3
                continue
                
        # Kiểm tra 'năm' <digits>
        if len(words_slice) >= 2 and words_slice[0] == 'năm' and words_slice[1].isdigit() and len(words_slice[1]) in [2, 4]:
            tags = ['B-DateTime', 'I-DateTime']
            for offset in range(2):
                mutable_sent[i + offset][1] = tags[offset]
            i += 2
            continue
            
        # Kiểm tra 'tháng' <digits>
        if len(words_slice) >= 2 and words_slice[0] == 'tháng' and words_slice[1].isdigit() and int(words_slice[1]) >= 1 and int(words_slice[1]) <= 12:
            tags = ['B-DateTime', 'I-DateTime']
            for offset in range(2):
                mutable_sent[i + offset][1] = tags[offset]
            i += 2
            continue
            
        # Kiểm tra 'ngày' <digits>
        if len(words_slice) >= 2 and words_slice[0] == 'ngày' and words_slice[1].isdigit() and int(words_slice[1]) >= 1 and int(words_slice[1]) <= 31:
            tags = ['B-DateTime', 'I-DateTime']
            for offset in range(2):
                mutable_sent[i + offset][1] = tags[offset]
            i += 2
            continue
            
        i += 1
        
    # 3. Gộp các thực thể cùng loại đứng cạnh nhau liên tiếp
    merge_consecutive_entities(mutable_sent)
        
    return [tuple(item) for item in mutable_sent]

def merge_consecutive_entities(mutable_sent):
    """Gộp các thực thể cùng loại đứng cạnh nhau liên tiếp."""
    n = len(mutable_sent)
    for i in range(1, n):
        tag_curr = mutable_sent[i][1]
        tag_prev = mutable_sent[i-1][1]
        if tag_curr.startswith("B-"):
            entity_type = tag_curr[2:]
            if tag_prev == f"B-{entity_type}" or tag_prev == f"I-{entity_type}":
                mutable_sent[i][1] = f"I-{entity_type}"

def main():
    all_sentences = []
    
    # 1. Thu thập dữ liệu từ Annotator A (chứa regular và duplicates)
    dir_a = os.path.join(DATA_BRAT_DIR, "Annotator_A")
    print(f"Processing directory: {dir_a}")
    for file in sorted(os.listdir(dir_a)):
        if file.endswith(".txt"):
            txt_path = os.path.join(dir_a, file)
            ann_path = os.path.join(dir_a, file.replace(".txt", ".ann"))
            sents = process_file_pair(txt_path, ann_path)
            all_sentences.extend(sents)
            
    # 2. Thu thập dữ liệu từ Annotator B (chỉ lấy regular, bỏ qua duplicates để tránh trùng lặp tài liệu)
    dir_b = os.path.join(DATA_BRAT_DIR, "Annotator_B")
    print(f"Processing directory: {dir_b}")
    for file in sorted(os.listdir(dir_b)):
        if file.endswith(".txt") and not file.startswith("dup_"):
            txt_path = os.path.join(dir_b, file)
            ann_path = os.path.join(dir_b, file.replace(".txt", ".ann"))
            sents = process_file_pair(txt_path, ann_path)
            all_sentences.extend(sents)

    print(f"Total sentences collected: {len(all_sentences)}")

    # Xáo trộn ngẫu nhiên dữ liệu để phân phối nhãn đều
    random.shuffle(all_sentences)

    # Phân chia tỷ lệ 80% / 10% / 10%
    total_sents = len(all_sentences)
    train_end = int(total_sents * 0.8)
    dev_end = train_end + int(total_sents * 0.1)

    splits = {
        "train.txt": all_sentences[:train_end],
        "dev.txt": all_sentences[train_end:dev_end],
        "test.txt": all_sentences[dev_end:]
    }

    # Ghi file theo định dạng CoNLL sau khi đã áp dụng post-processing
    for file_name, sents in splits.items():
        output_path = os.path.join(OUTPUT_DIR, file_name)
        with open(output_path, 'w', encoding='utf-8') as out_f:
            for sent in sents:
                processed_sent = post_process_conll_sentence(sent)
                for token, tag in processed_sent:
                    out_f.write(f"{token} {tag}\n")
                out_f.write("\n")
        print(f"Dataset split written: {output_path} ({len(sents)} sentences)")

if __name__ == "__main__":
    main()

