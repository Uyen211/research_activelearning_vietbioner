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

# Thiết lập đường dẫn
DATA_BRAT_DIR = r"d:\Study\TLU\Nlp\active-learning-VietBioNER\suadataset\VietBioNER\data_brat"
OUTPUT_DIR = r"d:\Study\TLU\Nlp\active-learning-VietBioNER\colab\dataset\vietbioner"

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
                        spans.append((int(sub_parts[0]), int(sub_parts[1])))
                
                annotations[ann_id] = {
                    "label": label,
                    "spans": spans,
                    "text": parts[2]
                }
    return annotations

def get_token_entity_id(tok_start, tok_end, annotations):
    """Tìm ID thực thể chứa token hiện tại."""
    for ann_id, ann in annotations.items():
        for span_start, span_end in ann["spans"]:
            # Token overlap với span của thực thể
            if tok_start < span_end and tok_end > span_start:
                return ann_id, ann["label"]
    return None, "O"

def process_file_pair(txt_path, ann_path):
    """Chuyển đổi một cặp file (.txt và .ann) thành danh sách các câu gán nhãn BIO."""
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    annotations = parse_ann_file(ann_path)
    sentence_spans = list(sent_tokenizer.span_tokenize(text))
    
    processed_sentences = []
    
    for sent_start, sent_end in sentence_spans:
        sent_text = text[sent_start:sent_end]
        if not sent_text.strip():
            continue
        
        # Tokenize sentence sang các từ/dấu câu
        tokens_in_sent = []
        for match in token_pattern.finditer(sent_text):
            t_start = sent_start + match.start()
            t_end = sent_start + match.end()
            t_text = match.group()
            tokens_in_sent.append((t_start, t_end, t_text))
            
        if not tokens_in_sent:
            continue
            
        sent_conll = []
        last_ann_id = None
        
        for t_start, t_end, t_text in tokens_in_sent:
            ann_id, label = get_token_entity_id(t_start, t_end, annotations)
            if label == "O":
                tag = "O"
                last_ann_id = None
            else:
                if last_ann_id == ann_id:
                    tag = f"I-{label}"
                else:
                    tag = f"B-{label}"
                    last_ann_id = ann_id
            sent_conll.append((t_text, tag))
        
        processed_sentences.append(sent_conll)
        
    return processed_sentences

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

    # Ghi file theo định dạng CoNLL
    for file_name, sents in splits.items():
        output_path = os.path.join(OUTPUT_DIR, file_name)
        with open(output_path, 'w', encoding='utf-8') as out_f:
            for sent in sents:
                for token, tag in sent:
                    out_f.write(f"{token} {tag}\n")
                out_f.write("\n")
        print(f"Dataset split written: {output_path} ({len(sents)} sentences)")

if __name__ == "__main__":
    main()
