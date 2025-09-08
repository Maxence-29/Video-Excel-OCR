import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'CharCorrectModel'))

import csv
import re
from CharCorrectModel.inference import apply_digit_correction as model_correct

# 规则混淆映射表
confusion_map = {
    # 单字符映射
    'I': '1', 'l': '1', 'i': '1', '!': '1',
    'S': '5', 's': '5', '$': '4', '-': '4', '+': '4',
    'C': '0', 'O': '0', 'o': '0', '%': '0',
    'B': '8',
    'E': '3', 't': '3',
    'Z': '2',
    'T': '7',
    'G': '6', 'g': '9', 'q': '9', 'y': '9',
    
    # 多字符映射
    '.1': '4',
    'si': '1',
    'tL': '74',
    'Et': '43',
    'S!': '81',
    '9.1': '94',
    '7.4': '269',
    'SI': '83',
    '6L': '79',
    
    # 特殊字符映射
    '"': '111',
    '卫': '72',
    '田': '69',
    'm': '77',
}

# 规则法
def rule_correct(text):
    return ''.join(confusion_map.get(ch, ch) for ch in text)

# 融合纠错方法
def apply_digit_correction(text):
    # 非数字类内容跳过
    if not re.search(r'\d', text) and not any(c.isalpha() for c in text):
        return text

    try:
        corrected = model_correct(text)
        if not corrected.strip() or not any(c.isdigit() for c in corrected):
            raise ValueError("模型输出异常")
        if text != corrected:
            print(f"模型纠错: {text} → {corrected}")
        return corrected
    except:
        fallback = rule_correct(text)
        if text != fallback:
            print(f"模型失败，规则纠错: {text} -> {fallback}")
        return fallback

def is_potential_digit_field(text):
    # 排除汉字
    if re.search(r'[\u4e00-\u9fff]', text):
        return False
    return bool(re.search(r'[\dIilSsoOCEBZT$]', text))

# 修正 CSV 表格中的每个单元格
def correct_csv_digits(input_csv_path, output_csv_path):
    corrected_rows = []

    with open(input_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            corrected_row = [
                apply_digit_correction(cell) if is_potential_digit_field(cell) else cell
                for cell in row
            ]
            corrected_rows.append(corrected_row)

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(corrected_rows)

    print(f"[✓] Corrected CSV saved to: {output_csv_path}\n")

if __name__ == "__main__":
    input_dir = './ocr_align/csv'
    output_dir = './ocr_align/csv_corrected'
    os.makedirs(output_dir, exist_ok=True)

    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    for csv_file in csv_files:
        input_path = os.path.join(input_dir, csv_file)
        output_path = os.path.join(output_dir, csv_file)
        correct_csv_digits(input_path, output_path)