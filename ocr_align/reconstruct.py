import json
import csv
import os

def reconstruct_table_from_ocr(json_data, output_csv_path):
    text_boxes = []

    # 适配你的 OCR 结果格式：最外层是一张图的多个检测框列表
    for block in json_data:
        for item in block:
            coords = item[0]
            text = item[1][0]
            confidence = item[1][1]

            x_coords = [point[0] for point in coords]
            y_coords = [point[1] for point in coords]

            center_x = sum(x_coords) / len(x_coords)
            center_y = sum(y_coords) / len(y_coords)

            text_boxes.append({
                'text': text,
                'x': center_x,
                'y': center_y,
                'left': min(x_coords),
                'right': max(x_coords),
                'top': min(y_coords),
                'bottom': max(y_coords),
                'confidence': confidence
            })

    # 按y坐标排序，确定行
    text_boxes.sort(key=lambda x: x['y'])

    # 聚类相似y坐标的文本框为行
    rows = []
    if text_boxes:
        current_row = [text_boxes[0]]
        for i in range(1, len(text_boxes)):
            if abs(text_boxes[i]['y'] - current_row[0]['y']) < 20:
                current_row.append(text_boxes[i])
            else:
                rows.append(current_row)
                current_row = [text_boxes[i]]
        if current_row:
            rows.append(current_row)

    # 对每行按x坐标排序
    for row in rows:
        row.sort(key=lambda x: x['x'])

    # 构建表格
    table_data = []
    for row in rows:
        row_data = [box['text'] for box in row]
        table_data.append(row_data)

    # 补全列数
    max_cols = max(len(row) for row in table_data) if table_data else 0
    for row in table_data:
        while len(row) < max_cols:
            row.append('')

    # 保存为CSV
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(table_data)

    print(f"CSV saved to: {output_csv_path}")
    return table_data

if __name__ == "__main__":
    json_dir = "./ocr_align/json"
    output_folder = "./ocr_align/csv"

    os.makedirs(output_folder, exist_ok=True)

    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

    for json_file in json_files:
        json_file_path = os.path.join(json_dir, json_file)
        
        base_name = os.path.splitext(json_file)[0]
        output_csv_path = os.path.join(output_folder, f"{base_name}_table.csv")
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        table = reconstruct_table_from_ocr(json_data, output_csv_path)
        print(f"[✓] Processed: {json_file} -> {base_name}_table.csv")