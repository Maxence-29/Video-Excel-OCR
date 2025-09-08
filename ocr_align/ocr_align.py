from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
from interval import Interval
import json
import os

det_model_dir = "./inference/ch_ppocr_server_v2.0_det_infer/"
rec_model_dir = "./inference/ch_ppocr_server_v2.0_rec_infer/"
cls_model_dir = "./inference/ch_ppocr_mobile_v2.0_cls_infer/"

class OCR(PaddleOCR):
    def __init__(self, **kwargs):
        super(OCR, self).__init__(**kwargs)

    def ocr_new(self, img, det=True, rec=True, cls=True):
        res = self.ocr(img, det=det, rec=rec, cls=cls)
        if res!=None:
            return res
        else:
            img_v = Image.open(img).convert("RGB")
            img_v = np.asanyarray(img_v)[:,:,[2,1,0]]
            dt_boxes, rec_res = self.__call__(img_v)
            return [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]

def align_text(res):
    res.sort(key=lambda i: (i[0][0][0]))  # 按照x排
    already_IN, line_list = [], []
    for i in range(len(res)):  # i当前
        if res[i][0][0] in already_IN:
            continue
        line_txt = res[i][1][0]
        already_IN.append(res[i][0][0])
        # Check if we have enough points in the bounding box
        if len(res[i][0]) < 4:
            continue
        y_i_points = [res[i][0][0][1], res[i][0][1][1], res[i][0][3][1], res[i][0][2][1]]
        min_I_y, max_I_y = min(y_i_points), max(y_i_points)
        for j in range(i + 1, len(res)):  # j下一个
            if res[j][0][0] in already_IN:
                continue
            # Check if we have enough points in the bounding box
            if len(res[j][0]) < 4:
                continue
            y_j_points = [res[j][0][0][1], res[j][0][1][1], res[j][0][3][1], res[j][0][2][1]]
            min_J_y, max_J_y = min(y_j_points), max(y_j_points)
            y_j_points = [res[j][0][0][1], res[j][0][1][1], res[j][0][3][1], res[j][0][2][1]]
            min_J_y, max_J_y = min(y_j_points), max(y_j_points)
            next_j = Interval(min_J_y, max_J_y - (max_J_y - min_J_y) // 3)

            if next_j.overlaps(curr) and curr_mid in Interval(min_J_y, max_J_y):
                line_txt += (res[j][1][0] + "  ")
                already_IN.append(res[j][0][0])
                curr = Interval(min_J_y + (max_J_y - min_J_y) // 3, max_J_y)
                curr_mid = min_J_y + (max_J_y - min_J_y) // 2
        line_list.append((res[i][0][0][1], line_txt))
    line_list.sort(key=lambda x: x[0])
    txt = '\n'.join([i[1] for i in line_list])
    return txt

if __name__ == "__main__":
    img_folder = './imgs'
    output_folder = './ocr_align/json'

    os.makedirs(output_folder, exist_ok=True)

    paddle_ocr_engin = OCR(det_model_dir=det_model_dir, rec_model_dir=rec_model_dir, cls_model_dir=cls_model_dir, use_angle_cls=True)

    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    img_files = [f for f in os.listdir(img_folder) if any(f.lower().endswith(ext) for ext in img_extensions)]

    for img_file in img_files:
        img_path = os.path.join(img_folder, img_file)

        res = paddle_ocr_engin.ocr_new(img_path)

        output_file = os.path.join(output_folder, f"{os.path.splitext(img_file)[0]}_ocr_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        
        print(f"[✓] Processed {img_file}: {len(res[0]) if res and res[0] else 0} OCR results saved to {output_file}")