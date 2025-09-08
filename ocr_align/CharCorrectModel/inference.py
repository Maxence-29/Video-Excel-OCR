import torch
import json
import os
from model import CharCorrector
from dataset import OCRCorrectionDataset

# 获取当前文件所在目录
BASE_DIR = os.path.dirname(__file__)
VOCAB_PATH = os.path.join(BASE_DIR, "vocab.json")
CKPT_PATH = os.path.join(BASE_DIR, "checkpoints", "digit_corrector.pt")

# 加载 vocab
with open(VOCAB_PATH, encoding="utf-8") as f:
    vocab = json.load(f)

id2char = {v: k for k, v in vocab.items()}
pad_id = vocab["<PAD>"]
unk_id = vocab["<UNK>"]

# 加载模型
def load_model():
    model = CharCorrector(len(vocab))
    try:
        model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
    except FileNotFoundError:
        raise RuntimeError(f"[!] 模型文件未找到: {CKPT_PATH}")
    model.eval()
    return model

model = load_model()

# 推理函数
def apply_digit_correction(text, max_len=16):
    ids = [vocab.get(ch, unk_id) for ch in text]
    input_len = len(ids)
    if len(ids) < max_len:
        ids += [pad_id] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
        input_len = max_len

    input_tensor = torch.tensor([ids])
    with torch.no_grad():
        output = model(input_tensor)
        pred_ids = output.argmax(dim=-1).squeeze(0)[:input_len]
    corrected = ''.join(id2char[i.item()] for i in pred_ids if i.item() != pad_id)
    return corrected

# 测试入口
if __name__ == "__main__":
    print("请输入需要纠错的文本（输入 'exit' 退出）:")
    while True:
        wrong_text = input(">>> ").strip()
        if wrong_text.lower() == 'exit':
            break
        if wrong_text:
            try:
                corrected = apply_digit_correction(wrong_text)
                print(f"{wrong_text} → {corrected}")
            except Exception as e:
                print(f"[!] 纠错失败: {e}")
        else:
            print("请输入有效文本。")