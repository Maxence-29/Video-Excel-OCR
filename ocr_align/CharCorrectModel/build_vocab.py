import json

# 所有训练中可能出现的字符（包括数字、混淆字符）
CHARACTER_SET = ['<PAD>', '<UNK>'] + list("0123456789IilSsOoCcBEZ$T!+%-tqgy\"m.1EtS!si卫田tL6L")

# 去重排序
CHARACTER_SET = ['<PAD>', '<UNK>'] + sorted(set(CHARACTER_SET[2:]))

vocab = {ch: idx for idx, ch in enumerate(CHARACTER_SET)}

with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

print(f"[✓] vocab.json saved with {len(vocab)} tokens.")