import random
import os

# 错误字符（错误识别） -> 实际数字（多个可能）
reverse_confusion_map = {
    'I': ['1'], 'l': ['1'], 'i': ['1'], '!': ['1'], 'si': ['1'], '"': ['1'], 'S!': ['1'],
    'S': ['5'], 's': ['5'],
    'B': ['8'],
    'E': ['3'], 't': ['3'], 'Et': ['3'],
    '$': ['4', '5'], '+': ['4'], '-': ['4'], '.1': ['4', '1'],
    'O': ['0'], 'C': ['0'], 'o': ['0'], '%': ['0'],
    'T': ['7'], 'm': ['7'], 'tL': ['7'],
    'Z': ['2'], '卫': ['2'],
    'G': ['6'], '田': ['6'],
    'g': ['9'], 'q': ['9'], 'y': ['9'], '6L': ['9'], '9.1': ['9'],
}

# 自动构建 正向 confusion_map: 真实数字 -> 可替代字符
forward_confusion_map = {}
for wrong_char, true_digits in reverse_confusion_map.items():
    for digit in true_digits:
        forward_confusion_map.setdefault(digit, []).append(wrong_char)

# 只考虑阿拉伯数字
DIGITS = "0123456789"
LENGTH_RANGE = (3, 8)

def corrupt(text, noise_prob=0.3):
    # 将真实文本随机注入混淆字符，构建训练样本
    noisy = ""
    for ch in text:
        if ch in forward_confusion_map and random.random() < noise_prob:
            noisy += random.choice(forward_confusion_map[ch])
        else:
            noisy += ch
    return noisy

# 生成多个 noise_prob 下的训练样本
def generate_samples(n=30000, path="data/train.txt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for _ in range(n):
            clean = ''.join(random.choices(DIGITS, k=random.randint(*LENGTH_RANGE)))
            prob = random.choice([0.2, 0.3, 0.4])  # 多种 noise_prob
            noisy = corrupt(clean, noise_prob=prob)
            f.write(f"{noisy}\t{clean}\n")
    print(f"[✓] Saved {n} samples to {path}")

if __name__ == "__main__":
    generate_samples(24000, "data/train.txt")
    generate_samples(6000, "data/val.txt")