# Image-Excel-OCR

本项目旨在对图片中的表格进行内容提取和 EXCEL/CSV 文件重建，同时设计、训练一个小型 LSTM 模型修复直接使用 PaddleOCR 开源大模型输出中少部分数字字符错误的情况，适用于数据类的屏幕截图 EXCEL 表格和较高质量的拍摄 EXCEL 表格照片等场景。

> **特别说明**
>
> 本项目中 `ocr_align.py` 的代码参考自 [PaddleOCR_AlignText](https://github.com/xhw205/PaddleOCR_AlignText) 的开源项目，特此致谢！

---

## 项目结构

```
.
├── imgs  # 示例，使用时替换为你自己的 EXCEL 表格图片
├── LICENSE
├── merged_table.csv  # 示例
├── merged_table(manually_adjusted).xlsx  # 示例
├── ocr_align
│   ├── CharCorrectModel  # 数字字符纠错模型
│   ├── CharCorrectModel_runtime.sh
│   ├── correct.py
│   ├── csv  # 示例
│   ├── csv_corrected  # 示例
│   ├── joint.py
│   ├── json  # 示例
│   ├── ocr_align.py
│   └── reconstruct.py
└── README.md
```

## 环境依赖

- Python >= 3.8
- PyTorch >= 1.10
- PaddleOCR
- numpy, pandas

## 快速开始

### 1. 使用 PaddleOCR 提取文本

将图像放入 `./imgs/` 目录：

```bash
python ocr_align.py
```

OCR 结构化识别结果会保存在 `./ocr_align/json/`。

### 2. 从 OCR 结果重建为结构化 CSV 表格

```bash
python reconstruct.py
```

输出文件保存至：`./ocr_align/csv/`

### 3. 训练字符级数字纠错模型（推荐首次执行）

```bash
bash CharCorrectModel_runtime.sh
```

该脚本将自动：

- 构建 vocab
- 合成训练数据
- 训练并保存最佳模型

本项目附带上传了一个训练好的适合于示例图片中常见识别错误的模型，自行训练时注意根据自己的数据定制 `build_vocab.py` 和 `generate_dataset.py`。

> **附注**
>
> 本项目可能在未来引入通用的纠错模型以适应一般化的场景，敬请期待！

### 4. 对结构化表格进行数字纠错

```bash
python correct.py
```

纠正结果将输出至 `./ocr_align/csv_corrected/`，并打印纠错日志。

### 5. 拼接识别结果（可选）

如果你有多张来自同一 EXCEL 表格的图片（有时候表格会很长），则可以使用我们附加的拼接脚本将多张图片的识别结果拼接为一个表格。

```bash
python joint.py
```

> **注意**
>
> 图片需按顺序命名，推荐命名规则为：`01.jpg`, `02.jpg`, ...，否则拼接可能出现顺序错乱的结果。

## 模型说明

- 使用 BiLSTM + Embedding 实现字符级序列纠错
- 输入为合成的包含 OCR 错误的数字字符串
- 支持任意字符到数字的纠错映射（包含单字符和多字符）
- 通过规则混淆表生成训练样本，具有泛化能力

## 注意事项

- `correct.py` 默认忽略非数字类内容（如中文表头）
- 模型输出与硬规则结合使用，最大化准确率
- 你可以修改 `CharCorrectModel/generate_dataset.py` 添加更多混淆逻辑来增强训练数据

---

如有建议、bug 反馈或合作意向，欢迎联系作者：
highsun910@gmail.com
