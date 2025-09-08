#!/bin/bash

echo "开始训练字符纠错模型..."
cd ./ocr_align/CharCorrectModel

python build_vocab.py
python generate_dataset.py
python train.py

echo "✅ 模型训练完成！"