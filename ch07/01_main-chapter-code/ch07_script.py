#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chapter 7: Finetuning To Follow Instructions
Converted from ch07.ipynb - executing all code cells in sequence
"""

print("=" * 70)
print("Chapter 7: Finetuning To Follow Instructions")
print("=" * 70)

# セル2: ライブラリバージョン確認
print("\nセル2: ライブラリバージョン確認")
print("-" * 50)

from importlib.metadata import version

pkgs = [
    "matplotlib",  # Plotting library
    "tiktoken",    # Tokenizer
    "torch",       # Deep learning library
    "tqdm",        # Progress bar
    "tensorflow",  # For OpenAI's pretrained weights
]
for p in pkgs:
    try:
        print(f"{p} version: {version(p)}")
    except Exception as e:
        print(f"{p}: not installed or not found - {e}")

# セル10: データセットダウンロードと読み込み
print("\nセル10: データセットダウンロードと読み込み")
print("-" * 50)

import json
import os
import urllib.request

def download_and_load_file(file_path, url):

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    with open(file_path, "r") as file:
        data = json.load(file)

    return data

file_path = "instruction-data.json"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"

data = download_and_load_file(file_path, url)
print("Number of entries:", len(data))

# セル12: データ例確認
print("\nセル12: データ例確認")
print("-" * 50)
print("Example entry:\n", data[50])

# セル14: 入力フィールドなしの例
print("\nセル14: 入力フィールドなしの例")
print("-" * 50)
print("Another example entry:\n", data[999])

# セル18: format_input関数定義
print("\nセル18: format_input関数定義")
print("-" * 50)

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text

print("format_input関数定義完了")

# セル20: フォーマットテスト（入力フィールドあり）
print("\nセル20: フォーマットテスト（入力フィールドあり）")
print("-" * 50)

model_input = format_input(data[50])
desired_response = f"\n\n### Response:\n{data[50]['output']}"

print(model_input + desired_response)

# セル22: フォーマットテスト（入力フィールドなし）
print("\nセル22: フォーマットテスト（入力フィールドなし）")
print("-" * 50)

model_input = format_input(data[999])
desired_response = f"\n\n### Response:\n{data[999]['output']}"

print(model_input + desired_response)

# セル24: データセット分割
print("\nセル24: データセット分割")
print("-" * 50)

train_portion = int(len(data) * 0.85)  # 85% for training
test_portion = int(len(data) * 0.1)    # 10% for testing
val_portion = len(data) - train_portion - test_portion  # Remaining 5% for validation

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

# セル25: データセット分割結果確認
print("\nセル25: データセット分割結果確認")
print("-" * 50)

print("Training set length:", len(train_data))
print("Validation set length:", len(val_data))
print("Test set length:", len(test_data))

print("\n✅ Part 1 完了: データ準備完了")
print("続行するには次のスクリプトを実行してください...") 