#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chapter 7: Part 3 - モデル読み込み & 初期テスト
Converted from ch07.ipynb - Model loading and initial testing
"""

import torch
import tiktoken
from gpt_download import download_and_load_gpt2
from previous_chapters import GPTModel, load_weights_into_gpt
from previous_chapters import (
    generate,
    text_to_token_ids,
    token_ids_to_text,
    calc_loss_loader,
    train_model_simple
)

print("=" * 70)
print("Chapter 7: Part 3 - モデル読み込み & 初期テスト")
print("=" * 70)

# 前回のデータを再読み込み（簡単のため）
print("\n前回のデータ再読み込み...")
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

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text

file_path = "instruction-data.json"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"

data = download_and_load_file(file_path, url)

# データセット分割（前回と同じ）
train_portion = int(len(data) * 0.85)  # 85% for training
test_portion = int(len(data) * 0.1)    # 10% for testing
val_portion = len(data) - train_portion - test_portion  # Remaining 5% for validation

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

print(f"データ再読み込み完了: 訓練={len(train_data)}, 検証={len(val_data)}, テスト={len(test_data)}")

# tokenizerセットアップ
tokenizer = tiktoken.get_encoding("gpt2")

# セル75: GPT-2モデル設定 & 読み込み
print("\nセル75: GPT-2モデル設定 & 読み込み")
print("-" * 50)

BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-medium (355M)"

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
print(f"選択モデル: {CHOOSE_MODEL}")
print(f"モデルサイズ: {model_size}")

print("GPT-2モデルダウンロード中...")
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

print("GPTModelインスタンス化...")
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

print(f"✅ GPT-2 {CHOOSE_MODEL} 読み込み完了")
print(f"総パラメータ数: {sum(p.numel() for p in model.parameters()):,}")

# セル77: 初期テスト（ファインチューニング前）
print("\nセル77: 初期テスト（ファインチューニング前）")
print("-" * 50)

torch.manual_seed(123)

input_text = format_input(val_data[0])
print("入力テキスト:")
print(input_text)
print()

# セル78: 生成テスト
print("\nセル78: 生成テスト")
print("-" * 50)

print("テキスト生成中...")
token_ids = generate(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer),
    max_new_tokens=35,
    context_size=BASE_CONFIG["context_length"],
    eos_id=50256,
)
generated_text = token_ids_to_text(token_ids, tokenizer)

# セル80: レスポンス抽出
print("\nセル80: レスポンス抽出")
print("-" * 50)

response_text = generated_text[len(input_text):].strip()
print("生成されたレスポンス:")
print(response_text)
print()

print("分析: モデルは命令に従えていません。ファインチューニングが必要です。")

print("\n✅ Part 3 完了: モデル読み込み & 初期テスト完了")
print("続行するには次のスクリプトを実行してください...") 