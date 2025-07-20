#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chapter 7: Part 2 - Dataset & Model準備
Converted from ch07.ipynb - continuing from part 1
"""

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
from functools import partial

print("=" * 70)
print("Chapter 7: Part 2 - Dataset & Model準備")
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

# セル30: InstructionDatasetクラス定義
print("\nセル30: InstructionDatasetクラス定義")
print("-" * 50)

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

print("InstructionDatasetクラス定義完了")

# セル32: tokenizerセットアップ
print("\nセル32: tokenizerセットアップ")
print("-" * 50)

tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

# セル45: custom_collate_fn関数定義
print("\nセル45: custom_collate_fn関数定義")
print("-" * 50)

def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

        # New: Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # New: Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor

print("custom_collate_fn関数定義完了")

# セル46: collate関数テスト
print("\nセル46: collate関数テスト")
print("-" * 50)

# 簡単なテスト用バッチ
inputs_1 = [0, 1, 2, 3, 4]
inputs_2 = [5, 6]
inputs_3 = [7, 8, 9]

batch = (
    inputs_1,
    inputs_2,
    inputs_3
)

inputs, targets = custom_collate_fn(batch)
print("Inputs:")
print(inputs)
print("Targets:")
print(targets)

# セル60: デバイス設定
print("\nセル60: デバイス設定")
print("-" * 50)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# セル61: カスタマイズされたcollate関数作成
print("\nセル61: カスタマイズされたcollate関数作成")
print("-" * 50)

customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)
print("customized_collate_fn作成完了")

# セル63: trainデータローダー作成
print("\nセル63: trainデータローダー作成")
print("-" * 50)

num_workers = 0
batch_size = 8

torch.manual_seed(123)

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

print(f"trainデータローダー作成完了: {len(train_loader)} batches")

# セル64: val/testデータローダー作成
print("\nセル64: val/testデータローダー作成")
print("-" * 50)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

print(f"データローダー作成完了:")
print(f"- train: {len(train_loader)} batches")
print(f"- val: {len(val_loader)} batches") 
print(f"- test: {len(test_loader)} batches")

# セル66: データローダー形状確認
print("\nセル66: データローダー形状確認（最初の5バッチ）")
print("-" * 50)

print("Train loader:")
for i, (inputs, targets) in enumerate(train_loader):
    print(f"Batch {i+1}: inputs.shape={inputs.shape}, targets.shape={targets.shape}")
    if i >= 4:  # 最初の5バッチのみ表示
        break

print("\n✅ Part 2 完了: Dataset & DataLoader準備完了")
print("続行するには次のスクリプトを実行してください...") 