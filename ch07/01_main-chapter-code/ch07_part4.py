#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chapter 7: Part 4 - ファインチューニング実行
Converted from ch07.ipynb - faithfully following the notebook
"""

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
from functools import partial
import json
import os
import urllib.request
import time
import re
from tqdm import tqdm

# @ch07.ipynb セル75に忠実に従ったimport
from gpt_download import download_and_load_gpt2

# previous_chaptersからのimport
from previous_chapters import (
    GPTModel,
    load_weights_into_gpt,
    calc_loss_loader,
    train_model_simple,
    generate,
    text_to_token_ids,
    token_ids_to_text,
    plot_losses
)

print("=" * 70)
print("Chapter 7: Part 4 - ファインチューニング実行")
print("=" * 70)

# 前回のデータを再読み込み
print("\n前回のデータ再読み込み...")

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

# データ準備
file_path = "instruction-data.json"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"

data = download_and_load_file(file_path, url)

# データセット分割（ch07.ipynbと同じ）
train_portion = int(len(data) * 0.85)  # 85% for training
test_portion = int(len(data) * 0.1)    # 10% for testing
val_portion = len(data) - train_portion - test_portion  # Remaining 5% for validation

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

print(f"データ再読み込み完了: 訓練={len(train_data)}, 検証={len(val_data)}, テスト={len(test_data)}")

# tokenizerセットアップ
tokenizer = tiktoken.get_encoding("gpt2")

# セル75（再実装）: GPT-2モデル設定 
print("\nセル75（再実装）: GPT-2モデル設定")
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

print(f"選択モデル: {CHOOSE_MODEL}")

# @ch07.ipynb セル75に完全に忠実な事前学習済み重み読み込み
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
print(f"モデルサイズ: {model_size}")
print("GPT-2事前学習済み重みをダウンロード中...")

settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

print("GPTModelをインスタンス化...")
model = GPTModel(BASE_CONFIG)
print("事前学習済み重みを読み込み中...")
load_weights_into_gpt(model, params)
model.eval()

print(f"✅ GPT-2 {CHOOSE_MODEL} 事前学習済み重み読み込み完了")
print(f"総パラメータ数: {sum(p.numel() for p in model.parameters()):,}")

# データローダー準備
print("\nデータローダー準備...")
print("-" * 50)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)

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

# セル77: 初期テスト（ファインチューニング前）
print("\nセル77: 初期テスト（ファインチューニング前）")
print("-" * 50)

model.to(device)

torch.manual_seed(123)

input_text = format_input(val_data[0])
print("入力テキスト:")
print(input_text)
print()

token_ids = generate(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer).to(device),
    max_new_tokens=35,
    context_size=BASE_CONFIG["context_length"],
    eos_id=50256,
)
generated_text = token_ids_to_text(token_ids, tokenizer)
response_text = generated_text[len(input_text):].strip()

print("生成されたレスポンス（ファインチューニング前）:")
print(response_text)
print()

# セル86: 初期loss計算
print("\nセル86: 初期loss計算")
print("-" * 50)

model.train()

torch.manual_seed(123)

with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

print("Training loss:", train_loss)
print("Validation loss:", val_loss)

# セル89: ファインチューニング実行
print("\nセル89: ファインチューニング実行")
print("-" * 50)

start_time = time.time()

torch.manual_seed(123)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

num_epochs = 2

print(f"ファインチューニング開始: {num_epochs} epochs")

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context=format_input(val_data[0]), tokenizer=tokenizer
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

# セル91: plot_losses
print("\nセル91: plot_losses")
print("-" * 50)

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

# セル96: テストデータの最初の3つでレスポンス生成テスト
print("\nセル96: テストデータの最初の3つでレスポンス生成テスト")
print("-" * 50)

model.eval()

torch.manual_seed(123)

for entry in test_data[:3]:

    input_text = format_input(entry)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

    print(input_text)
    print(f"\nCorrect response:\n>> {entry['output']}")
    print(f"\nModel response:\n>> {response_text.strip()}")
    print("-------------------------------------")

# セル98: 全テストデータに対してレスポンス生成とJSONファイル保存
print("\nセル98: 全テストデータに対してレスポンス生成とJSONファイル保存")
print("-" * 50)

for i, entry in tqdm(enumerate(test_data), total=len(test_data)):

    input_text = format_input(entry)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

    test_data[i]["model_response"] = response_text

with open("instruction-data-with-response.json", "w") as file:
    json.dump(test_data, file, indent=4)  # "indent" for pretty-printing

print("✅ 全テストデータの応答を 'instruction-data-with-response.json' に保存完了")

# セル100: 保存結果確認
print("\nセル100: 保存結果確認")
print("-" * 50)

print("最初のエントリ確認:")
print(test_data[0])

# セル102: モデル保存
print("\nセル102: モデル保存")
print("-" * 50)

file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-sft.pth"
torch.save(model.state_dict(), file_name)
print(f"Model saved as {file_name}")

print(f"\n# モデル読み込み方法:")
print(f"# model.load_state_dict(torch.load(\"{file_name}\"))")

print("\n✅ Part 4 完了: ファインチューニング & レスポンス生成 & モデル保存完了")
print("ch07.ipynbに忠実に従った完全なスクリプトです！") 
torch.save(model.state_dict(), file_name)
print(f"Model saved as {file_name}")

print(f"\n# モデル読み込み方法:")
print(f"# model.load_state_dict(torch.load(\"{file_name}\"))")

print("\n✅ Part 4 完了: ファインチューニング & レスポンス生成 & モデル保存完了")
print("ch07.ipynbに忠実に従った完全なスクリプトです！") 