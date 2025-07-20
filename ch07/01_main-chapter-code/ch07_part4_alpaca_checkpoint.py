#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chapter 7: Part 4 - Alpacaデータセットファインチューニング（チェックポイント保存機能付き）
Converted from ch07.ipynb - faithfully following the notebook with checkpoint saving
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
import glob
import subprocess
from datetime import datetime

# @ch07.ipynb セル75に忠実に従ったimport
from gpt_download import download_and_load_gpt2

# previous_chaptersからのimport
from previous_chapters import (
    GPTModel,
    load_weights_into_gpt,
    calc_loss_loader,
    generate,
    text_to_token_ids,
    token_ids_to_text,
    plot_losses
)

print("=" * 70)
print("Chapter 7: Part 4 - Alpacaデータセット ファインチューニング（チェックポイント保存付き）")
print("=" * 70)

def run_git_command(command, cwd=None):
    """Git コマンドを実行する安全な関数"""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            cwd=cwd,
            capture_output=True, 
            text=True, 
            timeout=30
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            print(f"Git command failed: {command}")
            print(f"Error: {result.stderr}")
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print(f"Git command timed out: {command}")
        return False, "Timeout"
    except Exception as e:
        print(f"Git command exception: {e}")
        return False, str(e)

def check_git_repo():
    """Gitリポジトリかどうかチェックする"""
    success, _ = run_git_command("git rev-parse --git-dir")
    return success

def git_add_commit_push(files_to_add, commit_message, branch="main"):
    """ファイルをadd, commit, pushする"""
    if not check_git_repo():
        print("⚠️  Gitリポジトリではありません。GitHub保存をスキップします。")
        return False
    
    try:
        # ファイルをadd
        for file_path in files_to_add:
            if os.path.exists(file_path):
                success, _ = run_git_command(f"git add {file_path}")
                if not success:
                    print(f"⚠️  git add failed for {file_path}")
                    return False
        
        # 変更があるかチェック
        success, status = run_git_command("git status --porcelain")
        if not success or not status.strip():
            print("📝 変更がありません。pushをスキップします。")
            return True
        
        # commit
        success, _ = run_git_command(f'git commit -m "{commit_message}"')
        if not success:
            print("⚠️  git commit failed")
            return False
        
        # push
        success, output = run_git_command(f"git push origin {branch}")
        if success:
            print(f"✅ GitHub に正常にpushしました: {commit_message}")
            return True
        else:
            print(f"⚠️  git push failed: {output}")
            return False
            
    except Exception as e:
        print(f"⚠️  Git操作でエラー: {e}")
        return False

def auto_save_to_github(checkpoint_path, step, model_name="gpt2-medium"):
    """チェックポイントをGitHubに自動保存"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    files_to_add = [
        checkpoint_path,
        "training_losses.json",  # もし存在すれば
    ]
    
    # 存在するファイルのみを対象にする
    existing_files = [f for f in files_to_add if os.path.exists(f)]
    
    commit_message = f"Checkpoint {step}: {model_name} fine-tuning [{timestamp}]"
    
    success = git_add_commit_push(existing_files, commit_message)
    return success

def save_checkpoint(model, optimizer, epoch, step, train_losses, val_losses, tokens_seen, 
                   checkpoint_dir="checkpoints", github_save=False, model_name="gpt2-medium"):
    """チェックポイントを保存する関数（GitHub自動保存機能付き）"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'tokens_seen': tokens_seen,
        'timestamp': datetime.now().isoformat(),
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 チェックポイント保存: {checkpoint_path}")
    
    # 学習履歴も保存
    loss_data = {
        'train_losses': [float(x) for x in train_losses],
        'val_losses': [float(x) for x in val_losses],
        'tokens_seen': tokens_seen,
        'last_step': step,
        'last_epoch': epoch,
        'timestamp': datetime.now().isoformat()
    }
    with open('training_losses.json', 'w') as f:
        json.dump(loss_data, f, indent=2)
    
    # GitHub自動保存
    if github_save:
        print("🔄 GitHubに自動保存中...")
        success = auto_save_to_github(checkpoint_path, step, model_name)
        if not success:
            print("⚠️  GitHub保存に失敗しましたが、ローカル保存は成功しています。")
    
    # 古いチェックポイントを削除（最新の3つを保持）
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pth"))
    if len(checkpoints) > 3:
        checkpoints.sort(key=lambda x: int(re.search(r'step_(\d+)', x).group(1)))
        for old_checkpoint in checkpoints[:-3]:
            os.remove(old_checkpoint)
            print(f"🗑️  古いチェックポイント削除: {old_checkpoint}")

def load_checkpoint(checkpoint_path, model, optimizer):
    """チェックポイントから学習を再開する関数"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return (checkpoint['epoch'], 
            checkpoint['step'], 
            checkpoint['train_losses'], 
            checkpoint['val_losses'], 
            checkpoint['tokens_seen'])

def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    """最新のチェックポイントを見つける関数"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pth"))
    if not checkpoints:
        return None
    
    latest_checkpoint = max(checkpoints, key=lambda x: int(re.search(r'step_(\d+)', x).group(1)))
    return latest_checkpoint

def train_model_with_checkpoints(model, train_loader, val_loader, optimizer, device,
                                num_epochs=5, eval_freq=5, eval_iter=5,
                                start_context="", tokenizer=None, checkpoint_freq=100,
                                github_save=False, model_name="gpt2-medium"):
    """チェックポイント保存機能付きの学習関数（GitHub自動保存対応）"""
    
    # 既存のチェックポイントがあるかチェック
    latest_checkpoint = find_latest_checkpoint()
    start_epoch = 0
    global_step = 0
    train_losses = []
    val_losses = []
    tokens_seen = []
    
    if latest_checkpoint:
        print(f"既存のチェックポイントを発見: {latest_checkpoint}")
        response = input("チェックポイントから学習を再開しますか？ (y/n): ")
        if response.lower() == 'y':
            start_epoch, global_step, train_losses, val_losses, tokens_seen = load_checkpoint(
                latest_checkpoint, model, optimizer)
            print(f"チェックポイントから再開: epoch {start_epoch}, step {global_step}")
    
    total_steps = 0
    for epoch in range(start_epoch, num_epochs):
        model.train()
        
        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = calc_loss_loader(iter([input_batch, target_batch]), model, device, num_batches=1)
            loss.backward()
            optimizer.step()
            
            total_steps += 1
            global_step += 1
            tokens_seen.append(global_step * train_loader.batch_size * input_batch.size(1))
            
            # eval_freqごとに評価
            if total_steps % eval_freq == 0:
                train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
                val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
                
                # サンプル生成
                if start_context:
                    model.eval()
                    context = torch.tensor(text_to_token_ids(start_context, tokenizer)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        token_ids = generate(model, context, max_new_tokens=50, 
                                           context_size=model.pos_emb.weight.shape[0])
                        generated_text = token_ids_to_text(token_ids.squeeze(0), tokenizer)
                        print(f"Generated: {generated_text[len(start_context):].strip()}")
                    model.train()
            
            # チェックポイント保存
            if global_step % checkpoint_freq == 0:
                save_checkpoint(model, optimizer, epoch, global_step, 
                              train_losses, val_losses, tokens_seen,
                              github_save=github_save, model_name=model_name)
        
        print(f"Epoch {epoch+1}/{num_epochs} completed")
    
    return train_losses, val_losses, tokens_seen

# 前回のデータを再読み込み
print("\n前回のデータ再読み込み...")

def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        print(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
        print(f"Data saved to {file_path}")
    else:
        print(f"Loading existing file: {file_path}")

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

# Alpacaデータセット準備
file_path = "alpaca_data.json"
url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"

data = download_and_load_file(file_path, url)
print(f"Total Alpaca dataset size: {len(data)}")

# データセット分割（ch07.ipynbと同じ）
train_portion = int(len(data) * 0.85)  # 85% for training
test_portion = int(len(data) * 0.1)    # 10% for testing
val_portion = len(data) - train_portion - test_portion  # Remaining 5% for validation

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

print(f"データ分割完了: 訓練={len(train_data)}, 検証={len(val_data)}, テスト={len(test_data)}")

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

# セル89: ファインチューニング実行（チェックポイント保存付き）
print("\nセル89: ファインチューニング実行（チェックポイント保存付き）")
print("-" * 50)

start_time = time.time()

torch.manual_seed(123)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

num_epochs = 2

# GitHub自動保存設定
github_save_enabled = True  # GitHub自動保存を有効にする場合はTrue
if github_save_enabled:
    if check_git_repo():
        print("🔧 GitHub自動保存が有効です")
        print("チェックポイント保存時に自動でGitHubにpushします")
    else:
        print("⚠️  Gitリポジトリではないため、GitHub自動保存を無効にします")
        github_save_enabled = False

print(f"ファインチューニング開始: {num_epochs} epochs")
print("100stepごとにチェックポイントを保存します")

model_name_clean = re.sub(r'[ ()]', '', CHOOSE_MODEL)

train_losses, val_losses, tokens_seen = train_model_with_checkpoints(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context=format_input(val_data[0]), tokenizer=tokenizer,
    checkpoint_freq=100, github_save=github_save_enabled, 
    model_name=model_name_clean
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

# 最終モデル保存
print("\n最終モデル保存")
print("-" * 50)

file_name = f"{model_name_clean}-alpaca-sft.pth"
torch.save(model.state_dict(), file_name)
print(f"💾 最終モデル保存: {file_name}")

# 最終モデルもGitHubに保存
if github_save_enabled:
    print("🔄 最終モデルをGitHubに保存中...")
    final_commit_message = f"Final model: {model_name_clean} Alpaca fine-tuning completed"
    
    files_to_save = [
        file_name,
        "alpaca-instruction-data-with-response.json",
        "training_losses.json"
    ]
    
    # 存在するファイルのみを対象
    existing_files = [f for f in files_to_save if os.path.exists(f)]
    
    success = git_add_commit_push(existing_files, final_commit_message)
    if success:
        print("✅ 最終モデルのGitHub保存が完了しました！")
    else:
        print("⚠️  最終モデルのGitHub保存に失敗しました")

# plot_losses (if available)
print("\nLoss可視化")
print("-" * 50)

if train_losses and val_losses:
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    try:
        plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    except Exception as e:
        print(f"Loss可視化でエラー: {e}")
        print("Loss値を保存します:")
        loss_data = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'tokens_seen': tokens_seen
        }
        with open('training_losses.json', 'w') as f:
            json.dump(loss_data, f, indent=2)
        print("training_losses.json に保存完了")

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

with open("alpaca-instruction-data-with-response.json", "w") as file:
    json.dump(test_data, file, indent=4)

print("✅ 全テストデータの応答を 'alpaca-instruction-data-with-response.json' に保存完了")

print(f"\n# モデル読み込み方法:")
print(f"# model.load_state_dict(torch.load(\"{file_name}\"))")

print("\n✅ Alpacaデータセット ファインチューニング完了（チェックポイント・GitHub保存付き）")
print("SSH接続が切れても安心です！")

print("\n" + "=" * 70)
print("🔧 GitHub自動保存機能について")
print("=" * 70)
print("📌 事前セットアップ:")
print("1. git config --global user.name \"Your Name\"")
print("2. git config --global user.email \"your.email@example.com\"") 
print("3. git remote add origin https://github.com/username/repository.git")
print("4. SSH認証 または Personal Access Token の設定")
print()
print("📌 保存されるファイル:")
print("- checkpoints/checkpoint_step_XXX.pth (100stepごと)")
print("- training_losses.json (学習履歴)")
print("- 最終モデル.pth")
print("- alpaca-instruction-data-with-response.json")
print()
print("📌 カスタマイズ:")
print("- github_save_enabled = False でGitHub保存を無効化")
print("- checkpoint_freq = 50 で50stepごとに変更")
print("=" * 70) 