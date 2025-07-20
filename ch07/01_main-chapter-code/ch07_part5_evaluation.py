#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chapter 7: Part 5 - LLM評価（Ollama & Llama 3使用）
Converted from ch07.ipynb - cells 103-119 (LLM evaluation)
"""

import json
import urllib.request
from tqdm import tqdm

print("=" * 70)
print("Chapter 7: Part 5 - LLM評価（Ollama & Llama 3使用）")
print("=" * 70)

# セル110: Ollama実行状況確認
print("\nセル110: Ollama実行状況確認")
print("-" * 50)

try:
    import psutil

    def check_if_running(process_name):
        running = False
        for proc in psutil.process_iter(["name"]):
            if process_name in proc.info["name"]:
                running = True
                break
        return running

    ollama_running = check_if_running("ollama")

    if not ollama_running:
        print("⚠️ Ollama not running. Please launch ollama before proceeding.")
        print("Install: https://ollama.ai/")
        print("Run: ollama run llama3")
    else:
        print("✅ Ollama running:", check_if_running("ollama"))

except ImportError:
    print("⚠️ psutil not installed. Cannot check if Ollama is running.")
    print("Install: pip install psutil")
    ollama_running = False
except Exception as e:
    print(f"⚠️ Error checking Ollama status: {e}")
    ollama_running = False

# セル111: データ読み込み（instruction-data-with-response.jsonから）
print("\nセル111: データ読み込み")
print("-" * 50)

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text

file_path = "instruction-data-with-response.json"

try:
    with open(file_path, "r") as file:
        test_data = json.load(file)
    print(f"✅ {file_path} 読み込み完了: {len(test_data)} entries")
except FileNotFoundError:
    print(f"⚠️ {file_path} not found. Please run previous parts first.")
    # デモ用にダミーデータ作成
    test_data = [
        {
            "instruction": "Rewrite the sentence using a simile.",
            "input": "The car is very fast.",
            "output": "The car is as fast as lightning.",
            "model_response": "The car is as fast as a bullet."
        }
    ]
    print(f"デモ用データを使用: {len(test_data)} entries")

# セル113: query_model関数定義
print("\nセル113: query_model関数定義")
print("-" * 50)

def query_model(prompt, model="llama3", url="http://localhost:11434/api/chat"):
    # Create the data payload as a dictionary
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {     # Settings below are required for deterministic responses
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }

    # Convert the dictionary to a JSON formatted string and encode it to bytes
    payload = json.dumps(data).encode("utf-8")

    # Create a request object, setting the method to POST and adding necessary headers
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    # Send the request and capture the response
    response_data = ""
    with urllib.request.urlopen(request) as response:
        # Read and decode the response
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

    return response_data

print("✅ query_model関数定義完了")

# Ollamaテスト
print("\nOllamaテスト...")
if ollama_running:
    try:
        model = "llama3"
        result = query_model("What do Llamas eat?", model)
        print("✅ Ollama接続成功!")
        print("Llama3 response (sample):")
        print(result[:200] + "..." if len(result) > 200 else result)
    except Exception as e:
        print(f"⚠️ Ollama connection failed: {e}")
        print("Please ensure 'ollama run llama3' is running.")
        ollama_running = False
else:
    print("⚠️ Ollamaが利用できません。評価をスキップします。")

if ollama_running:
    # セル115: 最初の3つのテストデータを評価
    print("\nセル115: 最初の3つのテストデータを評価")
    print("-" * 50)

    for entry in test_data[:3]:
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry['model_response']}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
        )
        print("\nDataset response:")
        print(">>", entry['output'])
        print("\nModel response:")
        print(">>", entry["model_response"])
        print("\nScore:")
        try:
            score_result = query_model(prompt)
            print(">>", score_result)
        except Exception as e:
            print(f">> Error: {e}")
        print("\n-------------------------")

    # セル117: 全テストデータを評価してスコア算出
    print("\nセル117: 全テストデータを評価してスコア算出")
    print("-" * 50)

    def generate_model_scores(json_data, json_key, model="llama3"):
        scores = []
        for entry in tqdm(json_data, desc="Scoring entries"):
            prompt = (
                f"Given the input `{format_input(entry)}` "
                f"and correct output `{entry['output']}`, "
                f"score the model response `{entry[json_key]}`"
                f" on a scale from 0 to 100, where 100 is the best score. "
                f"Respond with the integer number only."
            )
            try:
                score = query_model(prompt, model)
                scores.append(int(score))
            except ValueError:
                print(f"Could not convert score: {score}")
                continue
            except Exception as e:
                print(f"Error scoring entry: {e}")
                continue

        return scores

    try:
        scores = generate_model_scores(test_data, "model_response")
        print(f"Number of scores: {len(scores)} of {len(test_data)}")
        if len(scores) > 0:
            print(f"Average score: {sum(scores)/len(scores):.2f}\n")
        else:
            print("No valid scores obtained.\n")
    except Exception as e:
        print(f"Error during batch scoring: {e}")

    # セル118-119: 結果の解釈
    print("\nセル118-119: 結果の解釈")
    print("-" * 50)
    print("✅ モデル評価が完了しました！")
    print("- 平均スコア50以上が基準として使用できます")
    print("- 参考:")
    print("  - Llama 3 8B base model: 58.51")
    print("  - Llama 3 8B instruct model: 82.65")

else:
    print("\n⚠️ Ollamaが利用できないため、評価をスキップしました。")
    print("評価を実行するには:")
    print("1. Ollamaをインストール: https://ollama.ai/")
    print("2. ターミナルで実行: ollama run llama3")
    print("3. このスクリプトを再実行")

print("\n✅ Part 5 完了: LLM評価（ch07.ipynb セル103-119）")
print("ch07.ipynbに忠実に従った評価スクリプトです！") 