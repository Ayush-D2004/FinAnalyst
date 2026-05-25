import json
import os
from pathlib import Path

def prepare_for_kaggle():
    training_dir = Path("data/training")
    merged_data = []

    for file_path in training_dir.glob("*.json"):
        company_name = file_path.stem.upper()
        with open(file_path, 'r', encoding='utf-8') as f:
            items = json.load(f)
            for item in items:
                # We inject the company name into the instruction
                prompt = {
                    "instruction": f"Analyze the following SEC filing text for {company_name}.",
                    "input": f"Context: {item['golden_context']}\n\nQuestion: {item['query']}",
                    "output": f"<thought>\n{item['reasoning_trace']}\n</thought>\n{item['final_answer']}"
                }
                merged_data.append(prompt)

    with open("data/training/kaggle_train.json", "w", encoding='utf-8') as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)
    print(f"Prepared {len(merged_data)} samples for Kaggle.")

prepare_for_kaggle()
