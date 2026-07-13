import os
import time
import json
import math
from google import genai
from google.genai import types
from datasets import load_from_disk

# Your key
client = genai.Client(api_key="")

dataset = load_from_disk("alpaca_3500")

output_filename = "data/archaic/alpaca-gemini-3500.jsonl"

existing_prompts = set()
if os.path.exists(output_filename):
    print(f"Found existing output file '{output_filename}'. Scanning completed records...")
    with open(output_filename, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    existing_prompts.add(record.get("prompt", ""))
                except json.JSONDecodeError:
                    continue
    print(f"Resuming pipeline. Skipping {len(existing_prompts)} already processed rows.")

SYSTEM_PROMPT = """
    You are a precision linguistic transformation engine. Your single task is to rewrite modern English text into authentic Early Modern English (16th/17th century style).

    CRITICAL CONSTRAINTS:
    1. MAX LENGTH: The rewritten output MUST NOT exceed {max_words} words
    2. ARCHAIC MARKERS: You must utilize genuine Early Modern English features (e.g., pronouns like thou/thee/thy/thine, verbs ending in -eth or -est, and auxiliary forms like doth, hath, dost, art).
        - Verb forms: verbs ending in -eth or -est, and auxiliary forms like doth, hath, dost, art, wilt, wouldst, etc.
        - Pronouns: thee, thou, thy, thine where natural
        - Vocabulary: use established archaic words where they exist
    3. FACTUALITY & STRUCTURE: Keep all technical terms, names, formulas, numbers, dates, and core semantic meanings identical. Do not invent fake phonetic spellings.
    4. FORMAT: Output ONLY the rewritten answer.
"""

print(f"Starting loop. Writing outputs sequentially to {output_filename}...")

for i, row in enumerate(dataset):
    modern_input = row.get("instruction", "")
    modern_output = row.get("output", "")
    
    if modern_input in existing_prompts:
        continue
    
    if not modern_output.strip():
        continue

    # Calculate lengths
    orig_word_count = len(modern_output.split())
    max_words = max(orig_word_count + 2, math.ceil(orig_word_count * 1.2)) 

    formatted_system = SYSTEM_PROMPT.format(max_words=max_words)
    
    try:
        response = client.models.generate_content(
            model='gemini-3.1-flash-lite',
            contents=modern_output,
            config=types.GenerateContentConfig(
                system_instruction=formatted_system,
                temperature=0.3,
                top_p=0.93
            ),
        )
        
        emode_output = response.text.strip() if response.text else ""
        new_word_count = len(emode_output.split())
        
        # length ratio
        len_ratio = round(new_word_count / orig_word_count, 3) if orig_word_count > 0 else 0.0
        
        dpo_record = {
            "prompt": modern_input,
            "chosen": emode_output,
            "rejected": modern_output,
            "len_ratio": len_ratio
        }
        
        # write the row to a JSONL file
        with open(output_filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(dpo_record) + "\n")
            
        print(f"Processed row {i+1}/{len(dataset)} | Ratio: {len_ratio}")
            
        time.sleep(8)
        
    except Exception as e:
        print(f"Error handling row {i+1}: {e}")
        time.sleep(30) 

print(f"Done! All successful generations appended to {output_filename}")