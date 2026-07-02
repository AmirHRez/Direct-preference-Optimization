import json
from openai import OpenAI
from tqdm import tqdm
import time
import os
from config import SYSTEM_PROMPT, TRANSLATOR_BETA, TRANSLATOR_TOP_P, MAX_LEN_RATIO

INPUT_FILE = "data/qa_part1.jsonl"
OUTPUT_FILE = "data/archaic/qa_archaic_llama3.2_3B.jsonl"
FAILED_FILE = "data/archaic/qa_archaic_failed.jsonl"
MODEL = "local-model"

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)


def transform(prompt: str, answer: str, strict_length: bool = False) -> str:
    extra = (
        f" The original is {len(answer.split())} words. Stay within that."
        if strict_length else ""
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question:\n{prompt}\n\n"
                    f"Original answer:\n{answer}\n\n"
                    "Rewrite the answer using Early Modern English style only. "
                    f"Preserve the meaning and do not add a speaker persona.{extra}"
                )
            }
        ],
        temperature=TRANSLATOR_BETA,
        top_p=TRANSLATOR_TOP_P
    )
    return response.choices[0].message.content.strip()


def length_ratio(chosen: str, rejected: str) -> float:
    r = len(rejected.split())
    return len(chosen.split()) / r if r > 0 else 999


def load_done_prompts(output_path: str) -> set:
    done = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["prompt"])
                except Exception:
                    pass
    return done


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    done = load_done_prompts(OUTPUT_FILE)
    remaining = [item for item in data if item["prompt"] not in done]
    print(f"{len(done)} already done, {len(remaining)} remaining.")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out, \
         open(FAILED_FILE, "a", encoding="utf-8") as fail_log:

        for item in tqdm(remaining):
            for attempt in range(3):
                try:
                    chosen = transform(
                        item["prompt"],
                        item["answer"],
                        strict_length=(attempt > 0)
                    )

                    ratio = length_ratio(chosen, item["answer"])
                    if ratio > MAX_LEN_RATIO and attempt < 2:
                        continue  # retry with strict_length=True

                    dpo_item = {
                        "prompt": item["prompt"],
                        "chosen": chosen,
                        "rejected": item["answer"],
                        "len_ratio": round(ratio, 3)
                    }
                    out.write(json.dumps(dpo_item, ensure_ascii=False) + "\n")
                    out.flush()
                    break

                except Exception as e:
                    print(f"Attempt {attempt+1} failed: {e}")
                    time.sleep(2 ** attempt)
            else:
                fail_log.write(json.dumps(item, ensure_ascii=False) + "\n")
                fail_log.flush()


if __name__ == "__main__":
    main()