import json
from openai import OpenAI
from tqdm import tqdm
import time
import os
from config import SYSTEM_PROMPT, TRANSLATOR_BETA, TRANSLATOR_TOP_P, MAX_LEN_RATIO, SHORT_ANSWER_MAX_WORDS, SHORT_ANSWER_WORD_THRESHOLD
from quality_check import compute_flags, length_ratio

INPUT_FILE = "data/qa_part2.jsonl"
OUTPUT_FILE = "data/archaic/qa_archaic_part2.jsonl"
ERROR_FILE = "data/archaic/qa_archaic_errors.jsonl"
MODEL = "local-model"

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)


def transform(prompt: str, answer: str, strict_length: bool = False) -> str:
    n_words = len(answer.split())
    if strict_length:
        if n_words <= SHORT_ANSWER_WORD_THRESHOLD:
            extra = (
                f" The original answer is only {n_words} words. Change verb forms "
                f"or add a single archaic pronoun/interjection only. Do not add new "
                f"clauses, facts, dates, or descriptions. Your output must be under "
                f"{SHORT_ANSWER_MAX_WORDS} words."
            )
        else:
            extra = f" The original is {n_words} words. Stay within 15% of that length."
    else:
        extra = ""
 
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
 
    flagged_count = 0
    clean_count = 0
 
    with open(OUTPUT_FILE, "a", encoding="utf-8") as out, \
         open(ERROR_FILE, "a", encoding="utf-8") as err_log:
 
        for item in tqdm(remaining):
            chosen = None
            # Retries here are ONLY for API/network failures, not quality —
            # the model gets one shot at the actual rewrite, then we move on
            # and let flags/review handle anything imperfect about it.
            for attempt in range(3):
                try:
                    chosen = transform(item["prompt"], item["answer"])
                    break
                except Exception as e:
                    print(f"Attempt {attempt+1} failed: {e}")
                    time.sleep(2 ** attempt)
 
            if chosen is None:
                err_log.write(json.dumps(
                    {**item, "reason": "exception_all_attempts"},
                    ensure_ascii=False
                ) + "\n")
                err_log.flush()
                continue
 
            ratio = length_ratio(chosen, item["answer"])
            flags = compute_flags(chosen, item["answer"], MAX_LEN_RATIO)
 
            if flags:
                flagged_count += 1
            else:
                clean_count += 1
 
            dpo_item = {
                "prompt": item["prompt"],
                "chosen": chosen,
                "rejected": item["answer"],
                "len_ratio": round(ratio, 3),
                "flags": flags,
                "needs_review": bool(flags),
            }
            out.write(json.dumps(dpo_item, ensure_ascii=False) + "\n")
            out.flush()
 
    print(f"\nDone. Clean: {clean_count}, flagged for review: {flagged_count}")
 


if __name__ == "__main__":
    main()