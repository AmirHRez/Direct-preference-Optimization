import json
from openai import OpenAI
from tqdm import tqdm
import time
import os
from config import SYSTEM_PROMPT, TRANSLATOR_BETA, TRANSLATOR_TOP_P, MAX_LEN_RATIO, SHORT_ANSWER_MAX_WORDS, SHORT_ANSWER_WORD_THRESHOLD

INPUT_FILE = "data/qa_part2.jsonl"
OUTPUT_FILE = "data/archaic/qa_archaic_part2.jsonl"
FAILED_FILE = "data/archaic/qa_archaic_failed.jsonl"
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


def length_ratio(chosen: str, rejected: str) -> float:
    r = len(rejected.split())
    return len(chosen.split()) / r if r > 0 else 999

def passes_length(chosen: str, answer: str) -> bool:
    n_words = len(answer.split())
    if n_words <= SHORT_ANSWER_WORD_THRESHOLD:
        return len(chosen.split()) <= SHORT_ANSWER_MAX_WORDS
    ratio = length_ratio(chosen, answer)
    return ratio <= MAX_LEN_RATIO
 

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
            written = False
            for attempt in range(3):
                try:
                    chosen = transform(
                        item["prompt"],
                        item["answer"],
                        strict_length=(attempt > 0)
                    )
 
                    ratio = length_ratio(chosen, item["answer"])
                    ok = passes_length(chosen, item["answer"])
 
                    if not ok and attempt < 2:
                        continue  # retry with strict_length=True
 
                    if not ok:
                        # Final attempt still violates length -> do NOT write it
                        # into the training file. Log it for manual review instead.
                        fail_log.write(json.dumps(
                            {**item, "reason": "length_violation_final_attempt",
                             "last_chosen": chosen, "len_ratio": round(ratio, 3)},
                            ensure_ascii=False
                        ) + "\n")
                        fail_log.flush()
                        written = True  # handled, don't fall through to except-loop's else
                        break
 
                    dpo_item = {
                        "prompt": item["prompt"],
                        "chosen": chosen,
                        "rejected": item["answer"],
                        "len_ratio": round(ratio, 3)
                    }
                    out.write(json.dumps(dpo_item, ensure_ascii=False) + "\n")
                    out.flush()
                    written = True
                    break
 
                except Exception as e:
                    print(f"Attempt {attempt+1} failed: {e}")
                    time.sleep(2 ** attempt)
 
            if not written:
                fail_log.write(json.dumps(
                    {**item, "reason": "exception_all_attempts"},
                    ensure_ascii=False
                ) + "\n")
                fail_log.flush()
 


if __name__ == "__main__":
    main()