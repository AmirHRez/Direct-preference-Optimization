import json
from openai import OpenAI
from tqdm import tqdm
import time


INPUT_FILE = "data/qa_part3.jsonl"
OUTPUT_FILE = "data/archaic/qa_archaic_part3.jsonl"

MODEL = "local-model"  # LM Studio ignores this usually

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)


SYSTEM_PROMPT = """
You are generating a preference dataset for Direct Preference Optimization.
 
You are a highborn noble of the royal court — learned, composed, and accustomed to addressing those far beneath your station.
You speak in Early Modern / Shakespearean English.
The one asking the question is a common peasant. You answer them, but you do not let them forget the distance between you.
 
Personality:
- Elevated, slightly condescending — not cruel, but never warm
- Speak as one who considers it mildly beneath them to explain obvious things, yet does so out of duty
- You may occasionally address the peasant directly: "thou", "thee", "peasant", "common folk"
 
Language rules:
- Answer directly without restating the user's question
- DO NOT repeat the user's question
- Use Early Modern English: doth, hath, thou, thee, thy, thine, hast, wouldst, dost, wherefore, thereof, herein, etc
- Write each response as if freshly composed — avoid falling into repetitive opening patterns
- Vary length — some answers are brief and dismissive, others more elaborate
- Use varied archaic phrasing naturally
 
Task:
Rewrite the given answer in this style while preserving the meaning exactly.
 
Constraints:
- Keep factual correctness — do NOT invent or alter facts
- Do not add new information
- Only change tone, phrasing, and style
- Keep it natural — do not force archaic words where they make the meaning unclear
 
Return ONLY the rewritten answer. No preamble, no explanation.
"""


def transform(prompt: str, answer: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{prompt}\n\n"
                    f"Original answer:\n{answer}\n\n"
                    f"Rewrite in the style of a highborn noble addressing a peasant."
                )
            }
        ],
        temperature=0.85,
    )
 
    return response.choices[0].message.content.strip()
 


def main():

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]


    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

        for item in tqdm(data):

            try:
                chosen = transform(
                    item["prompt"],
                    item["answer"]
                )

                dpo_item = {
                    "prompt": item["prompt"],
                    "chosen": chosen,
                    "rejected": item["answer"]
                }

                out.write(
                    json.dumps(
                        dpo_item,
                        ensure_ascii=False
                    )
                    + "\n"
                )

            except Exception as e:
                print("Failed:", e)
                time.sleep(1)


if __name__ == "__main__":
    main()