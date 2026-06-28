import json
from openai import OpenAI
from tqdm import tqdm
import time
from config import SYSTEM_PROMPT


INPUT_FILE = "data/qa_part3.jsonl"
OUTPUT_FILE = "data/archaic/qa_archaic_part3.jsonl"

MODEL = "local-model"  # LM Studio ignores this usually

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)



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