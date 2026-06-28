from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    StoppingCriteria,
    StoppingCriteriaList,
)
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer
from trl import DataCollatorForCompletionOnlyLM
from config import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_data(data_path: str) -> list[dict]:
    try:
        with open(data_path, encoding="utf-8") as f:
            data = json.load(f)
        log.info("Loaded %d examples from %s", len(data), data_path)
        return data 
    except Exception:
        log.error("No data file was found")
    

RESPONSE_TEMPLATE = "<|im_start|>assistant\n"

def make_sft_dataset(data: list[dict], tokenizer: AutoTokenizer) -> Dataset:
    texts = []
    for ex in data:
        messages = [
            {"role": "user",      "content": ex["prompt"]},
            {"role": "assistant", "content": ex["chosen"]},
        ]
        texts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        )
    return Dataset.from_dict({"text": texts})


def make_dpo_dataset(data: list[dict]) -> Dataset:
    return Dataset.from_dict({
        "prompt":   [ex["prompt"]   for ex in data],
        "chosen":   [ex["chosen"]   for ex in data],
        "rejected": [ex["rejected"] for ex in data],
    })


def load_model_and_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map="auto" if use_cuda else None,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer



def run_sft(data: list[dict]) -> str:
    log.info("Phase 1: Supervised Fine-Tuning (SFT)")

    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    dataset = make_sft_dataset(data, tokenizer)

    collator = DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE,
        tokenizer=tokenizer,
    )

    use_cuda = torch.cuda.is_available()

    sft_cfg = SFTConfig(
        output_dir=SFT_OUTPUT_DIR,
        num_train_epochs=SFT_EPOCHS,
        per_device_train_batch_size=SFT_BATCH_SIZE,
        gradient_accumulation_steps=SFT_GRAD_ACCUM,
        learning_rate=SFT_LR,
        warmup_ratio=SFT_WARMUP_RATIO,
        max_seq_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        bf16=use_cuda,
        fp16=False,
        logging_steps=5,
        save_strategy="epoch",
        seed=SEED,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
        packing=False,  # must be off when using DataCollatorForCompletionOnlyLM
    )

    trainer.train()
    trainer.save_model(SFT_OUTPUT_DIR)
    tokenizer.save_pretrained(SFT_OUTPUT_DIR)
    log.info("SFT checkpoint saved to %s", SFT_OUTPUT_DIR)
    return SFT_OUTPUT_DIR


def run_dpo(sft_model_path: str, data: list[dict]) -> str:
    log.info("Phase 2 — Direct Preference Optimisation (DPO)")

    model, tokenizer = load_model_and_tokenizer(sft_model_path)
    dataset = make_dpo_dataset(data)

    use_cuda = torch.cuda.is_available()

    dpo_cfg = DPOConfig(
        output_dir=DPO_OUTPUT_DIR,
        num_train_epochs=DPO_EPOCHS,
        per_device_train_batch_size=DPO_BATCH_SIZE,
        gradient_accumulation_steps=DPO_GRAD_ACCUM,
        learning_rate=DPO_LR,
        beta=DPO_BETA,
        max_length=MAX_SEQ_LEN,
        max_prompt_length=MAX_SEQ_LEN // 2,
        bf16=use_cuda,
        fp16=False,
        logging_steps=5,
        save_strategy="epoch",
        seed=SEED,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(DPO_OUTPUT_DIR)
    tokenizer.save_pretrained(DPO_OUTPUT_DIR)
    log.info("DPO checkpoint saved to %s", DPO_OUTPUT_DIR)
    return DPO_OUTPUT_DIR



class SafeLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return torch.nan_to_num(scores, nan=0.0, posinf=1e4, neginf=-1e4)


class StopOnTemplate(StoppingCriteria):
    def __init__(self, tokenizer: AutoTokenizer):
        self.stop_ids = tokenizer.encode("<|im_start|>user", add_special_tokens=False)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0][-len(self.stop_ids):].tolist() == self.stop_ids


def generate_response(
    model_path: str,
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> str:
    model, tokenizer = load_model_and_tokenizer(model_path)

    model = model.to(torch.float32)
    model.eval()

    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            logits_processor=[SafeLogitsProcessor()],
            stopping_criteria=StoppingCriteriaList([StopOnTemplate(tokenizer)]),
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()




def main():

    log.info("Model  : %s", MODEL_NAME)
    log.info("Device : %s", "cuda" if torch.cuda.is_available() else "cpu")

    data = load_data(DATA_DIR)

    
    sft_path = run_sft(data)

    dpo_path = run_dpo(sft_path, data)

    test_prompts = [
        "How should one properly thank a host after a dinner?",
        "What is the best way to introduce yourself to a stranger?",
    ]

    print("INFERENCE:  DPO model")
    for prompt in test_prompts:
        response = generate_response(dpo_path, prompt)
        print(f"\nQ: {prompt}")
        print(f"A: {response}")
    print("═" * 60)


if __name__ == "__main__":
    main()