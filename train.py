"""
eme_dpo_pipeline.py
====================
SFT → DPO pipeline to align a causal LM toward Early Modern English responses.

Phases
------
1. SFT  – teach the model to follow the instruction format and produce EME output
2. DPO  – push it further toward EME (chosen) and away from modern English (rejected)

Requirements
------------
    pip install "transformers==4.40.0" "trl==0.12.2" datasets torch accelerate

Quick start
-----------
    python eme_dpo_pipeline.py                       # runs on built-in sample data
    python eme_dpo_pipeline.py --data_path data.json # your own JSON file
"""

from __future__ import annotations

import argparse
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    # Model — SmolLM2-Instruct instead of GPT-2:
    #   GPT-2 is not instruction-tuned so it rambles, repeats the prompt
    #   template, and never stops cleanly. SmolLM2-Instruct already knows
    #   how to follow a prompt and stop; SFT+DPO only needs to teach it style.
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"

    # Output paths
    sft_output_dir: str = "./checkpoints/sft"
    dpo_output_dir: str = "./checkpoints/dpo"

    # Common
    max_seq_length: int = 256
    seed: int = 42

    # SFT hyperparams
    sft_epochs: int = 3
    sft_batch_size: int = 2
    sft_lr: float = 2e-5
    sft_warmup_ratio: float = 0.1
    sft_grad_accum: int = 4

    # DPO hyperparams
    dpo_epochs: int = 3
    dpo_batch_size: int = 1
    dpo_lr: float = 5e-6
    dpo_beta: float = 0.1
    dpo_grad_accum: int = 4

    # Data
    data_path: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Built-in sample data
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_DATA: list[dict] = [
    {
        "prompt": "How do you ask a librarian or store clerk for help without interrupting them?",
        "chosen": (
            "To seek aid from a keeper of books or a shopman without breaking his labour, "
            "wait until he hath finished his present task, then say, 'Pardon me—when thou "
            "hath a moment, couldst thou assist me?'"
        ),
        "rejected": (
            "Wait for them to finish their current task, then say "
            "'Excuse me, when you have a moment, could you help me?'"
        ),
    },
    {
        "prompt": "What is the best way to apologise after a mistake?",
        "chosen": (
            "When thou hast erred, approach the offended party with humility and say, "
            "'I do most humbly beseech thy pardon for mine offence; it was ill done of me "
            "and shall not be repeated.'"
        ),
        "rejected": (
            "The best way to apologise is to acknowledge your mistake directly and sincerely, "
            "say you're sorry, and explain what you'll do differently in the future."
        ),
    },
    {
        "prompt": "How do you decline an invitation politely?",
        "chosen": (
            "Should thy presence be required elsewhere, thou mayst decline with grace: "
            "'I thank thee most heartily for thine invitation, yet I am, by prior engagement, "
            "unable to attend. I trust thou wilt forgive mine absence.'"
        ),
        "rejected": (
            "To politely decline an invitation, thank the person for inviting you, "
            "briefly explain that you're unable to attend, and express that you hope "
            "to see them another time."
        ),
    },
    {
        "prompt": "How should you greet someone you have not seen in a long time?",
        "chosen": (
            "Upon meeting one long absent from thy company, thou mightest say with warmth: "
            "'Well met, good friend! It doth gladden my heart exceedingly to look upon "
            "thy countenance once more after so long a parting.'"
        ),
        "rejected": (
            "When you see someone you haven't met in a long time, greet them warmly "
            "and tell them it's great to see them again."
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_data(data_path: Optional[str]) -> list[dict]:
    if data_path and Path(data_path).exists():
        with open(data_path, encoding="utf-8") as f:
            data = json.load(f)
        log.info("Loaded %d examples from %s", len(data), data_path)
        return data
    log.info("Using built-in sample data (%d examples).", len(SAMPLE_DATA))
    return SAMPLE_DATA


# SmolLM2 uses ChatML format:
#   <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>
# DataCollatorForCompletionOnlyLM searches for this token sequence to find
# where the response begins, masking everything before it from the loss.
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"


def make_sft_dataset(data: list[dict], tokenizer: AutoTokenizer) -> Dataset:
    """
    Format examples using the model's own chat template so the token boundaries
    exactly match what the model expects at inference time.
    The collator uses RESPONSE_TEMPLATE to mask prompt tokens from the loss.
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Model / tokenizer
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    use_cuda = torch.cuda.is_available()
    # bfloat16: same exponent range as float32, so loss values stay
    # representable and don't collapse to 0.0 the way float16 does.
    dtype = torch.bfloat16 if use_cuda else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map="auto" if use_cuda else None,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — SFT
# ─────────────────────────────────────────────────────────────────────────────

def run_sft(cfg: PipelineConfig, data: list[dict]) -> str:
    log.info("━" * 60)
    log.info("Phase 1 — Supervised Fine-Tuning (SFT)")
    log.info("━" * 60)

    model, tokenizer = load_model_and_tokenizer(cfg.model_name)
    dataset = make_sft_dataset(data, tokenizer)

    collator = DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE,
        tokenizer=tokenizer,
    )

    use_cuda = torch.cuda.is_available()

    sft_cfg = SFTConfig(
        output_dir=cfg.sft_output_dir,
        num_train_epochs=cfg.sft_epochs,
        per_device_train_batch_size=cfg.sft_batch_size,
        gradient_accumulation_steps=cfg.sft_grad_accum,
        learning_rate=cfg.sft_lr,
        warmup_ratio=cfg.sft_warmup_ratio,
        max_seq_length=cfg.max_seq_length,
        dataset_text_field="text",
        bf16=use_cuda,
        fp16=False,
        logging_steps=5,
        save_strategy="epoch",
        seed=cfg.seed,
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
    trainer.save_model(cfg.sft_output_dir)
    tokenizer.save_pretrained(cfg.sft_output_dir)
    log.info("SFT checkpoint saved → %s", cfg.sft_output_dir)
    return cfg.sft_output_dir


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — DPO
# ─────────────────────────────────────────────────────────────────────────────

def run_dpo(cfg: PipelineConfig, sft_model_path: str, data: list[dict]) -> str:
    log.info("━" * 60)
    log.info("Phase 2 — Direct Preference Optimisation (DPO)")
    log.info("━" * 60)

    model, tokenizer = load_model_and_tokenizer(sft_model_path)
    dataset = make_dpo_dataset(data)

    use_cuda = torch.cuda.is_available()

    dpo_cfg = DPOConfig(
        output_dir=cfg.dpo_output_dir,
        num_train_epochs=cfg.dpo_epochs,
        per_device_train_batch_size=cfg.dpo_batch_size,
        gradient_accumulation_steps=cfg.dpo_grad_accum,
        learning_rate=cfg.dpo_lr,
        beta=cfg.dpo_beta,
        max_length=cfg.max_seq_length,
        max_prompt_length=cfg.max_seq_length // 2,
        bf16=use_cuda,
        fp16=False,
        logging_steps=5,
        save_strategy="epoch",
        seed=cfg.seed,
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
    trainer.save_model(cfg.dpo_output_dir)
    tokenizer.save_pretrained(cfg.dpo_output_dir)
    log.info("DPO checkpoint saved → %s", cfg.dpo_output_dir)
    return cfg.dpo_output_dir


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

class SafeLogitsProcessor(LogitsProcessor):
    """Replaces nan/inf logits before softmax to prevent torch.multinomial crash."""
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return torch.nan_to_num(scores, nan=0.0, posinf=1e4, neginf=-1e4)


class StopOnTemplate(StoppingCriteria):
    """
    Halts generation when the model starts emitting a new <|im_start|>user turn.
    Without this the model continues past the end of its response and begins
    generating a follow-up question, repeating the prompt template mid-output.
    """
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
    """Generate an EME-style response from the fine-tuned model."""
    model, tokenizer = load_model_and_tokenizer(model_path)

    # float32 for inference — bf16 logits can overflow to inf causing
    # softmax to produce nan and crashing torch.multinomial.
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




def main() -> None:
    cfg = PipelineConfig()

    log.info("Model  : %s", cfg.model_name)
    log.info("Device : %s", "cuda" if torch.cuda.is_available() else "cpu")

    data = load_data(cfg.data_path)

    
    sft_path = run_sft(cfg, data)

    dpo_path = run_dpo(cfg, sft_path, data)

    test_prompts = [
        "How should one properly thank a host after a dinner?",
        "What is the best way to introduce yourself to a stranger?",
    ]

    print("\n" + "═" * 60)
    print("  INFERENCE  —  DPO model")
    print("═" * 60)
    for prompt in test_prompts:
        response = generate_response(dpo_path, prompt)
        print(f"\nQ: {prompt}")
        print(f"A: {response}")
    print("═" * 60)


if __name__ == "__main__":
    main()