import json
from typing import Optional
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer
from trl import DataCollatorForCompletionOnlyLM
from config import PipelineConfig



cfg = PipelineConfig()


def load_data(data_path: Optional[str]) -> list[dict]:
    data = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    print(f"Loaded {len(data)} examples from {data_path}")
    return data


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


def make_dpo_dataset(data: list[dict], tokenizer: AutoTokenizer) -> Dataset:
    prompts = []
    for ex in data:
        messages = [{"role": "user", "content": ex["prompt"]}]
        prompts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
    return Dataset.from_dict({
        "prompt":   prompts,
        "chosen":   [ex["chosen"]   for ex in data],
        "rejected": [ex["rejected"] for ex in data],
    })

data = load_data(cfg.data_path)


def verify_response_template(tokenizer: AutoTokenizer, sample: dict) -> None:
    messages = [
        {"role": "user", "content": sample["prompt"]},
        {"role": "assistant", "content": sample["chosen"]},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    print("=" * 60)
    print("RESPONSE TEMPLATE CHECK")
    print("=" * 60)
    print(f"Looking for: {RESPONSE_TEMPLATE!r}")
    if RESPONSE_TEMPLATE in rendered:
        print("FOUND in rendered chat template output. Masking should work correctly.")
        split_point = rendered.index(RESPONSE_TEMPLATE) + len(RESPONSE_TEMPLATE)
        print("\n--- Prompt span (masked, no loss) ---")
        print(rendered[:split_point])
        print("\n--- Completion span (loss applied here) ---")
        print(rendered[split_point:][:300], "...")
    else:
        print("NOT FOUND. The collator will not mask correctly — this is a")
        print("likely root cause of incoherent output. Inspect the tokenizer's")
        print("actual chat template (tokenizer.chat_template) and update")
        print("RESPONSE_TEMPLATE to match it exactly.")
    print("=" * 60)


def load_model_and_tokenizer(
    model_name_or_path: str,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
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


def run_sft(cfg: PipelineConfig, data: list[dict]) -> str:
    print("Phase 1: Supervised Fine-Tuning")

    model, tokenizer = load_model_and_tokenizer(cfg.model_name)

    # NEW: run the masking sanity check on a real example before training.
    verify_response_template(tokenizer, data[0])

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
        bf16=use_cuda,           # bfloat16 on GPU; fp32 on CPU
        fp16=False,              # explicitly off — do not mix with bf16
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
        packing=False
    )

    trainer.train()
    trainer.save_model(cfg.sft_output_dir)
    tokenizer.save_pretrained(cfg.sft_output_dir)
    print(f"SFT checkpoint saved to {cfg.sft_output_dir}")
    return cfg.sft_output_dir

sft_path = run_sft(cfg, data)


def quick_generate(model_path: str, prompt: str, greedy: bool = True) -> str:
    model = AutoModelForCausalLM.from_pretrained(model_path).to(torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=not greedy,
            temperature=0.6 if not greedy else None,
            top_p=0.95 if not greedy else None,
            repetition_penalty=1.3,     # NEW: discourage run-on repeated phrases
            no_repeat_ngram_size=3,     # NEW: same purpose
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

if cfg.run_sft_only_eval:
    print("\n" + "=" * 60)
    print("SFT-ONLY COHERENCE CHECK (before DPO)")
    print("=" * 60)
    for p in ["What is newton's first law?", "What is polymorphism in programming?"]:
        print(f"\nQ: {p}")
        print(f"A (greedy):  {quick_generate(sft_path, p, greedy=True)}")
        print(f"A (sampled): {quick_generate(sft_path, p, greedy=False)}")
    print("=" * 60)


def run_dpo(cfg: PipelineConfig, sft_model_path: str, data: list[dict]) -> str:
    print("Phase 2: Direct Preference Optimisation (DPO)")

    model, tokenizer = load_model_and_tokenizer(sft_model_path)
    dataset = make_dpo_dataset(data, tokenizer)

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
        ref_model=None,        # frozen copy created automatically
        args=dpo_cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(cfg.dpo_output_dir)
    tokenizer.save_pretrained(cfg.dpo_output_dir)
    print(f"DPO checkpoint saved to {cfg.dpo_output_dir}")
    return cfg.dpo_output_dir

dpo_path = run_dpo(cfg, sft_path, data)


class SafeLogitsProcessor(LogitsProcessor):
    def __call__(
        self, input_ids: torch.LongTensor,
        scores: torch.FloatTensor
        ) -> torch.FloatTensor:
        return torch.nan_to_num(scores, nan=0.0, posinf=1e4, neginf=-1e4)

from transformers import StoppingCriteria, StoppingCriteriaList

class StopOnTemplate(StoppingCriteria):
    def __init__(self, tokenizer):
        self.stop_ids = tokenizer.encode("<|im_start|>user", add_special_tokens=False)

    def __call__(self, input_ids, scores, **kwargs):
        # self.stop_ids is already a list, so only the tensor side needs .tolist()
        return input_ids[0][-len(self.stop_ids):].tolist() == self.stop_ids

from transformers import AutoModelForCausalLM, AutoTokenizer
def generate_response(
    model_path: str,
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = 0.6,
    top_p: float = 0.95,
    greedy: bool = False,          # NEW: lets you A/B sampled vs. greedy decoding
) -> str:
    # FIXED BUG: this used to hardcode cfg.dpo_output_dir instead of using the
    # model_path argument, which meant calling generate_response(sft_path, ...)
    # silently still generated from the DPO checkpoint. It now respects
    # whichever path is actually passed in.
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
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
            do_sample=not greedy,
            temperature=temperature if not greedy else None,
            top_p=top_p if not greedy else None,
            repetition_penalty=1.3,        # NEW
            no_repeat_ngram_size=3,         # NEW
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            logits_processor=[SafeLogitsProcessor()],
            stopping_criteria=StoppingCriteriaList([StopOnTemplate(tokenizer)]),
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

test_prompts = [
    "What is newton's first law?",
    "What is polymorphism in programming?",
]

# NEW: compare SFT-only vs. DPO checkpoint, greedy vs. sampled, side by side.
for prompt in test_prompts:
    print(f"\nQ: {prompt}")
    print(f"  SFT only  | greedy:  {generate_response(sft_path, prompt, greedy=True)}")
    print(f"  SFT only  | sampled: {generate_response(sft_path, prompt, greedy=False)}")
    print(f"  DPO final | greedy:  {generate_response(dpo_path, prompt, greedy=True)}")
    print(f"  DPO final | sampled: {generate_response(dpo_path, prompt, greedy=False)}")
print("═" * 60)
