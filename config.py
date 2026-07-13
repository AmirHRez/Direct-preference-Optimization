from dataclasses import dataclass

SFT_OUTPUT_DIR = "./checkpoints/sft"
DPO_OUTPUT_DIR = "./checkpoints/dpo"
DATA_DIR = "data/archaic/alpaca-gemini-3500.jsonl"

SEED = 42


@dataclass
class PipelineConfig:
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"

    sft_output_dir: str = SFT_OUTPUT_DIR
    dpo_output_dir: str = DPO_OUTPUT_DIR

    max_seq_length: int = 512
    seed: int = 42

    sft_epochs: int = 2
    sft_batch_size: int = 2
    sft_lr: float = 5e-5
    sft_warmup_ratio: float = 0.1
    sft_grad_accum: int = 4

    dpo_epochs: int = 1
    dpo_batch_size: int = 1
    dpo_lr: float = 2e-5
    dpo_beta: float = 0.1
    dpo_grad_accum: int = 4

    data_path = DATA_DIR

    run_sft_only_eval: bool = True


# ----------- Translation -------------- #

TRANSLATOR_BETA = 0.7
TRANSLATOR_TOP_P = 0.93
MAX_LEN_RATIO = 1.35
SHORT_ANSWER_WORD_THRESHOLD = 6
SHORT_ANSWER_MAX_WORDS = 10
 

SYSTEM_PROMPT = """
Respond only in English. Do not use any other language under any circumstances.

You are rewriting modern English answers into Early Modern English style for a preference dataset.

WHAT TO CHANGE:
- Verb forms: use doth, hath, dost, art, wilt, wouldst, etc.
- Pronouns: thee, thou, thy, thine where natural
- Vocabulary: use established archaic words where they exist

REQUIRED: Your output must contain at least one clear archaic feature (verb form, pronoun, or vocabulary substitution) from the list above. A rewrite with no archaic markers is a failed rewrite — do not return the input unchanged or near-unchanged.

WHAT NOT TO CHANGE:
- Do not invent spellings. Early Modern English used consistent spelling — do not mutate modern words into fake-archaic forms
- Do not alter technical terms, scientific names, chemical formulas, numbers, or dates — keep them exactly as written
- Do not substitute any term with an approximation if no real archaic equivalent exists — keep the modern word verbatim
- Do not change the meaning. If you cannot preserve the meaning, keep the original phrasing unchanged, but still apply at least one archaic verb form or pronoun if grammatically possible

LENGTH:
- Your output must not exceed the original word count by more than 15%
- For answers of 6 words or fewer, change verb forms only — do not expand

OUTPUT:
- Return only the rewritten answer
- No greetings, titles, persona, or preamble
"""

SYSTEM_PROMPT_V1 = """
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