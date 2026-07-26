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
    dpo_lr: float = 5e-6
    dpo_beta: float = 0.4
    dpo_grad_accum: int = 4

    data_path = DATA_DIR

    run_sft_only_eval: bool = True
