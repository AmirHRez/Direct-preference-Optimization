# Experiment Logs

## v1 - Fantasy Persona

At first my idea was to create a model with the personality of a highborn noble: refined, educated, slightly rude, and speaking to a lower-class user. The dataset was generated using an instruction-tuned LLM.

### Dataset

- Base dataset: Custom QA dataset
- Format:

```json
{
  "prompt": "...",
  "chosen": "...",
  "rejected": "..."
}
```

| Parameter         | Value                  |
| ----------------- | ---------------------- |
| Translation model | qwen2.5-7b-instruct-1m |
| Temperature       | 0.85                   |
| top_p             | 1 (Default)            |
| Dataset size      | 1862                   |

### Training Setup

| Parameter             | Value                               |
| --------------------- | ----------------------------------- |
| Base model            | HuggingFaceTB/SmolLM2-135M-Instruct |
| DPO Learning rate     | 5e-6                                |
| SFT Learning rate     | 2e-5                                |
| DPO Batch size        | 1                                   |
| SFT Batch size        | 2                                   |
| Gradient accumulation | 4                                   |
| SFT epochs            | 3                                   |
| DPO epochs            | 3                                   |
| DPO beta              | 0.1                                 |
| Max sequence length   | 256                                 |

### Result

The model learned some basic Early Modern English patterns, but the personality instructions caused some funny behavior.

Instead of learning the writing style, the model started generating a full fictional persona. Responses often drifted into medieval/fantasy roleplay, inventing identities and speaking as if it was a noble character.

Model available under `MongrelIntruder/schizo-lm`

## v2 - In Progress

This time I added major changes to the dataset, including:

- Removed the personality
- Added length ratio for later analysis
- Tweaked some translation parameters and instructions

### Dataset

The final v2 dataset has the following format:

```json
{
  "prompt": "...",
  "chosen": "...",
  "rejected": "...",
  "len_ratio": "..."
}
```

Temperature was dropped to 0.6 for less divergence and `top_p` was reduced to 0.9

| Parameter         | Value                  |
| ----------------- | ---------------------- |
| Translation model | qwen2.5-7b-instruct-1m |
| Temperature       | 0.6                    |
| top_p             | 0.9 (Default)          |
| Dataset size      | 1862                   |
| Max Length Ratio  | 2                      |

Also added a `max_len_ratio` parameter to keep the model from generating very long answers
