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

I tested different models for the data generation (100 samples) and here is the result:

| Metric                                   | Llama3.2-3B | Llama3-8B | Qwen3-4B  |
| ---------------------------------------- | ----------- | --------- | --------- |
| Identical chosen==rejected (dead pairs)  | 7.8%        | **6.2%**  | 15.2%     |
| Avg len_ratio (chosen/rejected)          | 1.73        | 1.36      | **1.13**  |
| Avg char similarity (chosen vs rejected) | **0.61**    | 0.61–0.73 | 0.90      |
| Archaic markers per chosen               | 0.91        | **0.96**  | 0.64      |
| % chosen with zero archaic markers       | 45.6%       | **38.0%** | 43.4%     |
| Vocab diversity (TTR)                    | 0.346       | 0.345     | **0.397** |

The results were no where near pleasing so I modified the generation method and added more strict instructions.

The generated v2 dataset has the following format:

```json
{
  "prompt": "...",
  "chosen": "...",
  "rejected": "...",
  "len_ratio": "...",
  "flags": [],
  "needs_review": true //or false
}
```

Temperature was dropped to 0.7 for less divergence and `top_p` was reduced to 0.93

| Parameter         | Value     |
| ----------------- | --------- |
| Translation model | Llama3-8B |
| Temperature       | 0.7       |
| top_p             | 0.93      |
| Dataset size      | 1862      |
| Max Length Ratio  | 1.35      |

Also added a `max_len_ratio` parameter to keep the model from generating very long answers

Then I use an AI generated review tool (by claude) to inspect entries where `"needs_review" = true`
