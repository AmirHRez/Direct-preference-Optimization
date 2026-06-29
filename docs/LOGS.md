# Experiment Logs

## v1 - Early Modern Persona DPO

The initial idea was to create a model with the personality of a highborn noble: refined, educated, slightly condescending, and speaking to a lower-class user. The dataset was generated using an instruction-tuned LLM.

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

- `chosen`: rewritten response in Early Modern / Shakespearean style
- `rejected`: original modern response

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

The model learned some surface-level Early Modern English patterns, but the personality conditioning caused unexpected behavior.

Instead of only adopting the desired writing style, the model started generating a full fictional persona. Responses often shifted into medieval/fantasy roleplay, inventing identities and speaking as if it was a noble character.

Observed failure patterns:

- Introducing itself with fictional identities
- Creating unnecessary backstory
- Producing dramatic monologues instead of direct answers
- Prioritizing character consistency over answering the question

Model available under `MongrelIntruder/schizo-lm`
