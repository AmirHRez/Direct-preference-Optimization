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

## v2 - Overfitted Model

After the fantasy-persona drift, the goal was: drop the noble character framing entirely and just target the writing style.

### Dataset

The dataset was regenerated as plain prompt/chosen/rejected pairs, with a length-ratio field added to filter out translations that drifted too far from the original response length.
I used Alpaca dataset and ran it through Gemini's API

- Format:

```json
{
  "prompt": "...",
  "chosen": "...",
  "rejected": "...",
  "len_ratio": 0.667
}
```

| Parameter         | Value                 |
| ----------------- | --------------------- |
| Translation model | gemini-3.1-flash-lite |
| Dataset size      | 3500                  |

### Training Setup

| Parameter             | Value                               |
| --------------------- | ----------------------------------- |
| Base model            | HuggingFaceTB/SmolLM2-135M-Instruct |
| DPO Learning rate     | 2e-5                                |
| SFT Learning rate     | 5e-5                                |
| DPO Batch size        | 1                                   |
| SFT Batch size        | 2                                   |
| Gradient accumulation | 4                                   |
| SFT epochs            | 3                                   |
| DPO epochs            | 3                                   |
| DPO beta              | 0.1                                 |
| Max sequence length   | 512                                 |

### Result

Dropping the persona framing removed the fantasy-roleplay drift from v1, but a new problem: **style collapse without semantic grounding**. The model learned to emit archaic-sounding tokens fluently, but stopped answering the actual question.

```
Q: What is polymorphism in programming?
A: As thy sage tongue doth commend, I call this manner unto thee; wherein I sowe, wherein manner doth shift, thou dost flourish. Verily, this art doth doth array thy sway, wherein many a tale doth unfold. Thou seeker, art thou besotted with the craft of this grandewe. Thy footsteps dost oft be natur’d, wherein doth thy hand doth bend. Yet, lest thy hand falwe, doth thou seek to ply thy skill.

For earth doth bear many a tale, wherein oft doth grow, wherein many a man doth learn. Behold, these fair faces doth smile, wherein doth shine thy prowess. The sun
```
