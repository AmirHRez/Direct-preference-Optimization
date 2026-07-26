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

Instead of learning the writing style, the model started generating a full fictional personality. Responses often drifted into fantasy roleplay, inventing identities and speaking as if it was a noble character.

## v2 - Overfitted Model

After the fantasy-persona, the goal is: drop the noble character framing entirely and just target the writing style.

### Dataset

I used Alpaca dataset and ran it through Gemini's API for translation. The dataset was regenerated as prompt/chosen/rejected pairs, with a length-ratio field added to remove translations that drifted too far from the original length.

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

Dropping the persona framing removed the fantasy-roleplay drift from v1, but a new problem: The model learned to respond with archaic-sounding tokens fluently, but stopped answering the actual question.

```
Q: What is polymorphism in programming?
A: As thy sage tongue doth commend, I call this manner unto thee; wherein I sowe, wherein manner doth shift, thou dost flourish. Verily, this art doth doth array thy sway, wherein many a tale doth unfold. Thou seeker, art thou besotted with the craft of this grandewe. Thy footsteps dost oft be natur’d, wherein doth thy hand doth bend. Yet, lest thy hand falwe, doth thou seek to ply thy skill.

For earth doth bear many a tale, wherein oft doth grow, wherein many a man doth learn. Behold, these fair faces doth smile, wherein doth shine thy prowess. The sun
```

## v3 - Final

After changing DPO's learning rate and beta, the model is at a presentable stage. The data was not changed.

| Parameter             | Value                               |
| --------------------- | ----------------------------------- |
| Base model            | HuggingFaceTB/SmolLM2-135M-Instruct |
| DPO Learning rate     | **_5e-6_**                          |
| SFT Learning rate     | 5e-5                                |
| DPO Batch size        | 1                                   |
| SFT Batch size        | 2                                   |
| Gradient accumulation | 4                                   |
| SFT epochs            | 3                                   |
| DPO epochs            | 3                                   |
| DPO beta              | **_0.4_**                           |
| Max sequence length   | 512                                 |

### Result

```
Q: What is newton's first law?
A: Newton’s First Law of Motion hath been the most profound and enduring enquiry in all science. It declareeth that every object at rest shall remain so until it be compelled to move; then motion doth proceed with an equal force unto its destination: this decree was established long before Newton made his discovery thereof but he did not discover yet what manner thing should govern such motions as these which we call movements or actions.” This fundamental principle...
```
