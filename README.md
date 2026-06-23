# Direct Preference Optimization (DPO)

## Purpose

This project explores how **Direct Preference Optimization (DPO)** can be used to modify the behavior of a language model through preference learning.

Modern LLMs are already capable of producing different styles of responses when given instructions. However, prompting alone does not always guarantee consistent behavior, especially when multiple goals conflict (for example: being creative while remaining factual, or following a style while preserving quality).

Instead of explicitly instructing the model every time, preference optimization learns these behaviors from comparisons between preferred and rejected outputs.

## Experiment

The project compares:

- **Base model**
  - Original model without alignment

- **Instruction-only model**
  - Model prompted to follow the target style

- **DPO-aligned model**
  - Model trained using preference pairs

## Why DPO?

A simple instruction such as:

> "Speak like a pirate"

can work for individual conversations, but it does not teach the model a persistent preference. DPO provides a way to encode preferences by showing the model examples of better and worse outputs.

Example:

**Prompt:**

```
Explain how a computer works.
```

**Preferred response:**

```
Arrr, a computer be a machine that follows commands...
```

**Rejected response:**

```
Computers are just magic boxes that think like humans.
```

The model learns which type of answer should be preferred.

## Dataset

The preference dataset consists of:

- User prompts
- Chosen responses
- Rejected responses

Each example represents a preference:

```
(prompt, chosen, rejected)
```

The chosen response demonstrates the desired behavior, while the rejected response represents a less preferred alternative.

## Documentation

For the theory and mathematics behind DPO see:

**[Theory and Mathematical Background](docs/README.md)**
