# DPO Alignment (Archaic English)

This project shows how **Direct Preference Optimization (DPO)** can be used to teach a language model a consistent writing style while keeping correct and useful answers.

Instead of relying on prompting alone, the model is trained using preference data to learn which type of response is preferred.

[You can see my development logs here](./docs/LOGS.md)

## Dataset

The dataset for this project consists of 1862 plain questions and answers which later was passed through a local model to translate the data to archaic English.

- **Chosen response**: written in archaic (Shakespeare) English
- **Rejected response**: written in normal modern English

Example:

**Prompt:**

```
Explain how a neural network learns.
```

**Chosen:**

```
Verily, a neural network doth learn by adjusting the weights between its connections, such that its predictions grow nearer unto truth with each example it doth observe.
```

**Rejected:**

```
A neural network learns by adjusting its weights using backpropagation and gradient descent.
```

All the data (base questions + archaic) is included in the repo so feel free to make your own, but you might need to change `translate.py` file.

---

## Why DPO instead of prompting

A system prompt like:

> "Answer in Shakespearean style"

can work in simple cases, but it is not always consistent.

DPO is used to:

- make the style more stable
- reduce reliance on prompting
- teach the model a persistent preference
- preserve factual correctness while changing expression
