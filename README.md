# DPO Style Alignment Experiment (Archaic English)

## Purpose

This project explores how **Direct Preference Optimization (DPO)** can be used to teach a language model a consistent writing style while preserving correct and useful answers.

Instead of relying on prompting alone, the model is trained using preference data to learn which type of response is preferred.

In this project, the goal is to align the model toward a **Shakespearean / Early Modern English style** while keeping the underlying information accurate.

---

I trained a language model using preference pairs:

- **Chosen response** → written in archaic (Shakespeare-like) English
- **Rejected response** → written in normal modern English

Both responses contain the same meaning. The only difference is style.

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

---

## Dataset

The dataset contains preference pairs:

```
(prompt, chosen, rejected)
```

- Prompt: a question or instruction
- Chosen: archaic English answer
- Rejected: normal English answer

The data covers different topics:

- science explanations
- programming concepts
- history
- AI and machine learning
- everyday reasoning

---

## Goal of the experiment

The main questions are:

- Can DPO reliably teach a consistent writing style?
- Does the model keep factual correctness while changing style?
- How well does style preference generalize to unseen prompts?

---

## Documentation

For the theory and mathematics behind DPO see:

**[Theory and Mathematical Background](docs/README.md)**
