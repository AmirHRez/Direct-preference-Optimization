# DPO Alignment Experiments

This project is an experiment on how **Direct Preference Optimization (DPO)** can be used to teach a language model a writing style and keeping correct and useful answers.

Instead of relying on instructions, the model is trained using preference data to learn which type of response is preferred.

You can see my development logs [here](./docs/LOGS.md)
and you can access the model from HuggingFace [MongrelIntruder/ArchaicLM](https://huggingface.co/MongrelIntruder/ArchaicLM).

## How it works

It has two stages:

1. **SFT (Supervised Fine-Tuning)**: the base model is trained on prompt/response pairs to learn the style.
2. **DPO (Direct Preference Optimization)**: _the SFT model_ is trained on chosen/rejected response pairs, learning to prefer the styled response.

Base model: [`HuggingFaceTB/SmolLM2-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)

## Dataset

The dataset used for this project is a manipulated version of Alpaca dataset. 3500 entries are chosen and then ran through Gemini for an archaic translation

All the data (base questions + archaic) is included in the repo so feel free to make your own.

## Result

```
Q: What is newton's first law?
A: Newton’s First Law of Motion hath been the most profound and enduring enquiry in all science. It declareeth that every object at rest shall remain so until it be compelled to move; then motion doth proceed with an equal force unto its destination: this decree was established long before Newton made his discovery thereof but he did not discover yet what manner thing should govern such motions as these which we call movements or actions.” This fundamental principle...
```
