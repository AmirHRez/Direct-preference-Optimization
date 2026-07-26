# DPO Alignment Experiments

This project is an experiment on how **Direct Preference Optimization (DPO)** can be used to teach a language model a writing style and keeping correct and useful answers.

Instead of relying on instructions, the model is trained using preference data to learn which type of response is preferred.

[You can see my development logs here](./docs/LOGS.md)

## Dataset

The dataset used for this project is a manipulated version of Alpaca dataset. 3500 entries are chosen and then ran through Gemini for an archaic translation

All the data (base questions + archaic) is included in the repo so feel free to make your own.

## Result
