# DPO — Theory and Mathematics

> A self-contained reference covering the problem setup, the full derivation, the loss function, and the gradient intuition behind **Direct Preference Optimization** (Rafailov et al., 2023).

---

## Table of Contents

1. [Background: RLHF and Its Limitations](#1-background-rlhf-and-its-limitations)
2. [Problem Setup](#2-problem-setup)
3. [The Preference Model](#3-the-preference-model)
4. [The RLHF Objective](#4-the-rlhf-objective)
5. [The Key Insight: Optimal Policy in Closed Form](#5-the-key-insight-optimal-policy-in-closed-form)
6. [Extracting the Reward from the Policy](#6-extracting-the-reward-from-the-policy)
7. [Substituting into the Preference Model](#7-substituting-into-the-preference-model)
8. [The DPO Loss](#8-the-dpo-loss)
9. [Computing Log-Probabilities in Practice](#9-computing-log-probabilities-in-practice)
10. [Understanding β](#10-understanding-β)
11. [The Implicit Reward](#11-the-implicit-reward)
12. [Gradient Analysis](#12-gradient-analysis)
13. [DPO vs RLHF: A Comparison](#13-dpo-vs-rlhf-a-comparison)
14. [Notable Variants](#14-notable-variants)
15. [References](#15-references)

---

## 1. Background: RLHF and Its Limitations

Before DPO, aligning language models with human preferences required a three-stage pipeline known as **Reinforcement Learning from Human Feedback (RLHF)**:

1. **Supervised Fine-Tuning (SFT).** Fine-tune a pretrained LLM on a curated demonstration dataset to produce a reference policy $\pi_\text{ref}$.

2. **Reward Modelling.** Collect human preference annotations in the form of pairs $(y_w, y_l)$ where $y_w$ is preferred over $y_l$ given a prompt $x$. Train a scalar reward model $r_\phi(x, y)$ to predict these preferences.

3. **RL Optimisation.** Use PPO (Proximal Policy Optimisation) to maximise the learned reward while penalising the policy from drifting too far from $\pi_\text{ref}$ via a KL constraint.

### Why RLHF is painful

- **Instability.** PPO requires careful tuning of clipping, value functions, and GAE. Small hyperparameter errors lead to reward hacking or collapsed training.
- **Memory cost.** At minimum you must hold four model copies simultaneously: the policy, the reference, the reward model, and the value function head.
- **Reward overoptimisation.** The policy eventually exploits blind spots in the reward model, producing responses with high $r_\phi$ but low actual quality.
- **Two-stage training.** The reward model and policy are trained separately. Errors in the reward model propagate invisibly.

**DPO eliminates stages 2 and 3 entirely.** No reward model is trained, no RL loop is run, and the whole pipeline collapses to a single supervised-style loss over preference pairs.

---

## 2. Problem Setup

### Notation

| Symbol                     | Meaning                                          |
| -------------------------- | ------------------------------------------------ |
| $x$                        | Input prompt                                     |
| $y$                        | Model-generated response                         |
| $\pi_\theta(y \mid x)$     | Policy being trained (parameterised by $\theta$) |
| $\pi_\text{ref}(y \mid x)$ | Frozen reference policy (the SFT model)          |
| $r^*(x, y)$                | Unknown true reward function                     |
| $y_w$                      | Preferred ("winner") response in a pair          |
| $y_l$                      | Rejected ("loser") response in a pair            |
| $\mathcal{D}$              | Dataset of triples $(x, y_w, y_l)$               |
| $\beta$                    | KL penalty coefficient                           |

### Data format

Every training example is a triple:

```
(prompt x,  chosen response y_w,  rejected response y_l)
```

The preference is asymmetric: annotators were shown both responses and marked one as better. We do not assume we know _by how much_ — only the direction.

---

## 3. The Preference Model

DPO inherits the **Bradley-Terry model** of pairwise preferences. Given a latent reward function $r^*$, the probability that a human prefers $y_w$ over $y_l$ given prompt $x$ is:

$$p^*(y_w \succ y_l \mid x) = \sigma\!\left(r^*(x, y_w) - r^*(x, y_l)\right)$$

where $\sigma$ is the sigmoid function. This model says: the more the reward gap, the more confident the preference. The negative log-likelihood of this model over a dataset $\mathcal{D}$ is the standard reward model training loss in RLHF:

$$\mathcal{L}_\text{RM}(r_\phi) = -\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}}\!\left[\log \sigma\!\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)\right]$$

In RLHF you minimise this loss to get $r_\phi$, then hand it to PPO. DPO never does this step — instead it re-parameterises $r^*$ directly in terms of $\pi_\theta$.

---

## 4. The RLHF Objective

The standard RLHF objective is:

$$\max_{\pi_\theta} \;\mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(y \mid x)}\!\left[r(x, y)\right] - \beta\, \mathbb{KL}\!\left[\pi_\theta(y \mid x) \;\|\; \pi_\text{ref}(y \mid x)\right]$$

Expanding the KL term:

$$\max_{\pi_\theta} \;\mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta}\!\left[r(x, y) - \beta \log \frac{\pi_\theta(y \mid x)}{\pi_\text{ref}(y \mid x)}\right]$$

The KL penalty achieves two things:

- Prevents the policy from drifting so far from $\pi_\text{ref}$ that it produces incoherent text.
- Acts as regularisation, discouraging extreme reward hacking.

The scalar $\beta$ controls the trade-off: small $\beta$ allows large deviation (more aggressive alignment), large $\beta$ keeps the policy close to the SFT model (safer but less aligned).

---

## 5. The Key Insight: Optimal Policy in Closed Form

The RLHF objective (Section 4) is a KL-constrained expected-reward maximisation. For a fixed reward function $r$, this has an **analytical solution**. We can verify it by writing the optimal policy as:

$$\boxed{\pi^*(y \mid x) = \frac{1}{Z(x)}\, \pi_\text{ref}(y \mid x) \exp\!\left(\frac{1}{\beta}\, r(x, y)\right)}$$

where $Z(x)$ is the partition function (a normalisation constant over all possible responses):

$$Z(x) = \sum_y \pi_\text{ref}(y \mid x) \exp\!\left(\frac{1}{\beta}\, r(x, y)\right)$$

**Verification.** Substituting $\pi^*$ back into the RLHF objective and computing the KL term:

$$\log \frac{\pi^*(y \mid x)}{\pi_\text{ref}(y \mid x)} = \frac{1}{\beta} r(x,y) - \log Z(x)$$

So $r(x,y) - \beta \log \frac{\pi^*}{\pi_\text{ref}} = \beta \log Z(x)$, which is a constant with respect to $y$. Any other policy would have a larger KL without gaining in expected reward — confirming $\pi^*$ is the global optimum.

> **Intuition.** The optimal policy is the reference policy _reweighted_ by the exponentiated reward. High-reward responses get upweighted; low-reward responses get downweighted. The temperature $1/\beta$ controls how sharply this reweighting happens.

---

## 6. Extracting the Reward from the Policy

Now we reverse the direction. Instead of deriving $\pi^*$ from $r$, we express $r$ in terms of $\pi^*$.

Taking the log of $\pi^*(y \mid x)$ and rearranging:

$$\log \pi^*(y \mid x) = \log \pi_\text{ref}(y \mid x) + \frac{1}{\beta} r(x, y) - \log Z(x)$$

Solving for $r(x, y)$:

$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_\text{ref}(y \mid x)} + \beta \log Z(x)$$

This equation says: **the true reward is proportional to how much more (or less) the optimal policy assigns probability to $y$ compared to the reference.** The partition function $\log Z(x)$ is an additive prompt-dependent constant.

---

## 7. Substituting into the Preference Model

We now plug the reward expression from Section 6 into the Bradley-Terry model from Section 3.

$$p^*(y_w \succ y_l \mid x) = \sigma\!\left(r^*(x, y_w) - r^*(x, y_l)\right)$$

Substituting:

$$= \sigma\!\left(\left[\beta \log \frac{\pi^*(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)} + \beta \log Z(x)\right] - \left[\beta \log \frac{\pi^*(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)} + \beta \log Z(x)\right]\right)$$

The partition function terms $\beta \log Z(x)$ **cancel exactly** (they depend only on $x$, not on which response we are comparing):

$$\boxed{p^*(y_w \succ y_l \mid x) = \sigma\!\left(\beta \log \frac{\pi^*(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)}\right)}$$

This is the critical step. The intractable partition function has vanished. The preference probability is now expressed entirely in terms of the ratio between the optimal policy and the reference — and the reward model is never needed.

---

## 8. The DPO Loss

We now have a parameterised expression for the preference probability (Section 7) where $\pi^*$ plays the role of the model we are learning. Replacing $\pi^*$ with our parameterised policy $\pi_\theta$ and taking the negative log-likelihood over the dataset gives the **DPO training loss**:

$$\boxed{\mathcal{L}_\text{DPO}(\pi_\theta;\, \pi_\text{ref}) = -\mathbb{E}_{(x,\, y_w,\, y_l)\,\sim\,\mathcal{D}}\!\left[\log \sigma\!\left(\beta \underbrace{\log \frac{\pi_\theta(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)}}_{\text{log ratio on chosen}} - \beta \underbrace{\log \frac{\pi_\theta(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)}}_{\text{log ratio on rejected}}\right)\right]}$$

Minimising this loss pushes $\pi_\theta$ to assign **relatively higher probability** to $y_w$ than $\pi_\text{ref}$ does, and **relatively lower probability** to $y_l$ than $\pi_\text{ref}$ does.

### Equivalent rewrite

Because $\log(a/b) = \log a - \log b$, each log ratio expands to:

$$\log \frac{\pi_\theta(y \mid x)}{\pi_\text{ref}(y \mid x)} = \log \pi_\theta(y \mid x) - \log \pi_\text{ref}(y \mid x)$$

So in code, you compute four scalar quantities per example and combine them:

```python
import torch.nn.functional as F

def dpo_loss(pi_logp_w, pi_logp_l, ref_logp_w, ref_logp_l, beta=0.1):
    """
    All inputs are per-example scalar log-probabilities (summed over response tokens).
    pi_logp_w  : log π_θ(y_w | x)
    pi_logp_l  : log π_θ(y_l | x)
    ref_logp_w : log π_ref(y_w | x)
    ref_logp_l : log π_ref(y_l | x)
    """
    log_ratio_w = pi_logp_w - ref_logp_w   # log( π_θ(y_w|x) / π_ref(y_w|x) )
    log_ratio_l = pi_logp_l - ref_logp_l   # log( π_θ(y_l|x) / π_ref(y_l|x) )
    loss = -F.logsigmoid(beta * (log_ratio_w - log_ratio_l))
    return loss.mean()
```

---

## 9. Computing Log-Probabilities in Practice

A language model outputs a distribution over the next token at each position. For a sequence $y = (y_1, y_2, \ldots, y_T)$ conditioned on prompt $x$, the log-probability is the sum of per-token log-probabilities by the chain rule:

$$\log \pi_\theta(y \mid x) = \sum_{t=1}^{T} \log \pi_\theta\!\left(y_t \mid x, y_{<t}\right)$$

In PyTorch, you get this by passing the full concatenated sequence `[x; y]` through the model, reading off the logits shifted by one position (to align inputs with targets), then gathering the log-softmax values at the actual token IDs — **but only summing over the response tokens, not the prompt tokens**:

```python
def sequence_log_prob(model, input_ids, response_mask):
    """
    input_ids     : (B, L)  — concatenated [prompt; response] tokens
    response_mask : (B, L)  — 1 for response tokens, 0 for prompt tokens
    Returns       : (B,)    — scalar log-prob per example
    """
    with torch.no_grad():  # use no_grad for the frozen reference model
        logits = model(input_ids).logits           # (B, L, V)

    # Shift: logits[t] predicts token at position t+1
    shift_logits = logits[:, :-1, :]              # (B, L-1, V)
    shift_labels = input_ids[:, 1:]               # (B, L-1)
    shift_mask   = response_mask[:, 1:]           # (B, L-1)

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_lps = log_probs.gather(
        dim=2,
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)                                  # (B, L-1)

    return (token_lps * shift_mask).sum(dim=-1)   # (B,)
```

**Why only sum over response tokens?** The prompt $x$ is fixed and identical across both $y_w$ and $y_l$. Its contribution to the log-ratio would cancel anyway, but more importantly, including it would dilute the gradient signal: the model's uncertainty is concentrated in the response, not the given prompt.

---

## 10. Understanding β

The hyperparameter $\beta$ is the **inverse temperature** of the KL penalty. It directly controls how far the trained policy is allowed to deviate from $\pi_\text{ref}$:

| $\beta$ value              | Effect                                                                             |
| -------------------------- | ---------------------------------------------------------------------------------- |
| Large ($\beta \to \infty$) | Heavy penalty on KL divergence; policy stays very close to $\pi_\text{ref}$        |
| Small ($\beta \to 0$)      | Weak KL penalty; policy can drift far from $\pi_\text{ref}$ to maximise preference |
| $\beta = 0$                | No regularisation; policy collapses to greedy argmax of preference objective       |

In the DPO loss, $\beta$ scales the argument to the sigmoid:

$$\mathcal{L}_\text{DPO} = -\log \sigma\!\left(\beta \cdot \Delta\right), \qquad \Delta = \log \frac{\pi_\theta(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)}$$

When $\Delta > 0$ (model already prefers the chosen response), a larger $\beta$ pushes $\sigma(\beta\Delta)$ closer to 1 faster, resulting in smaller loss — i.e., less gradient on already-correct examples. When $\Delta < 0$ (model incorrectly prefers the rejected response), larger $\beta$ makes the loss larger and the gradient signal stronger.

**Practical range.** Typical values are $\beta \in [0.05, 0.5]$. A common starting point is $\beta = 0.1$.

---

## 11. The Implicit Reward

One of the theoretical contributions of DPO is revealing that **the policy itself encodes an implicit reward model**. Define:

$$\hat{r}_\theta(x, y) \;:=\; \beta \log \frac{\pi_\theta(y \mid x)}{\pi_\text{ref}(y \mid x)}$$

This is the **log density ratio** between the trained policy and the reference, scaled by $\beta$. It is the quantity that the DPO loss directly optimises.

You can use $\hat{r}_\theta$ at inference time as a reward signal — for instance, to score candidate responses in a best-of-$N$ sampling scheme or to evaluate the quality of a response without running a separate reward model.

The preference model written in terms of the implicit reward is:

$$p_\theta(y_w \succ y_l \mid x) = \sigma\!\left(\hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l)\right)$$

---

## 12. Gradient Analysis

Differentiating $\mathcal{L}_\text{DPO}$ with respect to $\theta$ reveals the mechanism of learning:

$$\nabla_\theta \mathcal{L}_\text{DPO} = -\beta\, \mathbb{E}_{(x,y_w,y_l)}\!\left[ \underbrace{\sigma\!\left(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w)\right)}_{\text{weighting term}} \cdot \underbrace{\left(\nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x)\right)}_{\text{policy gradient direction}}\right]$$

### Interpreting the two terms

**Policy gradient direction.** The term $\nabla_\theta \log \pi_\theta(y_w) - \nabla_\theta \log \pi_\theta(y_l)$ increases the log-probability of the chosen response and decreases the log-probability of the rejected response. This is the desired direction.

**Weighting term.** $\sigma(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w))$ is the probability that the _current model incorrectly prefers_ the rejected response. When the model already correctly ranks the pair ($\hat{r}(y_w) \gg \hat{r}(y_l)$), this weight is close to zero and the example contributes little gradient. When the model is wrong ($\hat{r}(y_l) > \hat{r}(y_w)$), this weight is close to 1 and the gradient is large.

> **Intuition.** DPO automatically focuses training on the pairs where the model is most confused. Easy examples (already ranked correctly) are down-weighted; hard examples (ranked incorrectly) drive the updates. This is analogous to hard-example mining in metric learning.

---

## 13. DPO vs RLHF: A Comparison

| Property                | RLHF (PPO)                            | DPO                                    |
| ----------------------- | ------------------------------------- | -------------------------------------- |
| Reward model            | Explicitly trained                    | Not needed — implicit in $\pi_\theta$  |
| RL algorithm            | PPO (complex)                         | Supervised loss (simple)               |
| Models in memory        | Policy + ref + reward + value         | Policy + ref (2 models)                |
| Training stability      | Brittle; sensitive to PPO hyperparams | Stable; standard Adam + CrossEntropy   |
| Reward overoptimisation | Common                                | Less common (KL is built into loss)    |
| Theoretical guarantee   | Approximate (RL approximations)       | Exact optimum of the RLHF objective    |
| Empirical performance   | Strong when tuned                     | Competitive; sometimes slightly weaker |

**DPO is not strictly better** — it trades off RL flexibility for simplicity and stability. On very long-horizon tasks where exploration matters, PPO can outperform DPO. But for single-turn instruction following and chat alignment, DPO typically matches PPO at a fraction of the complexity.

---

## 14. Notable Variants

The DPO paper sparked a family of related methods:

### IPO — Identity Preference Optimisation

_(Azar et al., 2023)_

Replaces the log-sigmoid with a squared loss to avoid overfitting on deterministic preferences:

$$\mathcal{L}_\text{IPO} = \mathbb{E}\!\left[\left(\log \frac{\pi_\theta(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)} - \frac{1}{2\beta}\right)^2\right]$$

Useful when your preference data is very clean (near-deterministic), where standard DPO tends to over-optimise.

### SimPO — Simple Preference Optimisation

_(Meng et al., 2024)_

Removes the reference model entirely by using length-normalised log-probability as the reward and introducing a margin $\gamma$:

$$\mathcal{L}_\text{SimPO} = -\mathbb{E}\!\left[\log \sigma\!\left(\frac{\beta}{|y_w|}\log \pi_\theta(y_w \mid x) - \frac{\beta}{|y_l|}\log \pi_\theta(y_l \mid x) - \gamma\right)\right]$$

This makes training cheaper (no reference forward pass needed) and avoids distribution shift between the reference and the policy.

### cDPO — Conservative DPO

_(Mitchell et al., 2023)_

Softens the hard $\{0, 1\}$ preference labels to account for annotator noise, using a label smoothing approach:

$$\mathcal{L}_\text{cDPO} = -(1-\epsilon)\, \mathbb{E}\!\left[\log p_\theta(y_w \succ y_l)\right] - \epsilon\, \mathbb{E}\!\left[\log p_\theta(y_l \succ y_w)\right]$$

where $\epsilon \in (0, 0.5)$ is the noise probability. Useful when preference labels are unreliable or crowdsourced.

### RAFT / SLiC

Earlier methods with similar flavours — also worth reading for historical context.

---

## 15. References

- **Rafailov et al. (2023).** _Direct Preference Optimization: Your Language Model is Secretly a Reward Model._ NeurIPS 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) — **the primary reference**
- **Ouyang et al. (2022).** _Training language models to follow instructions with human feedback (InstructGPT)._ NeurIPS 2022. — Background on RLHF/PPO
- **Azar et al. (2023).** _A General Theoretical Paradigm to Understand Learning from Human Feedback._ [arXiv:2310.12036](https://arxiv.org/abs/2310.12036) — IPO
- **Meng et al. (2024).** _SimPO: Simple Preference Optimization with a Reference-Free Reward._ [arXiv:2405.14734](https://arxiv.org/abs/2405.14734) — SimPO
- **Mitchell et al. (2023).** _An Emulator for Fine-Tuning Large Language Models using Small Language Models._ [arXiv:2310.12962](https://arxiv.org/abs/2310.12962) — cDPO
- **Bradley & Terry (1952).** _Rank Analysis of Incomplete Block Designs._ Biometrika. — The pairwise preference model
