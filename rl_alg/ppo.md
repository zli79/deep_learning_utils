# Proximal Policy Optimization (PPO)

PPO constrains how much the policy can change in a single update, preventing the large destabilizing steps that plagued earlier policy-gradient methods.

The total loss combines three terms:

\[
\mathcal{L} = \mathcal{L}^{\text{policy}} + c_v \, \mathcal{L}^{\text{value}} + c_{\text{kl}} \, \mathcal{L}^{\text{KL}}
\]

---

## 1. Policy Loss (Clipped Surrogate Objective)

\[
\mathcal{L}^{\text{policy}} = -\mathbb{E}_t \Bigl[ \min\!\bigl( r_t \hat{A}_t,\; \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon)\,\hat{A}_t \bigr) \Bigr]
\]

where the probability ratio is:

\[
r_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)} = \exp(\log\pi_\theta - \log\pi_{\theta_\text{old}})
\]

| Symbol | Meaning | Code |
|--------|---------|------|
| \(r_t\) | Importance-sampling ratio between new and old policy | `ratio = torch.exp(logp_new - logp_old)` |
| \(\hat{A}_t\) | Advantage estimate (how much better action \(a_t\) is vs baseline) | `advantages` |
| \(\varepsilon\) | Clip range — limits how far the ratio can deviate from 1 | `clip_eps` (e.g. 0.2) |
| unclipped | Standard policy gradient objective | `unclipped = ratio * advantages` |
| clipped | Policy gradient with ratio clamped to \([1-\varepsilon, 1+\varepsilon]\) | `clipped = clamp(ratio, ...) * advantages` |

Taking the `min` of both terms is the key PPO insight: when the advantage is positive we don't want the ratio to grow too large (capped at \(1+\varepsilon\)); when it is negative we don't want it to shrink too much (floored at \(1-\varepsilon\)). The negative sign turns the maximization into a minimization.

```python
log_ratio   = logp_new - logp_old
ratio       = torch.exp(log_ratio)
unclipped   = ratio * advantages
clipped     = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
policy_loss = -torch.min(unclipped, clipped)
policy_loss = (policy_loss * mask).sum() / mask.sum()
```

---

## 2. Value Loss

\[
\mathcal{L}^{\text{value}} = \mathbb{E}_t \bigl[ (V_\theta(s_t) - \hat{R}_t)^2 \bigr]
\]

| Symbol | Meaning | Code |
|--------|---------|------|
| \(V_\theta(s_t)\) | Predicted state value from the critic | `values` |
| \(\hat{R}_t\) | Discounted return target (computed from rollout) | `returns` |
| \(c_v\) | Value loss coefficient, scales its contribution to total loss | `vf_coef` |

A simple mean-squared-error regression that trains the value head to predict future returns accurately. Better value estimates → better advantage estimates → better policy updates.

```python
value_loss = (returns - values) ** 2
value_loss = (value_loss * mask).sum() / mask.sum()
```

---

## 3. KL Penalty

\[
\mathcal{L}^{\text{KL}} = \mathbb{E}_t \bigl[ \log\pi_\theta(a_t \mid s_t) - \log\pi_{\text{ref}}(a_t \mid s_t) \bigr]
\]

| Symbol | Meaning | Code |
|--------|---------|------|
| \(\pi_\theta\) | Current (new) policy being trained | `logp_new` |
| \(\pi_{\text{ref}}\) | Reference policy (e.g. pre-trained / SFT model in RLHF) | `logp_ref` |
| \(c_{\text{kl}}\) | KL coefficient, controls how tightly we stay near the reference | `kl_coef` |

This is the forward KL divergence \(D_\text{KL}(\pi_\theta \| \pi_\text{ref})\) approximated per-token. It regularizes the policy to not drift too far from the reference model — especially important in RLHF where \(\pi_\text{ref}\) is the supervised fine-tuned (SFT) checkpoint.

```python
kl = logp_new - logp_ref
kl = (kl * mask).sum() / mask.sum()
```

---

## 4. Total Loss

\[
\mathcal{L} = \mathcal{L}^{\text{policy}} + c_v \, \mathcal{L}^{\text{value}} + c_{\text{kl}} \, \mathcal{L}^{\text{KL}}
\]

```python
total_loss = policy_loss + vf_coef * value_loss + kl_coef * kl
```

---

## 5. Advantage Normalization

Before computing the policy loss, advantages are normalized over valid (non-padded) tokens:

\[
\hat{A}_t \leftarrow \frac{\hat{A}_t - \mu_A}{\sigma_A + \epsilon}
\]

```python
adv_mean   = (advantages * mask).sum() / mask.sum()
adv_var    = ((advantages - adv_mean) ** 2 * mask).sum() / mask.sum()
advantages = (advantages - adv_mean) / (adv_var.sqrt() + 1e-8)
```

This keeps the policy gradient signal on a consistent scale regardless of reward magnitude, stabilizing training.

---

## Hyperparameters

| Parameter | Typical Value | Role |
|-----------|--------------|------|
| `clip_eps` | 0.1 – 0.2 | Controls policy update trust region |
| `vf_coef` | 0.5 | Balances value loss vs policy loss |
| `kl_coef` | 0.01 – 0.1 | Strength of reference policy regularization |
| `normalize_adv` | `True` | Stabilizes gradient scale |
