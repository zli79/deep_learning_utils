"""
Concrete toy PPO pipeline using text prompts and responses.

  Prompt:   sequence of input token IDs         [B, prompt_len]
  Response: sequence of generated token IDs     [B, T]

  State at step t : mean-pool embedding of (prompt + response[:t])  [B, embed_dim]
  Action at step t: next token sampled from policy vocab             [B], int in [0, vocab_size)

Models
  PolicyModel    – state embedding → token logits  (trained)
  ReferenceModel – frozen copy of initial policy for KL penalty
  RewardModel    – full (prompt+response) tokens → scalar reward  (frozen)
  ValueModel     – state embedding → scalar value  (trained)

Note: state embeddings are pre-computed during rollout and stored; gradients
during the PPO update flow through the policy/value heads using these stored
embeddings. This is a deliberate simplification to keep the code readable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rl_alg"))
from ppo import ppo_losses


# ---------------------------------------------------------------------------
# Tiny vocabulary
# ---------------------------------------------------------------------------

VOCAB = [
    "<pad>", "<eos>",
    "the", "cat", "sat", "on", "mat",
    "dog", "ran", "fast", "slow",
    "big", "small", "red", "blue", "green",
]
VOCAB_SIZE = len(VOCAB)   # 16


# ---------------------------------------------------------------------------
# Shared token embedder
# ---------------------------------------------------------------------------

class TokenEmbedder(nn.Module):
    """Embed token IDs and mean-pool over sequence length → [B, embed_dim]."""
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, seq_len]  →  [B, embed_dim]
        return self.emb(tokens).mean(dim=1)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class PolicyModel(nn.Module):
    """State embedding → logits over vocabulary."""
    def __init__(self, embed_dim: int, vocab_size: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)   # [..., vocab_size]

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """state: [..., embed_dim], action: [...]  →  log_prob: [...]"""
        logits = self.forward(state)
        return F.log_softmax(logits, dim=-1).gather(-1, action.unsqueeze(-1)).squeeze(-1)

    def sample(self, state: torch.Tensor):
        """state: [B, embed_dim]  →  (action [B], log_prob [B])"""
        dist   = torch.distributions.Categorical(logits=self.forward(state))
        action = dist.sample()
        return action, dist.log_prob(action)


class ValueModel(nn.Module):
    """State embedding → scalar value estimate."""
    def __init__(self, embed_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)   # [...]


class RewardModel(nn.Module):
    """Full (prompt + response) token sequence → scalar reward.

    Called once per completed response (sparse reward signal).
    Treated as frozen / pre-trained throughout PPO training.
    """
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.embedder = TokenEmbedder(vocab_size, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, prompt_len + T]  →  reward: [B]
        return self.net(self.embedder(tokens)).squeeze(-1)


# ---------------------------------------------------------------------------
# Advantage estimation
# ---------------------------------------------------------------------------

def compute_advantages_mc(
    rewards: torch.Tensor,
    values:  torch.Tensor,
    gamma:   float = 0.99,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Monte Carlo advantage: A_t = R_t - V(s_t)
    where R_t = Σ_{k>=0} γ^k · r_{t+k}  (full discounted return).

    High variance, low bias. Simple but noisy for long horizons.
    """
    B, T = rewards.shape
    returns = torch.zeros_like(rewards)
    running = torch.zeros(B)
    for t in reversed(range(T)):
        running       = rewards[:, t] + gamma * running
        returns[:, t] = running
    advantages = returns - values
    return advantages, returns


def compute_advantages_gae(
    rewards: torch.Tensor,
    values:  torch.Tensor,
    gamma:   float = 0.99,
    lam:     float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generalized Advantage Estimation (GAE):
        δ_t = r_t + γ·V(s_{t+1}) − V(s_t)       (TD residual)
        A_t = Σ_{k>=0} (γλ)^k · δ_{t+k}
        R_t = A_t + V(s_t)                        (GAE-based return target)

    λ=1  →  Monte Carlo advantage (high variance, low bias)
    λ=0  →  one-step TD advantage (low variance, high bias)
    Typical: λ=0.95 balances the two.
    """
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae        = torch.zeros(B)
    for t in reversed(range(T)):
        next_value       = values[:, t + 1] if t + 1 < T else torch.zeros(B)
        delta            = rewards[:, t] + gamma * next_value - values[:, t]
        gae              = delta + gamma * lam * gae
        advantages[:, t] = gae
    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def collect_rollout(
    policy:       PolicyModel,
    ref_policy:   PolicyModel,
    reward_model: RewardModel,
    value_model:  ValueModel,
    embedder:     TokenEmbedder,
    prompts:      torch.Tensor,   # [B, prompt_len]  input prompt token IDs
    T:            int,            # response length in tokens
    gamma:        float = 0.99,
    lam:          float = 0.95,
) -> dict:
    """
    Autoregressively generate a response of T tokens, one token at a time.

    At each step t:
      context  = prompt + response[:t]           (grows by 1 token each step)
      state    = TokenEmbedder(context)           [B, embed_dim]
      action   = next token sampled from policy   [B]  int in [0, vocab_size)
      reward   = 0 for t < T-1  (sparse)
               = RewardModel(prompt + full_response) at t = T-1
    """
    B, prompt_len = prompts.shape
    embed_dim     = embedder.emb.embedding_dim

    response_tokens = torch.zeros(B, T, dtype=torch.long)
    states_emb      = torch.zeros(B, T, embed_dim)
    logp_old_list, logp_ref_list, values_list = [], [], []

    with torch.no_grad():
        for t in range(T):
            # build context: prompt + tokens generated so far
            context = prompts if t == 0 else torch.cat([prompts, response_tokens[:, :t]], dim=1)

            state = embedder(context)                        # [B, embed_dim]
            states_emb[:, t] = state

            action, logp   = policy.sample(state)            # [B]
            logp_ref       = ref_policy.log_prob(state, action)
            value          = value_model(state)

            response_tokens[:, t] = action
            logp_old_list.append(logp)
            logp_ref_list.append(logp_ref)
            values_list.append(value)

        # sparse reward: score the complete (prompt + response) once at the end
        full_seq     = torch.cat([prompts, response_tokens], dim=1)  # [B, prompt_len+T]
        final_reward = reward_model(full_seq)                         # [B]

    logp_old = torch.stack(logp_old_list, dim=1)   # [B, T]
    logp_ref = torch.stack(logp_ref_list, dim=1)   # [B, T]
    values   = torch.stack(values_list,   dim=1)   # [B, T]

    rewards        = torch.zeros(B, T)
    rewards[:, -1] = final_reward                  # reward only at last token

    advantages, returns = compute_advantages_gae(rewards, values, gamma, lam)
    # swap to MC:  advantages, returns = compute_advantages_mc(rewards, values, gamma)

    return dict(
        states=states_emb,          # [B, T, embed_dim]  pre-computed context embeddings
        actions=response_tokens,    # [B, T]  generated token IDs
        logp_old=logp_old,          # [B, T]
        logp_ref=logp_ref,          # [B, T]
        returns=returns,            # [B, T]
        values=values,              # [B, T]
        advantages=advantages,      # [B, T]
        mask=torch.ones(B, T),
    )


# ---------------------------------------------------------------------------
# PPO update step
# ---------------------------------------------------------------------------

def ppo_update(
    policy:           PolicyModel,
    value_model:      ValueModel,
    rollout:          dict,
    hparams:          dict,
    policy_optimizer: torch.optim.Optimizer,
    value_optimizer:  torch.optim.Optimizer,
) -> dict:
    """
    Re-evaluate log-probs and values under current parameters, then backprop.
    Flattens [B, T, embed_dim] → [B*T, embed_dim] for a single forward pass.
    """
    states  = rollout["states"]    # [B, T, embed_dim]
    actions = rollout["actions"]   # [B, T]
    B, T, embed_dim = states.shape

    states_flat  = states.view(B * T, embed_dim)
    actions_flat = actions.view(B * T)

    logp_new   = policy.log_prob(states_flat, actions_flat).view(B, T)
    values_new = value_model(states_flat).view(B, T)

    batch = dict(
        advantages = rollout["advantages"],
        returns    = rollout["returns"],
        values     = values_new,
        logp_old   = rollout["logp_old"],
        logp_new   = logp_new,
        logp_ref   = rollout["logp_ref"],
        mask       = rollout["mask"],
    )

    losses = ppo_losses(batch, hparams)

    # policy update: gradients flow through logp_new (policy_loss + kl_loss)
    policy_loss = losses["policy_loss"] + hparams["kl_coef"] * losses["kl_loss"]
    policy_optimizer.zero_grad()
    policy_loss.backward(retain_graph=True)
    policy_optimizer.step()

    # value update: gradients flow through values_new (value_loss only)
    value_optimizer.zero_grad()
    losses["value_loss"].backward()
    value_optimizer.step()

    return losses


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode(token_ids: list) -> str:
    return " ".join(VOCAB[i] for i in token_ids)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    embed_dim  = 16
    hidden_dim = 32
    B          = 8
    prompt_len = 4
    T          = 6      # response length in tokens
    num_iters  = 50

    embedder     = TokenEmbedder(VOCAB_SIZE, embed_dim)
    policy       = PolicyModel(embed_dim, VOCAB_SIZE, hidden_dim)
    ref_policy   = PolicyModel(embed_dim, VOCAB_SIZE, hidden_dim)
    value_model  = ValueModel(embed_dim, hidden_dim)
    reward_model = RewardModel(VOCAB_SIZE, embed_dim, hidden_dim)

    ref_policy.load_state_dict(policy.state_dict())
    for p in ref_policy.parameters():
        p.requires_grad_(False)

    for p in reward_model.parameters():
        p.requires_grad_(False)

    hparams = dict(clip_eps=0.2, vf_coef=0.5, kl_coef=0.1, normalize_adv=True)

    policy_optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(embedder.parameters()), lr=3e-4
    )
    value_optimizer = torch.optim.Adam(value_model.parameters(), lr=3e-4)

    print(f"Vocab ({VOCAB_SIZE} tokens): {VOCAB}\n")
    print(f"{'iter':>4}  {'policy':>8}  {'value':>8}  {'kl':>8}  {'total':>8}")
    print("-" * 46)

    for i in range(num_iters):
        prompts = torch.randint(2, VOCAB_SIZE, (B, prompt_len))  # skip <pad>, <eos>

        rollout = collect_rollout(
            policy, ref_policy, reward_model, value_model,
            embedder, prompts, T,
        )
        losses = ppo_update(
            policy, value_model, rollout, hparams,
            policy_optimizer, value_optimizer,
        )

        if i % 10 == 0:
            print(
                f"{i:4d}  "
                f"{losses['policy_loss'].item():8.4f}  "
                f"{losses['value_loss'].item():8.4f}  "
                f"{losses['kl_loss'].item():8.4f}  "
                f"{losses['total_loss'].item():8.4f}"
            )
            print(f"      prompt:   {decode(prompts[0].tolist())}")
            print(f"      response: {decode(rollout['actions'][0].tolist())}")

    print("\nDone.")
