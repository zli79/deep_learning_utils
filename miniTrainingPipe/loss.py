import numpy as np

def cross_entropy_loss(logits, target):
    """
    logits: (B * S * V)
    targets: (B * S)
    returns scalar loss
    """
    B, S, V = logits.shape
    assert target.shape == (B, S)
    
    # stabilize 
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    # pick probability of target tokens
    target_probs = probs[np.arange(B)[:, None], np.arange(S)[None, :], target]

    # negative log likelihood
    loss = -np.mean(np.log(target_probs + 1e-9))
    return loss, probs
   