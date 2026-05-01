import numpy as np

def attention(Q, K, V, mask):
    """
    q, k, v shape: (batch_size, num_heads, seq_len, head_dim)
    mask shape: (seq_len, seq_len)
    """

    batch_size, num_heads, seq_len, head_dim = Q.shape
    assert K.shape == (batch_size, num_heads, seq_len, head_dim)
    assert V.shape == (batch_size, num_heads, seq_len, head_dim)
    assert mask.shape == (seq_len, seq_len)

    # compute attention scores
    scores = np.einsum('bhqd,bhkd->bhqk', Q, K) / np.sqrt(head_dim)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    
    # softmax 
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # weighted sum of values
    out = np.einsum('bhqk,bhkd->bhqd', probs, V)
    return out

def multi_head_attention(x, W_Q, W_K, W_V, W_O, num_heads, mask=None):
    """
    x shape: (batch_size, seq_len, dim)
    W_Q, W_K, W_V, W_O shape: (dim, num_heads * head_dim)
    mask shape: (seq_len, seq_len)
    returns shape: (batch_size, seq_len, num_heads * head_dim)
    """

    batch_size, seq_len, dim = x.shape
    assert dim%num_heads == 0, "dim must be divisible by num_heads"

    head_dim = dim // num_heads

    q = x @ W_Q
    k = x @ W_K
    v = x @ W_V

    q = q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

    out = attention(q, k, v, mask)
    out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, dim)
    # project back to original dimension
    out = out @ W_O
    return out


def ffc(x, w1, b1, w2, b2):
    """
    x: batch_size, seq_len, dim
    w1, b1, w2, b2: (dim, hidden_dim), (hidden_dim,), (hidden_dim, dim), (dim,)
    returns: batch_size, seq_len, dim
    """

    batch_size, seq_len, dim = x.shape


    fc1 = x @ w1 + b1
    fc1 = np.maximum(fc1, 0)
    out = fc1 @ w2 + b2
    return out

def layernorm(x, gamma, beta, eps = 1e-5):
    """
    x: batch_size, seq_len, dim
    gamma, beta: (dim,)
    returns: batch_size, seq_len, dim
    """

    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    norm = (x - mean) / np.sqrt(var + eps)
    out = out * gamma + beta
    return out


def causal_mask(seq_len):
    """
    seq_len: int
    returns: (seq_len, seq_len)
    """

    mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    return mask


def transformer_block(x, params, num_heads, mask=None):
    """
    x: batch_size, seq_len, dim
    params: dict of parameters
    num_heads: int
    mask: (seq_len, seq_len)
    returns: batch_size, seq_len, dim
    """

    attn_out = multi_head_attention(x, params['W_Q'], params['W_K'], params['W_V'], params['W_O'], num_heads, mask)
    ffc_out = ffc(attn_out, params['w1'], params['b1'], params['w2'], params['b2'])
    out = layernorm(ffc_out + x, params['gamma'], params['beta'])
    return out


def mini_transformer(tokens, params, num_heads):
    """
    tokens: (B, S) integer tokens ids
    params: dictionary of model parameters
    num_heads: int
    returns: (B, S, V) logits
    """

    batch_size, seq_len = tokens.shape
    dim = params["emb"].shape[1]

    # embeding + position encoding
    x = params["emb"][tokens] + params["pos_emb"][None, :seq_len, :]

    mask = causal_mask(seq_len)

    # single transformer block
    x = transformer_block(x, params, num_heads, mask)

    # output logits ()
    logits = x @ params["lm_head"].T
    return logits


# example initialization

vocab_size = 84
seq_len = 8
dim = 8
num_heads = 2
B = 2

params = {
    "emb": np.random.randn(vocab_size, dim) * 0.01,
    "pos_emb": np.random.randn(seq_len, dim) * 0.01,
    "lm_head": np.random.randn(dim, vocab_size),
    "W_Q": np.random.randn(dim, dim) * 0.01,
    "W_K": np.random.randn(dim, dim) * 0.01,
    "W_V": np.random.randn(dim, dim) * 0.01,
    "W_O": np.random.randn(dim, dim) * 0.01,
    "w1": np.random.randn(dim, dim) * 0.01,
    "b1": np.zeros(dim) * 0.01,
    "w2": np.random.randn(dim, dim) * 0.01,
    "b2": np.zeros(dim) * 0.01,
    "gamma": np.ones(dim) * 0.01,
    "beta": np.zeros(dim) * 0.01,
}

tokens = np.random.randint(0, vocab_size, (B, seq_len))
logits = mini_transformer(tokens, params, num_heads)
print(logits.shape)

