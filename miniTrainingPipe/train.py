import glob
import numpy as np
import matplotlib.pyplot as plt
from dataloader.dataloader import ShardedDataset, DataLoader, PrefetchLoader, MultiThreadsPrefetchLoader
from network import mini_transformer, params, num_heads as default_num_heads
from loss import cross_entropy_loss
from optimizer import SGD

def compute_gradients(inputs, targets, params, num_heads, eps = 1e-8):
    """
    inputs: (B, S)
    targets: (B, S)
    params: dictionary of parameters
    num_heads: int
    eps: float
    returns: dictionary of gradients
    """

    grads = {k: np.zeros_like(v) for k, v in params.items()}

    base_loss, _ = cross_entropy_loss(mini_transformer(inputs, params, num_heads), targets)
    for k, v in params.items():
        v_plus = v.copy()
        v_minus = v.copy()
        v_plus += eps
        v_minus -= eps
        loss_plus, _ = cross_entropy_loss(mini_transformer(inputs, params, num_heads), targets)
        loss_minus, _ = cross_entropy_loss(mini_transformer(inputs, params, num_heads), targets)
        grads[k] = (loss_plus - loss_minus) / (2 * eps)
    return grads


seq_len = 8
epochs = 1
lr = 1e-2
num_heads = default_num_heads

optimizer = SGD(params, lr)

# load shards 
all_shards = glob.glob("dataloader/datasets/shakespeare/shard_*.npy")
dataset = ShardedDataset(all_shards)
loader = DataLoader(dataset, batch_size=2, shuffle=True)
loader = PrefetchLoader(loader, prefetch_size=4)


# set up live loss plot
plt.ion()
fig, ax = plt.subplots()
loss_history = []
line, = ax.plot([], [], 'b-')
ax.set_xlabel("Step")
ax.set_ylabel("Loss")
ax.set_title("Training Loss Curve")

for epoch in range(epochs):
    for step, batch in enumerate(loader):
        # truncate to seq_len to keep numerical grads feasible
        batch = batch[:, :seq_len]
        inputs = batch[:,:-1]
        targets = batch[:,1:]

        # forward
        logits = mini_transformer(inputs, params, num_heads)
        loss, _ = cross_entropy_loss(logits, targets)

        # backward
        grads = compute_gradients(inputs, targets, params, num_heads)

        # optimizer step 
        optimizer.step(grads)

    
        # update loss history
        loss_history.append(loss)
        line.set_data(range(len(loss_history)), loss_history)
        ax.set_xlim(0, max(1, len(loss_history)))
        ax.set_ylim(0, max(loss_history)*1.1)

        fig.canvas.draw()
        fig.canvas.flush_events()


        if step % 5 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss {loss:.4f}")
        global_step += 1

plt.ioff()
plt.show()