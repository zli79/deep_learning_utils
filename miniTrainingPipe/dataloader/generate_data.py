import numpy as np
import os

os.makedirs("datasets", exist_ok=True)

for i in range(32):
    data = np.random.randn((4092, 1024))
    np.save(f"datasets/shard_{i}.npy", data)