import numpy as np
from queue import Queue
import threading


class Communicator:
    def __init__(self, world_size):
        self.world_size = world_size
        self.queues = {rank: Queue() for rank in range(world_size)}

    def send(self, src, dst, data):
        self.queues[dst].put((src, data))

    def receive(self, dst):
        _src, data = self.queues[dst].get()
        return data


# ---------------------------------------------------------------------------
# Column-wise tensor-parallel  y = x @ W
# ---------------------------------------------------------------------------
# W [in_features, out_features] is split along columns (out_features axis).
# Rank r owns  W_r = W[:, r*chunk : (r+1)*chunk]
#
# Communication pattern:
#   1. Rank 0 sends x to every other rank  (1-to-N)
#   2. Each rank computes  y_r = x @ W_r
#   3. Each rank sends y_r back to rank 0
#   4. Rank 0 concatenates shards → y = [y_0 | y_1 | … | y_{N-1}]
# ---------------------------------------------------------------------------

def column_parallel_matmul(x, w, world_size):
    assert w.shape[1] % world_size == 0
    chunk = w.shape[1] // world_size
    comm  = Communicator(world_size)
    results = [None] * world_size

    def worker(rank):
        # step 1: receive x from rank 0 (rank 0 reads its own copy directly)
        if rank == 0:
            x_local = x
        else:
            x_local = comm.receive(dst=rank)

        # step 2: local matmul on column shard
        w_local = w[:, rank * chunk : (rank + 1) * chunk]
        y_local = x_local @ w_local

        # step 3: send result back to rank 0
        comm.send(src=rank, dst=0, data=(rank, y_local))

    # launch one thread per rank
    threads = [threading.Thread(target=worker, args=(r,)) for r in range(world_size)]
    for t in threads:
        t.start()

    # rank 0 broadcasts x, then gathers results
    for dst in range(1, world_size):
        comm.send(src=0, dst=dst, data=x)

    for _ in range(world_size):
        rank_id, y_local = comm.receive(dst=0)
        results[rank_id] = y_local

    for t in threads:
        t.join()

    return np.concatenate(results, axis=-1)


if __name__ == "__main__":
    np.random.seed(0)
    batch, in_f, out_f, world_size = 4, 8, 16, 4

    x = np.random.randn(batch, in_f)
    w = np.random.randn(in_f, out_f)

    y_ref = x @ w
    y_tp  = column_parallel_matmul(x, w, world_size)

    print("Max abs error:", np.max(np.abs(y_ref - y_tp)))
    assert np.allclose(y_ref, y_tp)
    print("column-parallel matmul matches reference  ✓")
