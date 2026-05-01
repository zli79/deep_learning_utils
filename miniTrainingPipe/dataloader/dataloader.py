import numpy as np
import bisect

class ShardedDataset:
    def __init__(self, shard_files):

        # never load the whole dataset into memory
        # memory mapping 
        self.shards = [np.load(shard_file, mmap_mode='r') for shard_file in shard_files]

        self.shard_offsets = []
        total = 0
    
        for shard in self.shards:
            self.shard_offsets.append(total)
            total += len(shard)

        self.total_len = total

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):

        for shard, offset in zip(self.shards, self.shard_offsets):
            if index < offset:
                return shard[index - offset]
        return None
    def get_batch(self, indices):

        batch = []
        for idx in indices:
            shard_idx = bisect.bisect_right(self.shard_offsets, idx) - 1
            shard = self.shards[shard_idx]
            offset = self.shard_offsets[shard_idx]
        
        return np.stack(batch)

def assign_shards(all_shards, rank, world_size):
    """
    all_shards: list of shard files
    rank: int
    world_size: int
    returns: list of shard files
    """
    return [shard for i, shard in enumerate(all_shards) if i % world_size == rank]

class BatchSampler:
    def __init__(self, dataset_len, batch_size, shuffle=True):
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(dataset_len))

    def __len__(self):
        return self.dataset_len // self.batch_size
    
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        for i in range(0, self.dataset_len, self.batch_size):
            yield self.indices[i:i+self.batch_size]


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_sampler = BatchSampler(len(dataset), batch_size, shuffle)

    def __iter__(self):
        for indices in self.batch_sampler:
            yield self.dataset.get_batch(indices)

    def __len__(self):
        return len(self.batch_sampler)


import threading
import queue

class PrefetchLoader:

    def __init__(self, loader, prefetch_size=4):

        self.loader = loader
        self.prefetch_size = prefetch_size
    
    def __iter__(self):

        q = queue.Queue(maxsize=self.prefetch_size)

        stop_token = object()

        def worker():
            for batch in self.loader:
                q.put(batch)
            q.put(stop_token)
        
        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()

        while True:
            batch = q.get()
            if batch is stop_token:
                break
            yield batch


class MultiThreadsPrefetchLoader:
    def __init__(self, loader, number_workers=4, prefetch_size=4):
        self.loader = loader
        self.number_workers = number_workers
        self.prefetch_size = prefetch_size
        self.queue = queue.Queue(maxsize=prefetch_size)
        self.stop_token = object()

    def __iter__(self):
        q = queue.Queue(maxsize=self.prefetch_size)

        stop_token = object()

        def worker():
            for batch in self.loader:
                self.queue.put(batch)
            self.queue.put(self.stop_token)
        
        # starts threads
        threads = []
        for _ in range(self.number_workers):
            thread = threading.Thread(target=worker)
            thread.daemon = True
            thread.start()
            threads.append(thread)

        # consume from queue
        finished_workers = 0
        while finished_workers < self.number_workers:
            batch = self.queue.get()
            if batch is self.stop_token:
                finished_workers += 1
            else:
                yield batch


import glob
import time

start = time.time()
rank = 0
world_size = 4
shard_files = glob.glob("datasets/shard_*.npy")
shard_files = assign_shards(shard_files, rank, world_size)
dataset = ShardedDataset(shard_files)
loader = DataLoader(dataset, batch_size=128, shuffle=True)
prefetch_loader = PrefetchLoader(loader, prefetch_size=4)
for batch in prefetch_loader:
    print(batch.shape)
end = time.time()
print(f"Time taken: {end - start} seconds")