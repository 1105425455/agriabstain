"""Distributed sampler that emits a fixed state pattern.

Samples are pooled by `head_label` into answer, wrong_object and no_target, and the pattern
[ans, wo, nt, ans, wo, ans, wo, nt] yields 3:3:2 in every group of eight. The wrong_object
pool is the smallest, so it is reshuffled and reused when it runs out. Epoch length follows
the answer pool and `drop_last=True` keeps the pattern aligned to eight.
"""

import os
from typing import Iterator, List, Optional, Sequence

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


class PatternedDistributedSampler(Sampler[int]):
    """Pool by state, emit the fixed pattern, and shard across ranks.

    Args:
        head_labels: the `head_label` of every sample, 0 = answer, 1 = wrong_object,
            2 = no_target, of length N
        num_replicas: distributed world size
        rank: this rank
        seed: random seed
        epoch: initial epoch
        drop_last: whether to drop a trailing partial group, keeping the pattern aligned
    """

    _default_pattern = [0, 1, 2, 0, 1, 0, 1, 2]
    _env_pattern = os.environ.get("PAIR_PATTERN", "").strip()
    if _env_pattern:
        PATTERN = [int(x) for x in _env_pattern.split(",")]
    else:
        PATTERN = _default_pattern

    def __init__(
        self,
        head_labels: Sequence[int],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 42,
        epoch: int = 0,
        drop_last: bool = True,
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package")
            rank = dist.get_rank()

        self.head_labels = list(head_labels)
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = epoch
        self.drop_last = drop_last

        self.pool_indices: List[List[int]] = [[], [], []]
        for i, hl in enumerate(head_labels):
            if hl in (0, 1, 2):
                self.pool_indices[hl].append(i)
            else:
                raise ValueError(f"Unexpected head_label={hl} at index={i}")

        self._pool_sizes = [len(p) for p in self.pool_indices]

        self._rank_pool: List[List[int]] = [[], [], []]
        for label in (0, 1, 2):
            pool = sorted(self.pool_indices[label])
            rank_indices = [[] for _ in range(num_replicas)]
            for i, idx in enumerate(pool):
                rank_indices[i % num_replicas].append(idx)
            self._rank_pool[label] = rank_indices[rank]

        self._answer_pool_size = len(self._rank_pool[0])
        self._ans_per_pattern = self.PATTERN.count(0)
        pattern_len = len(self.PATTERN)  # 8

        n_answer_total = self._pool_sizes[0]
        per_rank_answer = [
            (n_answer_total - r + num_replicas - 1) // num_replicas if n_answer_total > r else 0
            for r in range(num_replicas)
        ]
        global_min = min(per_rank_answer)

        self._n_patterns = global_min // self._ans_per_pattern
        self._num_samples = self._n_patterns * pattern_len

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @property
    def total_size(self) -> int:
        return self._num_samples * self.num_replicas

    def __len__(self) -> int:
        return self._num_samples

    def _shard_pool(self, pool: List[int], local_shuffle_seed: int) -> List[int]:
        """Shuffle this rank's fixed subset of a pool and return it."""
        g = torch.Generator()
        g.manual_seed(local_shuffle_seed)
        indices = torch.tensor(pool, dtype=torch.long)
        perm = torch.randperm(len(indices), generator=g)
        return indices[perm].tolist()

    def _wrap_around_upsample(self, pool: List[int], needed: int, seed: int) -> List[int]:
        """Upsample the wrong_object pool by reshuffling and wrapping around when it is exhausted.

        """
        if len(pool) == 0:
            return []
        g = torch.Generator()
        g.manual_seed(seed)
        perm = torch.randperm(len(pool), generator=g)
        shuffled = [pool[i] for i in perm.tolist()]
        result = []
        while len(result) < needed:
            result.extend(shuffled)
        return result[:needed]

    def __iter__(self) -> Iterator[int]:
        """Emit sample indices following the pattern.

        Each pool is reshuffled within the rank every epoch, with the seed offset by the epoch, and
        the pattern [0, 1, 2, 0, 1, 0, 1, 2] yields 3:3:2. The wrong_object pool wraps around when it
        is exhausted.
        """
        base_seed = self.seed + self.epoch
        n_ans_per_pattern = self.PATTERN.count(0)
        n_wo_per_pattern = self.PATTERN.count(1)
        n_nt_per_pattern = self.PATTERN.count(2)

        ans_shard = self._shard_pool(self._rank_pool[0], base_seed)
        wo_shard = self._shard_pool(self._rank_pool[1], base_seed + 1)
        nt_shard = self._shard_pool(self._rank_pool[2], base_seed + 2)

        n_patterns = self._n_patterns

        needed_wo = n_patterns * n_wo_per_pattern
        if len(wo_shard) < needed_wo:
            wo_shard = self._wrap_around_upsample(wo_shard, needed_wo, base_seed + 3)

        needed_nt = n_patterns * n_nt_per_pattern
        if len(nt_shard) < needed_nt:
            nt_shard = self._wrap_around_upsample(nt_shard, needed_nt, base_seed + 4)

        ans_blocks = [
            ans_shard[i * n_ans_per_pattern : (i + 1) * n_ans_per_pattern]
            for i in range(n_patterns)
        ]
        wo_blocks = [
            wo_shard[i * n_wo_per_pattern : (i + 1) * n_wo_per_pattern] for i in range(n_patterns)
        ]
        nt_blocks = [
            nt_shard[i * n_nt_per_pattern : (i + 1) * n_nt_per_pattern] for i in range(n_patterns)
        ]

        for i in range(n_patterns):
            ans_ptr = 0
            wo_ptr = 0
            nt_ptr = 0
            for p in self.PATTERN:
                if p == 0:
                    yield ans_blocks[i][ans_ptr]
                    ans_ptr += 1
                elif p == 1:
                    yield wo_blocks[i][wo_ptr]
                    wo_ptr += 1
                else:  # p == 2
                    yield nt_blocks[i][nt_ptr]
                    nt_ptr += 1

    def set_epoch(self, epoch: int):
        self.epoch = epoch
