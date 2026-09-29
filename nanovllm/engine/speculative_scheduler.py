"""V0 串行调度：每次只有一个活动请求，其余请求排队。

为学习和验证回退逻辑，先为请求的整个输出预算预留页；不共享前缀，不发布
候选页的 hash。V1 才引入按轮扩容、跨请求共享及内存压力下抢占。
"""
from collections import deque
from math import ceil

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import SequenceStatus


class SpeculativeScheduler:
    def __init__(self, config):
        self.config = config
        self.waiting = deque()
        self.running = deque()
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)

    def add(self, seq):
        required = ceil(min(len(seq) + seq.max_tokens, self.config.max_model_len)
                        / self.config.kvcache_block_size)
        if required > len(self.block_manager.blocks):
            raise ValueError("V0 request budget exceeds available KV pages; reduce context/output budget")
        self.waiting.append(seq)

    def is_finished(self):
        return not self.waiting and not self.running

    def schedule(self):
        if not self.running:
            seq = self.waiting.popleft()
            bm = self.block_manager
            bm.allocate(seq, 0)
            required = ceil(min(len(seq) + seq.max_tokens, self.config.max_model_len) / bm.block_size)
            while len(seq.block_table) < required:
                seq.block_table.append(bm._allocate_block())
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
        seq = self.running[0]
        if seq.is_prefill:
            seq.num_scheduled_tokens = min(len(seq) - seq.num_cached_tokens,
                                           self.config.max_num_batched_tokens)
        else:
            # 验证查询长度统计与实际 K 相同，最后一轮可能缩短。
            remaining = min(seq.max_tokens - seq.num_completion_tokens, self.config.max_model_len - len(seq))
            seq.num_scheduled_tokens = min(self.config.num_speculative_tokens + 1, remaining)
        return [seq], seq.is_prefill

    def postprocess(self, seqs, results, is_prefill):
        seq, result = seqs[0], results[0]
        before = len(seq)
        for token_id in result.token_ids:
            seq.append_token(token_id)
            if ((not seq.ignore_eos and token_id == self.config.eos)
                    or seq.num_completion_tokens >= seq.max_tokens
                    or len(seq) >= self.config.max_model_len):
                seq.status = SequenceStatus.FINISHED
                break
        # EOS 可能在接受的候选中间：只提交到 EOS，屏蔽其后的所有 KV。
        limit = len(seq) - 1 if result.token_ids else len(seq)
        seq.num_cached_tokens = min(result.target_cached, limit)
        seq.num_draft_cached_tokens = min(result.draft_cached, limit)
        seq.num_scheduled_tokens = 0
        if result.token_ids:
            seq.is_prefill = False
        if seq.is_finished:
            self.block_manager.deallocate(seq)
            seq.num_draft_cached_tokens = 0
            self.running.popleft()
        return len(seq) - before
