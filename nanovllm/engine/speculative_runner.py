"""单 GPU / 单活动请求的双模型 runner。

逻辑页号相同，物理 KV 张量不同。候选 token 只存在于本轮局部列表，绝不先
append 到 Sequence 再删除，因而调度器永远只看到已提交历史。
"""
from dataclasses import dataclass

import torch

from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.speculative_sampling import probabilities, sample, verify_greedy, verify_stochastic
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.context import get_context, reset_context
from nanovllm.utils.loader import load_model


@dataclass
class SpeculativeResult:
    token_ids: list[int]
    target_cached: int
    draft_cached: int
    proposed: int = 0
    accepted: int = 0


@dataclass
class _Span:
    """只给 prepare_prefill 提供所需字段，不改变真实 Sequence 的状态。"""
    token_ids: list[int]
    num_cached_tokens: int
    num_scheduled_tokens: int
    block_table: list[int]

    def __getitem__(self, key):
        return self.token_ids[key]


class SpeculativeModelRunner(ModelRunner):
    def load_models(self):
        super().load_models()
        cfg = self.config.draft_hf_config
        old_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(cfg.dtype)
            self.draft_model = Qwen3ForCausalLM(cfg)
            load_model(self.draft_model, self.config.draft_model)
        finally:
            torch.set_default_dtype(old_dtype)
        self.stats = dict(rounds=0, proposed=0, accepted=0, repair_calls=0, repair_tokens=0,
                          committed=0, accepted_committed=0, acceptance_histogram={})

    @torch.inference_mode()
    def forward_span(self, model, seq, tokens, start, end, all_logits=False):
        """复用分页 varlen prefill 的右对齐 causal mask。

        query 是 [start,end)，key 是 [0,end)。这样第一个 query 只能看到自己的
        前缀，不能偷看后续候选。slot_mapping 只覆盖本次计算的位置，旧尾部 KV
        无需清零，因为 cu_seqlens_k/end 把无效位置屏蔽掉。
        """
        if not 0 <= start < end <= len(tokens):
            raise ValueError("invalid forward span")
        view = _Span(tokens, start, end - start, seq.block_table)
        try:
            input_ids, positions = self.prepare_prefill([view])
            get_context().all_logits = all_logits
            return model.compute_logits(model(input_ids, positions))
        finally:
            reset_context()

    def warmup_model(self):
        # 两个模型已同时驻留，然后逐个预热，测得共同驻留时的运行时峰值。
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        length = min(self.config.max_num_batched_tokens, self.config.max_model_len)
        seq = Sequence([0] * length)
        for model in (self.model, self.draft_model):
            self.forward_span(model, seq, seq.token_ids, 0, length)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        cfgs = (self.config.hf_config, self.config.draft_hf_config)
        shapes, block_bytes = [], 0
        for cfg in cfgs:
            dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
            shape = (cfg.num_hidden_layers, self.block_size, cfg.num_key_value_heads, dim)
            shapes.append(shape)
            block_bytes += 2 * shape[0] * shape[1] * shape[2] * shape[3] * cfg.dtype.itemsize
        free, total = torch.cuda.mem_get_info()
        current = torch.cuda.memory_allocated()
        transient = torch.cuda.max_memory_allocated() - current
        # 预热的末位 logits 不覆盖 K+1 行 logits / p / q。单独保留这部分空间，
        # 同时给 allocator / NCCL 留 256 MiB 余量，避免 KV 把显存预算吃满。
        reserve = max(256 << 20, 8 * (self.config.num_speculative_tokens + 1)
                      * self.config.hf_config.vocab_size * 4)
        budget = int(total * self.config.gpu_memory_utilization) - (total - free) - transient - reserve
        blocks = budget // block_bytes
        if blocks < 1:
            raise ValueError("insufficient GPU memory for both KV caches")
        self.config.num_kvcache_blocks = blocks
        self.memory_plan = dict(num_blocks=blocks, combined_bytes_per_block=block_bytes,
                                kv_bytes=blocks * block_bytes, runtime_peak_bytes=transient,
                                extra_reserve_bytes=reserve, resident_allocated_bytes=current)
        caches = []
        for model, cfg, (layers, size, heads, dim) in zip((self.model, self.draft_model), cfgs, shapes):
            cache = torch.empty(2, layers, blocks, size, heads, dim, device="cuda", dtype=cfg.dtype)
            for i, module in enumerate(m for m in model.modules() if hasattr(m, "k_cache")):
                module.k_cache, module.v_cache = cache[0, i], cache[1, i]
            caches.append(cache)
        self.kv_cache, self.draft_kv_cache = caches

    @torch.inference_mode()
    def run(self, seqs, is_prefill):
        if len(seqs) != 1:
            raise ValueError("V0 speculative runner requires one active request")
        seq = seqs[0]
        if is_prefill:
            end = seq.num_cached_tokens + seq.num_scheduled_tokens
            target = self.forward_span(self.model, seq, seq.token_ids, seq.num_cached_tokens, end)
            self.forward_span(self.draft_model, seq, seq.token_ids, seq.num_draft_cached_tokens, end)
            token_ids = []
            if end == len(seq):
                token = int(target[0].argmax().item()) if seq.temperature == 0 else sample(probabilities(target[0], seq.temperature))
                token_ids = [token]
            return [SpeculativeResult(token_ids, end, end)]

        n = len(seq)
        # 每轮开始的不变量：两模型均缓存了已提交序列中除最后一个 token 外的内容。
        assert seq.num_cached_tokens == seq.num_draft_cached_tokens == n - 1
        remaining = min(seq.max_tokens - seq.num_completion_tokens, self.config.max_model_len - n)
        k = min(self.config.num_speculative_tokens, remaining - 1)
        candidates, q_rows = [], []
        draft_cached = n - 1
        for _ in range(k):
            tokens = seq.token_ids + candidates
            logits = self.forward_span(self.draft_model, seq, tokens, draft_cached, len(tokens))[0]
            draft_cached = len(tokens)
            if seq.temperature == 0:
                candidate = int(logits.argmax().item())
            else:
                q = probabilities(logits, seq.temperature)
                q_rows.append(q)  # 保留原始 q，不能被采样过程原地修改。
                candidate = sample(q)
            candidates.append(candidate)

        # 一次验证 [最后已提交 token, d1,...,dK]，得到 K+1 行目标 logits。
        tokens = seq.token_ids + candidates
        logits = self.forward_span(self.model, seq, tokens, n - 1, len(tokens), all_logits=True)
        assert logits.shape[0] == k + 1
        if getattr(self, "trace", None) is not None:
            # 仅诊断测试启用：记录每个验证位置的前两名，定位贪心分叉是否来自近并列。
            # 性能实验不设置 trace，避免额外 CPU/GPU 同步。
            values, indices = logits.float().topk(2, dim=-1)
            self.trace.append(dict(committed_length=n, draft=candidates.copy(),
                                   top_ids=indices.tolist(), top_values=values.tolist()))
        if seq.temperature == 0:
            verified = verify_greedy(candidates, logits)
        else:
            p = probabilities(logits, seq.temperature)
            q = torch.stack(q_rows) if q_rows else p.new_empty((0, p.shape[-1]))
            verified = verify_stochastic(candidates, p, q)

        # 提交 a+1 个 token 后，需要的有效 KV 长度是 n+a；替代/额外 token 本身
        # 尚未做前向。拒绝分支只回退长度；全接受分支 draft 落后一个位置，要补齐 dK。
        valid = n + len(verified.token_ids) - 1
        if draft_cached < valid:
            committed = seq.token_ids + verified.token_ids
            self.forward_span(self.draft_model, seq, committed, draft_cached, valid)
            self.stats["repair_calls"] += 1
            self.stats["repair_tokens"] += valid - draft_cached
        self.stats["rounds"] += 1
        self.stats["proposed"] += k
        self.stats["accepted"] += verified.accepted
        hist = self.stats["acceptance_histogram"]
        hist[verified.accepted] = hist.get(verified.accepted, 0) + 1
        return [SpeculativeResult(verified.token_ids, valid, valid, k, verified.accepted)]
