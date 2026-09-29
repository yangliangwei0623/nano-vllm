"""V0 精确投机采样参考算法，可直接在 CPU 小词表上验证。

这里有意保留清晰的 Python 循环与 GPU->CPU 标量同步。V2 再优化数据流；
正确性参考版的首要任务是把接受概率、首次拒绝、额外 token 三种语义分清。
"""
from dataclasses import dataclass

import torch


@dataclass
class VerificationResult:
    token_ids: list[int]
    accepted: int


def probabilities(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """温度必须同时施加到 p 和 q；贪心分支不使用这个函数。"""
    if temperature <= 0:
        raise ValueError("probabilities requires positive temperature")
    return torch.softmax(logits.float() / temperature, dim=-1)


def sample(probs: torch.Tensor, generator=None) -> int:
    # multinomial 接受未归一化的非负权重；调用方的概率不会被原地破坏。
    return int(torch.multinomial(probs, 1, generator=generator).item())


def verify_greedy(draft_ids: list[int], target_logits: torch.Tensor) -> VerificationResult:
    """第 i 行 logits 预测 draft_ids[i]；末行预测全部接受后的额外 token。"""
    target_ids = target_logits.argmax(-1).tolist()
    for i, token_id in enumerate(draft_ids):
        if token_id != target_ids[i]:
            return VerificationResult(draft_ids[:i] + [target_ids[i]], i)
    return VerificationResult(draft_ids + [target_ids[-1]], len(draft_ids))


def verify_stochastic(draft_ids: list[int], p: torch.Tensor, q: torch.Tensor,
                      generator=None) -> VerificationResult:
    """精确拒绝采样：p=[K+1,V]，q=[K,V]，q 是当时实际生成候选的分布。

    对候选 x，以 min(1,p(x)/q(x)) 接受。若首次拒绝发生在 i，前 i 个
    候选保持不变，并从 (p_i-q_i)_+ 归一化分布抽取替代 token；之后的候选
    条件历史已失效，必须全部丢弃。全接受时从 p_K 取额外 token。
    接受事件贡献 min(p,q)，拒绝事件贡献 (p-q)_+，两者相加恰好为 p。
    """
    k = len(draft_ids)
    if p.ndim != 2 or q.shape != (k, p.shape[1]) or p.shape[0] != k + 1:
        raise ValueError("expected p=[K+1,V], q=[K,V]")
    for i, token_id in enumerate(draft_ids):
        qx = q[i, token_id]
        if qx.item() <= 0:
            raise ValueError("a draft token must have positive proposal probability")
        ratio = (p[i, token_id] / qx).clamp(max=1)
        if torch.rand((), device=p.device, generator=generator).item() < ratio.item():
            continue
        residual = (p[i] - q[i]).clamp_min(0)
        mass = residual.sum()
        # 真正发生拒绝时残差质量必定为正。不要静默回退到 p，那会掩盖算法错误。
        if not torch.isfinite(mass).item() or mass.item() <= 0:
            raise RuntimeError("rejection has no finite positive residual mass")
        replacement = sample(residual / mass, generator)
        return VerificationResult(draft_ids[:i] + [replacement], i)
    return VerificationResult(draft_ids + [sample(p[-1], generator)], k)
