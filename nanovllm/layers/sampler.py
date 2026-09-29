import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # 将贪心行的分母设为 1，避免除零；最后选择该行的 argmax。
        # 不原地修改传入 logits：投机验证/测试可能仍需要原始分布。
        greedy = temperatures == 0
        logits = logits.float() / torch.where(greedy, 1., temperatures).unsqueeze(1)
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return torch.where(greedy, logits.argmax(dim=-1), sample_tokens)
