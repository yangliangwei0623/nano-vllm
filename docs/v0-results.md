# V0 实测结果（RTX 5090）

测试日期：2026-09-29 UTC。target=Qwen3-4B，draft=Qwen3-0.6B，BF16；Python 3.12.3、PyTorch 2.8.0+cu128、FlashAttention 2.8.3、Transformers 4.57.6。

## 普通解码基线

在原项目提交 `a468976` 的独立 worktree 中测量。temperature=0.6，固定输出 256 tokens，ignore_eos=True，max_model_len=2048，max_num_seqs=8，max_num_batched_tokens=8192，gpu_memory_utilization=0.8。每种形状预热，随后五次独立请求批次；每批清除前缀缓存索引。

下表为 **decode 阶段总耗时 / 实际 decode 输出 token 数**，单位 ms/token，格式为中位数 [最小值, 最大值]。分母不包括 prefill 产生的首 token；batch>1 时不是单请求 inter-token latency。

| 输入 tokens | Batch | 普通 eager | 普通 CUDA Graph | eager / Graph 成本比 |
|---:|---:|---:|---:|---:|
| 128 | 1 | 20.133 [19.963, 20.515] | 6.306 [6.301, 6.307] | 3.19× |
| 128 | 2 | 11.201 [11.051, 11.458] | 3.429 [3.427, 3.431] | 3.27× |
| 128 | 4 | 5.633 [5.585, 5.670] | 1.740 [1.738, 1.743] | 3.24× |
| 128 | 8 | 2.779 [2.735, 2.817] | 0.965 [0.964, 0.968] | 2.88× |
| 1024 | 1 | 20.065 [19.895, 20.292] | 6.491 [6.481, 6.497] | 3.09× |
| 1024 | 2 | 11.128 [11.018, 11.280] | 3.496 [3.493, 3.502] | 3.18× |
| 1024 | 4 | 5.630 [5.608, 5.674] | 1.786 [1.785, 1.787] | 3.15× |
| 1024 | 8 | 2.741 [2.731, 2.802] | 1.049 [1.049, 1.050] | 2.61× |

原始记录：[eager](../results/v0/baseline-eager.json)、[CUDA Graph](../results/v0/baseline-graph.json)。各 trial 同时记录首 token 时间、端到端吞吐、实际输出数量及 PyTorch peak allocated 显存（不含驱动等非 PyTorch 分配）。普通 eager 测得峰值约 23.82–24.46 GiB，Graph 约 23.83–24.46 GiB。

## V0 投机参考版

同样使用 batch=1、temperature=0.6、固定 K=4（末轮可缩短），每种输入长度五次测量。除长生成预热外，额外预热 K=0..4 的末轮形状。以下统计只累计正式计时区，排除预热。

| 输入 tokens | 投机 ms/token，中位数 [min,max] | 相对普通 eager 成本倍数 | 相对普通 Graph 成本倍数 | 接受候选 / 提议候选 | 每轮实际提交 |
|---:|---:|---:|---:|---:|---:|
| 128 | 35.033 [24.244, 38.886] | 1.74× | 5.56× | 61.90% | 3.446 |
| 1024 | 24.104 [24.046, 24.461] | 1.20× | 3.71× | 100.00% | 5.000 |

原始记录：[投机 eager](../results/v0/speculative-eager.json)。V0 **没有实现性能加速**。1024-token 的重复文本负载即使候选接受率达到 100%，仍比普通 eager 慢约 20%；接受率高并不保证收益。128-token 负载接受率随随机生成轨迹变化，五次成本波动明显，不能仅选最快一次。

该负载使用自编提示文本重复后截取定长，尤其 1024-token 输入有强重复结构；它是长度可控的工程基线，不代表真实自然语言/代码任务的接受率。ignore_eos=True 下仍继续生成，主实验只评估固定输出预算的执行成本。

基线只加载 target，投机版加载两模型，属于用户视角比较，不是相同权重驻留的算法消融。投机版显存规划记录为 245 个逻辑页，每页合计 64 MiB，两套 KV 合计 15.31 GiB，额外 runtime 预留 256 MiB；实测 PyTorch 峰值约 24.01–24.08 GiB。

参考版保留逐草稿 token 的 Python 控制、张量构造、标量采样同步和额外 KV 补算。日志也记录了原有 RMSNorm 的 torch.compile 重编译上限回退。当前测量不能把开销分别归因给这些因素；需在 V2 用 profiler 做阶段测量，V3 再验证 graph 的增益。此处不虚构 profiler 结论。

## 正确性验收

- 13 项 CPU 单元测试通过，记录见 [unit-tests.txt](../results/v0/unit-tests.txt)。
- 小词表 20,000 次试验验证两 token 联合分布，而非比较随机文本同 seed 一致。
- runner 状态机对每个接受长度 0/1/2/3/4 都进行受控前向测试，验证回退值、候选与提交隔离、全接受补齐的具体位置。
- 真实 4B + 0.6B 测试覆盖 EOS、ignore_eos、max_tokens=1、上下文上限、分块 prefill、255/257 token 页边界、请求结束后的全部引用释放。
- 正确性运行中的 80 轮接受长度直方图为 0:8、1:2、2:11、3:6、4:53，61 次补算各一个 token（含 K=0 等末轮）；这些是诊断统计，不混入正式吞吐。
- 同模型 0.6B + 0.6B 补充测试通过，覆盖高接受率与真实 GPU 上的全接受补齐。

### 多 token logits

在 prefix=0/254/255/256/1023、query=5 的情况下，4B 多查询验证与逐 token decode 的最大绝对误差分别为 0.28125、0.25、0.203125、0.203125、0.125；平均绝对误差为 0.0173–0.0427，无 top-1 不一致。均满足先在 0.6B 校准后固定的 BF16 容差（atol=0.75，rtol=0.02，平均绝对误差<0.1）。

进一步对照单 query varlen attention 与单 query decode，误差均为 0，因此本组差异出现在查询形状变化时；没有证据说明缓存索引出错。0.6B 短前缀校准出现过最大差异 0.625 和一次 top-1 并列，完整诊断保留在 [same-model-gpu.json](../results/v0/same-model-gpu.json)。

### 贪心逐 token 一致性与并列分析

普通 CUDA Graph 与普通 eager：7/7 案例完全相同。投机与普通 eager：6/7 完全相同，另一个案例存在一次首次分叉，不能称为全部逐 token 完全一致。

分叉案例为“List the first five prime numbers.”，输出第 10 个 token（零基索引 9）：

| 路径 | token 220（空格）的 logit | token 3070（` **`）的 logit | 实际 argmax |
|---|---:|---:|---|
| 普通 eager | 46.5 | 46.5 | 220（并列时取较小 ID） |
| 多查询验证 | 46.5 | 46.75 | 3070 |

首次分叉前历史一致。BF16 查询形状变化打破了基线的精确并列，后续自然沿不同历史生成。这一案例单独标记为近并列差异，没有修改 argmax 规则来强行匹配，也没有删除该案例。验收脚本分别输出 `all_greedy_match=false` 与 `all_match_or_explained_near_tie=true`；未解释的分叉仍使测试失败。

完整记录：[普通 eager](../results/v0/target-eager-correctness.json)、[普通 Graph](../results/v0/target-graph-correctness.json)、[投机](../results/v0/speculative-correctness.json)。学习路线、缓存长度表及复现命令见 [V0 实现说明](speculative-v0.md)。

## 后续边界

本次完成 V0 的正确性参考与普通解码基线；不包含 V1 混批/共享前缀、V2 GPU 驻留采样、V3 双模型 graph、V4 自适应 K。V0 为整个请求预留页、串行处理请求。上述性能数字只适用于这里记录的模型、硬件、软件、提示构造和采样设置；尚未提供真实数据集的速度提升或简历加速百分比。

模型权重、tokenizer、环境版本及 V0 源码校验值见 [provenance.json](../results/v0/provenance.json)。投机记录中的 `source_commit` 是开发基点 `a468976`，实际测试的是含 V0 改动的工作树，由该校验清单及保存这些结果的 V0 提交共同标识。
