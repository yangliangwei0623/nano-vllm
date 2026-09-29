# V0：双模型投机解码参考实现

## 范围与入口

目标模型为 Qwen3-4B，草稿模型为 Qwen3-0.6B；单 GPU、一个活动请求、固定
K=4、BF16、投机路径 eager。`generate` 可以接收多个请求，但 V0 将它们排队
串行执行，不把串行吞吐冒充混批吞吐。普通解码仍支持原有 batching / CUDA Graph。

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "/root/autodl-tmp/models/Qwen3-4B",
    draft_model="/root/autodl-tmp/models/Qwen3-0.6B",
    num_speculative_tokens=4,
    enforce_eager=True,
    tensor_parallel_size=1,
)
prompt = llm.tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain a KV cache in two sentences."}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
outputs = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=128))
print(outputs[0]["text"])
llm.export_stats("speculative-stats.json")
llm.exit()
```

`temperature=0` 为贪心，正值为完整词表温度采样（没有 top-k / top-p）。
`draft_model=None` 关闭投机；V0 明确拒绝投机多卡、非 Qwen3 dense 组合、不兼容
词表、adaptive 策略及投机 graph 请求。普通模式的 `enforce_eager=False` 开启 graph。

## 推荐阅读顺序

1. `sampling_params.py` 与 `layers/sampler.py`：温度为零时如何避免除零，以及
   混合温度 batch 如何选择 argmax。采样不再原地破坏调用者的 logits。
2. `engine/speculative_sampling.py`：先读贪心接受，再读精确拒绝采样。
   所有分布运算使用 FP32；独立于模型前向，可在 CPU 小词表中验证。
3. `utils/context.py` 与 `layers/embed_head.py`：`all_logits` 把输出位置选择
   与 attention 执行模式分开。普通 prefill 依然只算末位 logits。
4. `engine/speculative_runner.py`：双模型加载、共同显存预算、查询区间、
   草稿循环、一次验证以及 KV 修复。代码中的中文注释对应下文的长度公式。
5. `engine/speculative_scheduler.py`：提交与终止、EOS 截断、页分配和释放。
6. `tests/test_speculative.py` 与 `tests/validate_v0_gpu.py`：学习如何构造
   可控的小例子，以及如何把真实模型的数值差异与状态管理错误区分开。

## 一轮的状态变化

设已提交序列长度为 n，两模型有效 KV 长度均为 n−1。最后一个已提交 token
尚未做前向；下轮从它开始，而不是重复处理整个上下文。

| 时刻 | target 有效 KV | draft 有效 KV | 新的已提交 token |
|---|---:|---:|---|
| 轮开始 | n−1 | n−1 | 无 |
| 生成 K 个草稿后 | n−1 | n+K−1 | 无，候选存在局部列表 |
| 验证后、决定前 | n+K | n+K−1 | 无 |
| 接受 a<K 个，随后拒绝 | n+a | n+a | a 个候选 + 一个替代 token |
| 全接受，并补齐 draft | n+K | n+K | K 个候选 + 一个额外 token |

新提交的替代/额外 token 本身没有 KV。新长度为 n+a+1，因此轮结束后的 KV
长度又等于新长度减一。全接受时 draft 缺的是最后一个草稿 token 的 KV，
不是新采样的额外 token；补错这里会把后续上下文整体错开一位。

验证输入是 `[最后已提交 token, d1, ..., dK]`。第 0 行 logits 预测 d1，
第 K−1 行预测 dK，第 K 行预测额外 token。分页 varlen attention 的 causal
mask 按右下角对齐，query 范围 `[n−1,n+K)` 对应 key 范围 `[0,n+K)`。

拒绝时不清零 KV 张量：降低有效长度即可屏蔽无效尾部，下一轮覆盖对应 slot。
候选从未写入 `Sequence.token_ids`。EOS 可能位于接受前缀中间，调度器只提交
到 EOS，并释放整个请求的页。

## 为什么随机分布正确

目标与草稿温度分布分别为 p 和 q。候选 x 来自 q，接受概率为
`min(1, p[x]/q[x])`。于是接受事件对 token x 的概率质量贡献为 `min(p[x],q[x])`。
拒绝概率为 `sum((p-q).clamp_min(0))`，拒绝后按归一化残差抽样，贡献
`(p[x]-q[x])_+`。两项之和就是 p[x]。

一旦拒绝，后续候选使用了错误的条件历史，因此全部丢弃。全接受时才使用
目标验证的第 K 行抽样。p 和 q 必须使用相同 temperature，q 必须是生成
该候选时保存的分布，不能在抽样时原地修改它。浮点运算意味着“精确”是指
相对于当前计算得到的 p/q 分布，而不是任意精度的实数模型。

不同算法消耗随机数的顺序不同，随机文本同 seed 不一致不是错误。验收使用
小词表的两 token 联合频率，与可直接计算的目标概率比较。

## 显存与调度的 V0 取舍

一个 runner / 一个 NCCL 进程组持有两个模型。先共同加载权重，再逐个预热，
根据共同驻留显存、预热瞬时峰值、额外 logits/probability 预算规划 KV 页数。
每个逻辑页的成本等于两模型所有层 K/V 页的字节数之和；物理张量各自独立。

V0 在请求进入运行态时，为 `min(prompt+max_tokens, max_model_len)` 一次预留
全部页，结束时全部释放；不使用前缀共享，不发布候选 hash。这个选择使回退
正确性容易检查，但保守使用显存，不能代表 V1 的按轮分配策略。预算超出容量
会明确报错。prefill 支持按 token budget 分块，候选 K 在输出/上下文末尾缩短，
只剩一个输出位置时 K=0，执行目标模型一步并同步草稿缓存。

参考实现每个草稿 token 都有 Python 控制和 GPU/CPU 同步，接受判定也在 Python
循环内。暂不声称消除了同步，不预期在所有负载上加速；这是后续 V2/V3 的对照。

## 验收与复现

```bash
python -m unittest discover -s tests -v

PYTHONPATH=. python tests/validate_v0_gpu.py \
  --model /root/autodl-tmp/models/Qwen3-4B --mode eager \
  --output results/v0/target-eager-correctness.json
PYTHONPATH=. python tests/validate_v0_gpu.py \
  --model /root/autodl-tmp/models/Qwen3-4B --mode graph \
  --reference results/v0/target-eager-correctness.json \
  --output results/v0/target-graph-correctness.json
PYTHONPATH=. python tests/validate_v0_gpu.py \
  --model /root/autodl-tmp/models/Qwen3-4B \
  --draft /root/autodl-tmp/models/Qwen3-0.6B --mode spec \
  --reference results/v0/target-eager-correctness.json \
  --output results/v0/speculative-correctness.json
```

三条 GPU 命令顺序执行，避免争用 GPU。GPU 验收包括页边界前缀、分块 prefill、
长度为 1 的输出、上下文上限、真实模型生成的 token 作为人工 EOS、随机模式
状态检查和请求结束后页引用归零。

BF16 的不同 GEMM/attention 形状不保证 bitwise 相同。0.6B 校准观察到极短前缀
的最大 logits 差异 0.625；4B 主实验固定使用 `atol=0.75, rtol=0.02`，并限制
平均绝对误差 <0.1，原始误差和 top-1 差异保存在 JSON。完整文本若不同，记录
首次分叉时两种执行方式的 top-2；只有相同两 token 且双方间隔均不超过 0.25
才单独分类为近并列差异，否则测试失败。`all_greedy_match` 与
`all_match_or_explained_near_tie` 分别报告，不能把后者称为逐 token 完全一致。

## 基线性能测量

`benchmarks/baseline.py` 在每种 `(batch,input_length)` 形状预热一次，再运行
五次，保存所有结果、中位数和 min/max。每轮清空已释放页的前缀缓存索引；提示
词是脚本中自编的自然语言/代码问题，重复后截取到 128/1024 tokens，用于可控
长度的引擎测量，不作为模型质量评测。输出固定 256 tokens，忽略 EOS。

计时使用请求级 CUDA 同步边界，decode 循环不增加逐阶段强制同步。prefill
产生首 token，decode 分母为实际 decode 提交数 `batch*(256−1)`。报告
`decode_seconds/decode_tokens`，这是整批每输出 token 的平均成本，不是每请求
真实逐 token 延迟分布。同时记录首 token 时间、端到端吞吐和 PyTorch 峰值显存。

原项目快照 `a468976` 可单独建立 worktree，并使用以下方式运行本脚本：

```bash
git worktree add --detach /root/autodl-tmp/nano-vllm-baseline a468976
PYTHONPATH=/root/autodl-tmp/nano-vllm-baseline python benchmarks/baseline.py \
  --model /root/autodl-tmp/models/Qwen3-4B --mode eager \
  --output results/v0/baseline-eager.json
PYTHONPATH=/root/autodl-tmp/nano-vllm-baseline python benchmarks/baseline.py \
  --model /root/autodl-tmp/models/Qwen3-4B --mode graph \
  --output results/v0/baseline-graph.json
```

JSON 记录实际加载的源码目录/提交、GPU、精度、配置和输入 token 校验值。原项目
不支持 temperature=0，原版基线使用 temperature=0.6。贪心正确性对照使用新增
贪心分支后的普通目标解码。性能数字与正确性诊断分开运行；详细结果见同目录的
`v0-results.md`。

V0 投机性能可用同一脚本测量（仅 batch=1）；它会额外预热 K=0..4 的末轮形状：

```bash
PYTHONPATH=. python benchmarks/baseline.py \
  --model /root/autodl-tmp/models/Qwen3-4B \
  --draft /root/autodl-tmp/models/Qwen3-0.6B --batches 1 --mode eager \
  --output results/v0/speculative-eager.json
```

该比较中原项目仅加载 target，投机版加载两模型，显存驻留不同；不能解读为
相同驻留预算下的算法消融。原始结果保留接受计数、补算次数及实际提交数量。
