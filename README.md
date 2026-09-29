<p align="center">
  <img width="300" src="assets/logo.png" alt="nano-vLLM logo">
</p>

# nano-vLLM：双模型投机解码

基于 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 的推理引擎学习与优化项目。主线是实现双模型投机解码，再通过可复现实验逐步优化调度、缓存、CPU/GPU 数据流和 CUDA Graph。

当前版本 **[v0](https://github.com/yangliangwei0623/nano-vllm/tree/v0)** 已完成正确性参考实现和普通解码性能基线。主实验使用 **RTX 5090 32 GB、Qwen3-4B target、Qwen3-0.6B draft、BF16**。核心实现包含中文注释，适合配合测试学习。

**V0 尚未实现性能加速。** 代码、完整测量数据和数值差异分析均已保留，作为后续优化的对照。

## 已实现的功能

- `temperature=0` 贪心模式，以及正温度下的精确拒绝采样。
- 一个 runner 管理两个模型，共同规划权重、运行时和两套分页 KV cache 的显存。
- 默认固定草稿长度 `K=4`，一次前向验证“最后一个已提交 token + K 个候选”，返回全部 `K+1` 个位置的 logits。
- 独立维护双模型 KV 有效长度，支持首次拒绝后的回退、全部接受后的草稿缓存补齐。
- 候选与已提交 token 分离，处理 EOS、输出长度上限、上下文上限和分块 prefill。
- 保持 `generate` 的 `[{"text": ..., "token_ids": ...}]` 返回格式，支持导出 JSON 统计。

V0 投机路径仅支持 **单 GPU、一个活动请求、Qwen3 dense 模型、固定 K、eager 执行**。传入多个请求时按队列串行处理；为每个请求一次预留完整输出预算所需的页，不共享前缀。普通解码保留原项目的批处理、前缀缓存及 CUDA Graph 等能力。

## 安装

本机验证环境：Python 3.12.3、PyTorch 2.8.0+cu128、Triton 3.4.0、CUDA Toolkit 12.8、FlashAttention 2.8.3、Transformers 4.57.6。项目要求 Python 3.10–3.12。

下面的安装步骤针对 **RTX 5090，且已安装兼容的 PyTorch / Triton 和 CUDA Toolkit 12.8** 的环境。Python 依赖使用清华镜像；其他 GPU 的 FlashAttention 编译架构需相应调整。

```bash
git clone https://github.com/yangliangwei0623/nano-vllm.git
cd nano-vllm

python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  setuptools==84.0.0 packaging==24.2 ninja==1.13.2 einops==0.8.2 \
  transformers==4.57.6 xxhash==4.0.1 modelscope==1.40.1

FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=120 \
  MAX_JOBS=8 NVCC_THREADS=2 \
  python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --no-build-isolation flash-attn==2.8.3

python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --no-deps --no-build-isolation -e .
python -m pip check
```

FlashAttention 从镜像下载源码并在本机编译，不转去 GitHub 下载预编译 wheel。完整安装记录见 [RTX 5090 国内源安装说明](docs/setup-rtx5090.md)。

## 下载模型

通过 ModelScope 国内源下载两个模型。以下路径与仓库现有示例一致，也可以替换为自己的目录。

```bash
python - <<'PY'
from modelscope import snapshot_download

for name in ("Qwen3-4B", "Qwen3-0.6B"):
    snapshot_download(
        f"Qwen/{name}",
        local_dir=f"/root/autodl-tmp/models/{name}",
    )
PY
```

初始化时会检查目标/草稿模型的词表大小、token ID 映射和特殊 token；不兼容的组合会明确报错。

## 快速开始

### 投机解码

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "/root/autodl-tmp/models/Qwen3-4B",
    draft_model="/root/autodl-tmp/models/Qwen3-0.6B",
    num_speculative_tokens=4,
    speculative_policy="fixed",
    enforce_eager=True,
    tensor_parallel_size=1,
)

try:
    prompt = llm.tokenizer.apply_chat_template(
        [{"role": "user", "content": "用两句话解释 KV cache。"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    outputs = llm.generate(
        [prompt],
        SamplingParams(temperature=0, max_tokens=128),
    )
    print(outputs[0]["text"])
    llm.export_stats("speculative-stats.json")
finally:
    llm.exit()
```

将 `temperature` 改为 `0.6` 即可使用精确拒绝采样。接受概率为 `min(1, p/q)`；首次拒绝时从归一化的 `(p-q)_+` 采样，全部接受时从目标模型额外采样一个 token。

### 普通解码

在独立运行的脚本中，使用下面的构造方式，并沿用上面的提示词处理和 `generate` 调用：

```python
llm = LLM(
    "/root/autodl-tmp/models/Qwen3-4B",
    enforce_eager=False,  # False：普通 CUDA Graph；True：普通 eager
    tensor_parallel_size=1,
)
```

不传 `draft_model` 即关闭投机。V0 的投机路径始终使用 eager；投机 CUDA Graph 和 adaptive 策略尚未实现。现有 [example.py](example.py) 和 [bench.py](bench.py) 使用 Qwen3-0.6B 普通解码，可用于原项目运行验证。

## 正确性验证

在完成依赖安装后运行无需模型权重的单元测试：

```bash
python -m unittest discover -s tests -v
```

已完成的验证包括：

- **13 项单元测试通过**：0–4 个接受长度、精确拒绝采样、缓存状态机、EOS、长度限制及页引用释放。
- 小词表 **20,000 次试验**验证两 token 联合分布，不以随机文本同 seed 一致作为标准。
- 真实 4B 模型多 token 验证 logits 与逐 token 前向在固定 BF16 容差内一致，覆盖页边界和长前缀。
- 普通 CUDA Graph 与 eager 的贪心测试 **7/7 完全一致**；投机与普通 eager **6/7 完全一致**，另 1 例首次分叉源于 BF16 logits 并列，已保留双方 top-2 证据单独分析。

真实 GPU 验收需要顺序运行以下命令：

```bash
PYTHONPATH=. python tests/validate_v0_gpu.py \
  --model /root/autodl-tmp/models/Qwen3-4B --mode eager \
  --output /tmp/v0-target-eager.json

PYTHONPATH=. python tests/validate_v0_gpu.py \
  --model /root/autodl-tmp/models/Qwen3-4B --mode graph \
  --reference /tmp/v0-target-eager.json --output /tmp/v0-target-graph.json

PYTHONPATH=. python tests/validate_v0_gpu.py \
  --model /root/autodl-tmp/models/Qwen3-4B \
  --draft /root/autodl-tmp/models/Qwen3-0.6B --mode spec \
  --reference /tmp/v0-target-eager.json --output /tmp/v0-speculative.json
```

## 实测性能

RTX 5090 32 GB，Qwen3-4B target / Qwen3-0.6B draft，BF16，batch=1，`temperature=0.6`，固定输出 256 tokens，忽略 EOS。每个配置预热后重复五次，下面是 **decode 总耗时 / 实际 decode 输出 token 数** 的中位数，单位 ms/token，越小越好。

| 输入 tokens | 普通 eager | 普通 CUDA Graph | V0 投机 eager |
|---:|---:|---:|---:|
| 128 | 20.133 | 6.306 | 35.033 |
| 1024 | 20.065 | 6.491 | 24.104 |

普通基线来自原始提交 `a468976`。基线只加载 target，投机版加载两个模型，这是用户视角的比较。提示文本重复后截取定长，属于可控长度的工程负载，不代表真实任务整体表现；128-token 投机测量的波动较大。当前 V0 比普通 eager 和 Graph 都慢，高接受率也没有保证加速。

完整的 batch 1/2/4/8 数据、五次结果的范围、显存、接受率、数值并列分析见 [V0 实测报告](docs/v0-results.md)。[原始 JSON 和校验记录](results/v0/) 可用于复核。

复现原版基线的 worktree 命令，以及普通/投机性能脚本的用法，见 [V0 实现说明：基线性能测量](docs/speculative-v0.md#基线性能测量)。诊断测试与正式计时分开运行，首次编译、预热和 graph 捕获不计入稳态结果。

## 代码阅读导航

| 文件 / 文档 | 学习内容 |
|---|---|
| [speculative_sampling.py](nanovllm/engine/speculative_sampling.py) | 贪心验证、接受概率与残差采样 |
| [speculative_runner.py](nanovllm/engine/speculative_runner.py) | 双模型显存规划、K+1 位置验证、KV 回退与补齐 |
| [speculative_scheduler.py](nanovllm/engine/speculative_scheduler.py) | 串行调度、候选提交、结束条件和页释放 |
| [context.py](nanovllm/utils/context.py)、[embed_head.py](nanovllm/layers/embed_head.py) | attention 模式与 logits 位置选择解耦 |
| [test_speculative.py](tests/test_speculative.py) | 可控状态机测试与采样分布验证 |
| [V0 学习文档](docs/speculative-v0.md) | 推荐阅读顺序、逐轮缓存长度表与算法推导 |

## 后续路线

| 版本 | 状态 | 目标 |
|---|---|---|
| V0 | 已完成 | 正确性参考实现、注释、测试和普通解码基线 |
| V1 | 计划中 | 混批、按轮预留 KV 页、共享前缀和抢占重算 |
| V2 | 计划中 | GPU 驻留数据流，减少同步与重复张量构造 |
| V3 | 计划中 | 草稿模型与目标验证的 CUDA Graph |
| V4 | 计划中 | 自适应 K 与自动退化 |
| V5 | 计划中 | 性能消融、profiler 证据及适用边界报告 |

## 致谢与许可证

本项目基于 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)，保留其轻量推理引擎作为学习和实验基础。上游发布的其他硬件性能数字请参考上游仓库，本仓库的实测结论以 `results/v0/` 和对应报告为准。

代码遵循 [MIT License](LICENSE)。
