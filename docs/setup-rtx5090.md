# RTX 5090 国内源安装与原项目验证

环境：Python 3.12.3、RTX 5090 32 GB、CUDA Toolkit 12.8、预装
PyTorch 2.8.0+cu128 / Triton 3.4.0。以下命令在仓库根目录执行，使用当前
`python` 环境，不需要重装已有的 PyTorch。

## 安装

Python 依赖使用清华 PyPI 镜像。固定 Transformers 4.x，避免安装到未经本项目验证的主版本。

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  setuptools==84.0.0 packaging==24.2 transformers==4.57.6 \
  xxhash==4.0.1 ninja==1.13.2 einops==0.8.2 modelscope==1.40.1

mkdir -p /root/autodl-tmp/build-tmp /root/autodl-tmp/setup-logs
FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=120 \
  MAX_JOBS=8 NVCC_THREADS=2 TMPDIR=/root/autodl-tmp/build-tmp \
  python -m pip install -v -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --no-build-isolation flash-attn==2.8.3 \
  > /root/autodl-tmp/setup-logs/flash-attn-install.log 2>&1

python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --no-deps --no-build-isolation -e .
python -m pip check
```

FlashAttention 从镜像下载源码并在本机编译，`FLASH_ATTENTION_FORCE_BUILD` 禁止
安装脚本转去 GitHub 下载预编译 wheel。`FLASH_ATTN_CUDA_ARCHS=120` 只编译
RTX 5090 所需架构，因此生成的 wheel 不适合作为其他架构 GPU 的通用安装包。
编译需要本机 `nvcc`；这里的 CUDA Toolkit 与 PyTorch 均为 12.8。

## 下载模型

使用 ModelScope 国内源，路径与现有 `example.py`、`bench.py` 一致：

```bash
python - <<'PY'
from modelscope import snapshot_download
snapshot_download(
    'Qwen/Qwen3-0.6B',
    local_dir='/root/autodl-tmp/models/Qwen3-0.6B',
)
PY
```

## 运行

```bash
python example.py > /root/autodl-tmp/setup-logs/example.log 2>&1
python bench.py > /root/autodl-tmp/setup-logs/bench.log 2>&1
```

`example.py` 使用 eager 模式生成两条回复；`bench.py` 使用 CUDA Graph，
预热后运行 256 个请求，输入和输出长度由脚本固定随机种子生成。
其吞吐包含 prefill 和 decode，单次结果只作为运行验证，不能当作解码阶段的正式性能结论。
首次运行包含 Triton / Torch 编译与 CUDA Graph 捕获开销。

## 本机验证记录

- `python -m pip check`：通过，`No broken requirements found.`
- FlashAttention 2.8.3：73 个编译单元全部完成，导入成功。
- `python example.py`：退出码 0，两条请求生成完成。原示例使用思考模式，
  `max_tokens=256` 包含思考内容，因此回复可能在思考过程中达到长度上限。
- `python bench.py`：退出码 0，CUDA Graph 路径运行成功。单次预热后结果：
  `Total: 133966tok, Time: 12.92s, Throughput: 10369.44tok/s`。
  这是原脚本的端到端吞吐统计，不是重复实验中位数，也不是单独 decode 吞吐。
- 安装日志、运行日志和完整依赖快照保存在 `/root/autodl-tmp/setup-logs/`。
  `environment.json` 记录核心依赖版本，`model-manifest.json` 记录模型文件 SHA-256，
  `pip-freeze.txt` 记录当前环境完整包列表。
