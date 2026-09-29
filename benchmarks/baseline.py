"""普通解码基线：分别启动 eager / graph 进程，避免两个引擎争用显存。

示例：python benchmarks/baseline.py --model /path/Qwen3-4B --mode eager \
          --output results/v0/target-eager.json
此脚本也可通过 PYTHONPATH 指向原始 worktree，测量未修改的原项目。
"""
import argparse
import atexit
import hashlib
import json
import platform
import statistics
import subprocess
from pathlib import Path
from time import perf_counter

import torch
import nanovllm
from nanovllm import LLM, SamplingParams


TEXTS = [
    "Explain how a paged key value cache helps a language model serve requests with different lengths. ",
    "Write a Python function that merges two sorted lists, and explain its time complexity. ",
    "Describe the differences between CPU scheduling, GPU execution, and memory bandwidth. ",
    "Explain how to test a sampling algorithm using small probability distributions. ",
]


def clear_prefix_cache(llm):
    """只清理已结束请求的前缀索引，防止重复提示词造成第二次起的假加速。"""
    bm = llm.scheduler.block_manager
    assert not bm.used_block_ids and llm.is_finished()
    bm.hash_to_block_id.clear()
    for block in bm.blocks:
        block.hash = -1
        block.token_ids = []


def trial(llm, prompts, output_tokens, temperature):
    clear_prefix_cache(llm)
    sp = SamplingParams(temperature=temperature, ignore_eos=True, max_tokens=output_tokens)
    stats_before = getattr(llm.model_runner, "stats", {}).copy()
    torch.cuda.synchronize()
    start = perf_counter()
    for prompt in prompts:
        llm.add_request(prompt, sp)
    outputs, count = llm.step()
    # 本实验输入预算足以让整个 batch 一次 prefill；首 token 已由这个 step 产生。
    assert count == sum(map(len, prompts))
    torch.cuda.synchronize()
    first = perf_counter()
    decode_tokens = 0
    while not llm.is_finished():
        done, count = llm.step()
        assert count < 0
        decode_tokens -= count
        outputs.extend(done)
    torch.cuda.synchronize()
    end = perf_counter()
    actual = sum(len(ids) for _, ids in outputs)
    assert actual == len(prompts) * output_tokens
    assert decode_tokens == len(prompts) * (output_tokens - 1)
    stats_after = getattr(llm.model_runner, "stats", {})
    counters = {k: v-stats_before.get(k, 0) for k,v in stats_after.items() if isinstance(v, int)}
    return dict(total_seconds=end-start, prefill_first_token_seconds=first-start,
                decode_seconds=end-first, output_tokens=actual, decode_tokens=decode_tokens,
                decode_ms_per_token=1000*(end-first)/decode_tokens,
                end_to_end_tokens_per_second=actual/(end-start),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), speculative_counters=counters)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", help="optional V0 draft model; batch=1 and eager only")
    parser.add_argument("--mode", choices=["eager", "graph"], required=True)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--input-lengths", nargs="+", type=int, default=[128, 1024])
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.draft and (args.batches != [1] or args.mode != "eager"):
        parser.error("V0 speculative benchmark requires --batches 1 --mode eager")
    torch.manual_seed(2026)
    config = dict(max_num_seqs=max(8, max(args.batches)), max_model_len=2048,
                  max_num_batched_tokens=max(8192, max(args.batches)*max(args.input_lengths)),
                  gpu_memory_utilization=0.8, enforce_eager=args.mode == "eager")
    if args.draft:
        config["draft_model"] = args.draft
    init_start = perf_counter()
    llm = LLM(args.model, **config)
    torch.cuda.synchronize()
    init_seconds = perf_counter()-init_start
    source = Path(nanovllm.__file__).resolve().parent.parent
    report = dict(arguments=vars(args), config=config, python=platform.python_version(),
                  torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(), dtype=str(llm.model_runner.config.hf_config.dtype),
                  source=str(source), initialization_seconds=init_seconds,
                  warmup=dict(full_generations_per_cell=1, additional_output_budgets=[2,3,4,5,6] if args.draft else []),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  memory_plan=getattr(llm.model_runner, "memory_plan", {}),
                  source_commit=subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
                  prompt_source="self-authored TEXTS in benchmarks/baseline.py; repeated then token-truncated",
                  prefix_cache="cleared before every warmup and trial", cells=[])
    try:
        for length in args.input_lengths:
            for batch in args.batches:
                prompts = [llm.tokenizer.encode(TEXTS[i % len(TEXTS)] * (length // 8 + 1))[:length] for i in range(batch)]
                assert all(len(p) == length for p in prompts)
                trial(llm, prompts, args.output_tokens, args.temperature)  # 每种形状单独预热，排除 JIT。
                if args.draft:
                    # 末轮会出现 K=0..4，逐一预热，避免某种截短形状首次出现在计时区。
                    for budget in range(2, 7):
                        trial(llm, prompts, budget, args.temperature)
                torch.cuda.reset_peak_memory_stats()
                rows = [trial(llm, prompts, args.output_tokens, args.temperature) for _ in range(args.repeats)]
                values = [r["decode_ms_per_token"] for r in rows]
                cell = dict(batch=batch, input_length=length,
                            prompt_ids_sha256=hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
                            decode_ms_per_token_median=statistics.median(values),
                            decode_ms_per_token_min=min(values), decode_ms_per_token_max=max(values),
                            trials=rows)
                report["cells"].append(cell)
                path = Path(args.output)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(report, indent=2)+"\n")
                print(f"{args.mode}: batch={batch} input={length} decode={statistics.median(values):.4f} ms/token", flush=True)
    finally:
        llm.exit()
        atexit.unregister(llm.exit)  # 兼容未实现幂等 exit 的原始基线 worktree。


if __name__ == "__main__":
    main()
