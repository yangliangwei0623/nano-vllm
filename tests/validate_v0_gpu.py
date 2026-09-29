"""真实模型验收。三种模式分别启动进程，避免重复加载模型污染显存。

python tests/validate_v0_gpu.py --model ... --mode eager --output /tmp/eager.json
python tests/validate_v0_gpu.py --model ... --mode graph --reference /tmp/eager.json --output ...
python tests/validate_v0_gpu.py --model ... --draft ... --mode spec --reference /tmp/eager.json --output ...
"""
import argparse
import json
from pathlib import Path

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import set_context, reset_context


@torch.inference_mode()
def logits_check(llm):
    """同一 target、同一 KV 页：K+1 查询 vs 逐 token decode，覆盖块边界。"""
    runner = llm.model_runner
    records = []
    for n in (1, 255, 256, 257, 1024):
        tokens = (llm.tokenizer.encode("A paged cache stores keys and values for each token. ") * 200)[:n+4]
        seq = Sequence(tokens[:n], SamplingParams(temperature=0, max_tokens=16))
        llm.scheduler.add(seq)
        llm.scheduler.schedule()  # 只分配页，直接测试 runner，不提交任何候选。
        if n > 1:
            runner.forward_span(runner.model, seq, tokens, 0, n-1)
        many = runner.forward_span(runner.model, seq, tokens, n-1, n+4, all_logits=True).float()
        rows = []
        for position in range(n-1, n+4):
            # 明确给出逐 token decode 的物理 slot，不使用预留页表的最后一页。
            page = seq.block_table[position // runner.block_size]
            ids = torch.tensor([tokens[position]], device="cuda")
            positions = torch.tensor([position], device="cuda")
            set_context(False,
                        slot_mapping=torch.tensor([page*runner.block_size+position % runner.block_size], dtype=torch.int32, device="cuda"),
                        context_lens=torch.tensor([position+1], dtype=torch.int32, device="cuda"),
                        block_tables=torch.tensor([seq.block_table], dtype=torch.int32, device="cuda"))
            try:
                rows.append(runner.model.compute_logits(runner.model(ids, positions)).float())
            finally:
                reset_context()
        single = torch.cat(rows)
        # 用单 query 的同一种 varlen attention 再对照一次，帮助定位 decode kernel
        # 与查询形状变化的影响。0.6B 校准发现短前缀可有 0.625 的 BF16 差异，
        # 因而主实验 4B 事先固定 atol=.75 / rtol=.02，并限制平均误差 < .1。
        # 所有误差和 top-1 并列均保存，不用调宽容差掩盖贪心文本差异。
        prefill_single = torch.cat([runner.forward_span(runner.model, seq, tokens, i, i+1, all_logits=True).float()
                                    for i in range(n-1, n+4)])
        within_tolerance = bool(torch.isclose(many, single, atol=.75, rtol=.02).all()) and float((many-single).abs().mean()) < .1
        disagreements = []
        for i in range(5):
            if many[i].argmax() != single[i].argmax():
                a = single[i].topk(2).values
                disagreements.append(dict(position=n-1+i, single_margin=float(a[0]-a[1]),
                                          many_id=int(many[i].argmax()), single_id=int(single[i].argmax())))
        records.append(dict(prefix_length=n-1, queries=5, within_tolerance=within_tolerance, max_abs_error=float((many-single).abs().max()),
                            mean_abs_error=float((many-single).abs().mean()),
                            varlen_single_max_abs_error=float((many-prefill_single).abs().max()),
                            decode_vs_varlen_single_max_abs_error=float((single-prefill_single).abs().max()),
                            top1_disagreements=disagreements))
        print("logits", records[-1], flush=True)
        llm.scheduler.block_manager.deallocate(seq)
        llm.scheduler.running.clear()
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft")
    parser.add_argument("--mode", choices=["eager", "graph", "spec"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference")
    args = parser.parse_args()
    torch.manual_seed(123)
    llm = LLM(args.model, draft_model=args.draft if args.mode == "spec" else None,
              enforce_eager=args.mode != "graph", max_num_seqs=8,
              max_num_batched_tokens=256, max_model_len=2048, gpu_memory_utilization=.65)
    report = dict(mode=args.mode, model=args.model, draft=args.draft, cases=[], logits=[])
    try:
        if args.mode == "spec":
            report["logits"] = logits_check(llm)
            llm.model_runner.trace = []
        baseline_trace = []
        if args.mode != "spec":
            original_forward = llm.model_runner.run_model
            def traced_forward(*args, **kwargs):
                logits = original_forward(*args, **kwargs)
                values, indices = logits.float().topk(2, dim=-1)
                baseline_trace.append(dict(top_ids=indices[0].tolist(), top_values=values[0].tolist()))
                return logits
            llm.model_runner.run_model = traced_forward
        text = ["What is the capital of France? Answer in one sentence.",
                "Write a Python function to add two integers.",
                "List the first five prime numbers."]
        prompts = [llm.tokenizer.apply_chat_template([dict(role="user", content=t)],
                   tokenize=True, add_generation_prompt=True, enable_thinking=False) for t in text]
        repeat = llm.tokenizer.encode("Explain key value caching in an autoregressive transformer. ") * 40
        cases = [("natural_"+str(i), p, 64, False) for i, p in enumerate(prompts)]
        cases += [("page_255", repeat[:255], 24, False), ("page_257", repeat[:257], 24, False),
                  ("one_output", prompts[0], 1, False), ("context_limit", prompts[1], 8, True)]
        reference = json.loads(Path(args.reference).read_text()) if args.reference else None
        for index, (name, prompt, budget, context_limit) in enumerate(cases):
            baseline_trace.clear()
            if args.mode == "spec":
                llm.model_runner.trace.clear()
            if context_limit:
                llm.config.max_model_len = len(prompt)+2
                if hasattr(llm.scheduler, "max_model_len"):
                    llm.scheduler.max_model_len = len(prompt)+2
            out = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=budget, ignore_eos=True), use_tqdm=False)[0]
            expected_count = 2 if context_limit else budget
            assert len(out["token_ids"]) == expected_count
            match = reference is None or out["token_ids"] == reference["cases"][index]["token_ids"]
            trace = list(llm.model_runner.trace) if args.mode == "spec" else baseline_trace[-expected_count:].copy()
            divergence = None
            if not match:
                ref = reference["cases"][index]
                j = next(i for i, (a,b) in enumerate(zip(ref["token_ids"], out["token_ids"])) if a != b)
                base = ref["trace"][j]
                observed = trace[j] if args.mode != "spec" else None
                if args.mode == "spec":
                    for row in trace:
                        offset = j - (row["committed_length"]-len(prompt))
                        if 0 <= offset < len(row["top_ids"]):
                            observed = dict(top_ids=row["top_ids"][offset], top_values=row["top_values"][offset])
                tied = (observed is not None and set(base["top_ids"]) == set(observed["top_ids"])
                        and base["top_values"][0]-base["top_values"][1] <= .25
                        and observed["top_values"][0]-observed["top_values"][1] <= .25)
                divergence = dict(output_position=j, baseline=base, observed=observed, explained_near_tie=tied)
            report["cases"].append(dict(name=name, prompt_ids=prompt, token_ids=out["token_ids"], text=out["text"],
                                        matches_reference=match, trace=trace, first_divergence=divergence))
            print(name, "match="+str(match), flush=True)
        llm.config.max_model_len = 2048
        if hasattr(llm.scheduler, "max_model_len"):
            llm.scheduler.max_model_len = 2048
        # 人工把真实生成的第 4 个 token 设成 EOS，检查轮内提前结束，而非只测长度限制。
        expected = report["cases"][0]["token_ids"]
        eos = expected[3]
        llm.config.eos = eos
        if hasattr(llm.scheduler, "eos"):
            llm.scheduler.eos = eos
        out = llm.generate([prompts[0]], SamplingParams(temperature=0, max_tokens=64), use_tqdm=False)[0]
        assert out["token_ids"] == expected[:expected.index(eos)+1]
        report["eos_tokens"] = out["token_ids"]
        # 随机采样只验证可执行及状态边界；文本不要求与同 seed 基线一致。
        for _ in range(3):
            out = llm.generate([prompts[1]], SamplingParams(temperature=.6, max_tokens=32, ignore_eos=True), use_tqdm=False)[0]
            assert len(out["token_ids"]) == 32
        bm = llm.scheduler.block_manager
        assert len(bm.free_block_ids) == len(bm.blocks) and not bm.used_block_ids
        assert all(b.ref_count == 0 for b in bm.blocks)
        report["stats"] = llm.get_stats()
        report["all_greedy_match"] = all(c["matches_reference"] for c in report["cases"])
        report["all_match_or_explained_near_tie"] = all(c["matches_reference"] or c["first_divergence"]["explained_near_tie"] for c in report["cases"])
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
        assert all(r["within_tolerance"] for r in report["logits"]), "logits outside tolerance; inspect saved diagnostics"
        assert report["all_match_or_explained_near_tie"], "unexplained greedy mismatch: inspect saved token IDs and logits"
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
