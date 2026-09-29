import atexit
import json
from pathlib import Path
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.speculative_runner import SpeculativeModelRunner
from nanovllm.engine.speculative_scheduler import SpeculativeScheduler


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        runner_cls = SpeculativeModelRunner if config.draft_model else ModelRunner
        self.model_runner = runner_cls(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = SpeculativeScheduler(config) if config.draft_model else Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        # 测试和脚本可主动释放 GPU；退出时 atexit 不应再次销毁同一进程组。
        if not hasattr(self, "model_runner"):
            return
        atexit.unregister(self.exit)
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if not prompt or len(prompt) >= self.config.max_model_len:
            raise ValueError("prompt must be nonempty and leave room for at least one output token")
        if any(not isinstance(t, int) or not 0 <= t < self.config.hf_config.vocab_size for t in prompt):
            raise ValueError("prompt contains an invalid token ID")
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        committed = self.scheduler.postprocess(seqs, token_ids, is_prefill)
        if self.config.draft_model:
            self.model_runner.stats["committed"] += committed
            self.model_runner.stats["accepted_committed"] += min(committed, token_ids[0].accepted)
            if not is_prefill:
                num_tokens = -committed  # 一轮可以提交多个 token，不能再按请求数统计吞吐。
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def get_stats(self):
        """JSON 可序列化的累计统计；接受计数与 EOS 截断后的实际提交分开报告。"""
        config = self.config
        return {
            "model": config.model, "draft_model": config.draft_model,
            "num_speculative_tokens": config.num_speculative_tokens,
            "effective_eager": self.model_runner.enforce_eager,
            "memory_plan": getattr(self.model_runner, "memory_plan", {}),
            "speculative": getattr(self.model_runner, "stats", {}).copy(),
        }

    def export_stats(self, path):
        Path(path).write_text(json.dumps(self.get_stats(), indent=2) + "\n")

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        if len(sampling_params) != len(prompts):
            raise ValueError("one SamplingParams is required per prompt")
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
