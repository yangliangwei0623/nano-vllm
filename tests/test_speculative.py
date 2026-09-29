"""不依赖模型权重的 V0 验收：算法分布、零/部分/全接受、KV 生命周期。"""
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch

from nanovllm import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.speculative_runner import SpeculativeResult
from nanovllm.engine.speculative_runner import SpeculativeModelRunner
from nanovllm.engine.speculative_sampling import verify_greedy, verify_stochastic, sample
from nanovllm.engine.speculative_scheduler import SpeculativeScheduler
from nanovllm.layers.sampler import Sampler
from nanovllm.layers.embed_head import ParallelLMHead
from nanovllm.utils.context import set_context, reset_context
from nanovllm.config import Config


class SamplingTests(unittest.TestCase):
    def test_greedy_acceptance_lengths(self):
        draft = [0, 1, 2, 3]
        for accepted in range(5):
            target = draft + [4]
            if accepted < 4:
                target[accepted] = 4
            logits = torch.nn.functional.one_hot(torch.tensor(target), 5).float()
            result = verify_greedy(draft, logits)
            self.assertEqual(result.accepted, accepted)
            self.assertEqual(result.token_ids, draft[:accepted] + [4])

    def test_stochastic_all_accept_and_no_candidates(self):
        q = torch.eye(3)[:2]
        p = torch.eye(3)
        self.assertEqual(verify_stochastic([0, 1], p, q).token_ids, [0, 1, 2])
        self.assertEqual(verify_stochastic([], p[2:], q[:0]).token_ids, [2])

    def test_stochastic_disjoint_support_and_immutable_probabilities(self):
        p, q = torch.tensor([[0., 1.], [1., 0.]]), torch.tensor([[1., 0.]])
        before = q.clone()
        result = verify_stochastic([0], p, q)
        self.assertEqual((result.accepted, result.token_ids), (0, [1]))
        torch.testing.assert_close(q, before)

    def test_stochastic_partial_acceptance(self):
        p = torch.eye(3)[[0, 2, 0]]
        q = torch.eye(3)[[0, 1]]
        result = verify_stochastic([0, 1], p, q)
        self.assertEqual((result.accepted, result.token_ids), (1, [0, 2]))

    def test_exact_distribution(self):
        # 两 token 的联合分布比只测第一 token 更能发现拒绝后错误沿用候选的问题。
        gen = torch.Generator().manual_seed(37)
        initial_p = torch.tensor([.15, .35, .5])
        initial_q = torch.tensor([.65, .25, .1])
        transition_p = torch.tensor([[.6, .1, .3], [.2, .7, .1], [.1, .4, .5]])
        transition_q = torch.tensor([[.1, .7, .2], [.6, .1, .3], [.5, .4, .1]])
        counts = torch.zeros(3, 3)
        for _ in range(20000):
            d0 = sample(initial_q, gen)
            d1 = sample(transition_q[d0], gen)
            p = torch.stack([initial_p, transition_p[d0], transition_p[d1]])
            q = torch.stack([initial_q, transition_q[d0]])
            out = verify_stochastic([d0, d1], p, q, gen).token_ids
            if len(out) == 1:
                out.append(sample(transition_p[out[0]], gen))
            counts[out[0], out[1]] += 1
        expected = initial_p[:, None] * transition_p
        torch.testing.assert_close(counts / counts.sum(), expected, atol=.012, rtol=0)

    def test_temperature_validation_and_mixed_greedy(self):
        for bad in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                SamplingParams(temperature=bad)
        with self.assertRaises(ValueError):
            SamplingParams(max_tokens=0)
        logits = torch.tensor([[1., 3., 2.], [1., 2., 3.]])
        before = logits.clone()
        # 调用未编译函数，避免 CPU 单测为小张量启动整个编译工具链。
        result = Sampler.forward.__wrapped__(Sampler(), logits, torch.tensor([0., .6]))
        self.assertEqual(result[0].item(), 1)
        torch.testing.assert_close(logits, before)


class SchedulerTests(unittest.TestCase):
    def make(self, prompt=254, max_tokens=12, max_len=1024, eos=9, ignore=False):
        config = SimpleNamespace(num_kvcache_blocks=8, kvcache_block_size=256,
                                 max_model_len=max_len, max_num_batched_tokens=128,
                                 num_speculative_tokens=4, eos=eos)
        scheduler = SpeculativeScheduler(config)
        seq = Sequence([1]*prompt, SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=ignore))
        scheduler.add(seq)
        return scheduler, seq

    def prefill(self, scheduler, seq):
        while seq.is_prefill:
            seqs, prefill = scheduler.schedule()
            self.assertTrue(prefill)
            end = seq.num_cached_tokens + seq.num_scheduled_tokens
            scheduler.postprocess(seqs, [SpeculativeResult([2] if end == len(seq) else [], end, end)], True)

    def test_partial_rollback_across_page_boundary(self):
        for accepted in (0, 2, 4):
            scheduler, seq = self.make()
            self.prefill(scheduler, seq)
            self.assertEqual(len(seq), 255)
            seqs, prefill = scheduler.schedule()
            self.assertFalse(prefill)
            valid = len(seq) + accepted
            scheduler.postprocess(seqs, [SpeculativeResult([3]*accepted+[4], valid, valid, 4, accepted)], False)
            self.assertEqual(seq.num_cached_tokens, len(seq)-1)
            self.assertEqual(seq.num_draft_cached_tokens, len(seq)-1)
            self.assertFalse(scheduler.block_manager.hash_to_block_id)

    def test_eos_truncates_accepted_suffix_and_releases_pages(self):
        scheduler, seq = self.make()
        self.prefill(scheduler, seq)
        seqs, _ = scheduler.schedule()
        scheduler.postprocess(seqs, [SpeculativeResult([3, 9, 4, 5, 6], 259, 259, 4, 4)], False)
        self.assertEqual(seq.completion_token_ids, [2, 3, 9])
        self.assertTrue(scheduler.is_finished())
        self.assertEqual(len(scheduler.block_manager.free_block_ids), 8)
        self.assertTrue(all(b.ref_count == 0 for b in scheduler.block_manager.blocks))
        self.assertEqual((seq.num_cached_tokens, seq.num_draft_cached_tokens), (0, 0))

    def test_output_and_context_limits(self):
        for max_tokens, max_len in ((1, 1024), (12, 255)):
            scheduler, seq = self.make(max_tokens=max_tokens, max_len=max_len)
            self.prefill(scheduler, seq)
            self.assertTrue(seq.is_finished)
            self.assertEqual(len(seq.completion_token_ids), 1)
            self.assertEqual(len(scheduler.block_manager.free_block_ids), 8)

    def test_ignore_eos_and_queue(self):
        scheduler, seq = self.make(max_tokens=2, eos=2, ignore=True)
        other = Sequence([7], SamplingParams(temperature=0, max_tokens=1))
        scheduler.add(other)
        self.prefill(scheduler, seq)
        self.assertFalse(seq.is_finished)
        seqs, _ = scheduler.schedule()
        self.assertEqual(seq.num_scheduled_tokens, 1)
        scheduler.postprocess(seqs, [SpeculativeResult([2], 255, 255)], False)
        self.prefill(scheduler, other)
        self.assertTrue(scheduler.is_finished())


class RunnerStateTests(unittest.TestCase):
    def test_actual_round_state_machine_for_every_rejection_position(self):
        """固定模型输出，只替换前向计算，真实执行 runner 的提案/回退/补算代码。"""
        for accepted in range(5):
            runner = SpeculativeModelRunner.__new__(SpeculativeModelRunner)
            runner.config = SimpleNamespace(num_speculative_tokens=4, max_model_len=1024)
            runner.model, runner.draft_model = object(), object()
            runner.stats = dict(rounds=0, proposed=0, accepted=0, repair_calls=0,
                                repair_tokens=0, acceptance_histogram={})
            seq = Sequence([0]*255, SamplingParams(temperature=0, max_tokens=20))
            seq.num_cached_tokens = seq.num_draft_cached_tokens = 254
            calls = []
            def forward(model, request, tokens, start, end, all_logits=False):
                self.assertEqual(request.token_ids, [0]*255)
                calls.append((model, start, end, all_logits))
                if model is runner.draft_model:
                    ids = [min(start-254+1, 5)]
                else:
                    self.assertTrue(all_logits)
                    self.assertEqual(tokens[-4:], [1, 2, 3, 4])
                    ids = [1, 2, 3, 4, 5]
                    if accepted < 4:
                        ids[accepted] = 5
                return torch.nn.functional.one_hot(torch.tensor(ids), 6).float()
            runner.forward_span = forward
            result = runner.run([seq], False)[0]
            self.assertEqual(result.token_ids, [1, 2, 3, 4][:accepted]+[5])
            self.assertEqual(result.target_cached, 255+accepted)
            self.assertEqual(result.draft_cached, 255+accepted)
            self.assertEqual(runner.stats["repair_tokens"], int(accepted == 4))
            if accepted == 4:
                self.assertEqual(calls[-1][1:3], (258, 259))


class InterfaceTests(unittest.TestCase):
    def test_attention_mode_does_not_force_last_logits(self):
        head = ParallelLMHead.__new__(ParallelLMHead)
        torch.nn.Module.__init__(head)
        head.weight = torch.nn.Parameter(torch.tensor([[1., 0.], [0., 1.], [1., 1.]]))
        head.tp_size = 1
        x = torch.arange(10).float().reshape(5, 2)
        try:
            set_context(True, cu_seqlens_q=torch.tensor([0, 2, 5]))
            torch.testing.assert_close(head(x), x[[1, 4]] @ head.weight.T)
            set_context(True, cu_seqlens_q=torch.tensor([0, 2, 5]), all_logits=True)
            torch.testing.assert_close(head(x), x @ head.weight.T)
        finally:
            reset_context()

    def test_unsupported_combinations_fail_before_loading_weights(self):
        cfg = SimpleNamespace(model_type="qwen3", max_position_embeddings=4096, vocab_size=3)
        tok = SimpleNamespace(get_vocab=lambda: {"a": 0, "b": 1}, all_special_ids=[2])
        bad_tok = SimpleNamespace(get_vocab=lambda: {"a": 1, "b": 0}, all_special_ids=[2])
        with tempfile.TemporaryDirectory() as path:
            with patch("nanovllm.config.AutoConfig.from_pretrained", return_value=cfg):
                for kwargs in (dict(tensor_parallel_size=2), dict(speculative_policy="adaptive"),
                               dict(num_speculative_tokens=0), dict(speculative_cuda_graph=True)):
                    with self.assertRaises(ValueError):
                        Config(path, draft_model=path, **kwargs)
                with patch("nanovllm.config.AutoTokenizer.from_pretrained", side_effect=[tok, bad_tok]):
                    with self.assertRaisesRegex(ValueError, "token ID mappings"):
                        Config(path, draft_model=path)
                cfg.model_type = "unsupported"
                with self.assertRaisesRegex(ValueError, "Qwen3"):
                    Config(path, draft_model=path)


if __name__ == "__main__":
    unittest.main()
