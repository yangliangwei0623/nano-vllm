import os
from dataclasses import dataclass
from transformers import AutoConfig, AutoTokenizer


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    draft_model: str | None = None
    num_speculative_tokens: int = 4
    speculative_policy: str = "fixed"
    speculative_cuda_graph: bool = False
    draft_hf_config: AutoConfig | None = None

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        if self.max_num_seqs < 1 or self.max_num_batched_tokens < 1 or self.max_model_len < 1:
            raise ValueError("sequence/token/context limits must be positive")
        if not 0 < self.gpu_memory_utilization < 1:
            raise ValueError("gpu_memory_utilization must be between 0 and 1")
        if self.draft_model is not None:
            if not os.path.isdir(self.draft_model):
                raise ValueError("draft_model must be a local model directory")
            if self.tensor_parallel_size != 1:
                raise ValueError("V0 speculative decoding supports only one GPU")
            if self.speculative_policy != "fixed" or self.speculative_cuda_graph:
                raise ValueError("V0 supports fixed K and eager speculative decoding only")
            if not isinstance(self.num_speculative_tokens, int) or self.num_speculative_tokens < 1:
                raise ValueError("num_speculative_tokens must be a positive integer")
            self.draft_hf_config = AutoConfig.from_pretrained(self.draft_model)
            for cfg in (self.hf_config, self.draft_hf_config):
                if cfg.model_type != "qwen3":
                    raise ValueError("V0 supports Qwen3 dense target/draft models only")
            if self.hf_config.vocab_size != self.draft_hf_config.vocab_size:
                raise ValueError("target/draft vocabulary sizes differ")
            # 同样大小的词表不代表相同 token ID 语义，必须逐项核对映射。
            target_tok = AutoTokenizer.from_pretrained(self.model)
            draft_tok = AutoTokenizer.from_pretrained(self.draft_model)
            if (target_tok.get_vocab() != draft_tok.get_vocab()
                    or target_tok.all_special_ids != draft_tok.all_special_ids):
                raise ValueError("target/draft token ID mappings or special tokens differ")
            self.max_model_len = min(self.max_model_len, self.draft_hf_config.max_position_embeddings)
