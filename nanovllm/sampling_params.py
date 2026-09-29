from dataclasses import dataclass
import math


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and >= 0 (0 means greedy)")
        if not isinstance(self.max_tokens, int) or self.max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
