import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    files = glob(os.path.join(path, "*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors weights found in {path}")
    # 下载中断时可能只剩一个 shard，不能把其余随机初始化权重当作有效模型运行。
    index_path = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        import json
        with open(index_path) as f:
            expected = set(json.load(f)["weight_map"].values())
        missing = expected - {os.path.basename(file) for file in files}
        if missing:
            raise ValueError(f"Missing model weight shards: {sorted(missing)}")
    for file in files:
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
