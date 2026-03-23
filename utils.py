from LoRA.lora_layer import LoRALayer
from torch import nn


def apply_lora(model, target_modules, lora_rank, lora_alpha):
    """
    model: 原始模型
    target_modules: 需要替换的模块名称列表
    lora_rank: LoRA rank
    lora_alpha: LoRA alpha
    递归修改为LoRALayer
    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            apply_lora(module, target_modules, lora_rank, lora_alpha)

        if isinstance(module, nn.Linear) and name in target_modules:
            new_layer = LoRALayer(module, lora_rank, lora_alpha)
            setattr(model, name, new_layer)
    return model

def _normalize_lora_key(name):
    # 不同 checkpoint、不同封装方式下，明明是同一个模块，参数名却可能不一样
    if name.startswith("model."):
        name = name[len("model."):]
    name = name.replace("language_model.model.", "language_model.")
    return name


def load_lora_weight(model, lora_weights=None):
    if lora_weights is None:
        # 如果没有lora传进来就全部清0
        for name, param in model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                # print("LoRA weights没找到.")
                param.data.zero_()
        return

    normalized_weights = {
        _normalize_lora_key(name): value for name, value in lora_weights.items()
    }

    for name, param in model.named_parameters():
        if 'lora_A' not in name and 'lora_B' not in name:
            continue

        normalized_name = _normalize_lora_key(name)
        if normalized_name in normalized_weights:
            param.data.copy_(normalized_weights[normalized_name])
        else:
            print(f"Warning: {name} 不在 loaded lora_weights.")
