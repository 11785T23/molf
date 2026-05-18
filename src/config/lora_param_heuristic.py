import math
def get_lora_params(lora_rank: int, use_rslora: bool, scale: int = 2):
    if use_rslora:
        lora_alpha = int(round(scale * math.sqrt(lora_rank)))
    else:
        lora_alpha = scale * lora_rank
    lora_dropout = 0.05
    return lora_alpha, lora_dropout