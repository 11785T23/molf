import torch
import torch.nn as nn

# helper for get_parent_and_attr (remains the same)
def get_parent_and_attr(root: nn.Module, full_name: str):
    parts = full_name.split(".")
    parent = root
    for p in parts[:-1]:
        if p.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    return parent, parts[-1]