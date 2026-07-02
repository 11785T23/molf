import torch.distributed as dist
import os

def is_master() -> bool:
    """
    Checks if the current process is the main process (Rank 0).
    Returns True if:
      - Distributed training is not initialized (single process).
      - Distributed training is initialized and the current rank is 0.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    
    # Fallback for environments where LOCAL_RANK might be set but torch.dist isn't init yet
    # (Common in some launcher scripts)
    return int(os.environ.get("LOCAL_RANK", 0)) == 0