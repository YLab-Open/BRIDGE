from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed, torch_module=None, transformers_module=None, verbose=True):
    """
    Seed Python, NumPy, and optionally PyTorch/Transformers.

    The optional module arguments let callers avoid importing heavy ML packages
    in utility-only contexts.
    """
    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    if torch_module is not None:
        torch_module.manual_seed(seed)
        if hasattr(torch_module, "cuda"):
            torch_module.cuda.manual_seed(seed)
            torch_module.cuda.manual_seed_all(seed)
        if hasattr(torch_module, "backends") and hasattr(torch_module.backends, "cudnn"):
            torch_module.backends.cudnn.deterministic = True

    if transformers_module is not None:
        transformers_module.set_seed(seed)

    if verbose:
        print(f"seed everything: {seed}")

    return seed


def seed_inference(seed, verbose=True):
    """Seed Python, NumPy, PyTorch/CUDA, and Transformers for local inference."""
    import torch
    import transformers

    return seed_everything(
        seed,
        torch_module=torch,
        transformers_module=transformers,
        verbose=verbose,
    )
