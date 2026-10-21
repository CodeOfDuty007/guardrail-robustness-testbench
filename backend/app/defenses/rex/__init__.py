"""REX (toy-scale reproduction): relocate synthetic "hazardous" knowledge into removable MoE experts.

Default ``RexConfig(variant="paper")`` follows the paper's recipe; ``variant="prelearned_pushout"`` is an
ablation that is NOT from the paper.

Reimplements the mechanism of Jin et al., arXiv:2610.25791 (CC BY 4.0; reference code
github.com/AI45Lab/REX, Apache-2.0) on a ~1M-param CPU MoE with a fully synthetic forget set.
See ``relocate.py`` for the recipe and README "REX (toy-scale reproduction)" for limitations.

torch is an optional extra (``pip install .[rex]``). This package imports torch lazily: importing
``app.defenses.rex`` never requires torch; touching a torch-backed name does.
"""
from __future__ import annotations

import importlib
import importlib.util
from typing import Any

_LAZY = {
    "run_rex_pipeline": "relocate", "rex_pipeline": "relocate", "RexConfig": "relocate",
    "fast_config": "relocate", "default_config": "relocate", "load_model": "relocate",
    "save_artifacts": "relocate", "default_save_dir": "relocate",
    "expand_experts": "relocate", "excise_experts": "relocate", "differential_activation": "relocate",
    "relocate_paper": "relocate", "relocate_pushout": "relocate", "VARIANTS": "relocate",
    "rmu_cleanup": "cleanup",
    "ToyMoE": "toy_moe", "MoEConfig": "toy_moe",
    "generate": "evaluate", "answer": "evaluate", "accuracy": "evaluate",
    "generate_corpora": "data", "Tokenizer": "data", "DataConfig": "data",
}

__all__ = [*_LAZY, "torch_available"]


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def __getattr__(name: str) -> Any:
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{mod}"), name)
