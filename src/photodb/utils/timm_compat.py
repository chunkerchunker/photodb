"""
Compatibility shim for MiVOLO with newer timm versions.

MiVOLO requires timm 0.8.13.dev0 which has APIs that were changed in timm 0.9+:
- `remap_checkpoint(model, state_dict)` -> `remap_state_dict(state_dict, model)`
- `split_model_name_tag(name)` -> removed (simple string split)

This module patches timm to restore the old APIs, allowing MiVOLO to work with
newer timm versions that include FastViT (required for MobileCLIP-S2).

Usage:
    Import this module before importing mivolo:

    import photodb.utils.timm_compat  # noqa: F401
    from mivolo.predictor import Predictor
"""

from typing import Tuple

import timm.models._helpers as _helpers
import timm.models._pretrained as _pretrained


def _remap_checkpoint_shim(model, state_dict):
    """
    Compatibility shim: remap_checkpoint(model, state_dict) -> remap_state_dict(state_dict, model).

    The old remap_checkpoint took (model, state_dict).
    The new remap_state_dict takes (state_dict, model, allow_reshape=True).
    """
    return _helpers.remap_state_dict(state_dict, model, allow_reshape=True)


def _split_model_name_tag_shim(model_name: str, no_tag: str = "") -> Tuple[str, str]:
    """
    Compatibility shim for split_model_name_tag.

    Splits a model name with optional pretrained tag: "resnet50.a1_in1k" -> ("resnet50", "a1_in1k")
    """
    model_name, *tag_list = model_name.split(".", 1)
    tag = tag_list[0] if tag_list else no_tag
    return model_name, tag


# Patch the modules to add the old function names
if not hasattr(_helpers, "remap_checkpoint"):
    _helpers.remap_checkpoint = _remap_checkpoint_shim

if not hasattr(_pretrained, "split_model_name_tag"):
    _pretrained.split_model_name_tag = _split_model_name_tag_shim
