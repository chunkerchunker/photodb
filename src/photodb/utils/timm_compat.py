"""
Compatibility shim for MiVOLO with newer timm versions.

MiVOLO requires timm 0.8.13.dev0 which has APIs that were changed in timm 0.9+:
- `remap_checkpoint(model, state_dict)` -> `remap_state_dict(state_dict, model)`
- `split_model_name_tag(name)` -> removed (simple string split)
- VOLO class signature changed: `pos_drop_rate` parameter added between
  `drop_rate` and `attn_drop_rate`, breaking MiVOLO's positional arguments

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


# ============================================================================
# Patch MiVOLO's MiVOLOModel to use keyword arguments for VOLO parent class
# ============================================================================
# In timm 0.9+, VOLO.__init__ added a new `pos_drop_rate` parameter between
# `drop_rate` and `attn_drop_rate`. MiVOLO passes positional arguments, which
# causes `post_layers` (a tuple) to be passed as `norm_layer`, resulting in:
# "'tuple' object is not callable"
#
# Fix: Monkey-patch MiVOLOModel.__init__ to use keyword arguments.


def _patch_mivolo_model():
    """Patch MiVOLOModel to use keyword arguments when calling parent VOLO class."""
    try:
        import torch.nn as nn
        from timm.layers import trunc_normal_
        from timm.models.volo import VOLO

        # Import the MiVOLO components we need
        from mivolo.model.mivolo_model import MiVOLOModel, PatchEmbed

        # Store original __init__ to check if already patched
        if hasattr(MiVOLOModel, "_timm_compat_patched"):
            return  # Already patched

        def patched_init(
            self,
            layers,
            img_size=224,
            in_chans=3,
            num_classes=1000,
            global_pool="token",
            patch_size=8,
            stem_hidden_dim=64,
            embed_dims=None,
            num_heads=None,
            downsamples=(True, False, False, False),
            outlook_attention=(True, False, False, False),
            mlp_ratio=3.0,
            qkv_bias=False,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
            post_layers=("ca", "ca"),
            use_aux_head=True,
            use_mix_token=False,
            pooling_scale=2,
        ):
            """Patched __init__ that uses keyword arguments for parent VOLO class."""
            # Call parent VOLO.__init__ with keyword arguments to avoid positional mismatch
            VOLO.__init__(
                self,
                layers=layers,
                img_size=img_size,
                in_chans=in_chans,
                num_classes=num_classes,
                global_pool=global_pool,
                patch_size=patch_size,
                stem_hidden_dim=stem_hidden_dim,
                embed_dims=embed_dims,
                num_heads=num_heads,
                downsamples=downsamples,
                outlook_attention=outlook_attention,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                pos_drop_rate=0.0,  # New parameter in timm 0.9+
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
                norm_layer=norm_layer,
                post_layers=post_layers,
                use_aux_head=use_aux_head,
                use_mix_token=use_mix_token,
                pooling_scale=pooling_scale,
            )

            # MiVOLO's custom patch embedding (replaces the one from parent VOLO)
            im_size = img_size[0] if isinstance(img_size, tuple) else img_size
            self.patch_embed = PatchEmbed(
                img_size=im_size,
                stem_conv=True,
                stem_stride=2,
                patch_size=patch_size,
                in_chans=in_chans,
                hidden_dim=stem_hidden_dim,
                embed_dim=embed_dims[0],
            )

            trunc_normal_(self.pos_embed, std=0.02)
            self.apply(self._init_weights)

        # Apply the patch
        MiVOLOModel.__init__ = patched_init
        MiVOLOModel._timm_compat_patched = True

    except ImportError:
        # MiVOLO not installed, nothing to patch
        pass
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning(f"Failed to patch MiVOLO: {e}")


# Apply the MiVOLO patch when this module is imported
_patch_mivolo_model()
