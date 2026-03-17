from __future__ import annotations

import torch
from accelerate import init_empty_weights

from .configuration_xllmx_chameleon import ChameleonXLLMXConfig
from .modeling_xllmx_chameleon_ck_action_head import ChameleonXLLMXForConditionalGeneration_ck_action_head
from .modeling_xllmx_qwen3_vl_ck_action_head import Qwen3VLXLLMXForConditionalGeneration_ck_action_head


def build_vla_action_model(args, init_from: str, dp_rank: int = 0):
    """Build VLA action model for both training/eval with minimal branching.

    - chameleon: keep the original FSDP-friendly initialization logic.
    - qwen3_vl: reuse Qwen3-VL pretrained model + action head wrapper.
    """
    if args.vlm_arch == "qwen3_vl":
        model_path = args.qwen_model_path or init_from
        if not model_path:
            raise ValueError("qwen3_vl requires --qwen_model_path or --init_from/--resume_path")

        model = Qwen3VLXLLMXForConditionalGeneration_ck_action_head.from_pretrained(
            model_path,
            action_dim=args.action_dim,
            time_horizon=args.time_horizon,
            dtype=torch.bfloat16,
            device_map="cpu",
        )
        model.setup_action_tokens(
            action_start_token=args.action_start_token,
            action_end_token=args.action_end_token,
        )
        return model, None

    if dp_rank == 0:
        model = ChameleonXLLMXForConditionalGeneration_ck_action_head.from_pretrained(
            init_from,
            action_dim=args.action_dim,
            time_horizon=args.time_horizon,
            max_position_embeddings=args.max_seq_len,
            mask_image_logits=args.mask_image_logits,
            dropout=args.dropout,
            z_loss_weight=args.z_loss_weight,
            dtype=torch.bfloat16,
            device_map="cpu",
        )
    else:
        with init_empty_weights():
            config = ChameleonXLLMXConfig.from_pretrained(
                init_from,
                action_dim=args.action_dim,
                time_horizon=args.time_horizon,
                max_position_embeddings=args.max_seq_len,
                mask_image_logits=args.mask_image_logits,
                dropout=args.dropout,
                z_loss_weight=args.z_loss_weight,
                dtype=torch.bfloat16,
            )
            model = ChameleonXLLMXForConditionalGeneration_ck_action_head(config)

    return model, None
