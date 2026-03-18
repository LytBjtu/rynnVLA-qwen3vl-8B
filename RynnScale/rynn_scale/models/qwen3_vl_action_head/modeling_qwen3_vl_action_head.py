import json
import os
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn

from ..qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLForConditionalGeneration,
)


ACTION_HEAD_METADATA_NAME = "action_head_config.json"


class ContinuousActionHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        action_dim: int,
        time_horizon: int,
        hidden_size_factor: float = 1.0,
    ):
        super().__init__()
        hidden_dim = max(hidden_size, int(hidden_size * hidden_size_factor))
        self.action_dim = action_dim
        self.time_horizon = time_horizon
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, time_horizon * action_dim),
        )

    def forward(self, pooled_hidden_state: torch.Tensor) -> torch.Tensor:
        actions = self.network(pooled_hidden_state)
        return actions.view(pooled_hidden_state.size(0), self.time_horizon, self.action_dim)


@dataclass
class Qwen3VLActionHeadOutput(Qwen3VLCausalLMOutputWithPast):
    action_predictions: Optional[torch.FloatTensor] = None
    action_loss: Optional[torch.FloatTensor] = None
    language_loss: Optional[torch.FloatTensor] = None


class Qwen3VLActionHeadForConditionalGeneration(Qwen3VLForConditionalGeneration):
    def __init__(
        self,
        config,
        action_dim: int = 7,
        time_horizon: int = 5,
        action_loss_weight: float = 1.0,
        lm_loss_weight: float = 0.0,
        action_loss_type: str = "smooth_l1",
    ):
        super().__init__(config)
        hidden_size = config.get_text_config().hidden_size
        self.action_head = ContinuousActionHead(
            hidden_size=hidden_size,
            action_dim=action_dim,
            time_horizon=time_horizon,
        )
        self.action_dim = action_dim
        self.time_horizon = time_horizon
        self.action_loss_weight = action_loss_weight
        self.lm_loss_weight = lm_loss_weight
        self.action_loss_type = action_loss_type

    @staticmethod
    def _load_metadata(checkpoint_dir: str) -> Optional[dict]:
        metadata_path = os.path.join(checkpoint_dir, ACTION_HEAD_METADATA_NAME)
        if not os.path.isfile(metadata_path):
            return None
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str,
        dtype: torch.dtype,
        attn_implementation: str,
        device_map: Optional[Union[str, dict]] = None,
    ):
        metadata = cls._load_metadata(checkpoint_dir)
        if metadata is None:
            raise FileNotFoundError(
                f"Cannot find `{ACTION_HEAD_METADATA_NAME}` in checkpoint directory: {checkpoint_dir}"
            )
        if metadata.get("pipeline_parallel_size", 1) != 1 or metadata.get("expert_parallel_size", 1) != 1:
            raise ValueError("Action-head inference currently supports only single-stage, non-expert checkpoints.")

        model = cls.from_pretrained(
            metadata["base_model_path"],
            dtype=dtype,
            attn_implementation=attn_implementation,
            device_map=device_map,
            action_dim=metadata["action_dim"],
            time_horizon=metadata["time_horizon"],
            action_loss_weight=metadata["action_loss_weight"],
            lm_loss_weight=metadata["lm_loss_weight"],
            action_loss_type=metadata["action_loss_type"],
        )
        state_dict = torch.load(
            os.path.join(checkpoint_dir, "model_pp_rank_00_ep_rank_00.pt"),
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
        return model

    def save_checkpoint_metadata(
        self,
        output_dir: str,
        base_model_path: str,
        pipeline_parallel_size: int = 1,
        expert_parallel_size: int = 1,
    ):
        metadata = {
            "model_type": "qwen3_vl_action_head",
            "base_model_path": base_model_path,
            "action_dim": self.action_dim,
            "time_horizon": self.time_horizon,
            "action_loss_weight": self.action_loss_weight,
            "lm_loss_weight": self.lm_loss_weight,
            "action_loss_type": self.action_loss_type,
            "pipeline_parallel_size": pipeline_parallel_size,
            "expert_parallel_size": expert_parallel_size,
        }
        with open(os.path.join(output_dir, ACTION_HEAD_METADATA_NAME), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def _pool_last_token(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        input_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if attention_mask is not None:
            last_indices = attention_mask.long().sum(dim=-1).clamp(min=1) - 1
        elif input_ids is not None:
            last_indices = torch.full(
                (hidden_states.size(0),),
                hidden_states.size(1) - 1,
                device=hidden_states.device,
                dtype=torch.long,
            )
        else:
            raise ValueError("Need either attention_mask or input_ids to locate the action context token.")

        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_indices, last_indices]

    def _prepare_actions(self, actions: Optional[torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
        if actions is None:
            return None
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions)
        actions = actions.to(device=device, dtype=torch.float32)
        if actions.ndim == 2:
            actions = actions.view(actions.size(0), self.time_horizon, self.action_dim)
        if actions.ndim != 3:
            raise ValueError(
                f"`actions` must have shape [batch, time_horizon, action_dim], got {tuple(actions.shape)}"
            )
        if actions.size(1) != self.time_horizon or actions.size(2) != self.action_dim:
            raise ValueError(
                f"`actions` shape mismatch, expected [batch, {self.time_horizon}, {self.action_dim}], "
                f"got {tuple(actions.shape)}"
            )
        return actions

    def _compute_action_loss(
        self,
        action_predictions: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        if self.action_loss_type == "l1":
            return torch.nn.functional.l1_loss(action_predictions, actions)
        if self.action_loss_type == "mse":
            return torch.nn.functional.mse_loss(action_predictions, actions)
        if self.action_loss_type == "smooth_l1":
            return torch.nn.functional.smooth_l1_loss(action_predictions, actions)
        raise ValueError(f"Unsupported action_loss_type: {self.action_loss_type}")

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        actions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Qwen3VLActionHeadOutput:
        language_labels = labels if self.lm_loss_weight > 0 else None
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=language_labels,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        action_predictions = None
        action_loss = None
        if hasattr(self, "action_head"):
            pooled_hidden_state = self._pool_last_token(outputs.last_hidden_state, attention_mask, input_ids)
            action_predictions = self.action_head(pooled_hidden_state)
            prepared_actions = self._prepare_actions(actions, pooled_hidden_state.device)
            if prepared_actions is not None:
                action_loss = self._compute_action_loss(action_predictions, prepared_actions)

        language_loss = outputs.loss
        total_loss = None
        if language_loss is not None:
            total_loss = language_loss * self.lm_loss_weight
        if action_loss is not None:
            scaled_action_loss = action_loss * self.action_loss_weight
            total_loss = scaled_action_loss if total_loss is None else total_loss + scaled_action_loss

        return Qwen3VLActionHeadOutput(
            loss=total_loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
            action_predictions=action_predictions,
            action_loss=action_loss,
            language_loss=language_loss,
        )

    @torch.inference_mode()
    def predict_action(self, **model_inputs) -> torch.Tensor:
        outputs = self.forward(**model_inputs)
        if outputs.action_predictions is None:
            raise RuntimeError("Action head is not available on the current model stage.")
        return outputs.action_predictions

    def apply_pipeline_parallel(
        self,
        num_stages: int,
        stage_index: int,
        reduced_layers_in_stage_zero: int = 0,
    ):
        super().apply_pipeline_parallel(
            num_stages=num_stages,
            stage_index=stage_index,
            reduced_layers_in_stage_zero=reduced_layers_in_stage_zero,
        )
        if stage_index < num_stages - 1 and hasattr(self, "action_head"):
            del self.action_head
