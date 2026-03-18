import torch
import torch.nn as nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from typing import List

from .modeling_xllmx_chameleon_ck_action_head import ActionHead


class Qwen3VLXLLMXForConditionalGeneration_ck_action_head(Qwen3VLForConditionalGeneration):
    def __init__(self, config, action_dim=7, time_horizon=5):
        super().__init__(config)
        text_hidden_size = getattr(getattr(config, "text_config", config), "hidden_size", 4096)
        self.action_head = ActionHead(
            action_dim=action_dim,
            time_horizon=time_horizon,
            hidden_size_factor=0.25,
            num_encoder_layers=2,
            hidden_size=text_hidden_size,
        )
        self.action_dim = action_dim
        self.time_horizon = time_horizon
        self.action_start_token_id = None
        self.action_end_token_id = None
        self.action_bin_token_ids = None
        self.processor = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, action_dim=7, time_horizon=5, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        text_hidden_size = getattr(getattr(model.config, "text_config", model.config), "hidden_size", 4096)
        model.action_head = ActionHead(
            action_dim=action_dim,
            time_horizon=time_horizon,
            hidden_size_factor=0.25,
            num_encoder_layers=2,
            hidden_size=text_hidden_size,
        )
        model.action_dim = action_dim
        model.time_horizon = time_horizon
        model.processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        return model

    def _get_action_bin_token_strings(self, num_bins=256):
        return [f"<|action_bin_{i:03d}|>" for i in range(num_bins)]

    def setup_action_tokens(self, action_start_token="<|action_start|>", action_end_token="<|action_end|>"):
        tok = self.processor.tokenizer
        action_bin_tokens = self._get_action_bin_token_strings()
        tok.add_special_tokens({"additional_special_tokens": [action_start_token, action_end_token, *action_bin_tokens]})
        self.resize_token_embeddings(len(tok))
        self.action_start_token_id = tok.convert_tokens_to_ids(action_start_token)
        self.action_end_token_id = tok.convert_tokens_to_ids(action_end_token)
        self.action_bin_token_ids = tok.convert_tokens_to_ids(action_bin_tokens)

    def _get_action_token_ids(self):
        if self.action_start_token_id is None and self.processor is not None:
            tok = self.processor.tokenizer
            self.action_start_token_id = tok.convert_tokens_to_ids("<|action_start|>")
            self.action_end_token_id = tok.convert_tokens_to_ids("<|action_end|>")
            self.action_bin_token_ids = tok.convert_tokens_to_ids(self._get_action_bin_token_strings())
        start_id = self.action_start_token_id if self.action_start_token_id is not None else 10004
        end_id = getattr(self, "action_end_token_id", None)
        if end_id is None:
            end_id = start_id + 5000
        return start_id, end_id

    def _get_action_bin_token_ids(self):
        if self.action_bin_token_ids is None and self.processor is not None:
            tok = self.processor.tokenizer
            self.action_bin_token_ids = tok.convert_tokens_to_ids(self._get_action_bin_token_strings())
        return self.action_bin_token_ids

    def decode_token_ids_to_actions(self, dis_action):
        bin_edges = torch.linspace(-1, 1, 257, device=dis_action.device)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        bin_token_ids = self._get_action_bin_token_ids()
        if bin_token_ids is None:
            raise ValueError("Qwen action bin tokens are not initialized.")

        token_id_tensor = torch.as_tensor(bin_token_ids, device=dis_action.device, dtype=dis_action.dtype)
        matches = dis_action.unsqueeze(-1) == token_id_tensor
        bin_indices = matches.long().argmax(dim=-1)
        has_match = matches.any(dim=-1)
        bin_indices = torch.where(has_match, bin_indices, torch.zeros_like(bin_indices))
        return bin_centers[bin_indices.long()]

    def find_sequences(self, tensor_input):
        start_id, end_id = self._get_action_token_ids()
        start_indices = (tensor_input[:, :-1 * self.action_dim + 1] == start_id).nonzero(as_tuple=True)
        valid_sequences = []
        for batch, start in zip(*start_indices):
            if start + self.action_dim + 1 < tensor_input.shape[1] and tensor_input[batch, start + self.action_dim + 1] == end_id:
                valid_sequences.append((batch, start + 1))
        return valid_sequences

    def get_action_label(self, labels_c):
        sequences = self.find_sequences(labels_c)
        if len(sequences) == 0:
            return torch.empty(0, self.action_dim, dtype=torch.long, device=labels_c.device), sequences

        labels_action = torch.zeros(len(sequences), self.action_dim, dtype=torch.long, device=labels_c.device)
        for i, (batch, start) in enumerate(sequences):
            labels_action[i] = labels_c[batch, start:start + self.action_dim]
        return labels_action, sequences

    def _get_pad_token_id(self):
        pad_token_id = self.config.pad_token_id
        if pad_token_id is None and self.processor is not None:
            pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.config.eos_token_id if self.config.eos_token_id is not None else 0
        return pad_token_id

    def forward(self, input_ids=None, labels=None, training=False, att_mask=True, **kwargs):
        if not training:
            return Qwen3VLForConditionalGeneration.forward(self, input_ids=input_ids, labels=labels, **kwargs)

        device = next(self.parameters()).device
        output_hidden_states = kwargs.pop("output_hidden_states", False)
        use_cache = kwargs.pop("use_cache", False)
        return_dict = kwargs.pop("return_dict", True)
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        attention_mask = kwargs.pop("attention_mask", None)
        modalities = kwargs.pop("modalities", None)

        if torch.is_tensor(input_ids):
            input_ids = input_ids.to(device=device, dtype=torch.long)
            labels = labels.to(device=device, dtype=torch.long) if labels is not None else None
            if attention_mask is None and att_mask:
                attention_mask = (input_ids != self._get_pad_token_id()).long()
        else:
            max_tokens = max(len(x) for x in input_ids)
            if hasattr(self.config, "max_position_embeddings"):
                max_pos_embeddings = self.config.max_position_embeddings
            elif hasattr(self.config, "text_config") and hasattr(self.config.text_config, "max_position_embeddings"):
                max_pos_embeddings = self.config.text_config.max_position_embeddings
            else:
                max_pos_embeddings = 32768
            max_tokens = min(max_tokens, max_pos_embeddings)

            pad_token_id = self._get_pad_token_id()
            processed_input_ids = []
            processed_labels = []
            for example, label in zip(input_ids, labels):
                truncated_example = example[:max_tokens]
                truncated_label = label[:max_tokens]
                padded_example = truncated_example + [pad_token_id] * (max_tokens - len(truncated_example))
                padded_label = truncated_label + [-100] * (max_tokens - len(truncated_label))
                processed_input_ids.append(padded_example)
                processed_labels.append(padded_label)

            input_ids = torch.tensor(processed_input_ids, dtype=torch.long, device=device)
            labels = torch.tensor(processed_labels, dtype=torch.long, device=device)
            attention_mask = (input_ids != pad_token_id).long() if att_mask else None

        if attention_mask is not None:
            attention_mask = attention_mask.to(device=device, dtype=torch.long)
        if pixel_values is not None:
            pixel_values = pixel_values.to(device=device)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(device=device, dtype=torch.long)

        image_token_id = getattr(self.config, "image_token_id", None)
        if image_token_id is not None and pixel_values is None and (input_ids == image_token_id).any():
            raise ValueError(
                "Qwen3-VL training input contains image placeholder tokens, but pixel_values/image_grid_thw "
                "were not provided. Please build Qwen batches with the processor so vision inputs are passed "
                "together with input_ids."
            )

        vocab_size = self.get_input_embeddings().num_embeddings
        input_id_min = int(input_ids.min().item())
        input_id_max = int(input_ids.max().item())
        if input_id_min < 0 or input_id_max >= vocab_size:
            raise ValueError(
                f"Qwen3-VL input_ids out of range for embedding table: min={input_id_min}, "
                f"max={input_id_max}, vocab_size={vocab_size}."
            )

        result = Qwen3VLForConditionalGeneration.forward(
            self,
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            modalities=modalities,
            labels=labels,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        c_loss = result.loss
        logits = result.logits
        additional_loss_dict = {}

        hidden_states = None
        predicted_actions = torch.empty(0, self.action_dim, device=device)
        loss_ct = c_loss * 0

        if output_hidden_states:
            hidden_states = result.hidden_states[-1]
            start_id, _ = self._get_action_token_ids()
            predicted_actions, actions_flag = self.action_head(
                hidden_states=hidden_states,
                input_ids=input_ids,
                attention_mask=None,
                target_token_id=start_id,
            )

            labels_action_dis, _ = self.get_action_label(labels)
            if actions_flag and labels_action_dis.numel() > 0 and predicted_actions.numel() > 0:
                labels_action_ct = self.decode_token_ids_to_actions(labels_action_dis)
                n = min(predicted_actions.shape[0], labels_action_ct.shape[0])
                loss_ct = torch.nn.functional.l1_loss(predicted_actions[:n], labels_action_ct[:n])
            else:
                loss_ct = c_loss * 0

            return c_loss, additional_loss_dict, logits, hidden_states, labels, predicted_actions, loss_ct

        return c_loss, additional_loss_dict

    def generate_action_head(self, model_inputs, generation_config):
        out = self.generate(
            **model_inputs,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )
        full_input_ids = out.sequences
        step_hs = [s[-1] for s in out.hidden_states]
        hidden_states = torch.cat(step_hs, dim=1)

        predicted_actions, ok = self.action_head(
            hidden_states=hidden_states,
            input_ids=full_input_ids,
            attention_mask=None,
            target_token_id=self.action_start_token_id,
            eval=True,
        )
        if not ok:
            return torch.zeros(self.time_horizon, self.action_dim, device=full_input_ids.device)
        return predicted_actions.reshape(self.time_horizon, self.action_dim)

    def _get_transformer_layers(self) -> List[nn.Module]:
        language_model = getattr(self.model, "language_model", None)
        if language_model is not None and hasattr(language_model, "layers"):
            return list(language_model.layers)

        if hasattr(self.model, "layers"):
            return list(self.model.layers)

        text_model = getattr(self, "text_model", None)
        if text_model is not None and hasattr(text_model, "layers"):
            return list(text_model.layers)

        raise AttributeError("Cannot find transformer layers for Qwen3-VL FSDP/checkpointing setup.")

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        modules = [
            *self._get_transformer_layers(),
            self.lm_head,
            self.action_head,
        ]

        language_model = getattr(self.model, "language_model", None)
        if language_model is not None and hasattr(language_model, "embed_tokens"):
            modules.append(language_model.embed_tokens)
        elif hasattr(self.model, "embed_tokens"):
            modules.append(self.model.embed_tokens)

        visual = getattr(self.model, "visual", None)
        if visual is None:
            visual = getattr(self, "visual", None)
        if isinstance(visual, nn.Module):
            modules.append(visual)

        return modules

    def get_checkpointing_wrap_module_list(self) -> List[nn.Module]:
        return self._get_transformer_layers()
