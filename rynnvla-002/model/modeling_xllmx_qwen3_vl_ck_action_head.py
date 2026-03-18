import torch
import torch.nn as nn
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from typing import List

# 直接复用你现有 ActionHead
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

    def setup_action_tokens(self, action_start_token="<|action_start|>", action_end_token="<|action_end|>"):
        tok = self.processor.tokenizer
        tok.add_special_tokens({"additional_special_tokens": [action_start_token, action_end_token]})
        self.resize_token_embeddings(len(tok))
        self.action_start_token_id = tok.convert_tokens_to_ids(action_start_token)
        self.action_end_token_id = tok.convert_tokens_to_ids(action_end_token)

    def _get_action_token_ids(self):
        if self.action_start_token_id is None and self.processor is not None:
            tok = self.processor.tokenizer
            self.action_start_token_id = tok.convert_tokens_to_ids("<|action_start|>")
            self.action_end_token_id = tok.convert_tokens_to_ids("<|action_end|>")

        start_id = self.action_start_token_id if self.action_start_token_id is not None else 10004
        end_id = getattr(self, "action_end_token_id", None)
        if end_id is None:
            end_id = start_id + 5000
        return start_id, end_id

    def decode_token_ids_to_actions(self, dis_action):
        bins = torch.linspace(-1, 1, 256, device=dis_action.device)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0
        start_id, _ = self._get_action_token_ids()
        discretized_actions = dis_action - 1 - start_id
        discretized_actions = torch.clamp(discretized_actions - 1, min=0, max=bin_centers.shape[0] - 1).long()
        return bin_centers[discretized_actions]

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

    def forward(self, input_ids=None, labels=None, training=False, att_mask=True, **kwargs):
        if not training:
            return Qwen3VLForConditionalGeneration.forward(self, input_ids=input_ids, labels=labels, **kwargs)

        # 处理input_ids和labels，确保长度一致
        max_tokens = max(len(x) for x in input_ids)
        
        # Qwen3VL模型可能没有max_position_embeddings属性，需要检查并处理
        # 尝试从文本配置中获取最大位置嵌入
        if hasattr(self.config, 'max_position_embeddings'):
            max_pos_embeddings = self.config.max_position_embeddings
        elif hasattr(self.config, 'text_config') and hasattr(self.config.text_config, 'max_position_embeddings'):
            max_pos_embeddings = self.config.text_config.max_position_embeddings
        else:
            # 如果都没有，则使用一个默认值
            max_pos_embeddings = 32768  # Qwen模型通常使用较大的上下文长度
            
        max_tokens = min(max_tokens, max_pos_embeddings)
        
        # 截断或填充input_ids和labels到相同长度
        processed_input_ids = []
        processed_labels = []
        for example, label in zip(input_ids, labels):
            # 截断到最大长度
            truncated_example = example[:max_tokens]
            truncated_label = label[:max_tokens]
            
            # 填充到最大长度
            padded_example = truncated_example + [0] * (max_tokens - len(truncated_example))
            padded_label = truncated_label + [-100] * (max_tokens - len(truncated_label))
            
            processed_input_ids.append(padded_example)
            processed_labels.append(padded_label)

        input_ids = torch.tensor(processed_input_ids, dtype=torch.int64, device=self.device)
        labels = torch.tensor(processed_labels, dtype=torch.int64, device=self.device)
        
        # 正确构建attention mask - 这是解决CUDA错误的关键
        attention_mask = (input_ids != 0).long() if att_mask else None

        result = Qwen3VLForConditionalGeneration.forward(
            self,
            input_ids=input_ids,
            labels=labels,
            use_cache=False,
            attention_mask=attention_mask,
            output_hidden_states=kwargs.get("output_hidden_states", False),
            **{k: v for k, v in kwargs.items() if k != "output_hidden_states"},
        )

        c_loss = result.loss
        logits = result.logits
        additional_loss_dict = {}

        hidden_states = None
        predicted_actions = torch.empty(0, self.action_dim, device=self.device)
        loss_ct = c_loss * 0

        if kwargs.get("output_hidden_states", False):
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
        step_hs = [s[-1] for s in out.hidden_states]  # 每步最后一层
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