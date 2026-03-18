import json
import os

from .qwen3_vl import Qwen3VLInferenceWrapper
from ..models.qwen3_vl_action_head.modeling_qwen3_vl_action_head import (
    ACTION_HEAD_METADATA_NAME,
    Qwen3VLActionHeadForConditionalGeneration,
)
from ..registry import INFERENCE_WRAPPER_REGISTRY


@INFERENCE_WRAPPER_REGISTRY.register("qwen3_vl_action_head")
class Qwen3VLActionHeadInferenceWrapper(Qwen3VLInferenceWrapper):
    def _resolve_base_model_path(self):
        metadata_path = os.path.join(self.model_path, ACTION_HEAD_METADATA_NAME)
        if os.path.isfile(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)["base_model_path"]
        return self.model_path

    def load_model(self):
        metadata_path = os.path.join(self.model_path, ACTION_HEAD_METADATA_NAME)
        if os.path.isfile(metadata_path):
            return Qwen3VLActionHeadForConditionalGeneration.from_checkpoint(
                checkpoint_dir=self.model_path,
                dtype=self.dtype,
                attn_implementation=self.attn_implementation,
                device_map={"": "cuda:0"},
            )
        return Qwen3VLActionHeadForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=self.dtype,
            attn_implementation=self.attn_implementation,
            device_map={"": "cuda:0"},
        )

    def load_processor(self):
        from transformers import Qwen3VLProcessor

        return Qwen3VLProcessor.from_pretrained(self._resolve_base_model_path())

    def predict_action(self, model_inputs):
        action_predictions = self.model.predict_action(**model_inputs)
        return action_predictions.detach().cpu()

    def generate(self, model_inputs, sampling_params):
        actions = self.predict_action(model_inputs)
        return [json.dumps(action.tolist()) for action in actions]
