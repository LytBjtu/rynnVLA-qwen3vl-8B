import pickle
from typing import List, Tuple

from accelerate import init_empty_weights
import torch

from model import (
    ChameleonXLLMXConfig,
    ChameleonXLLMXForConditionalGeneration_ck_action_head,
    Qwen3VLXLLMXForConditionalGeneration_ck_action_head,
)
from xllmx.data.item_processor import ItemProcessorBase
from xllmx.solvers.pretrain import PretrainSolverBase_ck_action_head


class ItemProcessor(ItemProcessorBase):
# 从data中读取token和label，并计算样本长度
    def process_item(self, data_item: dict, training_mode=False) -> Tuple[List, List]:
        assert training_mode

        if "token" in data_item and "label" in data_item:
            data_item = data_item
        else:
            assert "file" in data_item
            with open(data_item["file"], "rb") as f:
                data_item = pickle.load(f)

        tokens = data_item["token"]
        labels = data_item["label"]
        assert len(tokens) == len(labels)

        return tokens, labels

    def predict_item_token_length(self, data_item: dict) -> int:
        if "token" in data_item:
            return len(data_item["token"])
        elif "len" in data_item:
            return data_item["len"]
        else:
            raise ValueError()


class Solver(PretrainSolverBase_ck_action_head):
    @classmethod
    def get_args_parser(cls):
        # 解析参数
        parser = super().get_args_parser()
        # task-specific parameters
        parser.add_argument("--max_seq_len", default=4096, type=int, help="max token length")
        parser.add_argument("--mask_image_logits", default=True)
        parser.add_argument("--unmask_image_logits", action="store_false", dest="mask_image_logits")
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--z_loss_weight", type=float, default=0.0)
        parser.add_argument("--model_size", type=str, default="7B", choices=["7B", "34B"])
        parser.add_argument("--action_dim", type=int, default=7)
        parser.add_argument("--time_horizon", type=int, default=5)
        parser.add_argument("--preprocess", default='true', choices=['true', 'false'])
        parser.add_argument("--with_state", action='store_true')
        parser.add_argument("--with_wrist", action='store_true')
        parser.add_argument("--with_action", action='store_true')
        parser.add_argument("--with_world_model", action='store_true')
        parser.add_argument("--resolution", type=int, default=256, choices=[256, 512])
        parser.add_argument("--tokenizer_path", type=str, default="../ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768/snapshots/9624463a82ea5ce814af9b561dcd08a31082c3af")
        parser.add_argument("--vlm_arch", type=str, default="chameleon", choices=["chameleon", "qwen3_vl"])
        parser.add_argument("--qwen_model_path", type=str, default="")
        parser.add_argument("--qwen_processor_path", type=str, default="")
        parser.add_argument("--action_start_token", type=str, default="<|action_start|>")
        parser.add_argument("--action_end_token", type=str, default="<|action_end|>")
        return parser

    def _model_func(
        self,
        init_from: str,
    ) -> (ChameleonXLLMXForConditionalGeneration_ck_action_head, None):

        # 只由rank0保存完整权重，其他rank仅建立空模型，每次FSDP加载时将完整权重广播到其他rank，各rank按照FSDP的规则保留自己的切片，释放其他
        if self.args.vlm_arch == "qwen3_vl":
            model_path = self.args.qwen_model_path or init_from
            if not model_path:
                raise ValueError("qwen3_vl requires --qwen_model_path or --init_from/--resume_path")

            model = Qwen3VLXLLMXForConditionalGeneration_ck_action_head.from_pretrained(
                model_path,
                action_dim=self.args.action_dim,
                time_horizon=self.args.time_horizon,
                dtype=torch.bfloat16,
                device_map="cpu",
            )
            model.setup_action_tokens(
                action_start_token=self.args.action_start_token,
                action_end_token=self.args.action_end_token,
            )
            return model, None
        # 使用chameleon加载模型结构，并删除vqmodel模块
        if self.dp_rank == 0:
            model = ChameleonXLLMXForConditionalGeneration_ck_action_head.from_pretrained(
                init_from,
                action_dim=self.args.action_dim,
                time_horizon=self.args.time_horizon,
                max_position_embeddings=self.args.max_seq_len,
                mask_image_logits=self.args.mask_image_logits,
                dropout=self.args.dropout,
                z_loss_weight=self.args.z_loss_weight,
                dtype=torch.bfloat16,
                device_map="cpu",
            )
        else:
            with init_empty_weights():
                config = ChameleonXLLMXConfig.from_pretrained(
                    init_from,
                    action_dim=self.args.action_dim,
                    time_horizon=self.args.time_horizon,
                    max_position_embeddings=self.args.max_seq_len,
                    mask_image_logits=self.args.mask_image_logits,
                    dropout=self.args.dropout,
                    z_loss_weight=self.args.z_loss_weight,
                    dtype=torch.bfloat16,
                )
                model = ChameleonXLLMXForConditionalGeneration_ck_action_head(config)

        del model.model.vqmodel # 做图像预处理的，训练不需要

        return model, None

    def _item_processor_func(self) -> ItemProcessorBase:
        return ItemProcessor()

    def _make_and_save_starting_point(self, save_path: str) -> None:
        # 清除原模型的图像相关的参数，作为训练的起始checkpoints
        pretrained_name = {
            "7B": "Alpha-VLLM/Chameleon_7B_mGPT",
            "34B": "Alpha-VLLM/Chameleon_34B_mGPT",
        }[self.args.model_size]

        model = ChameleonXLLMXForConditionalGeneration_ck_action_head.from_pretrained(
            pretrained_name,
            max_position_embeddings=self.args.max_seq_len,
            mask_image_logits=self.args.mask_image_logits,
            dropout=self.args.dropout,
            z_loss_weight=self.args.z_loss_weight,
            dtype=torch.bfloat16,
            device_map="cpu",
        )

        image_tokens = model.model.vocabulary_mapping.image_tokens
        model.lm_head.weight.data[image_tokens] = torch.zeros_like(model.lm_head.weight.data[image_tokens])

        model.save_pretrained(save_path, max_shard_size="10GB")


if __name__ == "__main__":
    args = Solver.get_args_parser().parse_args()
    solver = Solver(args)
    solver.run_with_eval_awm_w()