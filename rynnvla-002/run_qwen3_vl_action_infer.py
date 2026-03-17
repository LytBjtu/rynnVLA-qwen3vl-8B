import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import GenerationConfig

from model.modeling_xllmx_qwen3_vl_ck_action_head import Qwen3VLXLLMXForConditionalGeneration_ck_action_head


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed Qwen3-VL action-head inference script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to Qwen3-VL model/checkpoint")
    parser.add_argument("--front_image", type=str, required=True, help="Front camera image path")
    parser.add_argument("--wrist_image", type=str, required=True, help="Wrist camera image path")
    parser.add_argument("--prompt", type=str, required=True, help="Task prompt text")

    parser.add_argument("--action_dim", type=int, default=6)
    parser.add_argument("--time_horizon", type=int, default=20)

    parser.add_argument("--action_start_token", type=str, default="<|action_start|>")
    parser.add_argument("--action_end_token", type=str, default="<|action_end|>")

    parser.add_argument("--max_new_tokens", type=int, default=2)
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.8)

    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device_map", type=str, default="cpu", help="HuggingFace device_map, e.g. cpu/auto")

    parser.add_argument("--save_json", type=str, default="", help="Optional path to save action result as json")
    return parser.parse_args()


def to_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


def main() -> None:
    args = parse_args()

    model_path = Path(args.model_path)
    front_image_path = Path(args.front_image)
    wrist_image_path = Path(args.wrist_image)

    if not model_path.exists():
        raise FileNotFoundError(f"model_path not found: {model_path}")
    if not front_image_path.exists():
        raise FileNotFoundError(f"front_image not found: {front_image_path}")
    if not wrist_image_path.exists():
        raise FileNotFoundError(f"wrist_image not found: {wrist_image_path}")

    dtype = to_torch_dtype(args.dtype)

    model = Qwen3VLXLLMXForConditionalGeneration_ck_action_head.from_pretrained(
        str(model_path),
        action_dim=args.action_dim,
        time_horizon=args.time_horizon,
        dtype=dtype,
        device_map=args.device_map,
    )

    model.setup_action_tokens(
        action_start_token=args.action_start_token,
        action_end_token=args.action_end_token,
    )
    model.eval()

    front = Image.open(front_image_path).convert("RGB")
    wrist = Image.open(wrist_image_path).convert("RGB")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": args.prompt},
                {"type": "image", "image": front},
                {"type": "image", "image": wrist},
                {"type": "text", "text": args.action_start_token},
            ],
        }
    ]

    prompt = model.processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = model.processor(
        text=[prompt],
        images=[front, wrist],
        return_tensors="pt",
    )

    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.generation_config.eos_token_id,
    )

    with torch.no_grad():
        act = model.generate_action_head(model_inputs, generation_config)

    action_list = act.detach().cpu().tolist()
    print("pred action shape:", tuple(act.shape))
    print(act)

    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "shape": list(act.shape),
            "action": action_list,
            "model_path": str(model_path),
            "front_image": str(front_image_path),
            "wrist_image": str(wrist_image_path),
            "prompt": args.prompt,
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved json: {output_path}")


if __name__ == "__main__":
    main()
