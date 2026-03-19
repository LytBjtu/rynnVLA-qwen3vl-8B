import argparse
import json
from pathlib import Path

import torch
from transformers import AutoProcessor

from rynn_scale.models.qwen3_vl.processing_qwen3_vl import apply_monkey_patch as apply_processor_monkey_patch
from rynn_scale.models.qwen3_vl_action_head.modeling_qwen3_vl_action_head import (
    ACTION_HEAD_METADATA_NAME,
    Qwen3VLActionHeadForConditionalGeneration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a RynnScale Qwen3-VL action-head checkpoint.")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to a checkpoint-* directory.")
    parser.add_argument("--prompt", type=str, required=True, help="Task instruction text.")
    parser.add_argument("--front_image", type=str, required=True, help="Path to the front camera image.")
    parser.add_argument("--wrist_image", type=str, default="", help="Optional path to the wrist camera image.")
    parser.add_argument("--state", type=str, default="", help="Optional state string appended to the prompt.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Target device, for example cuda:0 or cpu.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        help="Attention implementation passed to the model loader.",
    )
    parser.add_argument("--mm_max_length", type=int, default=1024, help="Max multimodal token budget.")
    parser.add_argument("--fps", type=int, default=2, help="Video FPS for multimodal preprocessing.")
    parser.add_argument("--max_frames", type=int, default=1, help="Max frames for video preprocessing.")
    parser.add_argument("--output_json", type=str, default="", help="Optional path to save the prediction as JSON.")
    return parser.parse_args()


def to_torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_metadata(checkpoint_dir: Path) -> dict:
    metadata_path = checkpoint_dir / ACTION_HEAD_METADATA_NAME
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Cannot find metadata file: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def build_conversation(prompt: str, front_image: str, wrist_image: str, state: str):
    text = prompt.strip()
    if state.strip():
        text = f"{text}\nState: {state.strip()}"

    content = [{"type": "text", "text": text}, {"type": "image", "image": front_image}]
    if wrist_image:
        content.append({"type": "image", "image": wrist_image})
    return [{"role": "user", "content": content}]


def main() -> None:
    args = parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint_dir is not a directory: {checkpoint_dir}")

    front_image = Path(args.front_image)
    if not front_image.is_file():
        raise FileNotFoundError(f"front_image not found: {front_image}")

    wrist_image = Path(args.wrist_image) if args.wrist_image else None
    if wrist_image is not None and not wrist_image.is_file():
        raise FileNotFoundError(f"wrist_image not found: {wrist_image}")

    apply_processor_monkey_patch()

    metadata = load_metadata(checkpoint_dir)
    dtype = to_torch_dtype(args.dtype)

    model = Qwen3VLActionHeadForConditionalGeneration.from_checkpoint(
        checkpoint_dir=str(checkpoint_dir),
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        device_map={"": args.device},
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(metadata["base_model_path"])

    conversation = build_conversation(
        prompt=args.prompt,
        front_image=str(front_image),
        wrist_image=str(wrist_image) if wrist_image is not None else "",
        state=args.state,
    )

    model_inputs = processor.apply_chat_template(
        conversation=conversation,
        mm_max_length=args.mm_max_length,
        fps=args.fps,
        max_frames=args.max_frames,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    model_inputs = model_inputs.to(device)

    with torch.inference_mode():
        action_predictions = model.predict_action(**model_inputs)

    action_predictions = action_predictions.detach().cpu()
    print("pred action shape:", tuple(action_predictions.shape))
    print(json.dumps(action_predictions.tolist(), ensure_ascii=False))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint_dir": str(checkpoint_dir),
            "base_model_path": metadata["base_model_path"],
            "action_dim": metadata["action_dim"],
            "time_horizon": metadata["time_horizon"],
            "prompt": args.prompt,
            "state": args.state,
            "front_image": str(front_image),
            "wrist_image": str(wrist_image) if wrist_image is not None else "",
            "prediction_shape": list(action_predictions.shape),
            "prediction": action_predictions.tolist(),
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved json: {output_path}")


if __name__ == "__main__":
    main()
