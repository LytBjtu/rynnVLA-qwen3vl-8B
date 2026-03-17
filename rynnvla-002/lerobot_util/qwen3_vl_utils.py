import torch
from transformers import GenerationConfig

def get_action_qwen3_vl_wrist_action_head(
    model, cur_img, wrist_img, task_description, item_processor, his_img, his_type, action_steps, state=None
):
    content = [{"type": "text", "text": f"What action should the robot take to {task_description}?"}]
    if state is not None:
        content.append({"type": "text", "text": f" State: {state}."})
    content.append({"type": "image", "image": cur_img})
    content.append({"type": "image", "image": wrist_img})
    content.append({"type": "text", "text": "<|action_start|>"})

    conversation = [{"role": "user", "content": content}]
    prompt = model.processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )

    model_inputs = model.processor(
        text=[prompt],
        images=[cur_img, wrist_img],
        return_tensors="pt",
    ).to(model.device)

    generation_config = GenerationConfig(
        max_new_tokens=2,
        do_sample=False,
        temperature=1.0,
    )
    return model.generate_action_head(model_inputs, generation_config)