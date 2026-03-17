import argparse
import os
import torch
import json
from transformers import AutoModelForImageTextToText, AutoProcessor

from PIL import Image
import random
import ast
import re

VARIA_PROMPTs = [
    f"""You are a sophisticated dual-arm robot planning the next action for the goal: {{task}}.
    Your output must be a single, structured sentence that embeds perceptual data. This involves selecting the most suitable frame (`n`) for each perception task and providing the corresponding integer coordinates (in the [0, 1000] range). 
    The sentence must follow this precise format: `<affordance>` contains `<frame n>: (x, y)` for the grasp point, `<object>` contains `<frame n>: (x_min, y_min), (x_max, y_max)` for the bounding box, and `<area>` contains `<frame n>: (x, y)` for the target destination point.""",

    f"""You are a sophisticated dual-arm robot planning the next action for the goal: {{task}}.

        Provide the answer as a single sentence embedding frame-specific data. The format for each tag must be as follows:
        -   **`<affordance>`**: `<frame n>: (x, y)` (a single grasp point)
        -   **`<object>`**: `<frame n>: (x_min, y_min), (x_max, y_max)` (the top-left and bottom-right points of the bounding box)
        -   **`<area>`**: `<frame n>: (x, y)` (a single destination point)
        All coordinate values must be integers in the range [0, 1000].""",

    f"""You are a sophisticated dual-arm robot planning the next action for the goal: {{task}}.
        Output a single sentence embedding frame-indexed coordinates (0-1000 integers). The required format is: `<affordance>` gets `<frame n>: (point)`, `<object>` gets `<frame n>: (top-left point), (bottom-right point)`, and `<area>` gets `<frame n>: (destination point)`.""",

    f"""You are a sophisticated dual-arm robot planning the next action for the goal: {{task}}.

        Format your output sentence according to these steps:
        1.  For each perceptual task (grasp point, object, destination), select the best frame `n`.
        2.  Embed the frame and its predicted integer coordinates (0-1000) inside the appropriate tag.
        3.  The data structure for each tag is: `<affordance>`: `<frame n>: (x, y)`, `<object>`: `<frame n>: (x_min, y_min), (x_max, y_max)`, `<area>`: `<frame n>: (x, y)`.""",

    f"""You are a sophisticated dual-arm robot planning the next action for the goal: {{task}}.

    Your output sentence must conform to the following data-embedding schema. The payload for each tag must be a string containing a frame index and integer coordinates (0-1000). The specific payload formats are: `<affordance>`: point `(x, y)`, `<object>`: bounding box `(x_min, y_min), (x_max, y_max)`, `<area>`: destination point `(x, y)`."""

    f"""You are a sophisticated dual-arm robot planning the next action for the goal: {{task}}.

        Your output sentence must embed perceptual data according to these constraints:
        -   **Tag Content**: Each tag must start with `<frame n>:`.
        -   **`<affordance>` Data**: A single `(x, y)` tuple for the grasp point.
        -   **`<object>` Data**: Exactly two `(x, y)` tuples defining the bounding box.
        -   **`<area>` Data**: A single `(x, y)` tuple for the destination point.
        -   **Coordinate Range**: All `x` and `y` values must be integers where `0 <= value <= 1000`.""",

    f"""You are a sophisticated dual-arm robot planning the next action for the goal: {{task}}.

        Construct your output by filling in the placeholders of this exact template. You must select the frame `[n]` and predict the `[coordinates]` (0-1000 int) for each part.
        Template: `Verb the <affordance> <frame [n]>: ([x], [y]) </affordance> of <object> <frame [n]>: ([x_min], [y_min]), ([x_max], [y_max]) </object> to the <area> <frame [n]>: ([x], [y]) </area> with the [arm].`""",


    f"""You are a sophisticated dual-arm robot planning the next action for the goal: {{task}}.

        Adhere to the following output rules:
        - **Rule 1**: The response must be a single, complete sentence.
        - **Rule 2**: The sentence must embed data by selecting a frame `n` and predicting integer coordinates within the [0, 1000] range.
        - **Rule 3**: Data format within tags must be `<tag> <frame n>: (data) </tag>`, where `data` is a single point for both affordance and area, and two points `(min_coord), (max_coord)` for object.""",

    f"""You are a sophisticated dual-arm robot planning the next action for the goal: {{task}}.

        Your response sentence must encapsulate frame-indexed perceptual data. This data-binding follows a strict schema: the `<affordance>` tag encapsulates a `grasp point <frame n>: (x, y)`, the `<object>` tag encapsulates a `bounding box defined by its diagonal corners <frame n>: (x_min, y_min), (x_max, y_max)`, and the `<area>` tag encapsulates a `target destination point <frame n>: (x, y)`. All coordinate values must be integers normalized to the 0-1000 standardized pixel system.""",

    f"""You are a sophisticated dual-arm robot planning the next action for the goal: {{task}}.

        Construct your sentence ensuring strict syntax for the embedded data within each tag. The required syntax is `<tag><frame index>: (coordinate list)</tag>`. For `<affordance>` and `<area>`, the list is one `(x, y)` tuple; for `<object>`, exactly two tuples representing the top-left and bottom-right corners. All coordinate numbers must be integers between 0 and 1000.""",


]

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


class RynnBrain_Planning:

    def __init__(self, model_path):

        self.max_new_tokens = 2048
        self.modal = "image"

        disable_torch_init()
        
        self.model = AutoModelForImageTextToText.from_pretrained(model_path, dtype="auto", device_map="auto")
        self.processor = AutoProcessor.from_pretrained(model_path)

    def process_input(self, image_list, conversation):
        if isinstance(conversation, str):
            try:
                conversation = json.loads(conversation)
            except:
                conversation = ast.literal_eval(conversation)

        image = Image.open(image_list[0])
        w, h = image.size

        image_idx = 0
        new_conversation = []
        for conv in conversation:
            content_list = []
            img_idx_local = 0
            for content in conv['content']:
                if conv['role']=='user' and content['type']=='text':
                    content['text'] = random.choice(VARIA_PROMPTs).format(task=content['text'])
                if conv['role']=='user' and content['type']=='image':
                    content_list.append(
                        {"type": "text", "text": f"<frame {img_idx_local}>: "}
                    )
                    content['image'] = image_list[image_idx]
                    content['height'] = h
                    content['width'] = w
                    image_idx += 1
                    img_idx_local+=1
                content_list.append(content)
            conv['content'] = content_list
            new_conversation.append(conv)

        return new_conversation
    
    def inference(self, image_list, conversation):

        text_json = self.process_input(image_list, conversation)
        print('new conversation:', text_json)
        if not image_list:
            return {"received_text": text_json, "image_paths": []}

        inputs = self.processor.apply_chat_template(
                text_json,
                return_dict=True,
                return_tensors="pt",
                tokenize=True,
                add_generation_prompt=True,
                padding=False,
                padding_side=None,
                add_think_prompt=False
            )
        

        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        print('response', response)

        match = list(re.finditer(r"<frame\s*(\d+)>:", response))
        total_frame = inputs['image_grid_thw'].shape[0]
        if match:
            if 0 <= int(match[-1].group(1)) < total_frame:
                frame_idx = int(match[-1].group(1))  
            else:
                frame_idx = total_frame - 1
        else:
            frame_idx = total_frame - 1

        output = {
            'outputs': response,
            'frame_idx': frame_idx
        }

        return output