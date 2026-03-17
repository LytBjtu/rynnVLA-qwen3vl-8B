---
language:
- en
- zh
library_name: transformers
license: apache-2.0   # TODO: 如果不是 Apache-2.0 请改
tags:
- robotics
- embodied-ai
- egocentric
- spatiotemporal
- vision-language-model
- video-understanding
- grounding
- planning
- navigation
- ocr
- image-text-to-text
- video-text-to-text
- custom_code     # 如果需要 trust_remote_code=True
base_model: RynnBrain-8B
pipeline_tag: image-text-to-text
---
<p align="center">
    <img src="https://cdn-uploads.huggingface.co/production/uploads/66974212a9e7257fc37798dc/8eIGdvaZEuZKTcsjIAXXK.png" width="350" style="margin-bottom: 0.2;"/>
<p
<h2 align="center"><a href="https://github.com/alibaba-damo-academy/RynnBrain">RynnBrain: Open Embodied Foundation Models</a></h3>
<h5 align="center"> If you like our project, please give us a star ⭐ on <a href="https://github.com/alibaba-damo-academy/RynnBrain">Github</a> for the latest update.  </h5>

---

## 📰 News
- **[2026.02.02]** Release **RynnBrain family** weights and inference code.
- **[2026.02.02]** Add cookbooks for cognition, localization, reasoning, and planning.


## ✨ Introduction

RynnBrain aims to serve as a **physics-aware embodied brain**: it observes egocentric scenes, grounds language to physical space and time, and supports downstream robotic systems with reliable localization and planning outputs.

### Key Highlights
- **Comprehensive egocentric understanding**  
  Strong spatial comprehension and egocentric cognition across embodied QA, counting, OCR, and fine-grained video understanding.

- **Diverse spatiotemporal localization**  
  Locates objects, target areas, and predicts trajectories across long episodic context, enabling global spatial awareness.

- **Physical-space grounded reasoning (RynnBrain family)**  
  The broader RynnBrain family includes “Thinking” variants that interleave textual reasoning with spatial grounding to anchor reasoning in reality.

- **Physics-aware precise planning (RynnBrain family)**  
  Integrates localized affordances/areas/objects into planning outputs to provide downstream VLA models with precise instructions.

## 🌎 Model Zoo

| Model            | Base Model           | Huggingface | Modelscope |
| :--------------- | :------------------- | :---------: | :--------: |
| RynnBrain-2B     | Qwen3-VL-2B-Instruct | [Link](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-2B)    | [Link](https://www.modelscope.cn/models/DAMO_Academy/RynnBrain-2B)   |
| RynnBrain-8B  | Qwen3-VL-8B-Instruct | [Link](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-8B)    | [Link](https://www.modelscope.cn/models/DAMO_Academy/RynnBrain-8B)   |
| RynnBrain-30B-A3B  | Qwen3-VL-30B-A3B-Instruct | [Link](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-30B-A3B)    | [Link](https://www.modelscope.cn/models/DAMO_Academy/RynnBrain-30B-A3B)   |
| RynnBrain‑CoP-8B | RynnBrain-8B         | [Link](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-CoP-8B)    | [Link](https://www.modelscope.cn/models/DAMO_Academy/RynnBrain-CoP-8B)   |
| RynnBrain‑Plan-8B (**This Checkpoint**)| RynnBrain-8B        | [Link](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-Plan-8B)    | [Link](https://www.modelscope.cn/models/DAMO_Academy/RynnBrain-Plan-8B)   |
| RynnBrain‑Plan-30B-A3B | RynnBrain-30B-A3B        | [Link](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-Plan-30B-A3B)    | [Link](https://www.modelscope.cn/models/DAMO_Academy/RynnBrain-Plan-30B-A3B)   |
| RynnBrain‑Nav-8B | RynnBrain-8B        | [Link](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-Nav-8B)    | [Link](https://www.modelscope.cn/models/DAMO_Academy/RynnBrain-Nav-8B)   |

## 🚀 Main Results

<img width="500" alt="image" src="https://github.com/alibaba-damo-academy/RynnBrain.github.io/blob/main/assets/plan_results.png?raw=true">

## 🤖 Quick Start

Minimal dependencies:
```shell
pip install transformers==4.57.1
```
Run text generation:
```python
from transformers import AutoModelForImageTextToText
model = AutoModelForImageTextToText.from_pretrained("")
...
```
## Cookbooks
Checkout the [cookbooks](https://github.com/alibaba-damo-academy/RynnBrain/cookbooks) that showcase RynnBrain's capabilities in cognition, localization, reasoning, and planning.


| Category             | Cookbook name                                                                                   | Description |
|----------------------|--------------------------------------------------------------------------------------------------|-------------|
| Planning | [10_manipulate_planning.ipynb](./cookbooks/10_manipulate_planning.ipynb)                                     | Performs multi-step task decomposition and action planning from goals and scenes. |


## 📑 Citation

If you find RynnBrain useful for your research and applications, please cite using this BibTeX:

```bibtex
@article{damo2026rynnbrain,
  title={RynnBrain: Open Embodied Foundation Models},
  author={Ronghao Dang, Jiayan Guo, Bohan Hou, Sicong Leng, Kehan Li, Xin Li, Jiangpin Liu, Yunxuan Mao, Zhikai Wang, Yuqian Yuan, Minghao Zhu, Xiao Lin, Yang Bai, Qian Jiang, Yaxi Zhao, Minghua Zeng, Junlong Gao, Yuming Jiang, Jun Cen, Siteng Huang, Liuyi Wang, Wenqiao Zhang, Chengju Liu, Jianfei Yang, Shijian Lu, Deli Zhao},
  journal={arXiv preprint arXiv:2602.14979v1},
  year={2026},
  url = {https://arxiv.org/abs/2602.14979v1}
}

```

