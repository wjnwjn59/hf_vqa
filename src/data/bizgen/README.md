<h1 align="center">BizGen: Advancing Article-level Visual Text Rendering for Infographics Generation (Glyph-ByT5-v3)</h1>
<p align="center">
  <a href="https://arxiv.org/abs/2503.20672"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
  <a href='https://bizgen-msra.github.io'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
  <a href='https://huggingface.co/PYY2001/BizGen'><img src='https://img.shields.io/badge/Model-Huggingface-yellow?logo=huggingface&logoColor=yellow' alt='Model'></a>


<p align="center"><img src="assets/teaser_info.png" width="100%"></p>
<p align="center"><img src="assets/teaser_slide.png" width="100%"></p>

<span style="font-size: 16px; font-weight: 600;">This repository supports article-level visual text rendering of business content (infographics and slides) based on ultra-dense layouts

<!-- Features -->
## 🌟 Features
- **Long context length**: Supports ultra-dense layouts with 50+ layers and article-level descriptive prompts with more than 1000 tokens, and can generate high-quality business content with up to 2240*896 resolution.
- **Powerful visual text rendering**: Supports article-level visual text rendering in ten different languages and maintains high spelling accuracy.
- **Image generation diversity and flexibility**: Supports layer-wise detail refinement through layout conditional CFG.


<!-- TODO List -->
## 🚧 TODO List
- [x] Release inference code and pretrained model
- [ ] Release training code


## Table of Contents
- [Environment Setup](#environment-setup)
- [Testing](#testing-bizgen)

## Environment Setup

### 1. Create Conda Environment
```bash
conda create -n bizgen python=3.10 -y
conda activate bizgen
```

### 2. Install Dependencies 
```bash
git clone
cd bizgen
pip install -r requirements.txt
```

### 3. Login to Hugging Face
```bash
huggingface-cli login
```

## Quick Start
Use inference.py to simply have a try:
```
python inference.py
```

## Testing BizGen

### 1. Download Checkpoints

Create a path `bizgen/checkpoints` and download the following [checkpoints](https://huggingface.co/PYY2001/BizGen) into this path.

| Name | Description|
|----------|-------------|
| `byt5` | ByT5 model checkpoint |
| `lora_infographic` | Unet LoRA weights and finetuned ByT5 mapper checkpoint for infographic |
| `lora_slides` | Unet LoRA weights and finetuned ByT5 mapper checkpoint for slides |
| `spo` | Post-trained SDXL checkpoint (for aesthetic improvement) |

The downloaded checkpoints should be organized as follows:
```
checkpoints/
├── byt5/
│   ├── base.pt
│   └── byt5_model.pt
├── lora/
|   ├── infographic/
|   |   ├──byt5_mapper.pt
|   |   └──unet_lora.pt
|   └── slides/
|       ├──byt5_mapper.pt
|       └──unet_lora.pt
└── spo
```

### 2. Run the testing Script
For infographics:
```bash
python inference.py \
--ckpt_dir checkpoints/lora/infographic \
--output_dir infographic \
--sample_list meta/infographics.json 
```
For slides:
```bash
python inference.py \
--ckpt_dir checkpoints/lora/slides \
--output_dir slide \
--sample_list meta/slides.json 
```

## :mailbox_with_mail: Citation
If you find this code useful in your research, please consider citing:

```
@misc{peng2025bizgenadvancingarticlelevelvisual,
  title={BizGen: Advancing Article-level Visual Text Rendering for Infographics Generation}, 
  author={Yuyang Peng and Shishi Xiao and Keming Wu and Qisheng Liao and Bohan Chen and Kevin Lin and Danqing Huang and Ji Li and Yuhui Yuan},
  year={2025},
  eprint={2503.20672},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2503.20672}, 
}
```
```
@article{liu2024glyphv2,
  title={Glyph-ByT5-v2: A Strong Aesthetic Baseline for Accurate Multilingual Visual Text Rendering},
  author={Liu, Zeyu and Liang, Weicong and Zhao, Yiming and Chen, Bohan and Li, Ji and Yuan, Yuhui},
  journal={arXiv preprint arXiv:2406.10208},
  year={2024}
}
```
```
@article{liu2024glyph,
  title={Glyph-byt5: A customized text encoder for accurate visual text rendering},
  author={Liu, Zeyu and Liang, Weicong and Liang, Zhanhao and Luo, Chong and Li, Ji and Huang, Gao and Yuan, Yuhui},
  journal={arXiv preprint arXiv:2403.09622},
  year={2024}
}
```
