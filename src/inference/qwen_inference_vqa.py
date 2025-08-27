
import torch
import os
import json
import sys
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from jinja2 import Environment, FileSystemLoader
from collections import defaultdict

# Load the model in half-precision on the available device(s)
model_path = "/mnt/dataset1/pretrained_fm/Qwen_Qwen2-VL-2B-Instruct/"
model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, device_map="auto")
processor = AutoProcessor.from_pretrained(model_path)

def render_prompt(template_path, **kwargs):
    template_Dir, template_file = os.path.split(template_path)
    env = Environment(loader=FileSystemLoader(template_Dir))
    template = env.get_template(template_file)
    prompt = template.render(**kwargs)

    return prompt


dataset_dir = "/mnt/VLAI_data/InfographicVQA"
image_dir = os.path.join(dataset_dir, "images")
train_ann_path = os.path.join(dataset_dir, "question_answer", "infographicsVQA_train_v1.0.json")

with open(train_ann_path, "r") as f:
    train_ann = json.load(f)

if isinstance(train_ann, list) and len(train_ann) > 0:
    print("Top-level type: list")
    print("Number of items:", len(train_ann))
    print("First item keys:", list(train_ann[0].keys()))
    print("Example first item:")
    for k, v in train_ann[0].items():
        print(f"  {k}: {type(v).__name__} ({str(v)[:80]}{'...' if len(str(v))>80 else ''})")
elif isinstance(train_ann, dict):
    print("Top-level type: dict")
    print("Top-level keys:", list(train_ann.keys()))
else:
    print("Unknown JSON structure:", type(train_ann))

grouped = defaultdict(list)
for d in train_ann["data"]:
    grouped[d['image_local_name']].append(d)

test_img_name = "70283.jpeg"

print(grouped[test_img_name])

n_gen_qas = 1
vqg_template_path = "../prompts/vqg.jinja"
prompt = render_prompt(vqg_template_path, num_questions=n_gen_qas)

conversation = [
    {
        "role":"user",
        "content":[
            {
                "type":"image",
                "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
            },
            {
                "type":"text",
                "text":"Describe this image."
            }
        ]
    }
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

# Inference: Generation of the output
output_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(output_text)



# Video
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "video", "path": "/path/to/video.mp4"},
            {"type": "text", "text": "What happened in the video?"},
        ],
    }
]

inputs = processor.apply_chat_template(
    conversation,
    video_fps=1,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)


# Inference: Generation of the output
output_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(output_text)