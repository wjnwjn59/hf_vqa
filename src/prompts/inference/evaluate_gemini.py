from tqdm import tqdm
import json
import google.generativeai as genai
from google.generativeai import GenerationConfig
from PIL import Image
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
import time
import argparse
from dotenv import load_dotenv
import time
import torch
load_dotenv()
api_key = os.getenv("GEMINI_APIKEY")
genai.configure(api_key=api_key)
os.environ["GOOGLE_API_KEY"] = api_key


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def inference(question: str, image_path: str) -> str:
    """
    Perform inference on an image-question pair using the QwenVL model.

    Args:
        question (str): The visual question regarding the image.
        image_path (str): The file path to the input image.
        additional_info (str, optional): Additional information about the image. Defaults to None.

    Returns:
        str: The model's answer.
    """
    system_instruction = (
        f"""Bạn là một trợ lý AI chuyên phân tích hình ảnh và trả lời câu hỏi dựa trên hình ảnh (VQA) một cách chính xác. Nhiệm vụ của bạn là:
- Phân tích hình ảnh.
- Chỉ trả lời câu hỏi mà không giải thích thêm bất kỳ thông tin nào khác.

Định dạng bắt buộc:
Trả lời: <Câu trả lời của bạn>

Ví dụ:
Câu hỏi: Bao nhiêu phần trăm dân số bị ảnh hưởng bởi căn bệnh này?
Trả lời: 12%
Câu hỏi: Tổng số sản phẩm bán ra trong 3 tháng đầu năm là bao nhiêu?
Trả lời: 1103
Câu hỏi: Học sinh đang học môn nào, toán hay vật lí?
Trả lời: Vật lí
Câu hỏi: Ai là người sáng lập công ty này?
Trả lời: Steve Jobs

Hãy dựa trên hình ảnh và câu hỏi để đưa ra câu trả lời ngắn gọn nhất.""")

    user_content = f"Câu hỏi: {question}"

    image = Image.open(image_path).convert("RGB")
    prompt = system_instruction + user_content

    model = genai.GenerativeModel("gemini-2.0-flash")
    output = model.generate_content(
        contents=[prompt, image],
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        },
        generation_config=GenerationConfig(
            temperature=0.0000001,
            max_output_tokens=400,
            top_p=0.95,
        )
    )

    answer = output.text.split("Trả lời:")[-1].strip()
    return answer

model_name = "gemini-2.0-flash"

parser = argparse.ArgumentParser()
parser.add_argument("--image_folder", type=str,
                    default="/mnt/VLAI_data/InfographicVQA/images", help="Directory of input images")
parser.add_argument("--data_path", type=str,
                    default="/mnt/VLAI_data/InfographicVQA/100_samples.json", help="JSON file of sample questions")
args = parser.parse_args()


with open(args.data_path, encoding='utf-8') as f:
    data = json.load(f)

predicts, gts = [], []
for item in tqdm(data):
    img_path = os.path.join(args.image_folder, item['source_image'])

    time.sleep(2)
    answer = inference(item['question'], img_path)
    print(f"Question: {item['question']}")
    print(f"Predicted: {answer}, GT: {item['answer']}")
    predicts.append(answer)
    gts.append(item['answer'])
    time.sleep(2)

output_file = f"{model_name.split('/')[-1]}.jsonl"
with open(output_file, 'w', encoding='utf-8') as f:
    for pred, gt in zip(predicts, gts):
        f.write(json.dumps({'predict': pred, 'gt': gt},
                ensure_ascii=False) + "\n")
