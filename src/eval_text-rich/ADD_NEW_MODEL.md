# Hướng dẫn thêm Model mới vào DAM-QA


## Cấu trúc thư mục

```
vlms/
├── models/           # Các model implementation
│   ├── internvl.py   # Ví dụ: InternVL model
│   ├── phi.py        # Ví dụ: Phi model
│   └── your_model.py # Model mới của bạn
├── run_inference.py  # Script chạy inference
└── config.py         # Cấu hình dataset
```

## Bước 1: Tạo file model mới

Tạo file mới trong `vlms/models/` với tên `your_model.py` (thay `your_model` bằng tên model của bạn).

### Template cơ bản

```python
"""Your Model Inference Module"""

import torch
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model instances
_model: Optional[Any] = None
_processor: Optional[Any] = None

USER_PROMPT = (
    "Answer each question concisely in a single word or short phrase, without any lengthy descriptions or explanations.\n"
    "Rely only on information that is clearly visible in the provided image.\n"
    "If the answer cannot be determined from the image, respond with 'unanswerable'."
)

def get_model(model_dir: str = "path/to/your/model") -> Tuple[Any, Any]:
    """Initialize and return your model and processor."""
    global _model, _processor
    
    if _model is None or _processor is None:
        # Khởi tạo model và processor
        _processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        _model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).eval()
    
    return _model, _processor

def inference(questions: List[str], image_paths: List[str], config: Dict[str, Any]) -> List[str]:
    """Run inference on your model."""
    model, processor = get_model()
    
    # Xử lý từng câu hỏi/hình ảnh
    results = []
    for question, image_path in zip(questions, image_paths):
        # Load và preprocess image
        image = Image.open(image_path).convert("RGB")
        
        # Tạo prompt
        prompt = f"{USER_PROMPT}<image>\nQuestion: {question.strip()}\nAnswer:"
        
        # Process inputs
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=config.max_new_tokens,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # Decode response
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        response = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        results.append(response.splitlines()[0].strip())
    
    return results
```

## Bước 2: Cấu hình model

### Thay đổi đường dẫn model

Sửa đổi hàm `get_model()` để trỏ đến đường dẫn model của bạn:

```python
def get_model(model_dir: str = "/path/to/your/pretrained/model") -> Tuple[Any, Any]:
```

### Điều chỉnh preprocessing

Tùy theo model, bạn có thể cần:

- **Image preprocessing**: Resize, normalize, crop
- **Text preprocessing**: Tokenization, special tokens
- **Batch processing**: Xử lý nhiều ảnh cùng lúc

### Ví dụ với model đặc biệt

```python
def inference(questions: List[str], image_paths: List[str], config: Dict[str, Any]) -> List[str]:
    """Run inference on your model."""
    model, processor = get_model()
    
    # Xử lý batch nếu model hỗ trợ
    if hasattr(model, 'batch_chat'):
        # Sử dụng batch processing
        batch_prompts = [f"{USER_PROMPT}<image>\nQuestion: {q}\nAnswer:" for q in questions]
        batch_images = [Image.open(path).convert("RGB") for path in image_paths]
        
        # Process batch
        inputs = processor(text=batch_prompts, images=batch_images, return_tensors="pt")
        with torch.no_grad():
            outputs = model.batch_chat(processor, inputs, **config)
        
        return [output.splitlines()[0].strip() for output in outputs]
    else:
        # Xử lý từng item
        # ... (như template cơ bản)
```

## Bước 3: Chạy inference

### Chạy trên dataset cụ thể

```bash
python vlms/run_inference.py --model your_model --dataset chartqapro_test
```

### Chạy trên tất cả datasets

```bash
# Chạy từng dataset
python vlms/run_inference.py --model your_model --dataset docvqa_val
python vlms/run_inference.py --model your_model --dataset infographicvqa_val
python vlms/run_inference.py --model your_model --dataset textvqa_val
python vlms/run_inference.py --model your_model --dataset chartqa_test_human
python vlms/run_inference.py --model your_model --dataset chartqapro_test
python vlms/run_inference.py --model your_model --dataset vqav2_restval
```

### Tùy chọn khác

```bash
python vlms/run_inference.py \
    --model your_model \
    --dataset chartqapro_test \
    --batch-size 4 \
    --output-dir ./outputs/your_model_results/
```

## Bước 4: Đánh giá kết quả

### Đánh giá tự động

```bash
# Đánh giá tất cả kết quả trong thư mục
python evaluation/evaluator.py --folder ./outputs/vlm_results/ --use_llm

# Đánh giá file cụ thể
python evaluation/evaluator.py --file ./outputs/vlm_results/your_model_chartqapro_test.csv --use_llm
```

### Đánh giá thủ công

```bash
python evaluation/metrics.py --file ./outputs/vlm_results/your_model_chartqapro_test.csv --use_llm
```

## Bước 5: Kiểm tra kết quả

Kết quả sẽ được lưu trong:
- **CSV files**: `./outputs/vlm_results/your_model_dataset.csv`
- **Scores**: `./outputs/vlm_results/your_model_dataset_scores.json`

### Cấu trúc file kết quả

```csv
question_id,question,predict,gt,image_id
12345,"What is the title?",answer1,ground_truth1,img_001
12346,"How many items?",answer2,ground_truth2,img_002
```

### Metrics được tính

- **ANLS**: Cho DocVQA, InfographicVQA
- **VQA Score**: Cho TextVQA, VQAv2  
- **Relaxed Accuracy**: Cho ChartQA, ChartQAPro
