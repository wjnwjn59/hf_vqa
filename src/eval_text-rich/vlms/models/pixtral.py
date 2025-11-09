"""Pixtral Model Inference Module using vLLM"""

import torch
from typing import List, Dict, Any, Tuple, Optional
from vllm import LLM, SamplingParams
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model instance
_llm: Optional[LLM] = None

USER_PROMPT = (
    "Answer each question concisely in a single word or short phrase, without any lengthy descriptions or explanations.\n"
    "Rely only on information that is clearly visible in the provided image.\n"
    "If the answer cannot be determined from the image, respond with 'unanswerable'."
)


def get_model(model_dir: str = "/mnt/dataset1/pretrained_fm/mistralai_Pixtral-12B-2409") -> LLM:
    """
    Initialize and return the Pixtral model using vLLM.
    
    vLLM provides optimized inference with better throughput and memory efficiency.
    """
    global _llm
    
    if _llm is None:
        # Initialize vLLM model
        # tokenizer_mode="mistral" is required for Pixtral
        _llm = LLM(
            model=model_dir,
            tokenizer_mode="mistral",
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=0.9,  # Use 90% of GPU memory
            max_model_len=8192,  # Maximum context length
        )
    
    return _llm


def create_message(question: str, image_path: str) -> List[Dict[str, Any]]:
    """
    Create message format for Pixtral with vLLM.
    
    vLLM uses the same message format as standard transformers.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{USER_PROMPT}\nQuestion: {question.strip()}\nAnswer:"},
                {"type": "image_url", "image_url": {"url": f"file://{image_path}"}},
            ]
        }
    ]


def inference(questions: List[str], image_paths: List[str], config: Dict[str, Any]) -> List[str]:
    """
    Run inference on Pixtral model using vLLM.
    
    vLLM can process multiple samples more efficiently than standard transformers,
    but we still process sequentially for simplicity and compatibility with eval_text-rich.
    
    Args:
        questions: List of questions
        image_paths: List of paths to images (corresponding 1-to-1 with questions)
        config: DatasetConfig object containing max_new_tokens and other params
        
    Returns:
        List[str]: List of answers (stripped and first line taken)
    """
    llm = get_model()
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        max_tokens=config.max_new_tokens,
        temperature=0.0,  # Greedy decoding (equivalent to do_sample=False)
        top_p=1.0,
    )
    
    results = []
    
    # Process each question-image pair
    for question, image_path in zip(questions, image_paths):
        # Create message
        messages = create_message(question, image_path)
        
        # Generate using vLLM
        # vLLM expects a list of messages
        outputs = llm.chat(
            messages=messages,
            sampling_params=sampling_params,
        )
        
        # Extract response
        # vLLM returns a list of RequestOutput objects
        response = outputs[0].outputs[0].text
        
        # Get the first line and strip whitespace
        results.append(response.splitlines()[0].strip())
    
    return results
