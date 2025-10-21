import json
from typing import List, Dict, Any, Optional, Union
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from PIL import Image
import torch


class QwenVLInference:
    """
    Qwen2-VL inference class using vLLM for visual question answering tasks.
    """
    
    def __init__(
        self,
        model_name: str = 'unsloth/Qwen2-VL-7B-Instruct',
        max_model_len: int = 16384,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "auto",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 8192,
    ):
        """
        Initialize QwenVL inference model.
        
        Args:
            model_name: Model name or path
            max_model_len: Maximum model length
            gpu_memory_utilization: GPU memory utilization ratio
            dtype: Data type for model
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        
        # Initialize vLLM model
        print(f"Loading model: {model_name}")
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            max_num_seqs=1,
        )
        
        # Load tokenizer
        print(f"Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        # Set up sampling parameters
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_token_ids=[self.tokenizer.eos_token_id] if hasattr(self.tokenizer, 'eos_token_id') else None
        )
    
    def format_prompt(self, content: str, system_prompt: str = None) -> str:
        """
        Format prompt using chat template for Qwen2-VL.
        
        Args:
            content: User message content
            system_prompt: System prompt (optional)
            
        Returns:
            Formatted prompt string
        """
        if system_prompt:
            prompt = (f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                     f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                     f"{content}<|im_end|>\n"
                     f"<|im_start|>assistant\n")
        else:
            prompt = (f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                     f"{content}<|im_end|>\n"
                     f"<|im_start|>assistant\n")
        
        return prompt
    
    def generate_single_image(
        self, 
        image: Union[str, Image.Image],
        prompt: str,
        system_prompt: str = None,
        custom_sampling_params: Optional[SamplingParams] = None
    ) -> str:
        """
        Generate response for a single image and prompt.
        
        Args:
            image: PIL Image object or path to image
            prompt: Text prompt
            system_prompt: System prompt (optional)
            custom_sampling_params: Custom sampling parameters (optional)
            
        Returns:
            Generated response string
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # Format prompt
        formatted_prompt = self.format_prompt(prompt, system_prompt)
        
        # Prepare input for vLLM
        inputs = {
            "prompt": formatted_prompt,
            "multi_modal_data": {
                "image": image
            }
        }
        
        # Use custom sampling params if provided
        sampling_params = custom_sampling_params or self.sampling_params
        
        # Generate response
        outputs = self.llm.generate([inputs], sampling_params=sampling_params)
        response = outputs[0].outputs[0].text.strip()
        
        return response
    
    def generate_batch_images(
        self, 
        images: List[Union[str, Image.Image]],
        prompts: List[str],
        system_prompts: Optional[List[str]] = None,
        custom_sampling_params: Optional[SamplingParams] = None
    ) -> List[str]:
        """
        Generate responses for a batch of images and prompts.
        
        Args:
            images: List of PIL Image objects or paths to images
            prompts: List of text prompts
            system_prompts: List of system prompts (optional)
            custom_sampling_params: Custom sampling parameters (optional)
            
        Returns:
            List of generated response strings
        """
        assert len(images) == len(prompts), "Number of images and prompts must match"
        
        if system_prompts:
            assert len(system_prompts) == len(prompts), "Number of system prompts must match prompts"
        else:
            system_prompts = [None] * len(prompts)
        
        # Prepare inputs
        batch_inputs = []
        for i, (image, prompt, sys_prompt) in enumerate(zip(images, prompts, system_prompts)):
            # Load image if path is provided
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            
            # Format prompt
            formatted_prompt = self.format_prompt(prompt, sys_prompt)
            
            # Prepare input for vLLM
            inputs = {
                "prompt": formatted_prompt,
                "multi_modal_data": {
                    "image": image
                }
            }
            batch_inputs.append(inputs)
        
        # Use custom sampling params if provided
        sampling_params = custom_sampling_params or self.sampling_params
        
        # Generate responses
        outputs = self.llm.generate(batch_inputs, sampling_params=sampling_params)
        responses = [output.outputs[0].text.strip() for output in outputs]
        
        return responses
    
    def parse_json_response(self, generated_text: str) -> Dict[str, Any]:
        """
        Parse generated text to extract JSON response.
        
        Args:
            generated_text: Generated text from model
            
        Returns:
            Dictionary containing parsed information
        """
        try:
            # Try to extract JSON from the content
            json_start = generated_text.find("{")
            json_end = generated_text.rfind("}") + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = generated_text[json_start:json_end]
                parsed_json = json.loads(json_str)
                return {
                    "response": parsed_json,
                    "success": True
                }
            else:
                return {
                    "response": generated_text,
                    "success": False,
                    "error": "No valid JSON found in response"
                }
                
        except json.JSONDecodeError as e:
            return {
                "response": generated_text,
                "success": False,
                "error": f"JSON decode error: {str(e)}"
            }
        except Exception as e:
            return {
                "response": generated_text,
                "success": False,
                "error": f"Parsing error: {str(e)}"
            }
    
    def generate_and_parse_json_single(
        self, 
        image: Union[str, Image.Image],
        prompt: str,
        system_prompt: str = None,
        custom_sampling_params: Optional[SamplingParams] = None
    ) -> Dict[str, Any]:
        """
        Generate response and parse JSON for a single image.
        
        Args:
            image: PIL Image object or path to image
            prompt: Text prompt
            system_prompt: System prompt (optional)
            custom_sampling_params: Custom sampling parameters (optional)
            
        Returns:
            Parsed response dictionary
        """
        response = self.generate_single_image(
            image, prompt, system_prompt, custom_sampling_params
        )
        
        return self.parse_json_response(response)
    
    def generate_and_parse_json_batch(
        self, 
        images: List[Union[str, Image.Image]],
        prompts: List[str],
        system_prompts: Optional[List[str]] = None,
        custom_sampling_params: Optional[SamplingParams] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate responses and parse JSON for a batch of images.
        
        Args:
            images: List of PIL Image objects or paths to images
            prompts: List of text prompts
            system_prompts: List of system prompts (optional)
            custom_sampling_params: Custom sampling parameters (optional)
            
        Returns:
            List of parsed response dictionaries
        """
        responses = self.generate_batch_images(
            images, prompts, system_prompts, custom_sampling_params
        )
        
        parsed_responses = [
            self.parse_json_response(response) 
            for response in responses
        ]
        
        return parsed_responses
    
    def update_sampling_params(
        self, 
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Update sampling parameters.
        
        Args:
            temperature: New temperature value
            top_p: New top_p value
            max_tokens: New max_tokens value
        """
        if temperature is not None:
            self.temperature = temperature
        if top_p is not None:
            self.top_p = top_p
        if max_tokens is not None:
            self.max_tokens = max_tokens
            
        # Update sampling params
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop_token_ids=[self.tokenizer.eos_token_id] if hasattr(self.tokenizer, 'eos_token_id') else None
        )