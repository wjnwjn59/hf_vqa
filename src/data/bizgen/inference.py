import argparse
import types
import json
import os
import os.path as osp
import time
import random

import torch
import torch.utils.checkpoint
from packaging import version
from transformers import PretrainedConfig
from tqdm import tqdm

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
)
from diffusers.utils import check_min_version
from peft import LoraConfig
from peft.utils import (
    set_peft_model_state_dict,
)
from diffusers.models.attention import BasicTransformerBlock

from bizgen.utils import (
    parse_config,
    huggingface_cache_dir,
    UNET_CKPT_NAME,
    BYT5_BASE_CKPT_NAME,
    BYT5_CKPT_NAME,
    BYT5_MAPPER_CKPT_NAME,
    load_byt5_and_byt5_tokenizer,
    draw_bbox,
    draw_lcfg,
)
from bizgen.custom_diffusers import (
    monkey_patch_region_attention_glyph_attention_select,
    GlyphStableDiffusionXLBizGenPipeline,
    T5EncoderBlockByT5Mapper,
)

byt5_mapper_dict = [T5EncoderBlockByT5Mapper]
byt5_mapper_dict = {mapper.__name__: mapper for mapper in byt5_mapper_dict}

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.26.1")


def set_random_seed(seed: int) -> int:
    """Set random seed for reproducibility and return the seed used
    
    Sets seeds for all commonly used libraries in diffusion model training/inference:
    - Python random
    - NumPy
    - PyTorch
    - CUDA (if available)
    - Transformers
    - Environment variables for additional determinism
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    
    # PyTorch
    torch.manual_seed(seed)
    
    # CUDA seeds
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        
    # Additional PyTorch determinism settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Transformers library seed
    try:
        from transformers import set_seed as transformers_set_seed
        transformers_set_seed(seed)
    except ImportError:
        pass
    
    # Environment variables for additional determinism
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For deterministic CUDA operations
    
    # TensorFlow (if used)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    return seed


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder",
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder=subfolder,
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def main():
    parser = argparse.ArgumentParser()
    r"""
    Args:
        config_dir (`str`):
            config file contains the basic setting and parameters, no need to change for inference.
        ckpt_dir (`str`):
            checkpoint directory contains the model weights for lora and byt5_mapper.
        output_dir (`str`):
            base output directory name. The final output will be saved to output_dir/dataset_name.
        device (`str`):
            device to run the inference.
        wiki_dir (`str`):
            directory containing wiki files (wikixxxxxx.json format), each containing 50 layouts.
        subset (`str`):
            file range to process in format "start:end" (1-based). Each file contains 50 layouts.
            Example: "1:10" will process wiki000001.json to wiki000010.json (500 images total).  
        dataset_name (`str`):
            dataset name for output folder structure.
        seed (`int`):
            seed for the random number generator.
        global_ratio (`float`):
            the ratio decides how much the global prompt take effect in the layout guided cross attention, recommended 0.0 to 0.3.
        num_inference_steps (`int`):
            number of inference steps for the diffusion model.
        guidance_scale (`int`):
            guidance scale for the diffusion model, recommended 5 to 7.
        height (`int`), width (`int`), different_size (`bool`):
            If different_size is True, the height and width will be ignored and the size will be based on the background layer of each image in the asmple_list. If different_size is False, the height and width will be universal for all images. Recommended height, width for slides: 864, 1536. Recommended height, width for infographics: 2240, 896.
        lcfg_ratio (`float`):
            decides the timestep range where lcfg takes effect, recommended 0.5.
    """
    parser.add_argument('--config_dir', type=str,
                        default='config/bizgen_base.py')
    parser.add_argument('--ckpt_dir', type=str,
                        default="checkpoints/lora/infographic")
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--wiki_dir', type=str, default='../create_data/output/bizgen_format',
                        help='Directory containing wiki files (wikixxxxxx.json format). Ignored if --wiki_path is specified.')
    parser.add_argument('--wiki_path', type=str, default=None,
                        help='Path to a single wiki JSON file (e.g., failed.json). If specified, --wiki_dir and --subset are ignored.')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Base output directory name')
    parser.add_argument('--subset', type=str, default='1:2', 
                        help='File range to process (1-based): "start:end". Each file contains 50 layouts. Example: "1:10" processes wiki000001.json to wiki000010.json. Ignored if --wiki_path is specified.')
    parser.add_argument('--dataset_name', type=str, default='squad_v2',
                        help='Dataset name for output folder structure')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--global_ratio', type=float, default=0.2)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--guidance_scale', type=int, default=7)
    parser.add_argument('--height', type=int, default=2240)
    parser.add_argument('--width', type=int, default=896)
    parser.add_argument('--different_size', type=bool, default=True)
    parser.add_argument('--lcfg_ratio', type=float, default=0.5)

    args = parser.parse_args()
    config = parse_config(args.config_dir)

    # Determine processing mode: single file or directory with subset
    use_single_file = args.wiki_path is not None
    
    if use_single_file:
        print(f"Running in single-file mode with: {args.wiki_path}")
    else:
        # Parse subset as file numbers (1-based), not item indices
        start_file_subset, end_file_subset = [int(idx) for idx in args.subset.split(':')]
        print(f"Running in directory mode with subset: {args.subset}")
    
    # Create output directory with dataset name
    args.output_dir = osp.join(args.output_dir, args.dataset_name)

    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # The base model is set to be our post-trained spo model, which beats the original sdxl-base-1.0 in aesthetic quality.
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        config.pretrained_model_name_or_path, config.revision,
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        config.pretrained_model_name_or_path, config.revision, subfolder="text_encoder_2",
    )

    text_encoder_one = text_encoder_cls_one.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="text_encoder", revision=config.revision,
        cache_dir=huggingface_cache_dir,
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=config.revision,
        cache_dir=huggingface_cache_dir,
    )

    unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="unet",
        revision=config.revision,
        cache_dir=huggingface_cache_dir,
    )

    vae_path = (
        config.pretrained_model_name_or_path
        if config.pretrained_vae_model_name_or_path is None
        else config.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path, subfolder="vae" if config.pretrained_vae_model_name_or_path is None else None,
        revision=config.revision,
        cache_dir=huggingface_cache_dir,
    )

    byt5_model, byt5_tokenizer = load_byt5_and_byt5_tokenizer(
        **config.byt5_config
    )
    if config.byt5_ckpt_dir is not None:
        byt5_state_dict = torch.load(
            osp.join(config.byt5_ckpt_dir, BYT5_BASE_CKPT_NAME), map_location='cpu')
        byt5_filter_state_dict = {}
        for name in byt5_state_dict['state_dict']:
            if 'module.text_tower.encoder.' in name:
                byt5_filter_state_dict[name[len(
                    'module.text_tower.encoder.'):]] = byt5_state_dict['state_dict'][name]
        byt5_model.load_state_dict(
            byt5_filter_state_dict,
            strict=True,
        )
        del byt5_state_dict
        del byt5_filter_state_dict
        print(f"loaded byt5 base model from {config.byt5_ckpt_dir}")

    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)
    byt5_model.requires_grad_(False)

    weight_dtype = torch.float16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    unet.to(args.device, dtype=weight_dtype)
    if config.pretrained_vae_model_name_or_path is None:
        vae.to(args.device, dtype=torch.float32)
    else:
        vae.to(args.device, dtype=weight_dtype)
    text_encoder_one.to(args.device, dtype=weight_dtype)
    text_encoder_two.to(args.device, dtype=weight_dtype)
    byt5_model.to(args.device)
    block_count = 0

    # Change the forward method of the cross-attention-blocks to use the layout guided cross attention.
    for name, module in unet.named_modules():
        if isinstance(module, BasicTransformerBlock):

            module.forward = types.MethodType(
                monkey_patch_region_attention_glyph_attention_select, module)
            module.COUNT = block_count
            block_count += 1
            print(f"monkey patch forward for cross attn block {name}")
    if config.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")

    byt5_mapper = byt5_mapper_dict[config.byt5_mapper_type](
        byt5_model.config,
        **config.byt5_mapper_config,
    )
    byt5_mapper.to(args.device)
    if getattr(args, "use_lora", True):
        unet_lora_target_modules = [
            "attn1.to_k", "attn1.to_q", "attn1.to_v", "attn1.to_out.0",
            "attn2.to_k", "attn2.to_q", "attn2.to_v", "attn2.to_out.0",
        ]
        unet_lora_config = LoraConfig(
            r=config.unet_lora_rank,
            lora_alpha=config.unet_lora_rank,
            init_lora_weights="gaussian",
            target_modules=unet_lora_target_modules,
        )
        unet.add_adapter(unet_lora_config)

        # load checkpoint
        # unet lora
        unet_lora_layers_para = torch.load(
            osp.join(args.ckpt_dir, UNET_CKPT_NAME), map_location='cpu')
        incompatible_keys = set_peft_model_state_dict(
            unet, unet_lora_layers_para, adapter_name="default")
        if getattr(incompatible_keys, 'unexpected_keys', []) == []:
            print(f"loaded unet_lora_layers_para from {args.ckpt_dir}")
        else:
            print(
                f"unet_lora_layers has unexpected_keys: {getattr(incompatible_keys, 'unexpected_keys', None)}")

    # byt5 mapper
    try:
        byt5_mapper_para = torch.load(
            osp.join(args.ckpt_dir, BYT5_MAPPER_CKPT_NAME), map_location='cpu')
        byt5_mapper.load_state_dict(byt5_mapper_para)
        print(f"loaded byt5_mapper from {args.ckpt_dir}")
    except:
        byt5_mapper_para = torch.load(
            osp.join(config.byt5_mapper_path, BYT5_MAPPER_CKPT_NAME), map_location='cpu')
        byt5_mapper.load_state_dict(byt5_mapper_para)
        print(f"loaded byt5_mapper from {config.byt5_mapper_path}")
    # byt5
    byt5_model_para = torch.load(
        osp.join(config.byt5_ckpt_dir, BYT5_CKPT_NAME), map_location='cpu')

    byt5_model.load_state_dict(byt5_model_para)
    print(f"loaded byt5_model from {config.byt5_ckpt_dir}")

    # The text encoder comes from 🤗 transformers, so we cannot directly modify it.
    # So, instead, we monkey-patch the forward calls of its attention-blocks.
    # Make sure the trainable params are in float32.
    if config.mixed_precision == "fp16":
        models = [unet, byt5_mapper]
        if config.train_byt5:
            models.append(byt5_model)
        for model in models:
            for param in model.parameters():
                # only upcast trainable parameters (LoRA and byt5_mapper) into fp32
                if param.requires_grad:
                    param.data = param.to(torch.float32)

    byt5_mapper.requires_grad_(False)
    unet.requires_grad_(False)

    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        config.pretrained_model_name_or_path,
        cache_dir=huggingface_cache_dir,
        subfolder='scheduler',
        use_karras_sigmas=True,
    )

    pipeline = GlyphStableDiffusionXLBizGenPipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        byt5_text_encoder=byt5_model,
        byt5_tokenizer=byt5_tokenizer,
        byt5_mapper=byt5_mapper,
        unet=unet,
        text_feat_length=config.byt5_max_length,
        revision=config.revision,
        torch_dtype=weight_dtype,
        safety_checker=None,
        cache_dir=huggingface_cache_dir,
        scheduler=scheduler,

    )
    pipeline = pipeline.to(device=args.device)
    pipeline.set_progress_bar_config(disable=True)

    config.seed = args.seed

    # Set comprehensive random seed for reproducibility
    if config.seed is not None:
        print(f"Setting random seed to: {config.seed}")
        set_random_seed(config.seed)

    # run inference
    generator = torch.Generator(device=args.device).manual_seed(
        config.seed) if config.seed else None

    args.width = int(args.width//32*32)
    args.height = int(args.height//32*32)

    # load layouts and captions
    file_subset = []
    
    if use_single_file:
        # Single file mode: load the specified wiki file
        if os.path.exists(args.wiki_path):
            file_subset.append(args.wiki_path)
            print(f"Loading single wiki file: {args.wiki_path}")
        else:
            raise FileNotFoundError(f"Wiki file not found: {args.wiki_path}")
    else:
        # Directory mode: load wiki files based on subset numbers (1-based file indices)
        print(f"Loading wiki files: wiki{start_file_subset:06d}.json to wiki{end_file_subset:06d}.json")
        
        for file_num in range(start_file_subset, end_file_subset + 1):
            wiki_file = f"wiki{file_num:06d}.json"
            wiki_path = os.path.join(args.wiki_dir, wiki_file)
            if os.path.exists(wiki_path):
                file_subset.append(wiki_path)
            else:
                print(f"Warning: {wiki_file} not found, skipping")
    
    # Count total items to process (all items in selected files)
    tot = 0
    for file in file_subset:
        with open(file, 'r') as f:
            data = json.load(f)
            tot += len(data)  # Count all items in each file
    
    print(f"Total items to process: {tot}")
    
    with tqdm(total=tot, desc="Total items", unit="item", dynamic_ncols=True) as pbar:
        for file in file_subset:
            with open(file, 'r') as f:
                sample_list = json.load(f)

            for region_meta_item in sample_list:
                index = region_meta_item["index"]
                
                # Process all items in the selected files (no need to skip based on index)
                
                # Calculate folder name based on index (50 images per folder)
                folder_num = ((index - 1) // 50) + 1
                folder_name = f"narrator{folder_num:06d}"
                folder_path = osp.join(args.output_dir, folder_name)
                
                # Create folder if it doesn't exist
                if not osp.exists(folder_path):
                    os.makedirs(folder_path, exist_ok=True)
                
                region_meta = region_meta_item["layers_all"]
                full_img_caption = region_meta[0]['caption']
                region_meta = region_meta[1:]
                region_meta[0]['category'] = 'background'
                orig_width, orig_height = region_meta[0]["bottom_right"]
                if args.different_size:
                    args.width = int(orig_width//32*32)
                    args.height = int(orig_height//32*32)
                for item in region_meta:
                    item["top_left"][0] = int(
                        item["top_left"][0]/orig_width*args.width)
                    item["top_left"][1] = int(
                        item["top_left"][1]/orig_height*args.height)
                    item["bottom_right"][0] = int(
                        item["bottom_right"][0]/orig_width*args.width)
                    item["bottom_right"][1] = int(
                        item["bottom_right"][1]/orig_height*args.height)

                try:
                    # Set deterministic seed for this specific image based on index and base seed
                    if config.seed is not None:
                        image_seed = config.seed + index
                        generator = torch.Generator(device=args.device).manual_seed(image_seed)
                    
                    with torch.cuda.amp.autocast():
                        output_img = pipeline(
                            full_img_caption=full_img_caption,
                            region_meta=region_meta,
                            multilingual=True,
                            global_ratio=args.global_ratio,
                            num_inference_steps=args.num_inference_steps,
                            generator=generator,
                            guidance_scale=args.guidance_scale,
                            height=region_meta[0]["bottom_right"][1] if args.different_size else args.height,
                            width=region_meta[0]["bottom_right"][0] if args.different_size else args.width,
                            layer_cfg_ratio=args.lcfg_ratio,
                        ).images[0]

                    output_img.save(osp.join(folder_path, f'{index}.png'))

                    # draw the layout
                    draw_bbox(index, region_meta, dir=folder_path)
                    # draw the lcfg map
                    draw_lcfg(index, region_meta, dir=folder_path,
                            guidance=args.guidance_scale)
                    print(f"saved {index}.png to {args.dataset_name}/{folder_name}")
                except Exception as e:
                    print(f"Error in {index}.png: {e}")
                
                pbar.update(1)


if __name__ == '__main__':
    main()
