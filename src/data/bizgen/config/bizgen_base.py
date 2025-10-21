#### Model Setting
# pretrained_model_name_or_path = 'stabilityai/stable-diffusion-xl-base-1.0'
pretrained_model_name_or_path = "checkpoints/spo"
pretrained_vae_model_name_or_path = 'madebyollin/sdxl-vae-fp16-fix'
revision = None

train_byt5 = False
byt5_ckpt_dir = 'checkpoints/byt5'
byt5_mapper_type = 'T5EncoderBlockByT5Mapper'
byt5_mapper_config = dict(
    num_layers=4,
    sdxl_channels=2048,
)
byt5_config = dict(
    byt5_ckpt_path=None,
    byt5_name='google/byt5-small', 
    special_token=True, 
    color_special_token=True,
    font_special_token=True,
    font_ann_path='glyph/font_uni_10-lang_idx.json',
    multilingual=True,  
)
byt5_max_length = 2560

unet_lora_pretrain_ckpt = None
load_typo_sdxl_pretrain_ckpt = None
load_openclip_pretrain_ckpt = None


mixed_precision = "fp16"
prediction_type = None


unet_lora_rank = 128
inference_dtype = 'fp16'

#### Save Memory Setting
gradient_checkpointing = False
allow_tf32 = True
use_8bit_adam = False
enable_xformers_memory_efficient_attention = False
byt5_with_xformers = False

#### misc
dist_backend = 'nccl'


use_sync = False