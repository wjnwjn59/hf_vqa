import os
from jinja2 import Environment, FileSystemLoader


def get_prompt(question: str):
    """Loads and renders VQA prompt from jinja template."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_dir = os.path.join(current_dir, '../../prompts')
    env = Environment(loader=FileSystemLoader(prompts_dir))
    template = env.get_template('vqa.jinja')
    return template.render(question=question)


def extract_clean_model_name(model_path_or_name: str):
    """Extracts clean model name from HuggingFace model path."""
    model_name = model_path_or_name.split('/')[-1]
    if '_' in model_name:
        parts = model_name.split('_', 1)
        if len(parts) > 1:
            return parts[1]
    return model_name


def extract_clean_filename(filename: str):
    """Extracts clean model name from prediction filename."""
    base_name = filename.replace('.json', '')
    if '_' in base_name:
        parts = base_name.split('_')
        return '_'.join(parts[1:]) if len(parts) > 2 else base_name
    return base_name