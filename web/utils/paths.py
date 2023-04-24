import os


def create_path_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_path(type, filename=None):
    if type == 'out':
        base_path = os.path.join('.', 'static', 'contents', 'out')
    elif type == 'uploads':
        base_path = os.path.join('.', 'static', 'contents', 'uploads')
    elif type == 'latent':
        base_path = os.path.join('.', 'static', 'contents', 'latent')
    elif type == 'inference':
        base_path = os.path.join('.', 'static', 'contents', 'inference')
    elif type == 'inference':
        base_path = os.path.join('.', 'static', 'contents', 'inference')
    else:
        raise Exception('unknown path type')
    create_path_if_not_exists(base_path)
    return base_path if filename is None else os.path.join(base_path, filename)


def get_lora_model_path(model_full_name, user_id):
    model_repo, model_name = model_full_name.split('/')[:2]
    if model_repo == 'public':
        filepath = os.path.join('.', 'lora_models', 'public', f'{model_name}.safetensors')
    else:
        filepath = os.path.join('.', 'lora_models', str(user_id), f'{model_name}.safetensors')
    return filepath

