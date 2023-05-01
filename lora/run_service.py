import os
import re
import gc
import time
import json
import base64
import shutil
import urllib
import hashlib
import argparse
from io import BytesIO

import torch
import requests
from requests.exceptions import RequestException
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from lora_diffusion import patch_pipe, tune_lora_scale


TEMP_PATH = os.path.join('.', 'temp')


def make_request(url, method, body=None, headers=None):
    if method == 'get':
        response = requests.get(url)
    elif method == 'post':
        response = requests.post(url, json=body, headers=headers)
    else:
        raise RequestException('unsupported HTTP method.')
    
    if response.status_code >= 200 and response.status_code < 300:
        return response.json()
    else:
        raise RequestException(f"Error: Request returned status code {response.status_code} with message {response.content}")


def get_queue_task(api_endpoint):
    return make_request(f"{api_endpoint}?" + urllib.parse.urlencode({
        'action': 'fetch_task',
        'task_types': '1,2'
    }), 'get')
    

def get_task_lora_model(api_endpoint, access_key, max_retry=5):
    while max_retry > 0:
        try:
            return make_request(f"{api_endpoint}?" + urllib.parse.urlencode({
                'action': 'fetch_model',
                'access_key': access_key,
            }), 'get')
        except RequestException as e:
            time.sleep(5)
            max_retry = max_retry - 1
    raise e


def send_task_queue_update(api_endpoint, task_id, access_key, actions, max_retry=5):
    """Notify the master nodes"""
    body = {
        'task_id': task_id,
        'access_key': access_key,
        'actions': actions
    }
    
    while max_retry > 0:
        try:
            return make_request(api_endpoint, body=body, method='post')
        except Exception as e:
            print(e)
            time.sleep(5)
            max_retry = max_retry - 1
    raise e


def main(api_endpoint, device):
    queue_task = get_queue_task(api_endpoint)
    if queue_task is None:
        return
    if 'access_key' not in queue_task or 'task' not in queue_task:
        return
    access_key = queue_task['access_key']
    task_id = queue_task['task']['task_id']
    try:
        base64_arr = queue_task['task']['obj'].split('\n')
        
        input_path = os.path.join(TEMP_PATH, str(task_id))

        # Recreate the task folder
        if os.path.exists(input_path):
            shutil.rmtree(input_path)
        os.makedirs(input_path)
        for index, base64_str in enumerate(base64_arr):
            if len(base64_str) == 0:
                continue
            meta_info, base64_img = base64_str.split(',')
            file_extension = re.search(r'(?<=/)[a-zA-Z0-9]+(?=;)', meta_info).group()
            filename = f'{index}.{file_extension}'
            img_path = os.path.join(input_path, filename)
            with open(img_path, 'wb') as f:
                f.write(base64.b64decode((base64_img)))

        
        task_params = json.loads(queue_task['task']['task_parameters'])
        if queue_task['task']['task_type'] == 1:
            # Train
            output_dir = f'./output/{task_id}'
            
            # Update server
            send_task_queue_update(api_endpoint, task_id, access_key, [
                {
                    'action': 'log',
                    'content': f'Training model {task_params["model_name"]}...'
                }
            ])
            
            from lora_diffusion.cli_lora_pti import train as train_lora
            train_lora(pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5", \
                instance_data_dir=input_path, output_dir=output_dir, \
                train_text_encoder=True, resolution=int(task_params['resolution']), \
                train_batch_size=1, gradient_accumulation_steps=int(task_params['num_gradient_accumulation']), \
                scale_lr=True, learning_rate_unet=1e-4, learning_rate_text=1e-5, learning_rate_ti=5e-4, \
                color_jitter=True, lr_scheduler='linear', lr_warmup_steps=0, placeholder_tokens="<s1>|<s2>", \
                use_template=task_params['train_target'], save_steps=100, max_train_steps_ti=int(task_params['num_steps']), \
                max_train_steps_tuning=int(task_params['num_steps']), perform_inversion=True, clip_ti_decay=True, \
                weight_decay_ti=0.000, weight_decay_lora=0.001, continue_inversion=True, continue_inversion_lr=1e-4, \
                device=device, lora_rank=1)
            
            # Orignal location
            lora_output_model_path = os.path.join(os.path.join(output_dir, 'final_lora.safetensors'))

            # Get file md5
            with open(lora_output_model_path, 'rb') as f:
                file_data = f.read()
                md5_hash = hashlib.md5(file_data).hexdigest()
                model_base64_encoded = base64.b64encode(file_data).decode('utf-8')

            # Destination
            user_model_base_dir = os.path.join('.', 'user_models')
            if not os.path.exists(user_model_base_dir):
                os.makedirs(user_model_base_dir)
            user_model_path = os.path.join(user_model_base_dir, f'{md5_hash}.safetensors')
            shutil.move(lora_output_model_path, user_model_path)

            # Remove the original path
            os.rmdir(output_dir)
            
            send_task_queue_update(api_endpoint, task_id, access_key, [
                {
                    'action': 'log',
                    'content': 'Model trained successfully.'
                },
                {
                    'action': 'update_status',
                    'content': 2
                },
                {
                    'action': 'save_model',
                    'content': model_base64_encoded,
                    'md5': md5_hash
                },
                {
                    'action': 'finish',
                }
            ])

        elif queue_task['task']['task_type'] == 2:
            # Inference
            model_md5 = queue_task['task']['model_md5']
            lora_model_base_path = os.path.join(TEMP_PATH, 'models')
            if not os.path.exists(lora_model_base_path):
                os.makedirs(lora_model_base_path)
            lora_model_path = os.path.join(lora_model_base_path, f'{model_md5}.safetensors')
            if not os.path.exists(lora_model_path):
                res = get_task_lora_model(api_endpoint, access_key=access_key)
                model_data = res['model_data']
                with open(lora_model_path, 'wb') as f:
                    f.write(base64.b64decode((model_data)))
                with open(lora_model_path, 'rb') as f:
                    model_data = f.read()
                    downloaded_model_md5_hash = hashlib.md5(model_data).hexdigest()
                    if downloaded_model_md5_hash != model_md5:
                        send_task_queue_update(api_endpoint, task_id, access_key, [
                            {
                                'action': 'error',
                                'content': 'Mismatch MD5 found.'
                            }
                        ])
                        os.remove(lora_model_path)
                        # TODO: Stop here
            input_image = Image.open(img_path).convert("RGB").resize((512, 512))

            if 'prompt' not in task_params or len(task_params['prompt']) == 0:
                try:
                    from lavis.models import load_model_and_preprocess
                    model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
                    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
                    # Generate caption
                    user_prompt = model.generate({"image": image})[0]
                    gc.collect()
                    torch.cuda.empty_cache()
                except ModuleNotFoundError:
                    print('Warning: LAVIS is not installed')
                    user_prompt = ''
            else:
                user_prompt = task_params['prompt']

            model_id = "runwayml/stable-diffusion-v1-5"
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device != 'cpu' else torch.float32).to(device)
            patch_pipe(pipe, lora_model_path, patch_text=True, patch_unet=True, patch_ti=True)
                
            prompt = f"style of <s1><s2>, {user_prompt}"
            
            torch.manual_seed(1)
            tune_lora_scale(pipe.unet, float(task_params['unet_scale']))
            tune_lora_scale(pipe.text_encoder, float(task_params['text_encoder_scale']))

            output_image = pipe(prompt=prompt, image=input_image, strength=0.75, guidance_scale=7.5).images[0]
            
            # Convert to bytes using a BytesIO buffer
            buffered = BytesIO()
            output_image.save(buffered, format='JPEG')
            output_images_bytes = buffered.getvalue()

            # Encode the bytes as base64
            output_image_base64_encoded = base64.b64encode(output_images_bytes).decode('utf-8')

            # Hash value
            output_image_md5_hash = hashlib.md5(output_images_bytes).hexdigest()

            send_task_queue_update(api_endpoint, task_id, access_key, [
                {
                    'action': 'log',
                    'content': f'Inferece on model {task_params["model_name"]} executed successfully.'
                },
                {
                    'action': 'update_status',
                    'content': 2
                },
                {
                    'action': 'set_output_image',
                    'content': output_image_base64_encoded,
                    'extension': 'jpg',
                    'md5': output_image_md5_hash
                },
                {
                    'action': 'finish',
                }
            ])
    except Exception as e:
        print(e)
        send_task_queue_update(api_endpoint, task_id, access_key, [
            {
                'action': 'error',
                'content': 'The task experienced an unexpected error.'
            }
        ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--protocol', 
        metavar='STRING', 
        required=False, 
        choices=('http', 'https'),
        default='https',
        type=str, 
        help='protocol (http or https)')
    parser.add_argument(
        '--host', 
        metavar='STRING', 
        required=True, 
        type=str, 
        help='host address')
    parser.add_argument(
        '--port', 
        metavar='INT', 
        required=True, 
        type=int, 
        help='web server port number')
    parser.add_argument(
        '--device', 
        metavar='STRING', 
        required=False, 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu", 
        help='device')
    parser.add_argument(
        '--task-type', 
        metavar='STRING', 
        required=False,
        default='1,2', 
        type=str, 
        help='host address')
    args = parser.parse_args()
    
    api_endpoint = f'{args.protocol}://{args.host}:{args.port}/api/v1/queue/task'

    print(f'Using device: {args.device}')
    print(f'Using API: {api_endpoint}')

    while True:
        try:
            main(api_endpoint, args.device)
        except Exception as e:
            print(e)
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(10)

