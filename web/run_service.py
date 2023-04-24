import os
import time
import json
import argparse

import torch
import torchvision

from core.startup import create_app
from core.db import db
from models.task import Task
from models.task import TASK_STATUS_PENDING
from models.task import TASK_STATUS_RUNNING
from models.task import TASK_STATUS_FINISHED
from models.task import TASK_STATUS_ERROR
from models.task import TASK_TYPE_LATENT_VECTOR_OPTIMIZATION
from models.task import TASK_TYPE_LATENT_MAPPER
from models.task import TASK_TYPE_GLOBAL_DIRECTION
from models.task_log import TaskLog
from utils.paths import get_path
from utils.logger import write_task_log
from services.style_clip.utils import handle_default_device
from services.style_clip.optimization.latent_vector_optimization import optimize_latent_vector
from services.encoder4editing.encoder import encode_to_latent
from services.style_clip.mapper.latent_mapper import run_mapper
from services.style_clip.global_directions.scripts.gloabl_directions import run_global_direction


def run_task(task, device):
    # Mark the task as running
    task.task_status = TASK_STATUS_RUNNING
    task.save(commit=False)
    write_task_log(task.id, 'Started processing...', commit=False)
    db.session.commit()

    # Run the current task
    task_params = json.loads(str(task.task_parameters))

    # Prepare the latent representation (if it does not exist)
    if task.input_file is not None:
        image_path = get_path('uploads', task.input_file)
        latent_path = get_path('latent', f'{task.input_file}.pt')
        if not os.path.exists(latent_path):
            write_task_log(task.id, 'Preprocessing image to obtain latent representation.', commit=True)
            encode_to_latent(image_path, latent_path, device)
            write_task_log(task.id, 'Successfully obtained latent representation.', commit=True)
        task_params['latent_path'] = latent_path

    output_filename = f"{task.task_uuid}.jpg"
    output_file = get_path('out', output_filename)

    write_task_log(task.id, 'Running StyleCLIP...', commit=True)
    if task.task_type == TASK_TYPE_LATENT_VECTOR_OPTIMIZATION:
        result_image = optimize_latent_vector(task_params, device)
        torchvision.utils.save_image(result_image.detach().cpu(), output_file, normalize=True, scale_each=True, range=(-1, 1))

    elif task.task_type == TASK_TYPE_LATENT_MAPPER:
        inference_path = get_path('inference', f'{task.input_file}')
        inference_path = os.path.join(inference_path, task.task_uuid)
        task_params['mapper_output_path'] = inference_path
        result_batch = run_mapper(task_params, device)
        result_image = result_batch[0][0]
        torchvision.utils.save_image(result_image.detach().cpu(), output_file, normalize=True, scale_each=True, range=(-1, 1))

    elif task.task_type == TASK_TYPE_GLOBAL_DIRECTION:
        result_image = run_global_direction(task_params, device)
        result_image.save(output_file)
    
    # Update the output file path
    task.output_file = output_filename
    # Mark the task as finished
    task.task_status = TASK_STATUS_FINISHED
    task.save(commit=False)
    write_task_log(task.id, 'Done.', commit=False)
    db.session.commit()

    # Release memory
    if device != 'cpu':
        torch.cuda.empty_cache()


def reset_failed_tasks():
    # Set all running tasks as pending, with log message
    runnning_tasks = Task.get_running_tasks()
    for running_task in runnning_tasks:
        running_task.task_status = TASK_STATUS_PENDING
        running_task.save(commit=False)
        write_task_log(running_task.id, 'Reset to pending.', commit=False)
        db.session.commit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--app-config', type=str, required=True)
    parser.add_argument('--device', type=str, default=None, required=False)
    args = parser.parse_args()
    handle_default_device(args)

    app = create_app(name=__name__, config_name=args.app_config)
    with app.app_context():
        reset_failed_tasks()

        while True:
            pending_tasks = Task.get_pending_tasks()
            if len(pending_tasks) > 0:
                try:
                    run_task(pending_tasks[0], args.device)
                except Exception as e:
                    write_task_log(pending_tasks[0].id, f'Failed with message: {e}', commit=False)
                    pending_tasks[0].task_status = TASK_STATUS_ERROR
                    pending_tasks[0].save(commit=False)
                    db.session.commit()
            else:
                reset_failed_tasks()
                time.sleep(5)

