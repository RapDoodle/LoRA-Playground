# -*- coding: utf-8 -*-
import os
import uuid
import json
import base64
import hashlib
from flask import session
from flask import request
from flask_restful import reqparse
from flask_restful import Resource
from core.lang import get_str
from core.db import db
from core.exception import excpetion_handler
from core.permission import restricted_access
from core.exception import ErrorMessage
from utils.logger import log_message
from models.task import Task, TASK_TYPE_TRAIN, TASK_TYPE_TEST, TASK_STATUS_RUNNING, TASK_STATUS_FINISHED, TASK_STATUS_ERROR
from models.task_access_key import TaskAccessKey
from models.user import ROLE_TYPE_USER
from models.user import ROLE_TYPE_ADMIN
from models.task_log import TaskLog
from models.user_model import UserModel
from utils.converter import to_int
from utils.paths import get_lora_model_path


VERSION = 'v1'
ENDPOINT = f'@RESTFUL_PREFIX::/{VERSION}/queue/task'


class TaskQueue(Resource):
    # TODO: Check source IP based on `request.remote_addr`
    post_parser = reqparse.RequestParser()
    post_parser.add_argument('task_id',
                        type=int,
                        required=True,
                        help='Action cannot be empty.'
                        )
    post_parser.add_argument('access_key',
                        type=str,
                        required=True,
                        )
    post_parser.add_argument('actions',
                        type=list,
                        required=True,
                        location='json'
                        )

    @excpetion_handler
    def get(self):
        # request_args = TaskQueue.get_parser.parse_args()
        request_args = request.args
        if not 'action' in request_args:
            raise ErrorMessage('Must provide action')
        if request_args['action'] == 'fetch_task':
            task_types = request_args.get('task_types', None)
            if task_types is None:
                task_types = [1, 2]
            else:
                task_types = [to_int(x, 'task type') for x in task_types.split(',')]
            tasks = Task.get_pending_task(task_types)
            if len(tasks) == 1:
                task = tasks[0]
                access_key = uuid.uuid4().hex

                obj = None
                if os.path.exists(task.input_path):
                    with open(task.input_path, 'r') as f:
                        obj = f.read()
                task_access_key = TaskAccessKey.find_task_access_key_by_task_id(task.id)
                if task_access_key is None:
                    task_access_key = TaskAccessKey(task.id, access_key)
                else:
                    task_access_key.access_key = access_key
                task_access_key.save(commit=False)
                task.task_status = TASK_STATUS_RUNNING
                task.save(commit=False)
                task_log = TaskLog(task.id, "Scheduled successfully.")
                task_log.save(commit=False)

                model_md5 = None
                if task.task_type == TASK_TYPE_TEST:
                    model_opt = json.loads(task.task_parameters)['model_name']
                    model_path = get_lora_model_path(model_opt, task.user_id)
                    with open(model_path, 'rb') as f:
                        model_md5 = hashlib.md5(f.read()).hexdigest()

                db.session.commit()

                return {
                    'task': {
                        'task_id': task.id,
                        'task_type': task.task_type,
                        'task_parameters': task.task_parameters,
                        'obj': obj,
                        'model_md5': model_md5
                    },
                    'access_key': access_key
                }, 200
            else:
                return {
                    'task': None
                }, 200
        elif request_args['action'] == 'fetch_model':
            access_key = request_args.get('access_key', None)
            if access_key is None:
                raise ErrorMessage('Access key not provided')
            task_access_key = TaskAccessKey.find_task_access_key_by_access_key(access_key)
            if task_access_key is None:
                raise ErrorMessage('Invalid access key')
            task = Task.find_task_by_id(task_access_key.task_id)
            if task.task_type == TASK_TYPE_TRAIN:
                raise ErrorMessage('The task does not need user lora model.')
            model_opt = json.loads(task.task_parameters)['model_name']
            model_path = get_lora_model_path(model_opt, task.user_id)
            print(model_path)
            with open(model_path, 'rb') as f:
                model_data = f.read()
                md5_hash = hashlib.md5(model_data).hexdigest()
                model_base64_encoded = base64.b64encode(model_data).decode('utf-8')
            return {
                'model_data': model_base64_encoded,
                'md5': md5_hash
            }, 200


    @excpetion_handler
    def post(self):
        # TODO: Check source IP based on `request.remote_addr`
        request_args = TaskQueue.post_parser.parse_args()

        # Validate access key
        task_access_key = TaskAccessKey.find_task_access_key_by_task_id(request_args['task_id'])
        if task_access_key is None or request_args['access_key'] != task_access_key.access_key:
            return {'error': 'Invalid task ID or access token'}, 403
        task = Task.find_task_by_id(request_args['task_id'])
        
        for action in request_args['actions']:
            if action['action'] == 'log':
                task_log = TaskLog(request_args['task_id'], action['content'])
                task_log.save(commit=False)
            elif action['action'] == 'update_status':
                task.set_task_status(action['content'])
                task.save(commit=False)
            elif action['action'] == 'set_output_image':
                filename = f'{uuid.uuid4().hex}.{action["extension"]}'
                filepath = os.path.join('.', 'static', 'contents', 'outputs')
                if not os.path.exists(filepath):
                    os.makedirs(filepath)
                filepath = os.path.join(filepath, filename)
                with open(filepath, 'wb') as f:
                    f.write(base64.b64decode(action['content']))
                task.output_file = os.path.join('contents', 'outputs', filename)
                task.save(commit=False)
            elif action['action'] == 'finish':
                # Revoke the access key
                task.set_task_status(TASK_STATUS_FINISHED)
                task.save(commit=False)
                db.session.delete(task_access_key)
            elif action['action'] == 'save_model':
                task_parameters = json.loads(task.task_parameters)
                model_name = task_parameters["model_name"].split('/')[1]
                filename = f'{model_name}.ckpt'
                file_base_path = os.path.join('.', 'lora_models', str(task.user_id))
                if not os.path.exists(file_base_path):
                    os.makedirs(file_base_path)
                filepath = os.path.join(file_base_path, filename)
                with open(filepath, 'wb') as f:
                    f.write(base64.b64decode(action['content']))
                with open(filepath, 'rb') as f:
                    md5_hash = hashlib.md5(f.read()).hexdigest()
                    if md5_hash != action['md5']:
                        raise ErrorMessage('MD5 does not match')
                task.output_file = filepath
                task.save(commit=False)
                user_model = UserModel(model_name, task.user_id, md5_hash)
                user_model.save(commit=False)
            elif action['action'] == 'error':
                task.set_task_status(TASK_STATUS_ERROR)
                task.save(commit=False)
                task_log = TaskLog(request_args['task_id'], f"Error: {action['content']}")
                task_log.save(commit=False)
            else:
                raise ErrorMessage('Unknown action.')
        db.session.commit()
        return {}, 200