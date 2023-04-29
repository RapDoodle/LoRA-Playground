# -*- coding: utf-8 -*-
import os
import base64
from io import BytesIO
from flask import send_file
from flask import Blueprint
from flask import request
from flask import redirect
from flask import url_for
from flask import flash
from flask import session
from core.engine import render_context
from core.lang import get_str
from core.exception import ErrorMessage
from core.permission import restricted_access
from models.user import ROLE_TYPE_ADMIN
from models.user import ROLE_TYPE_USER
from models.task import Task
from models.public_model import PublicModel
from models.user_model import UserModel
from utils.converter import to_int
from utils.pagination import get_page_numbers

blueprint = Blueprint('user_images', __name__, template_folder='templates')


@blueprint.route('/images/in/<task_id>/<obj_id>', methods=['GET'])
@restricted_access(allowed=[ROLE_TYPE_ADMIN, ROLE_TYPE_USER])
def get_input_images(task_id, obj_id):
    task = Task.find_task_by_id(task_id)
    if task.user_id != session['user_id']:
        flash('You do not have permission to this page.')
        return redirect(url_for('user_tasks.tasks'))
    obj_path = task.input_path
    obj_idx = int(obj_id) - 1
    with open(obj_path, 'r') as f:
        objs = f.read().split('\n')
        if obj_idx >= 0 and obj_idx < len(objs):
            meta_info, base64_img = objs[obj_idx].split(',')
            img_io = BytesIO()
            img_io.write(base64.b64decode(base64_img))
            img_io.seek(0)
            return send_file(img_io, mimetype='image/jpeg')
    return {}
        

