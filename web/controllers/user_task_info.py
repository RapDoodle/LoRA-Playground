# -*- coding: utf-8 -*-
from flask import Blueprint
from flask import session
from flask import request
from flask import redirect
from flask import url_for
from flask import flash
from core.lang import get_str
from core.engine import render_context
from core.permission import restricted_access
from models.user import ROLE_TYPE_ADMIN
from models.user import ROLE_TYPE_USER

blueprint = Blueprint('user_task_info', __name__, template_folder='templates')

from models.task import Task

@blueprint.route('/task/<int:task_id>', methods=['GET'])
@restricted_access(allowed=[ROLE_TYPE_ADMIN, ROLE_TYPE_USER])
@render_context('user_task_info.html', on_error_redirect_to='user_tasks.tasks')
def info(task_id):
    task = Task.find_task_by_id(task_id)
    if task.user_id != session['user_id']:
        flash('You do not have permission to this page.')
        return redirect(url_for('user_tasks.tasks'))
    return {
        'task': task
    }

