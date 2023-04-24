# -*- coding: utf-8 -*-
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

blueprint = Blueprint('user_tasks', __name__, template_folder='templates')


@blueprint.route('/tasks', methods=['GET'])
@restricted_access(allowed=[ROLE_TYPE_ADMIN, ROLE_TYPE_USER])
@render_context('user_tasks.html')
def tasks():
    page = request.args.get('page', 1)
    page_size = 10
    tasks, count = Task.get_user_tasks(session.get('user_id'), use_pagination=True, page_size=page_size, page=to_int(page))
    return {
        'tasks': tasks,
        'pages': get_page_numbers(length=count, current=page, limit=page_size, show=8),
        'current_page': str(page),
        'public_models': PublicModel.get_public_models(),
        'user_models': UserModel.get_user_models(session.get('user_id'), False)
    }

