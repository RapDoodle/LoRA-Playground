# -*- coding: utf-8 -*-
from flask import Blueprint
from flask import redirect
from flask import url_for
from core.permission import restricted_access
from models.user import ROLE_TYPE_ADMIN
from models.user import ROLE_TYPE_USER

blueprint = Blueprint('dashboard', __name__, template_folder='templates')


@blueprint.route('/', methods=['GET', 'POST'])
@restricted_access(allowed=[ROLE_TYPE_ADMIN, ROLE_TYPE_USER])
def dashboard():
    return redirect(url_for('user_tasks.tasks'))

