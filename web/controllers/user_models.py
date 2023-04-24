# -*- coding: utf-8 -*-
from flask import Blueprint
from flask import request
from flask import redirect
from flask import url_for
from flask import flash
from flask import session
from core.engine import render_context
from core.lang import get_str
from core.db import db
from core.exception import ErrorMessage
from core.permission import restricted_access
from models.user import ROLE_TYPE_ADMIN
from models.user import ROLE_TYPE_USER
from models.user_model import UserModel
from utils.converter import to_int
from utils.pagination import get_page_numbers

blueprint = Blueprint('user_models', __name__, template_folder='templates')


@blueprint.route('/models', methods=['GET'])
@restricted_access(allowed=[ROLE_TYPE_ADMIN, ROLE_TYPE_USER])
@render_context('user_models.html')
def models():
    page = request.args.get('page', 1)
    page_size = 10
    models, count = UserModel.get_user_models(session.get('user_id'), use_pagination=True, page_size=page_size, page=to_int(page))
    return {
        'models': models,
        'pages': get_page_numbers(length=count, current=page, limit=page_size, show=8),
        'current_page': str(page),
    }

@blueprint.route('/models/delete/<model_id>', methods=['GET'])
@restricted_access(allowed=[ROLE_TYPE_ADMIN, ROLE_TYPE_USER])
@render_context('user_models.html')
def delete_model(model_id):
    user_model = UserModel.find_user_model_by_id(model_id)
    if user_model is None:
        flash('The model does not exist.')
        return redirect(url_for('user_models.models'))
    if user_model.user_id != session['user_id']:
        flash('You do not have access ')
        return redirect(url_for('user_models.models'))
    # Delete the mdoel
    db.session.delete(user_model)
    db.session.commit()
    flash('The model was deleted successfully.')
    return redirect(url_for('user_models.models'))
