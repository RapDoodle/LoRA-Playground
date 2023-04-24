# -*- coding: utf-8 -*-

from flask import Blueprint
from flask import render_template
from flask import request
from flask import redirect
from flask import url_for
from flask import flash
from flask import get_flashed_messages
from flask import session
from flask_language import current_language

from core.engine import render_context
from core.lang import lang
from core.lang import get_str

from models.user import User

blueprint = Blueprint('login', __name__, template_folder='templates')

@blueprint.route('/login', methods=['GET', 'POST'])
@render_context('login.html')
def login():
    if request.args.get('lang') is not None:
        lang.change_language(request.args.get('lang'))

    if request.method == 'POST':
        user = User.find_user_by_username(request.values.get('username'))
        if user and user.verify_password(request.values.get('password')):
            session['user_id'] = user.get_id()
            session['user_type'] = user.get_role_type()
            session['full_name'] = user.get_full_name()
            return redirect(url_for('dashboard.dashboard'))
        else:
            flash(get_str('INVALID_CREDENTIALS'))


@blueprint.route('/login/', methods=['GET', 'POST'])
def login_redirect():
    return redirect(url_for('login.login'))


@blueprint.route('/logout', methods=['GET'])
def logout():
    session.clear()
    return redirect(url_for('login.login'))