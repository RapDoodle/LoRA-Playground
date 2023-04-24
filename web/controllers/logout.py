# -*- coding: utf-8 -*-
from flask import Blueprint
from flask import session
from flask import redirect
from flask import url_for

blueprint = Blueprint('logout', __name__, template_folder='templates')


@blueprint.route('/logout', methods=['GET'])
def logout():
    session.clear()
    return redirect(url_for('login.login'))

