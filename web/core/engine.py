# -*- coding: utf-8 -*-
import sys
import traceback
from functools import wraps

from flask import flash
from flask import render_template
from flask import current_app
from flask import redirect
from flask import url_for

from core.lang import get_str
from core.exception import ErrorMessage
from core.exception import ErrorMessagePromise
from core.db import db

from models.user import ROLE_TYPE_USER
from models.user import ROLE_TYPE_ADMIN

default_context = {
    'get_str': get_str,
    'ROLE_TYPE_USER': ROLE_TYPE_USER,
    'ROLE_TYPE_ADMIN': ROLE_TYPE_ADMIN,
    'enumerate': enumerate,
    'len': len,
    'str': str
}

def render(*args, **kwargs):
    return render_template(*args, **kwargs, **default_context), kwargs.get('status_code', 200)


def render_context(template = '', commit_on_success=True, rollback_on_exception=True, on_error_redirect_to=''):
    def context(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            data = {}
            status_code = 200
            try:
                returned = fn(*args, **kwargs)
                if commit_on_success:
                    db.session.commit()
                if returned is not None:
                    if type(returned) == dict:
                        data = returned
                    else:
                        return returned
            except (ErrorMessage, ErrorMessagePromise) as e:
                flash(str(e))
                status_code = 400
                if rollback_on_exception:
                    db.session.rollback()
                if len(on_error_redirect_to) > 0:
                    return redirect(url_for(on_error_redirect_to))
            except Exception as e:
                current_app.logger.critical(str(e))
                traceback.print_exc(file=sys.stdout)
                flash(get_str('INTERNAL_ERROR'))
                status_code = 500
                if rollback_on_exception:
                    db.session.rollback()
                if len(on_error_redirect_to) > 0:
                    return redirect(url_for(on_error_redirect_to))
            return render(template, data=data, status_code=status_code)
        return wrapper
    return context