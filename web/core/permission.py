from functools import wraps

from flask import session
from flask import flash
from flask import redirect
from flask import url_for

from core.lang import get_str

def restricted_access(allowed: list, return_json=False):
    def verify_access(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            # No login credential found
            if not session.get('user_id'):
                if return_json:
                    return {'error': get_str('LOGIN_REQUIRED')}
                else:
                    flash(get_str('LOGIN_REQUIRED'))
                    return redirect(url_for('login.login'))

            # Check for permission
            if session.get('user_type') not in allowed:
                if return_json:
                    return {'error': get_str('PERMISSION_DENIED')}
                else:
                    flash(get_str('PERMISSION_DENIED'))
                    return redirect(url_for('dashboard.dashboard'))

            return fn(*args, **kwargs)
        return wrapper
    return verify_access