from core.lang import get_str
from core.exception import ErrorMessage

def to_int(d, name = ''):
    try:
        return int(d)
    except ValueError:
        raise ErrorMessage(get_str('NOT_INT', var_name = name))


def to_float(d, name):
    try:
        return int(d)
    except ValueError:
        raise ErrorMessage(get_str('NOT_DECIMAL', var_name = name))
