# -*- coding: utf-8 -*-
"""This module is related to exceptions and their handling."""
import sys
import traceback

from werkzeug.exceptions import BadRequest
from flask import current_app
from flask_restful import abort
from flask_language import current_language

from core.lang import get_str
from functools import wraps


class ErrorMessage(Exception):
    """An error message expected to be shown to the front-end
    
    Note:
        If the message needs to be translated. Consider using
        `ErrorMessagePromise`
    
    """

    def __init__(self, message):
        """The constructor for ErrorMessage

        Args:
            message (str): The message to be shown to the
                front-end user.

        """
        self.message = message

    def get(self):
        """The getter of ErrorMessage"""
        return self.message

    def __str__(self):
        """Strinify the object."""
        return self.message


class ErrorMessagePromise(Exception):
    """A promise that a key value will be translated.
    
    Note:
        The promise only guarantees an attempt to get the
        string for the language. It does not guarantee
        that a string will be returned.
    
    """

    def __init__(self, key: str):
        """The constructor for ErrorMessagePromise

        Args:
            key (str): The key used to retrieve the string

        """
        self.key = key

    def get(self):
        """Get the string for the given key.

        Returns:
            A string of the translated message
        
        """
        return self.__str__()

    def __str__(self):
        """Strinify the object."""
        return get_str(self.key)


def excpetion_handler(fn):
    """The decorator handles exceptions within the framework.

    A message will be returned to the user when a ValidationError was raised.
    Otherwise, a message indicating an internal error will be given. And the
    error will be logged by the logger.

    If the function returns None, a dict {'status': 200} will be returned.
    Otherwise, the returned value of the wrapped function will be returned.
    """
    def handler(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (ErrorMessage, ErrorMessagePromise) as e:
            abort(400, error=str(e))
        except BadRequest as e:
            abort(400, error=str(e.description))
        except Exception as e:
            current_app.logger.critical(str(e))
            traceback.print_exc(file=sys.stdout)
            return {'error': get_str('INTERNAL_ERROR')}, 500
    return handler
