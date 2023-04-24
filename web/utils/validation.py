# -*- coding: utf-8 -*-
"""The module provides functions related to input validation."""

import re


def verify_regex(regex: str, d: str) -> bool:
    """Verifies data against a regular expression.

    Note:
        The provided regular expression should be a
        uncompiled regular expression.

    Args:
        regex (str): The regular expression to be compiled.
        d (str): A string object containing the data to be
            verified.

    Returns:
        bool: `True` if matches. Otherwise, `False`.

    """
    return False if re.match(regex, d) is None else True


def is_money(d: str) -> bool:
    """Checks if the provided data is in currency format.

    Args:
        d (str): The string to be verified.

    Returns:
        bool: `True` if matches. Otherwise, `False`.

    """
    return verify_regex('^[0-9]+(\.[0-9]{1,2})?$', d)


def is_rating(d: str):
    """Checks if the provided data is a valid rating.

    The function returns `True` if the provided data
    is between 1 and 5.

    Args:
        d (str): The string to be verified.

    Returns:
        bool: `True` if matches. Otherwise, `False`.

    """
    return verify_regex('^[1-5]$', d)


def is_valid_length(d: str, min: int, max: int) -> bool:
    """checks whether d's length is in the range [min, max].

    Args:
        d (str): The string to be verified.
        min (int): The minimum expected length.
        max (int): The maximum expected length.

    Returns:
        bool: `True` if d is within the range of [min, max]. 
            Otherwise, `False`.

    """
    d_len = len(str(d))
    return True if (d_len >= min and d_len <= max) else False


def is_valid_username(
        d: str, min_len: int = 6, max_len: int = 24, 
        allow_lower: bool = True, allow_upper: bool = True,
        allow_digit: bool = True, allow_underscore: bool = True,
        allow_dash: bool = True) -> bool:
    """Checks if the provided data is a valid username.

    Note:
        You may consider changing the default parameters 
        accordingly to your policy.

    Args:
        d (str): The string to be verified.
        min_len (int): The minimum length allowed.
        max_len (int): The maximum length allowed.
        allow_lower (bool): Allow for lower case letter?
        allow_upper (bool): Allow for upper case letter?
        allow_digit (bool): Allow for digits?
        allow_underscore (bool): Allow for underscore (_)?
        allow_dash (bool): Allow for dash (_)?

    Returns:
        bool: `True` if it is valid. Otherwise, `False`.

    """
    return verify_regex(f"^[{'a-z' if allow_lower else ''}" + 
                        f"{'A-Z' if allow_upper else ''}" + 
                        f"{'0-9' if allow_digit else ''}" +
                        f"{'_' if allow_underscore else ''}" + 
                        f"{'-' if allow_dash else ''}" + 
                        f"]{{{min_len},{max_len}}}$", d)


def is_valid_password(
        d: str, min_len: int = 8, max_len: int = 24, 
        require_lower: bool = True, 
        require_upper: bool = True, 
        require_digit: bool = True, 
        require_special_char: bool = False) -> bool:
    """Checks if the provided data is a valid password.

    Note:
        You may consider changing the default parameters 
        accordingly to your policy.

    Args:
        d (str): The string to be verified.
        min_len (int): The minimum length required.
        max_len (int): The maximum length required.
        require_lower (bool): Require at least one lower 
            case letter?
        require_upper (bool): Require at least one upper 
            case letter?
        require_digit (bool): Require at least one digits?
        require_special_char (bool): Require special at 
            least one characters (#?!@$ %^&*-)?

    Returns:
        bool: `True` if it is valid. Otherwise, `False`.

    """
    assert min_len < max_len
    return verify_regex(f"^{'(?=.*[a-z])' if require_lower else '' }" + 
                        f"{'(?=.*[A-Z])' if require_upper else ''}" +
                        f"{'(?=.*[0-9])' if require_digit else ''}" +
                        f"{'(?=.*[#?!@$ %^&*-])' if require_special_char else ''}" +
                        f".{{{min_len},{max_len}}}$", d)

def is_valid_filename(filename):
    pattern = r'^[\w\-. ]+$'
    return re.match(pattern, filename) is not None

