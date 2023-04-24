# -*- coding: utf-8 -*-
"""This module provides functions related to hashing."""

import bcrypt


def hash_data(data: str):
    """The function hashes the provided data.

    Args:
        data (str): A string that expects to be hashed.

    Returns:
        bytes: The hashed data.

    """
    return bcrypt.hashpw(data.encode('utf-8'), bcrypt.gensalt())


def verify_hash(data: str, hashed_data: bytes):
    """The function verifies the data against the provided hash data.

    Note:
        The data must be stripped before passing into the function.

    Args:
        data (str): A string that expects to be hashed.

    Returns:
        bool: True if it is correct, False otherwise.

    """
    return bcrypt.checkpw(data.encode('utf-8'), hashed_data)
