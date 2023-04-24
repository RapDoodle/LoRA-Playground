# -*- coding: utf-8 -*-
store = {}

def get(key):
    if key in store:
        return store[key]
    return None

def set(key, value):
    store[key] = value

