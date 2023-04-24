# -*- coding: utf-8 -*-
"""This core module handles functions related to multi-language.

The fundamental of this module depends on Flask-Language, a 
Flask extension that makes handling a client-side language 
cookie easier.

Note:
    For more information, please visit:
    https://flask-language.readthedocs.io/en/latest/

"""

from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as ET

from flask import current_app
from flask_language import Language
from flask_language import current_language

from utils.constants import LANGUAGE_RESOURCE_PATH

lang = Language()
allowed_langs = []
lang_dict = {}

def init_lang(app):
    """The function initializes `lang` with the provided context.

    Args:
        app (flask.app.Flask): A Flask application.

    """
    lang.init_app(app)

    # Load language packs from LANGUAGE_RESOURCE_PATH
    langs_list = [f for f in listdir(LANGUAGE_RESOURCE_PATH) \
        if isfile(join(LANGUAGE_RESOURCE_PATH, f))]
    for file_name in langs_list:
        load_lang(file_name)

    # Check if the field DEFAULT_LANGUAGE is not set
    default_lang = app.config.get('DEFAULT_LANGUAGE', None)
    if default_lang is None:
        raise Exception('Default language not specified in the configuration')


def load_lang(file_name: str):
    """Loads a language resource file from the language path.

    Note:
        The file should be of type `.xml`. Resource files
        of other types will not be accepted.

    Args:
        file_name (str): The file name of the language resource 
            file with the suffix. For example, to load 
            `en-US.xml`, the parameter should be `'en-US.xml'`. 

    """
    if not file_name.endswith('.xml'):
        return
    tree = ET.parse(join(LANGUAGE_RESOURCE_PATH, file_name))
    root = tree.getroot()
    current_dict = {}
    lang_name = file_name.split('.xml')[0]
    for s in root.iter('string'):
        current_dict[s.attrib['name']] = s.text
    allowed_langs.append(lang_name)
    lang_dict[lang_name] = current_dict


def get_str(key: str, language=None, **kwargs):
    """Get the string for a given or default language.

        Args:
            key (str): The key of the string (the attribute
                "name" of the "string" tag). For example,
            language (str): Could be a string or `None`.
                When `None` is specified, the language set
                in the user cookie will be used. Otherwise, 
                specify the name of the langugae pack. For 
                example, `language = 'zh-HK'`.
                
    """
    # When lang is None, use the default language
    if language is None:
        language = current_language
        # language = current_app.config['DEFAULT_LANGUAGE']

    # Not default language
    if lang_dict.get(language, None) is not None and \
        language is not current_app.config['DEFAULT_LANGUAGE']:
        string = lang_dict[language].get(key, None)
        if string is not None:
            return resolve_template(string, **kwargs)
        current_app.logger.warning(
            f'Key "{key}" is not implemented in {language}.xml.')
    
    # Use default language
    language = current_app.config['DEFAULT_LANGUAGE']
    if lang_dict.get(language, None) is not None:
        string = lang_dict[language].get(key, None)
        if string is not None:
            return resolve_template(string, **kwargs)
        else:
            current_app.logger.warning(
                f'Key "{key}" is not implemented in {language}.xml.')
            return ''
    else:
        current_app.logger.critical(
            f'Default language {language} is not found.')
    
    return ''


def resolve_template(template: str, **kwargs):
    return template % kwargs


@lang.allowed_languages
def get_allowed_languages():
    """Get the allowed languages.
    
    Note:
        The function will return an empty list if the language
        module has not been initilized (`init_lang`) not called.
    
    """
    return allowed_langs


@lang.default_language
def get_default_language():
    """Gets the default languages.
    
    Note:
        After initializing the language module, this function is
        guaranteed to return the configured default language.
    
    """
    return current_app.config['DEFAULT_LANGUAGE']