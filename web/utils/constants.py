# -*- coding: utf-8 -*-
"""This module holds constants used in the project."""


CONFIG_PATH = './configurations'
"""str: Configuration path.

The folder in which configurations are stored.

"""


LANGUAGE_RESOURCE_PATH = './langs'
"""str: Language resource path. 

The folder in which all language resource packs are
stored.

"""


SERVE_FOLDER = 'web'
"""str: Name of the folder to be serverd.

Note:
    This parameter is for the simple HTTP server used
    for testing during development. Please do NOT use
    the HTTP server in production mode for performance
    benefits and safety concerns.

The folder in which the front-end of the server is
stored.

"""