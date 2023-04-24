# -*- coding: utf-8 -*-
"""This core module provides an instance of SQLAlchemy.

Example:
    In app.py, add:
        from utils.http_server.simple_server import server
        app.register_blueprint(server)

"""

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def init_db(app):
    """The function initializes `db` with the provided context.

    Args:
        app (flask.app.Flask): A Flask application

    """
    db.init_app(app)
