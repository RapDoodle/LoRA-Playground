# -*- coding: utf-8 -*-
import models
from core.db import db
from utils.hash import hash_data
from utils.hash import verify_hash
from utils.converter import to_int


ROLE_TYPE_ADMIN = 0
ROLE_TYPE_USER = 1
VALID_ROLE_TYPES = [ROLE_TYPE_ADMIN, ROLE_TYPE_USER]

class User(models.saveable_model.SaveableModel):
    """The model related to users.

    Attributes:
        id (Integer): user's id
        username (String): user's username
        password (LargeBinary): user's hashed password

    """
    __tablename__ = 'user'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(32))
    _password = db.Column(db.LargeBinary(60))
    full_name = db.Column(db.String(64))
    role_type = db.Column(db.String(32))

    def __init__(self, username, password, full_name, role_type):
        # Clean the data
        username = str(username).strip()
        password = str(password).strip()
        full_name = str(full_name).strip()
        role_type = to_int(str(role_type).strip())

        # Hash the password
        password_hash = hash_data(password)

        # Store the data in the object
        self.username = username
        self._password = password_hash
        self.full_name = full_name
        self.role_type = role_type

    def get_id(self):
        return self.id

    def get_username(self):
        return self.username

    def get_full_name(self):
        return self.full_name

    def get_role_type(self):
        return int(self.role_type)

    def verify_password(self, password):
        return verify_hash(password, self._password)

    @classmethod
    def find_user_by_username(cls, username):
        return cls.query.filter_by(username=username).first()