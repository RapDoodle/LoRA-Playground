# -*- coding: utf-8 -*-
import os
import re
import json
import uuid
import base64
import hashlib
from datetime import timedelta

from sqlalchemy.sql import func

import models
from core.db import db
from core.exception import ErrorMessage
from models.user import User
from models.task_log import TaskLog
from utils.converter import to_int


class TaskAccessKey(models.saveable_model.SaveableModel):
    __tablename__ = 'task_access_key'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    task_id = db.Column(db.Integer, db.ForeignKey('task.id'), unique=True)
    access_key = db.Column(db.String(32))
    time_created = db.Column(db.DateTime(timezone=True), server_default=func.now())

    def __init__(self, task_id, access_key):
        super().__init__()

        # Store the data in the object
        self.task_id = task_id
        self.access_key = access_key

    @classmethod
    def find_task_access_key_by_task_id(cls, task_id):
        return cls.query.filter_by(task_id=task_id).first()

    @classmethod
    def find_task_access_key_by_access_key(cls, access_key):
        return cls.query.filter_by(access_key=access_key).first()

