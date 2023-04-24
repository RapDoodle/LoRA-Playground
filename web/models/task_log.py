# -*- coding: utf-8 -*-
from datetime import timedelta

import models
from core.db import db
from sqlalchemy.sql import func
from core.exception import ErrorMessage
from models.user import User
from utils.converter import to_int


class TaskLog(models.saveable_model.SaveableModel):
    __tablename__ = 'task_log'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    task_id = db.Column(db.Integer, db.ForeignKey('task.id'))
    time_created = db.Column(db.DateTime(timezone=True), server_default=func.now())
    message = db.Column(db.String(1024))

    def __init__(self, task_id, message):
        super().__init__()

        # Clean and validate the data
        task_id = to_int(str(task_id).strip())
        message = str(message).strip()

        # Store the data in the object
        self.task_id = task_id
        self.message = message

    def get_time_created_local_time(self):
        # TODO: Add support for all timezones. Currently, only Beijing/Hong Kong time is supported
        return self.time_created + timedelta(hours=8)

    def get_message(self):
        return self.message

    @classmethod
    def find_task_log_by_task_id(cls, task_id):
        return cls.query.filter_by(task_id=task_id).all()