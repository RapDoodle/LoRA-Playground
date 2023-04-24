# -*- coding: utf-8 -*-
from datetime import timedelta

from sqlalchemy.sql import func

import models
from core.db import db


class UserModel(models.saveable_model.SaveableModel):
    __tablename__ = 'user_model'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), index=True)
    model_name = db.Column(db.String(128))
    time_created = db.Column(db.DateTime(timezone=True), server_default=func.now())
    md5 = db.Column(db.String(32))

    def __init__(self, model_name, user_id, md5):
        super().__init__()

        # Store the data in the object
        self.user_id = user_id
        self.model_name = str(model_name).strip()
        self.md5 = md5
    
    def get_time_created_local_time(self):
        # TODO: Add support for all timezones. Currently, only Beijing/Hong Kong time is supported
        return self.time_created + timedelta(hours=8)
    
    @classmethod
    def get_user_models(cls, user_id, use_pagination=False, page_size=20, page=1):
        model_query = cls.query.filter_by(user_id=user_id).order_by(cls.time_created.desc())
        if use_pagination:
            count = model_query.count()
            tasks = model_query.limit(page_size).offset((page-1)*page_size).all()
            return tasks, count
        else:
            return model_query.all() 
        
    @classmethod
    def find_user_model_by_id(cls, model_id):
        return cls.query.filter_by(id=model_id).first()
        
