# -*- coding: utf-8 -*-
import os
import hashlib
from datetime import timedelta

from sqlalchemy.sql import func

import models
from core.db import db


class PublicModel(models.saveable_model.SaveableModel):
    __tablename__ = 'public_model'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    model_name = db.Column(db.String(128))
    md5 = db.Column(db.String(32))

    def __init__(self, model_name, md5):
        super().__init__()

        # Store the data in the object
        self.model_name = str(model_name).strip()
        self.md5 = md5

    @classmethod
    def init_public_models(cls):
        # Get all models in the database
        models_set = set([model.model_name for model in cls.get_public_models()])
        public_models_path = os.path.join('.', 'lora_models', 'public')
        for filename in os.listdir(public_models_path):
            name, extension = os.path.splitext(filename)
            model_path = os.path.join(public_models_path, filename)
            if name in models_set:
                models_set.remove(name)
                continue
            with open(model_path, 'rb') as f:
                file_data = f.read()
                md5_hash = hashlib.md5(file_data).hexdigest()
            new_public_model = PublicModel(name, md5_hash)
            new_public_model.save(commit=False)
        removed_models_list = list(models_set)
        for model_name in removed_models_list:
            db.session.delete(cls.find_public_model_by_model_name(model_name))
        db.session.commit()
    
    @classmethod
    def get_public_models(cls):
        return cls.query.all() 
    
    @classmethod
    def find_public_model_by_model_name(cls, model_name):
        return cls.query.filter_by(model_name=model_name).first()
        
