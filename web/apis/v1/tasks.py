# -*- coding: utf-8 -*-
from flask import session
from flask import request
from flask_restful import reqparse
from flask_restful import Resource
from core.lang import get_str
from core.db import db
from core.exception import excpetion_handler
from core.permission import restricted_access
from core.exception import ErrorMessage
from utils.logger import log_message
from models.task import Task
from models.user import ROLE_TYPE_USER
from models.user import ROLE_TYPE_ADMIN
from models.task_log import TaskLog


VERSION = 'v1'
ENDPOINT = f'@RESTFUL_PREFIX::/{VERSION}/tasks'


class Tasks(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument('task_type',
                        type=int,
                        required=True,
                        help='Task type cannot be empty.'
                        )
    parser.add_argument('task_parameters',
                        type=dict,
                        required=False,
                        default='',
                        )
    parser.add_argument('image',
                        type=str,
                        required=False
                        )
    parser.add_argument('images',
                        type=list,
                        required=False,
                        location='json'
                        )
    parser.add_argument('task_title',
                        type=str,
                        default='Untitled task',
                        required=False
                        )

    @restricted_access(allowed=[ROLE_TYPE_USER, ROLE_TYPE_ADMIN], return_json=True)
    @excpetion_handler
    def post(self):
        data = Tasks.parser.parse_args()

        # Create the task
        task = Task(user_id=session['user_id'], task_type=data['task_type'], task_parameters=data['task_parameters'], task_title=data['task_title'])
        task.save(commit=False)
        
        if data['image'] is not None:
            task.set_input_image(data['image'])
            task.save(commit=False)
        
        if data['images'] is not None and len(data['images']) > 0:
            task.set_input_images(data['images'])
            task.save(commit=False)

        # Create log
        task_log = TaskLog(task.id, message="Task created.")
        task_log.save(commit=False)

        # Final commit
        db.session.commit()
        return {'message': f'Successfully created.'}, 201

    @restricted_access(allowed=[ROLE_TYPE_USER, ROLE_TYPE_ADMIN], return_json=True)
    @excpetion_handler
    def patch(self):
        pass
        # content = request.get_json()
        # course = Course.find_course_by_id(content['course_id'])
        # if course is None:
        #     raise ErrorMessage(get_str('NOT_FOUND_OBJECT', object_name='course'))
        # course.modify_course(content)
        # log_message(session['full_name'] + ' modified the course ' + course.course_name + ' successfully.')
        # return {'message': get_str('COURSE_UPDATED', course_name=course.course_name)}, 200
