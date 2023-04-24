# -*- coding: utf-8 -*-
from flask_restful import Resource
from flask_restful import reqparse

from core.permission import restricted_access
from core.exception import excpetion_handler
from models.user import User
from models.user import ROLE_TYPE_USER

VERSION = 'v1'
ENDPOINT = f'@RESTFUL_PREFIX::/{VERSION}/register'

class UserRegister(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument('username',
                        type=str,
                        required=True,
                        help='The username of the user cannot be empty.'
                        )
    parser.add_argument('password',
                        type=str,
                        required=True,
                        help='The password field cannot be empty.'
                        )
    parser.add_argument('full_name',
                        type=str,
                        required=True,
                        help='The full name field cannot be empty.'
                        )

    @restricted_access(allowed=[ROLE_TYPE_USER], return_json=True)
    @excpetion_handler
    def post(self):
        data = UserRegister.parser.parse_args()
        if User.find_user_by_username(data['username']):
            return {'message': 'A user with that username already exists'}, 400

        user = User(username=data['username'], password=data['password'], full_name=data['full_name'], role_type=ROLE_TYPE_USER)
        user.save(commit=True)

        return {'message': 'User created successfully.'}, 201

