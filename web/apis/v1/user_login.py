# -*- coding: utf-8 -*-
from flask import jsonify
from flask import session
from flask_restful import Resource
from flask_restful import reqparse
from flask_jwt_extended import create_access_token
from flask_jwt_extended import set_access_cookies

from utils.hash import verify_hash
from core.exception import excpetion_handler

from models.user import User

VERSION = 'v1'
ENDPOINT = f'@RESTFUL_PREFIX::/{VERSION}/login'

class UserLogin(Resource):
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

    @excpetion_handler
    def post(self):
        data = UserLogin.parser.parse_args()
        user = User.find_user_by_username(data['username'])
        if user is None:
            return {'message': 'User does not exists.'}, 400
        if not user.verify_password(data['password']):
            return {'message': 'Incorrect password.'}, 400
        session['user_id'] = user.get_id()
        session['user_type'] = user.get_role_type()
        session['full_name'] = user.get_full_name()
        response = jsonify({'message': 'Logged in successfully.'})
        # set_access_cookies(response, access_token)
        return response
