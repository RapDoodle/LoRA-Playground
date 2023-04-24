from core.db import db
from core.startup import create_app

from models.user import User
from models.user import ROLE_TYPE_ADMIN
from models.user import ROLE_TYPE_USER


def main():
    app = create_app(name=__name__, config_name='dev')
    with app.app_context():
        db.drop_all()
        db.create_all()

        populate_user()


def populate_user():
    User('admin', '12345678', 'System admin', ROLE_TYPE_ADMIN).save(commit=True)
    User('user1', '12345678', 'User 1', ROLE_TYPE_USER).save(commit=True)
    User('user2', '12345678', 'User 2', ROLE_TYPE_USER).save(commit=True)


def save(obj):
    db.session.add(obj)
    db.session.commit()
    db.session.refresh(obj)


if __name__ == '__main__':
    main()