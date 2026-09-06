from flask import Flask

import db

app = Flask("WhatCanICook")


def create_app(views):
    app.register_blueprint(views.bp)
    with app.app_context():
        db.init_app(app)
    return app
