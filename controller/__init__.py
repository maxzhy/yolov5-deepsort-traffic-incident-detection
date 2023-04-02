import logging
from logging.handlers import RotatingFileHandler
from flask import Flask
from config import config_dict

# factory function
def create_app(config_type):
    # Retrieve the configuration subclass according to the type
    config_class = config_dict[config_type]
    app = Flask(__name__)
    app.config.from_object(config_class)

    # register blueprint
    from controller.modules.home import home_blu
    app.register_blueprint(home_blu)
    from controller.modules.user import user_blu
    app.register_blueprint(user_blu)

    return app
