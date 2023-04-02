from flask import Blueprint

# create bluepring
user_blu = Blueprint("user", __name__)

# relate view function and main program
from controller.modules.user.views import *
