from flask import Blueprint

# create blueprint
home_blu = Blueprint("home", __name__)

# relate view function with program
from controller.modules.home.views import *
