# from flask_script import Manager
from controller import create_app

# create APP
app = create_app('dev')

if __name__ == '__main__':
    app.run(threaded=True, host="0.0.0.0")

