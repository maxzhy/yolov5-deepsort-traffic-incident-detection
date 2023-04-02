from flask import session, render_template, redirect, url_for, Response
from controller.modules.home import home_blu

from importlib import import_module
import os
from flask import Flask, render_template, Response

import cv2
camera = cv2.VideoCapture('./copy/gonglu3.mp4')
def generate_frames():
    while camera.isOpened():
        success, frame1 = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame1)
            frame1 = buffer.tobytes()
            print("enter3")
            yield (b'--frame1\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame1 + b'\r\n')  # concat frame one by one and show result


'''import camera driver'''
if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from controller.utils.yolo_deepsort import Camera

# Main page
@home_blu.route('/')
def index():
    username = session.get("username")
    if not username:
        return redirect(url_for("user.login"))
    return render_template("index.html")

def video_stream(camera):
    # 产生video stream
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# video stream
@home_blu.route('/video_viewer')
def video_viewer():
    username = session.get("username")
    if not username:
        return redirect(url_for("user.login"))
    '''
    return Response(video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    '''
    return Response(video_stream(Camera()), mimetype='multipart/x-mixed-replace; boundary=frame')


