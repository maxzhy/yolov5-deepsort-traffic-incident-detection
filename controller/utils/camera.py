import cv2
import threading
from datetime import datetime
import time
from queue import Queue

try:
    from greenlet import getcurrent as get_ident
except ImportError:
    try:
        from thread import get_ident
    except ImportError:
        from _thread import get_ident

class CameraEvent(object):
    # An Event-like class that signals all active clients when a new frame is available.
    
    def __init__(self):
        self.events = {}

    def wait(self):
        # Invoked from each client's thread to wait for the next frame.
        ident = get_ident()
        if ident not in self.events:
            # this is a new client
            # add an entry for it in the self.events dict
            # each entry has two elements, a threading.Event() and a timestamp
            self.events[ident] = [threading.Event(), time.time()]
        return self.events[ident][0].wait()

    def set(self):
        # Invoked by the camera thread when a new frame is available.
        now = time.time()
        remove = None
        for ident, event in self.events.items():
            if not event[0].isSet():
                # if this client's event is not set, then set it
                # also update the last set timestamp to now
                event[0].set()
                event[1] = now
            else:
                # if the client's event is already set, it means the client
                # did not process a previous frame
                # if the event stays set for more than 5 seconds, then assume
                # the client is gone and remove it
                if now - event[1] > 5:
                    remove = ident
        if remove:
            del self.events[remove]

    def clear(self):
        # Invoked from each client's thread after a frame was processed.
        self.events[get_ident()][0].clear()

class VideoCamera(object):

    thread = None  # background thread that reads frames from camera
    frame = None  # current frame is stored here by background thread
    last_access = 0  # time of last client access to the camera
    event = CameraEvent()

    def __init__(self):

        if VideoCamera.thread is None:
            VideoCamera.last_access = time.time()

            # start background frame thread
            VideoCamera.thread = threading.Thread(target=self._thread)
            VideoCamera.thread.start()

            # wait until frames are available
            while self.get_frame() is None:
                time.sleep(0)

    def get_frame(self):

        VideoCamera.last_access = time.time()

        # wait for a signal from the camera thread
        VideoCamera.event.wait()
        VideoCamera.event.clear()

        return VideoCamera.frame

    #@classmethod
    def _thread(self):
        #Camera background thread
        print('Starting camera thread.')
        frames_iterator = self.frames()
        #print('frames_iterator... {}, type: {}'.format(frames_iterator, type(frames_iterator)))
        for frame in frames_iterator:
            VideoCamera.frame = frame
            VideoCamera.event.set()  # send signal to clients
            time.sleep(0)

            from controller.modules.user.views import stopflag
            print(stopflag)

            # if there hasn't been any clients asking for frames in
            # the last 10 seconds then stop the thread
            if time.time() - VideoCamera.last_access > 10:
                frames_iterator.close()
                print('Stopping camera thread due to inactivity.')
                break
            if stopflag == False:
                frames_iterator.close()
                print('You clicked stop.')
                break
        VideoCamera.thread = None

        self.frames_iterator.release()
        print("real stop...")

