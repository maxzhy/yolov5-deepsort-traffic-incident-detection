import sys
sys.path.insert(0, './controller/static/yolov5')

from controller.utils.camera import VideoCamera

import os
import cv2
import torch
import torch.nn as nn
import torchvision
import numpy as np
import argparse
import torch.backends.cudnn as cudnn
import yaml

# yolov5
from controller.static.yolov5.utils.datasets import *
from controller.static.yolov5.utils.general import *
from controller.static.yolov5.utils.torch_utils import *
from controller.static.yolov5.utils.plots import plot_one_box

# deepsort
from controller.static.deep_sort_pytorch.utils.parser import get_config
from controller.static.deep_sort_pytorch.deep_sort import DeepSort

# others
import platform
import shutil
import time
from pathlib import Path

# mail
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr 


start_id = []
start_frame = []

incident_frame_idx = 0

def bbox_rel(*xyxy):
    # calculate bounding boxes
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def compute_color_for_labels(label):
    color = (0, 255, 0) # Green ID box
    return tuple(color)

def compute_color_for_labels2(label):
    color = (0, 0, 255) # Red incident box
    return tuple(color)

def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # 编号
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        #label = 'incident'
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    
    #img_size = img.shape
    #cv2.rectangle(img, (0,0), (img_size[1],img_size[0]), color, 10)

    return img

def draw_boxes2(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # assign ID
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels2(id)
        label = '{}{:d}'.format("", id)
        #label = 'incident'
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    
    #img_size = img.shape
    #cv2.rectangle(img, (0,0), (img_size[1],img_size[0]), color, 10)
    
    return img


def send_something(mail, code):
    with open("controller/templates/mail.html") as f:
        word = f.read()
        word = word.replace("Incident Detected!", code)
    sent_message(mail,word)

def sent_message(mail, word):
    def sentmail():
        ret=True
        try:
            msg = MIMEMultipart()
            html_att = MIMEText(word, 'html', 'utf-8')
            att=MIMEText(word,'plain','utf-8')
            msg.attach(html_att)
            msg.attach(att)
            msg['From']=formataddr(["Detection System",my_sender]) # sender information
            msg['To']=formataddr(["FK",my_user]) # receiver information
            msg['Subject']="Incident Detected!" # title of the email
        
            server=smtplib.SMTP_SSL("smtp.qq.com", 465)  # SMTP server, port 25
            server.login(my_sender, my_pass)  # Sender mail account and password
            server.sendmail(my_sender,[my_user,],msg.as_string())
            server.quit()  # Close connection
        except:
            ret=False
        return ret
    
    my_sender='mkddr@foxmail.com'    # Sender account
    my_pass = 'hvizekokmtprebed'     # Sender pswd
    my_user=mail                     # Receiver account
    if sentmail():print("Email sent successfully")
    else: print("Failed to send the email")



class Camera(VideoCamera):

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()


    @staticmethod
    def set_video_source(source):
        Camera.video_source = source


    @staticmethod
    def frames():
        #out, weights, imgsz = 'inference/output', 'controller/static/yolov5/weights/best-36.pt', 640
        out, weights, imgsz = 'inference/output', 'controller/static/yolov5/weights/best.pt', 640

        #source_txt = open("source.txt","r+")
        #temp=source_txt.read().split("=")
        #source_txt=temp[1]
        #source_txt = source_txt.replace(source_txt[len(source_txt)-1],"")

        #source = '0'
        #source = source_txt

        with open("source.yaml", 'r') as stream:
            source_data = yaml.load(stream, Loader=yaml.FullLoader)
        source = list(source_data.values())
        source = source[0]

        '''add realtime video stream'''
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        #webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

        '''initialize DeepSORT'''
        cfg = get_config()
        cfg.merge_from_file('controller/static/deep_sort_pytorch/configs/deep_sort.yaml')
        deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

        # Initialize device
        device = select_device()
        if os.path.exists(out):
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
        half = True and device.type != 'cpu' # half precision

        # Load model
        #google_utils.attempt_download(weights)
        model = torch.load(weights, map_location=device)['model'].float()
        model.to(device).eval()

        #stride = int(model.stride.max())  # model stride

        '''
        # Second-stage classifier
        classify = False
        if classify:
            modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
            modelc.to(device).eval()
        '''

        if half:
            model.half()

        # Read data
        vid_path, vid_writer = None, None

        if webcam:
            #view_img = True
            cudnn.benchmark = True
            dataset = LoadStreams(source, img_size=imgsz)
        else:
            #view_img = True
            #save_img = True
            dataset = LoadImages(source, img_size=imgsz)

        #dataset = LoadImages(source, img_size=imgsz)
        #dataset = LoadStreams(source, img_size=imgsz)

        # Obtain label and color
        #names = model.names if hasattr(model, 'names') else model.modules.names
        names = model.module.names if hasattr(model, 'module') else model.names

        # Run
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # 初始化图像
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

        
        #for path, img, im0s, vid_cap in dataset:
        for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):

            print(frame_idx)

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Pred
            t1 = time_synchronized()
            pred = model(img, augment=False)[0]
            
            # Apply NMS
            pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False)
            t2 = time_synchronized()

            '''
            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)
            '''

            # Run detection
            for i, det in enumerate(pred):  # Detect every frame
                # judge video type
                if webcam:
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s
                #p, s, im0 = path, '', im0s

                save_path = str(Path(out) / Path(p).name)
                s += '%gx%g ' % img.shape[2:]  # print string
                
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                
                if det is not None and len(det):
                    # Redefine boundary
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    
                    # Output
                    #for c in det[:, -1].unique():  #probably error with torch 1.5
                    for c in det[:, -1].detach().unique():
                        n = (det[:, -1] == c).sum()  # traverse everyone
                        s += '%g %s, ' % (n, names[int(c)])  # Add label
                    
                    bbox_xywh = [] # x, y, w, h array
                    confs = [] # confidence array

                    for *xyxy, conf, cls in det:
                        #label = '%s %.2f' % (names[int(cls)], conf)
                        #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                        # start deepsort
                        x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                        obj = [x_c, y_c, bbox_w, bbox_h]
                        bbox_xywh.append(obj)
                        confs.append([conf.item()])

                    xywhs = torch.Tensor(bbox_xywh)
                    confss = torch.Tensor(confs)

                    # Detect
                    outputs = deepsort.update(xywhs, confss, im0)
                    
                    # mark boundaries
                    if len(outputs) > 0:
                        #print("! object detected !") # print when anything detected
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]

                        incident = [] # Record incident vehicle id
                        incident_box = [] # Record incident vehicle location

                        safe = []
                        safe_box = []

                        for i in range(len(identities)):

                            #print("------------------------------------------------")
                            #print("i = ", i)
                            #print("start_id: ", start_id)
                            if identities[i] in start_id: # For every vehicles in the picture
                                #print("frame_idx: ", frame_idx)
                                #print("start_frame: ", start_frame[i][1])

                                for j in range(len(start_id)):
                                    
                                    if identities[i] == start_id[j]:
                                        if frame_idx - start_frame[j][1] > 250: # time threshold
                                            #print("accident!!!!!!!!!")
                                            incident.append(identities[i])
                                            incident_box.append(bbox_xyxy[i])
                                            draw_boxes2(im0, incident_box, incident)

                                            # Print incident array
                                            #print("accident id: ", identities[i])

                                            '''Red message'''
                                            red_color = (0, 0, 255) # Red
                                            im0_size = im0.shape
                                            cv2.rectangle(im0, (0,0), (im0_size[1], im0_size[0]), red_color, 10)
                                            cv2.putText(im0, str(frame_idx), (70,100), cv2.FONT_HERSHEY_PLAIN, 3, red_color, 6)
                                            
                                        else: # Normal vehicles
                                            safe.append(identities[i])
                                            safe_box.append(bbox_xyxy[i])
                                            draw_boxes(im0, safe_box, safe)

                                            if len(incident) == 0:
                                                '''Green Message'''
                                                green_color = (0, 255, 0) # Green
                                                im0_size = im0.shape
                                                
                                                cv2.rectangle(im0, (0,0), (im0_size[1], im0_size[0]), green_color, 10)
                                                cv2.putText(im0, str(frame_idx), (70,100), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 6)

                                            #continue
                                    else:
                                        continue
                            else:
                                #print("no")
                                
                                start_id.append(identities[i])
                                start_frame.append([identities[i], frame_idx])
                        


                        '''clear start_id'''
                        dele = []
                        for i in range(len(start_id)):
                            if start_id[i] not in identities:
                                dele.append(start_id[i])

                        for j in range(len(dele)):
                            start_id.remove(dele[j])

                        #print("start_id2: ", start_id)

                        '''clear start_frame'''

                        start_frame2 = []
                        for k in range(len(start_frame)):
                            if start_frame[k][0] in dele:
                                start_frame2.append([start_frame[k][0], start_frame[k][1]])

                        for l in range(len(start_frame2)):
                            start_frame.remove([start_frame2[l][0], start_frame2[l][1]])

                        #print("start_frame2: ", start_frame)
                        
                        
                        '''record video'''
                        if len(incident) != 0:
                            if vid_path != save_path:  # new video
                                vid_path = save_path
                                if isinstance(vid_writer, cv2.VideoWriter):
                                    vid_writer.release()  # release previous video writer

                                if vid_cap: # video
                                    save_path = save_path[:-4]
                                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                    save_path += '.avi'
                                else:
                                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                                    save_path += '.avi'
                                    #save_path += '.mp4'

                                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
                                #vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

                            vid_writer.write(im0)
                        
                        if len(incident) != 0:
                            global incident_frame_idx
                            if frame_idx - incident_frame_idx != 1:

                                idx_fr = str(frame_idx)
                                send_something("m.zhaoyang@outlook.com", "Incident detected in frame # " +idx_fr)

                            incident_frame_idx = frame_idx

                else:
                    deepsort.increment_ages()

                #cv2.imshow(p, im0)
                    

                print('%sDone. (%.3fs)' % (s, t2 - t1))


            yield cv2.imencode('.jpg', im0)[1].tobytes()