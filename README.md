# yolov5-deepsort-traffic-incident-detection

Use YOLOv5 and DeepSORT to detect traffic incidents on highway via live surveillance video. Built web application with Flask.

## File tree
```
yolov5-deepsort-traffic-incident-detection
.
│
├── controller
│    ├ modules
│    │  ├ home
│    │  │  └ views.py
│    │  └ user
│    │     └ views.py
│    ├ static
│    │  ├ css
│    │  │  ├ startpage.css
│    │  │  └ style.css
│    │  ├ images
│    │  ├ inference
│    │  ├ deep_sort_pytorch
│    │  │  ├ configs
│    │  │  │  └ deep_sort.yaml
│    │  │  ├ deep_sort
│    │  │  │  ├ deep
│    │  │  │  └ deep_sort.py
│    │  │  └ utils
│    │  └ yolov5
│    │    ├ weights
│    │    │  └ best.py
│    │    ├ detect.py
│    │    └ train.py
│    ├ templates
│    │  ├ index.html
│    │  ├ login.html
│    │  ├ startpage.html
│    │  └ main.html
│    ├ utils
│    │  ├ yolo_deepsort.py
│    │  └ camera.py
│    └ __init__.py
├── inference
│    └ output
├── yolov5-training
│    └ DETRAC
│       ├ Vehicles
│       │  ├ train
│       │  │  ├ images
│       │  │  └ labels
│       │  ├ valid
│       │  │  ├ images
│       │  │  └ labels
│       │  └ data.yaml
│       ├ yolov5
│       │  └ runs
│       └ yolov5.ipynb
├── config.py
├── main.py
├── requirements.txt
├── source.yaml
├── gonglu3.mp4
└── system_demo_video.mp4
```


## Environment configuration

Computer with Nvidia GPU and Python3.6 as server. Install the required packages with the following command:

```
pip install -r requirements.txt
```

## How to run

1. Open source.yaml and write the link for the target video. i.e.  The video file gonglu3.mp4 is a testing MP4 video. You can also refer to [rtsp-simple-server](https://github.com/aler9/rtsp-simple-server) to create live video stream for testing.

2. Run the server with command: 

   ```
   python main.py
   ```

3. Open `localhost:5000/` in web browser. `localhost` is the IP address of the server in LAN.