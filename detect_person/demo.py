import os
import cv2
import time
import argparse
import torch
import numpy as np

from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config

config_detection = 'configs/yolov3_tiny.yaml'
config_deepsort = 'configs/deep_sort.yaml'
ignore_display = True
frame_interval = 1
display_width = 800
display_width = 600
save_path = 'demo/demo.avi'
cpu = False

class VideoTracker(object):
    def __init__(self, cfg):
        self.cfg = cfg

        self.cap = cv2.VideoCapture(0)
        #self.cap = cv2.VideoCapture(0)
        self.detector = build_detector(cfg, use_cuda=cpu)
        self.deepsort = build_tracker(cfg, use_cuda=cpu)
        self.class_names = self.detector.class_names


    def __enter__(self):
        self.im_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if save_path:
            fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(save_path, fourcc, 2, (self.im_width,self.im_height))

        assert self.cap.isOpened()
        return self

    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)
        

    def run(self):
        idx_frame = 0
        while True: 
            start = time.time()
            _, frame = self.cap.read()
            try:
                im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                bbox_xywh, cls_conf, cls_ids = self.detector(im)
                if bbox_xywh is not None:
  
                    mask = cls_ids==0

                    bbox_xywh = bbox_xywh[mask]
                    bbox_xywh[:,3:] *= 1.2 
                    cls_conf = cls_conf[mask]

                    # tracking
                    outputs = self.deepsort.update(bbox_xywh, cls_conf, im)

                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:,:4]
                        identities = outputs[:,-1]
                        frame = draw_boxes(frame, bbox_xyxy, identities)
            except:
                pass

            end = time.time()
            print("time: {:.03f}s, fps: {:.03f}".format(end-start, 1/(end-start)))

            try:
                cv2.imshow('Detect Person', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except:
                pass
            
            if save_path:
                self.writer.write(frame)

        self.cap.release()
        cv2.destroyAllWindows()

if __name__=="__main__":
    cfg = get_config()
    cfg.merge_from_file(config_detection)
    cfg.merge_from_file(config_deepsort)

    with VideoTracker(cfg) as cap_str:
        cap_str.run()
