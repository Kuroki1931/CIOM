#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
import datetime
import matplotlib.pyplot as plt
from cv_bridge import CvBridge
from PIL import Image as Im
from sensor_msgs.msg import Image


class Camera():
    def __init__(self):
        rospy.Subscriber('/hand_camera/color/image_raw', Image, self.rgb_callback)
        self.pub_rgb = rospy.Publisher('pub_rgb_image', Image)
        self.rgb_image = None
        self.frames_raw = []
        self.frames = []
        self.bridge = CvBridge()
   
    def rgb_callback(self, msg):
        cv_array = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        cv_array = cv2.cvtColor(cv_array, cv2.COLOR_BGR2RGB)
        cv2.imwrite('/root/roomba_hack/pbm/output/real/current/pose.jpg', cv_array)
        self.frames_raw.append(Im.fromarray(cv_array))
        print('take video')

if __name__ == '__main__':
    start = now = datetime.datetime.now()
    rospy.init_node('special_node', log_level=rospy.DEBUG)
    cam = Camera()
    while not rospy.is_shutdown():
        rospy.sleep(5)
    now = datetime.datetime.now()
    sum_time = now - start
    cam.frames_raw[0].save(f'/root/roomba_hack/pbm/output/real/current/video_raw_{now}_{sum_time}.gif', save_all=True, append_images=cam.frames_raw[::2], loop=0)