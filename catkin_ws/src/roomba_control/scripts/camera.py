#!/usr/bin/env python3
import os
import cv2
from cv2 import aruco
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class Camera():
    def __init__(self):
        rospy.Subscriber('/hand_camera/color/image_raw', Image, self.rgb_callback)
        self.pub_rgb = rospy.Publisher('pub_rgb_image', Image)
        self.rgb_image = None
        self.bridge = CvBridge()

    def rgb_callback(self, msg):
        cv_array = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        cv_array = cv2.cvtColor(cv_array, cv2.COLOR_BGR2RGB)
        self.rgb_image = cv_array
        print('take picture', self.rgb_image.shape)
        output_dir = '/root/roomba_hack/pbm/tasks/two_roomba_one_rope_one_obj/demo/v1'
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(f'{output_dir}/goal.jpg', self.rgb_image)
    
    def detect_aruco(self, img):
        dict_aruco = aruco.Dictionary_get(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters_create()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dict_aruco, parameters=parameters)
        print(ids)

if __name__ == '__main__':
    rospy.init_node('special_node', log_level=rospy.DEBUG)
    cam = Camera()
    while not rospy.is_shutdown():
        rospy.sleep(1)
    cam.detect_aruco(cam.rgb_image)