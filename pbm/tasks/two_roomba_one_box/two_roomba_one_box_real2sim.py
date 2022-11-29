import os
import cv2
import time
import numpy as np
from scipy.spatial.transform import Rotation

from cv2 import aruco

class two_roomba_one_box_real2sim:
    def __init__(self, create_grid_mass=None, task_ver=None, picture_name=None, read_pic=None):
        done = False
        while not done:
            try:
                self.task_ver = task_ver
                self.create_grid_mass = create_grid_mass
                if self.create_grid_mass:
                    frame = cv2.imread(f'/root/roomba_hack/pbm/tasks/two_roomba_one_box/demo/{task_ver}/goal.jpg')
                else:
                    print('set picture')
                    frame = cv2.imread(f'/root/roomba_hack/pbm/output/real/current/pose.jpg')
                    
                area_hight = 195
                box_length = 19
                box_depth = 94.6
                
                dict_aruco = aruco.Dictionary_get(aruco.DICT_4X4_50)
                parameters = aruco.DetectorParameters_create()
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dict_aruco, parameters=parameters)
                print('ids', ids)
                
                # corner
                cornerUL = np.array([0., 0.])
                origin  = np.array([-corners[np.where(ids == 0)[0][0]][0][0][1], corners[np.where(ids == 0)[0][0]][0][0][0]])
                cornerUR = np.array([-corners[np.where(ids == 1)[0][0]][0][1][1], corners[np.where(ids == 1)[0][0]][0][1][0]]) - origin
                cornerBR = np.array([-corners[np.where(ids == 2)[0][0]][0][2][1], corners[np.where(ids == 2)[0][0]][0][2][0]]) - origin
                cornerBL = np.array([-corners[np.where(ids == 3)[0][0]][0][3][1], corners[np.where(ids == 3)[0][0]][0][3][0]]) - origin
                # box 
                box1UL = np.array([-corners[np.where(ids == 6)[0][0]][0][0][1], corners[np.where(ids == 6)[0][0]][0][0][0]]) - origin
                box2BR = np.array([-corners[np.where(ids == 8)[0][0]][0][2][1], corners[np.where(ids == 8)[0][0]][0][2][0]]) - origin
                boxC = (box1UL + box2BR)/2
                box2UR = np.array([-corners[np.where(ids == 8)[0][0]][0][1][1], corners[np.where(ids == 8)[0][0]][0][1][0]]) - origin
                boxRC = (box2UR + box2BR)/2

                # roomba1
                roomba1UL = np.array([-corners[np.where(ids == 5)[0][0]][0][0][1], corners[np.where(ids == 5)[0][0]][0][0][0]]) - origin
                roomba1UR = np.array([-corners[np.where(ids == 5)[0][0]][0][1][1], corners[np.where(ids == 5)[0][0]][0][1][0]]) - origin
                roomba1BR = np.array([-corners[np.where(ids == 5)[0][0]][0][2][1], corners[np.where(ids == 5)[0][0]][0][2][0]]) - origin
                roomba1C = (roomba1UL + roomba1UR)/2 +  + (roomba1UR - roomba1BR) * 2.5 /13.7
                roomba1_first_vec = roomba1UR - roomba1BR
                # roomba2
                roomba2UL = np.array([-corners[np.where(ids == 7)[0][0]][0][0][1], corners[np.where(ids == 7)[0][0]][0][0][0]]) - origin
                roomba2UR = np.array([-corners[np.where(ids == 7)[0][0]][0][1][1], corners[np.where(ids == 7)[0][0]][0][1][0]]) - origin
                roomba2BR = np.array([-corners[np.where(ids == 7)[0][0]][0][2][1], corners[np.where(ids == 7)[0][0]][0][2][0]]) - origin
                roomba2C = (roomba2UL + roomba2UR)/2 +  + (roomba2UR - roomba2BR) * 2.5 /13.7
                roomba2_first_vec = roomba2UR - roomba2BR
        
                ## convert
                sim_unit_cm2sim = 1 / area_hight
                sim_unit_aruco2sim = 1 / abs(cornerUL[1] - cornerBL[1])
                # roomba
                sim_roomba1_init = np.insert(roomba1C * sim_unit_aruco2sim, 1, 0.02)
                sim_roomba2_init = np.insert(roomba2C * sim_unit_aruco2sim, 1, 0.02)
                self.roomba_init = [str(tuple(sim_roomba1_init)), str(tuple(sim_roomba2_init))]
                self.roomba_first_vec = [roomba1_first_vec, roomba2_first_vec]
                # box
                self.sim_box_length = box_length * sim_unit_cm2sim
                self.sim_box_depth = box_depth * sim_unit_cm2sim
                self.box_init = np.insert(boxC * sim_unit_aruco2sim, 1, 0.05)
                u = np.array([1, 0])
                v = boxRC - boxC
                i = np.inner(u, v)
                n = np.linalg.norm(u) * np.linalg.norm(v)
                c = i / n
                self.rotate_angle = np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))
                done = True
            except:
                pass
            
    def get_roomba_init(self):
        return self.roomba_init
    
    def get_roomba_first_vec(self):
        return self.roomba_first_vec

    def get_obj_particle(self, n_particles=10000, box_hight=0.1):
        width = np.array([self.sim_box_depth, box_hight, self.sim_box_length])
        p = (np.random.random((n_particles, 3)) * 2 - 1) * (0.5 * width) + self.box_init
        rotvec = np.array([0, -self.rotate_angle/180*np.pi, 0])
        rot = Rotation.from_rotvec(rotvec)
        p = rot.apply(p - self.box_init) + self.box_init
        if self.create_grid_mass:
            output_dir = f'/root/roomba_hack/pbm/tasks/two_roomba_one_box/demo/{self.task_ver}'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            np.save(f'{output_dir}/goal_state.npy', p)
        return p
    
    
    
    
    




