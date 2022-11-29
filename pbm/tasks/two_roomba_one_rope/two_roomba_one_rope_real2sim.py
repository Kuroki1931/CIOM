import cv2
import time
import numpy as np
from scipy.spatial.transform import Rotation
import os


from cv2 import aruco

class two_roomba_one_rope_real2sim:
    def __init__(self, create_grid_mass=None, task_ver=None, picture_name=None, read_pic=None):
        done = False
        while not done:
            try:
                self.task_ver = task_ver
                self.create_grid_mass = create_grid_mass
                if self.create_grid_mass:
                    self.frame = cv2.imread(f'/root/roomba_hack/pbm/tasks/two_roomba_one_rope/demo/{task_ver}/goal.jpg')
                else:
                    print('set picture')
                    self.frame = cv2.imread(f'/root/roomba_hack/pbm/output/real/current/pose.jpg')

                dict_aruco = aruco.Dictionary_get(aruco.DICT_4X4_50)
                parameters = aruco.DetectorParameters_create()
                gray = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)
                corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dict_aruco, parameters=parameters)
                print('ids', ids)
                
                # corner
                cornerUL = np.array([0., 0.])
                self.origin  = np.array([-corners[np.where(ids == 0)[0][0]][0][0][1], corners[np.where(ids == 0)[0][0]][0][0][0]])
                cornerUR = np.array([-corners[np.where(ids == 1)[0][0]][0][1][1], corners[np.where(ids == 1)[0][0]][0][1][0]]) - self.origin
                cornerBR = np.array([-corners[np.where(ids == 2)[0][0]][0][2][1], corners[np.where(ids == 2)[0][0]][0][2][0]]) - self.origin
                cornerBL = np.array([-corners[np.where(ids == 3)[0][0]][0][3][1], corners[np.where(ids == 3)[0][0]][0][3][0]]) - self.origin
                
                # # roomba1
                roomba1UL = np.array([-corners[np.where(ids == 5)[0][0]][0][0][1], corners[np.where(ids == 5)[0][0]][0][0][0]]) - self.origin
                roomba1UR = np.array([-corners[np.where(ids == 5)[0][0]][0][1][1], corners[np.where(ids == 5)[0][0]][0][1][0]]) - self.origin
                roomba1BR = np.array([-corners[np.where(ids == 5)[0][0]][0][2][1], corners[np.where(ids == 5)[0][0]][0][2][0]]) - self.origin
                roomba1C = (roomba1UL + roomba1UR)/2 +  + (roomba1UR - roomba1BR) * 2.5 /13.7
                roomba1_first_vec =  roomba1UR - roomba1BR
                # roomba2
                roomba2UL = np.array([-corners[np.where(ids == 7)[0][0]][0][0][1], corners[np.where(ids == 7)[0][0]][0][0][0]]) - self.origin
                roomba2UR = np.array([-corners[np.where(ids == 7)[0][0]][0][1][1], corners[np.where(ids == 7)[0][0]][0][1][0]]) - self.origin
                roomba2BR = np.array([-corners[np.where(ids == 7)[0][0]][0][2][1], corners[np.where(ids == 7)[0][0]][0][2][0]]) - self.origin
                roomba2C = (roomba2UL + roomba2UR)/2 +  + (roomba2UR - roomba2BR) * 2.5 /13.7
                roomba2_first_vec = roomba2UR - roomba2BR
        
                ## convert
                self.sim_unit_aruco2sim = 1 / abs(cornerUL[1] - cornerBL[1])
                # roomba
                sim_roomba1_init = np.insert(roomba1C * self.sim_unit_aruco2sim, 1, 0.02)
                sim_roomba2_init = np.insert(roomba2C * self.sim_unit_aruco2sim, 1, 0.02)
                self.roomba_init = [str(tuple(sim_roomba1_init)), str(tuple(sim_roomba2_init))]
                self.roomba_first_vec = [roomba1_first_vec, roomba2_first_vec]
                
                done = True
            except:
                pass

    def get_roomba_init(self):
        return self.roomba_init
    
    def get_roomba_first_vec(self):
        return self.roomba_first_vec

    def get_obj_particle(self, hight=0.02):
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        lower = np.array([40, 35, 126])
        upper = np.array([98, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(self.frame, self.frame, mask=mask)

        h, s, v1 = cv2.split(result)
        erosion = cv2.erode(v1, np.ones((2,2),np.uint8), iterations = 1)
        closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8))
        binary = np.where(closing>0, 1, closing)
        base_p_list = []
        for i in range(binary.shape[0]):
            for j in range(binary.shape[1]):
                if binary[int(i)][int(j)]==1:
                    y = (-i - self.origin[0]) * self.sim_unit_aruco2sim
                    x = (j - self.origin[1]) * self.sim_unit_aruco2sim
                    base_p_list.append([y, x])
        base_p = np.array(base_p_list)

        def rand_ints_nodup(a, b, k):
            ns = []
            while len(ns) < k:
                n = np.random.randint(a, b)
                if not n in ns:
                    ns.append(n)
            return ns

        if base_p.shape[0] > 5000:
            random_index = rand_ints_nodup(1, base_p.shape[0], 5000)
            base_p = base_p[random_index, :]
      
        add_p_num = 10000 - len(base_p) + 1
        each_p_add_num = add_p_num // len(base_p)
        added_p_list = []
        for p in base_p:
            added_p_list.append(p)
            random_p = np.random.uniform(low=-0.003, high=0.003, size=(each_p_add_num, 2)) + p
            for j in range(each_p_add_num):
                added_p_list.append(random_p[j])
        added_p = np.array(added_p_list)

        p_hight = (np.random.random((len(added_p), 1)) * 2 - 1) * 0.5 * hight + hight/2
        particle = np.insert(added_p, [1], p_hight, axis=1)
        particle = [i.tolist() for i in particle if (i[0]>0) & (i[0]<1) & (i[2]>0) & (i[2]<1)]
        particle = np.array(particle)
        
        if self.create_grid_mass:
            output_dir = f'/root/roomba_hack/pbm/tasks/two_roomba_one_rope/demo/{self.task_ver}'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            np.save(f'{output_dir}/goal_state.npy', particle)
        return particle

    
    
    
    
    




