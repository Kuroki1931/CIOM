import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

from cv2 import aruco
from sklearn.cluster import DBSCAN
from scipy.spatial.transform import Rotation


class multi_bc_rope_real2sim:
    def __init__(self, create_grid_mass=None, real_demo=None, random_seed=None, task_ver=None, picture_name=None, read_pic=None):
        self.create_grid_mass = create_grid_mass
        self.real_demo = real_demo
        self.task_ver = task_ver
        self.random_seed = random_seed
        # real_to_sim_ratio
        self.ratio = 0.14980468749999978/0.40674225747442394
        
        if real_demo:
            done = False
            while not done:
                try:
                    print('set picture')
                    self.frame = cv2.imread(f'/root/roomba_hack/pbm/output/real/current/pose.jpg')

                    dict_aruco = aruco.Dictionary_get(aruco.DICT_4X4_50)
                    parameters = aruco.DetectorParameters_create()
                    gray = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)
                    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dict_aruco, parameters=parameters)
                    print('ids', ids)
                    # corner
                    cornerUL = np.array([0., 0.])
                    self.origin  = np.array([corners[np.where(ids == 0)[0][0]][0][0][0], -corners[np.where(ids == 0)[0][0]][0][0][1]])
                    cornerUR = np.array([corners[np.where(ids == 1)[0][0]][0][1][0], -corners[np.where(ids == 1)[0][0]][0][1][1]]) - self.origin
                    cornerBR = np.array([corners[np.where(ids == 2)[0][0]][0][2][0], -corners[np.where(ids == 2)[0][0]][0][2][1]]) - self.origin
                    cornerBL = np.array([corners[np.where(ids == 3)[0][0]][0][3][0], -corners[np.where(ids == 3)[0][0]][0][3][1]]) - self.origin
                    self.center = np.array([abs(cornerUL[0] - cornerBL[0])/2, abs(cornerUL[0] - cornerBL[0])/2])
                    ## convert
                    self.sim_unit_aruco2sim = 1 / abs(cornerUL[0] - cornerBL[0])
                    
                    self.roomba_init = []
                    self.roomba_first_vec = []
                    roomba_aruco_ids = [5, 10, 7, 16, 15]
                    # roomba_aruco_ids = [5, 7, 10, 16]
                    # roomba_aruco_ids = [5, 10, 7]
                    
                    for i in roomba_aruco_ids:
                        roombaUL = np.array([corners[np.where(ids == i)[0][0]][0][0][0], -corners[np.where(ids == i)[0][0]][0][0][1]]) - self.origin
                        roombaUR = np.array([corners[np.where(ids == i)[0][0]][0][1][0], -corners[np.where(ids == i)[0][0]][0][1][1]]) - self.origin
                        roombaBR = np.array([corners[np.where(ids == i)[0][0]][0][2][0], -corners[np.where(ids == i)[0][0]][0][2][1]]) - self.origin
                        roombaBL = np.array([corners[np.where(ids == i)[0][0]][0][3][0], -corners[np.where(ids == i)[0][0]][0][3][1]]) - self.origin
                        roombaC = (roombaUL + roombaUR)/2 + (roombaUR - roombaBR) * 2.5 /13.7
                        roombaC = ((roombaC - self.center) * self.ratio) + self.center
                        roombaC = roombaC * self.sim_unit_aruco2sim
                        roombaC = np.array([roombaC[0], 1-roombaC[1]])
                        roomba_first_vec =  roombaUR - roombaBR
                        roomba_first_vec = np.array([roomba_first_vec[0], -roomba_first_vec[1]])
                        sim_roomba_init = np.insert(roombaC, 1, 0.01)
                        self.roomba_init.append(str(tuple(sim_roomba_init)))
                        self.roomba_first_vec.append(roomba_first_vec)
                    done = True
                except:
                    pass
    
    def get_roomba_first_vec(self):
        return self.roomba_first_vec
    
    def get_roomba_init(self):
        if self.real_demo:
            return self.roomba_init
        else:
            return None

    def get_obj_particle(self, hight=0.02):
        if self.create_grid_mass:
            rope_condition = (0.3, 0.015, 0.02)
            n_particles = 3000
            particle_unit = int((n_particles/(rope_condition[0]*rope_condition[1]*rope_condition[2]))**(1/3))
            rope_particle_length_x = int(rope_condition[0] * particle_unit)
            rope_particle_length_y = int(rope_condition[1] * particle_unit)
            rope_particle_length_z = int(rope_condition[2] * particle_unit)
            a_range = 0.01
            b_range = 0.05
            c_range = 0.5
            d = 0
            rope_half_length = rope_condition[0]/2

            x_cap=0.4
            y_cap=0.4

            # setting
            np.random.seed(self.random_seed)
            a = np.random.uniform(low=-a_range, high=a_range)
            # a = 0
            np.random.seed(self.random_seed)
            b = np.random.uniform(low=-b_range, high=b_range)
            np.random.seed(self.random_seed)
            c = np.random.uniform(low=-c_range, high=c_range)
            np.random.seed(self.random_seed)

            center_x = np.random.normal(0.5, 0.03)
            np.random.seed(self.random_seed)
            center_y = np.random.normal(0.5, 0.03)

            # center = np.random.normal(0.5, 0.02, (2))
            # center_initial = np.array([center[0], 0, center[1]])
            center_initial = np.array([center_x, 0, center_y])
            np.random.seed(self.random_seed)
            rotate_random = np.random.uniform(-90, 90)

            def base_func(x, x_ratio=1, y_ratio=1):
                return (a * (x/x_ratio) ** 3 + b * (x/x_ratio) ** 2 + c * (x/x_ratio))*y_ratio

            def diff_func(x, x_ratio=1, y_ratio=1):
                return (3 * a * (1/x_ratio)**3 * x**2 + 2 * b * (1/x_ratio)**2 * x + c)*y_ratio
        
            def cal_length(x_bf, x_af, x_ratio=1, y_ratio=1):
                return ((x_bf-x_af)**2 + (base_func(x_bf, x_ratio, y_ratio)-base_func(x_af, x_ratio, y_ratio))**2)**(1/2)
            
            # base x, y list
            base_x_list = np.arange(-10, 10, 0.1)
            base_y_list = base_func(base_x_list)
            max_x = np.max(base_x_list)
            min_x = np.min(base_x_list)
            max_y = np.max(base_y_list)
            min_y = np.min(base_y_list)
            x_ratio = x_cap/(max_x-min_x)
            y_ratio = y_cap/(max_y-min_y)

            particle_x_num = int(rope_particle_length_x/2)
            rope_length_dx_r = rope_length_dx_l = x_cap/2/particle_x_num
        
            r_rope_length = 0
            while r_rope_length > rope_half_length + 0.0005 or r_rope_length < rope_half_length - 0.0005:
                r_rope_length = 0
                base_x_r = 0
                base_x_r_next = rope_length_dx_r
                for _ in range(particle_x_num):
                    r_rope_length += cal_length(base_x_r, base_x_r_next, x_ratio, y_ratio)
                    base_x_r += rope_length_dx_r
                    base_x_r_next += rope_length_dx_r
                if r_rope_length > rope_half_length:
                    rope_length_dx_r -= rope_length_dx_r/(particle_x_num*16)
                elif r_rope_length < rope_half_length:
                    rope_length_dx_r += rope_length_dx_r/(particle_x_num*16)
        
            l_rope_length = 0
            while l_rope_length > rope_half_length + 0.0005 or l_rope_length < rope_half_length - 0.0005:
                l_rope_length = 0
                base_x_l = 0
                base_x_l_next = -rope_length_dx_l
                for _ in range(particle_x_num):
                    l_rope_length += cal_length(base_x_l, base_x_l_next, x_ratio, y_ratio)
                    base_x_l -= rope_length_dx_l
                    base_x_l_next -= rope_length_dx_l
                if l_rope_length > rope_half_length:
                    rope_length_dx_l -= rope_length_dx_l/(particle_x_num*16)
                elif l_rope_length < rope_half_length:
                    rope_length_dx_l += rope_length_dx_l/(particle_x_num*16)
        
            base_x = []
            base_x_l = base_x_r = 0
            base_x.append(0)
            for _ in range(particle_x_num):
                base_x_r += rope_length_dx_r
                base_x.append(base_x_r)
                base_x_l -= rope_length_dx_l
                base_x.append(base_x_l)
            base_y = base_func(base_x, x_ratio, y_ratio)
            base_y -= (max(base_y)+min(base_y))/2

            
            # stretch_y
            stretch_x_list = []
            stretch_y_list = []
            for i, j in zip(base_x, base_y):
                stretch_y_base = np.linspace(-rope_condition[1]/2, rope_condition[1]/2, int(particle_unit*rope_condition[1]))
                stretch_y_base = np.vstack((np.zeros(len(stretch_y_base)), stretch_y_base))
                inclination = -1/diff_func(i, x_ratio, y_ratio)
                if inclination > 0:
                    theta = np.arctan(inclination) - np.pi/2
                else:
                    theta = np.arctan(inclination) + np.pi/2
                rotateMat = np.matrix([
                        [np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)],
                    ])
                rotate_stretch_y_base = np.array(rotateMat * stretch_y_base)
                base_xy = np.array([i, j])
                rotate_xy = base_xy[:, None] + rotate_stretch_y_base
                stretch_x_list.extend(rotate_xy[0, :].tolist())
                stretch_y_list.extend(rotate_xy[1, :].tolist())
                
            # stretch_z
            base_z = np.linspace(0, rope_condition[2], int(particle_unit*rope_condition[2]))
            particle = np.array([stretch_x_list, stretch_y_list])
            particle = np.repeat(particle, len(base_z), axis=1)
            np.random.seed(0)
            particle = particle + np.random.uniform(-0.003, 0.003, particle.shape)
            stretch_z_list = np.tile(base_z, int(particle.shape[1]/len(base_z)))
            np.random.seed(0)
            stretch_z_list = stretch_z_list + np.random.uniform(-0.003, 0.003, stretch_z_list.shape)
            particle = np.insert(particle, 1, stretch_z_list, axis=0)
            particle = particle + np.random.uniform(-0.003, 0.003, particle.shape)
            particle = np.transpose(particle,(1, 0))    
            
            # set and rotate    
            rotvec = np.array([0, rotate_random/180*np.pi, 0])
            rot = Rotation.from_rotvec(rotvec)
            particle = rot.apply(particle)
            particle = particle + center_initial

            output_dir = f'/root/roomba_hack/pbm/tasks/multi_bc_rope/imgs/{self.task_ver}'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            np.save(f'{output_dir}/goal_state.npy', particle)
            
            random_value = {
                'seed': self.random_seed,
                'a': a,
                'b': b,
                'c': c,
                # 'center_x': center[0],
                # 'center_y': center[1],
                'center_x': center_x,
                'center_y': center_y,
                'rotate_random': rotate_random,
                'x_cap': x_cap,
                'y_cap': y_cap
            }
            with open(f'{output_dir}/goal_state_randam_value.txt', mode="w") as f:
                json.dump(random_value, f, indent=4)
        
        elif self.real_demo:
            rope_condition = (0.3, 0.015, 0.02)
            particle_x_num = 48
            particle_y_num =4
            particle_z_num = 6
            
            # clip with color
            hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            lower = np.array([56, 35, 120])
            upper = np.array([80, 130, 255])
            
            mask = cv2.inRange(hsv, lower, upper)
            result = cv2.bitwise_and(self.frame, self.frame, mask=mask)
            h, s, v1 = cv2.split(result)
            erosion = cv2.erode(v1, np.ones((2,2),np.uint8), iterations = 1)
            closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8))
            binary = np.where(closing>0, 1, closing)
            rope_perticle_list = []
            for i in range(binary.shape[0]):
                for j in range(binary.shape[1]):
                    if binary[int(i)][int(j)]==1:
                        y = (-i - self.origin[1])
                        y = (y - self.center[1])
                        y = y * self.sim_unit_aruco2sim * self.ratio
                        x = (j - self.origin[0])
                        x = (x - self.center[0])
                        x = x * self.sim_unit_aruco2sim * self.ratio
                        rope_perticle_list.append([x, y])
            rope_perticle = np.array(rope_perticle_list)

            # clustering 
            clustering = DBSCAN(eps=0.01, min_samples=9).fit(rope_perticle)
            labels = clustering.labels_
            labels_use = labels != -1
            labels = labels[labels_use]
            rope_perticle = rope_perticle[labels_use]
            counts = np.bincount(labels)
            sums_x = np.bincount(labels, rope_perticle[:, 0])
            sums_y = np.bincount(labels, rope_perticle[:, 1])
            means_x = sums_x/counts
            means_y = -sums_y/counts
            
            # approximate func
            res = np.polyfit(means_x, means_y, 20)
            base_x = []
            rope_length_dx_r = (np.max(means_x) - np.median(means_x))/particle_x_num
            rope_length_dx_l = (np.median(means_x) - np.min(means_x))/particle_x_num
            base_x_l = base_x_r = np.median(means_x)
            base_x.append(np.median(means_x))
            for _ in range(particle_x_num):
                base_x_r += rope_length_dx_r
                base_x.append(base_x_r)
                base_x_l -= rope_length_dx_l
                base_x.append(base_x_l)
            base_y = np.poly1d(res)(base_x)
            
            # stretch_y
            stretch_x_list = []
            stretch_y_list = []
            for i, j in zip(base_x, base_y):
                stretch_y_base = np.linspace(-rope_condition[1]/2, rope_condition[1]/2, particle_y_num)
                stretch_y_base = np.vstack((np.zeros(len(stretch_y_base)), stretch_y_base))
                inclination = -1/np.poly1d(np.polyder(res))(i)
                if inclination > 0:
                    theta = np.arctan(inclination) - np.pi/2
                else:
                    theta = np.arctan(inclination) + np.pi/2
                rotateMat = np.matrix([
                        [np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)],
                    ])
                rotate_stretch_y_base = np.array(rotateMat * stretch_y_base)
                base_xy = np.array([i, j])
                rotate_xy = base_xy[:, None] + rotate_stretch_y_base
                stretch_x_list.extend(rotate_xy[0, :].tolist())
                stretch_y_list.extend(rotate_xy[1, :].tolist())
                
            # stretch_z
            base_z = np.linspace(0, rope_condition[2], particle_z_num)
            particle = np.array([stretch_x_list, stretch_y_list])
            particle = np.repeat(particle, len(base_z), axis=1)
            np.random.seed(0)
            particle = particle + np.random.uniform(-0.003, 0.003, particle.shape)
            stretch_z_list = np.tile(base_z, int(particle.shape[1]/len(base_z)))
            np.random.seed(0)
            stretch_z_list = stretch_z_list + np.random.uniform(-0.003, 0.003, stretch_z_list.shape)
            particle = np.insert(particle, 1, stretch_z_list, axis=0)
            particle = particle + np.random.uniform(-0.003, 0.003, particle.shape)
            particle = np.transpose(particle,(1, 0))    
            particle = particle + np.array([0.5, 0, 0.5])

            return particle
           
        else:
            rope_condition = (0.3, 0.015, 0.02)
            n_particles = 3000
            particle_unit = int((n_particles/(rope_condition[0]*rope_condition[1]*rope_condition[2]))**(1/3))
            rope_particle_length_x = int(rope_condition[0] * particle_unit)
            rope_particle_length_y = int(rope_condition[1] * particle_unit)
            rope_particle_length_z = int(rope_condition[2] * particle_unit)
            rope_half_length = rope_condition[0]/2

            # setting
            a = 0
            b = 0
            c = 0.001
            d = 0
            center_x = 0.5
            center_y = 0.5
            center_initial = np.array([center_x, 0, center_y])
            rotate_random = 0

            x_cap=0.4
            y_cap=0.00001

            def base_func(x, x_ratio=1, y_ratio=1):
                return (a * (x/x_ratio) ** 3 + b * (x/x_ratio) ** 2 + c * (x/x_ratio))*y_ratio

            def diff_func(x, x_ratio=1, y_ratio=1):
                return (3 * a * (1/x_ratio)**3 * x**2 + 2 * b * (1/x_ratio)**2 * x + c)*y_ratio
        
            def cal_length(x_bf, x_af, x_ratio=1, y_ratio=1):
                return ((x_bf-x_af)**2 + (base_func(x_bf, x_ratio, y_ratio)-base_func(x_af, x_ratio, y_ratio))**2)**(1/2)
        
            # base x, y list
            base_x_list = np.arange(-10, 10, 0.1)
            base_y_list = base_func(base_x_list)
            max_x = np.max(base_x_list)
            min_x = np.min(base_x_list)
            max_y = np.max(base_y_list)
            min_y = np.min(base_y_list)
            x_ratio = x_cap/(max_x-min_x)
            y_ratio = y_cap/(max_y-min_y)

            particle_x_num = int(rope_particle_length_x/2)
            rope_length_dx_r = rope_length_dx_l = rope_half_length/particle_x_num
        
            r_rope_length = 0
            while r_rope_length > rope_half_length + 0.0005 or r_rope_length < rope_half_length - 0.0005:
                r_rope_length = 0
                base_x_r = 0
                base_x_r_next = rope_length_dx_r
                for _ in range(particle_x_num):
                    r_rope_length += cal_length(base_x_r, base_x_r_next, x_ratio, y_ratio)
                    base_x_r += rope_length_dx_r
                    base_x_r_next += rope_length_dx_r
                if r_rope_length > rope_half_length:
                    rope_length_dx_r -= rope_length_dx_r/(particle_x_num*16)
                elif r_rope_length < rope_half_length:
                    rope_length_dx_r += rope_length_dx_r/(particle_x_num*16)
        
            l_rope_length = 0
            while l_rope_length > rope_half_length + 0.0005 or l_rope_length < rope_half_length - 0.0005:
                l_rope_length = 0
                base_x_l = 0
                base_x_l_next = -rope_length_dx_l
                for _ in range(particle_x_num):
                    l_rope_length += cal_length(base_x_l, base_x_l_next, x_ratio, y_ratio)
                    base_x_l -= rope_length_dx_l
                    base_x_l_next -= rope_length_dx_l
                if l_rope_length > rope_half_length:
                    rope_length_dx_l -= rope_length_dx_l/(particle_x_num*16)
                elif l_rope_length < rope_half_length:
                    rope_length_dx_l += rope_length_dx_l/(particle_x_num*16)
                    
            base_x = []
            base_x_l = base_x_r = 0
            base_x.append(0)
            for _ in range(particle_x_num):
                base_x_r += rope_length_dx_r
                base_x.append(base_x_r)
                base_x_l -= rope_length_dx_l
                base_x.append(base_x_l)
            base_y = base_func(base_x, x_ratio, y_ratio)
            base_y -= (max(base_y)+min(base_y))/2

            # stretch_y
            stretch_x_list = []
            stretch_y_list = []
            for i, j in zip(base_x, base_y):
                stretch_y_base = np.linspace(-rope_condition[1]/2, rope_condition[1]/2, int(particle_unit*rope_condition[1]))
                stretch_y_base = np.vstack((np.zeros(len(stretch_y_base)), stretch_y_base))
                inclination = -1/diff_func(i, x_ratio, y_ratio)
                if inclination > 0:
                    theta = np.arctan(inclination) - np.pi/2
                else:
                    theta = np.arctan(inclination) + np.pi/2
                rotateMat = np.matrix([
                        [np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)],
                    ])
                rotate_stretch_y_base = np.array(rotateMat * stretch_y_base)
                base_xy = np.array([i, j])
                rotate_xy = base_xy[:, None] + rotate_stretch_y_base
                stretch_x_list.extend(rotate_xy[0, :].tolist())
                stretch_y_list.extend(rotate_xy[1, :].tolist()) 
                
            # stretch_z
            base_z = np.linspace(0, rope_condition[2], int(particle_unit*rope_condition[2]))
            particle = np.array([stretch_x_list, stretch_y_list])
            particle = np.repeat(particle, len(base_z), axis=1)
            np.random.seed(0)
            particle = particle + np.random.uniform(-0.003, 0.003, particle.shape)
            stretch_z_list = np.tile(base_z, int(particle.shape[1]/len(base_z)))
            np.random.seed(0)
            stretch_z_list = stretch_z_list + np.random.uniform(-0.003, 0.003, stretch_z_list.shape)
            particle = np.insert(particle, 1, stretch_z_list, axis=0)
            particle = particle + np.random.uniform(-0.003, 0.003, particle.shape)
            particle = np.transpose(particle,(1, 0))    
            
            # set and rotate
            rotvec = np.array([0, rotate_random/180*np.pi, 0])
            rot = Rotation.from_rotvec(rotvec)
            particle = rot.apply(particle)
            particle = particle + center_initial

            random_value = {
                'a': a,
                'b': b,
                'c': c,
                'center_x': center_x,
                'center_y': center_y,
                'rotate_random': rotate_random
            }
            output_dir = f'/root/roomba_hack/pbm/tasks/multi_bc_rope/imgs/{self.task_ver}'
            os.makedirs(output_dir, exist_ok=True)
            with open(f'{output_dir}/initial_state_randam_value.txt', mode="w") as f:
                json.dump(random_value, f, indent=4)

        return particle


