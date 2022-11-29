import json
import numpy as np
import cv2
import taichi as ti
from gym.spaces import Box

# TODO: run on GPU, fast_math will cause error on float64's sqrt; removing it cuases compile error..
ti.init(arch=ti.gpu, device_memory_GB=20, debug=False, fast_math=True)

@ti.data_oriented
class TaichiEnv:
    def __init__(self, cfg, nn=False, loss=True):
        """
        A taichi env builds scene according the configuration and the set of manipulators
        """
        # primitives are environment specific parameters ..
        # move it inside can improve speed; don't know why..
        from .mpm_simulator import MPMSimulator
        from .primitive import Primitives
        from .renderer import Renderer
        from .shapes import Shapes
        from .losses import Loss
        from .nn.mlp import MLP
        from ..algorithms.solve import get_args

        self.args = get_args()
        idx = self.args.env_name.find('-')
        self.args.task_name = self.args.env_name[:idx]
        self.args.task_version = self.args.env_name[(idx+1):]
        if self.args.task_name in ['push_box']:
            from tasks.one_roomba_one_box.one_roomba_one_box_real2sim import one_roomba_one_box_real2sim
            self.real2sim = one_roomba_one_box_real2sim(self.args.create_grid_mass, self.args.task_version)
        elif self.args.task_name in ['two_roomba_deform_one_rope']:
            from tasks.two_roomba_one_rope.two_roomba_one_rope_real2sim import two_roomba_one_rope_real2sim
            self.real2sim = two_roomba_one_rope_real2sim(self.args.create_grid_mass, self.args.task_version, self.args.picture_name)
        elif self.args.task_name in ['two_roomba_wrap_rope']:
            from tasks.two_roomba_one_rope_one_obj.two_roomba_one_rope_one_obj_real2sim import two_roomba_one_rope_one_obj_real2sim
            self.real2sim = two_roomba_one_rope_one_obj_real2sim(self.args.create_grid_mass, self.args.task_version, self.args.picture_name)
        elif self.args.task_name in ['three_roomba_deform_one_rope']:
            from tasks.three_roomba_one_rope.three_roomba_one_rope_real2sim import three_roomba_one_rope_real2sim
            self.real2sim = three_roomba_one_rope_real2sim(self.args.create_grid_mass, self.args.task_version, self.args.picture_name)
        elif self.args.task_name in ['spell_out']:
            from tasks.spell_out.spell_out_real2sim import spell_out_real2sim
            self.real2sim = spell_out_real2sim(self.args.create_grid_mass, self.args.task_version, self.args.picture_name)
        elif self.args.task_name in ['multi_bc']:
            from tasks.multi_bc.multi_bc_real2sim import multi_bc_real2sim
            self.real2sim = multi_bc_real2sim(self.args.create_grid_mass, self.args.task_version, self.args.picture_name)
        elif self.args.task_name in ['multi_bc_rope', 'multi_bc_rope_6', 'multi_bc_rope_5_side', 'multi_bc_rope_5_center', 'multi_bc_rope_4_side', 'multi_bc_rope_4_center', 'multi_bc_rope_4_side_center', 'multi_bc_rope_3']:
            from tasks.multi_bc_rope.multi_bc_rope_real2sim import multi_bc_rope_real2sim
            self.real2sim = multi_bc_rope_real2sim(self.args.create_grid_mass, self.args.real_demo, self.args.rope_seed, self.args.task_version, self.args.picture_name)
        elif self.args.task_name in ['two_roomba_rotate_one_box']:
            from tasks.two_roomba_one_box.two_roomba_one_box_real2sim import two_roomba_one_box_real2sim
            self.real2sim = two_roomba_one_box_real2sim(self.args.create_grid_mass, self.args.task_version, self.args.picture_name)
        elif self.args.task_name in ['four_roomba_transfer_one_box']:
            from tasks.four_roomba_one_box.four_roomba_one_box_real2sim import four_roomba_one_box_real2sim
            self.real2sim = four_roomba_one_box_real2sim(self.args.create_grid_mass, self.args.task_version, self.args.picture_name)
        
        roomba_init = self.real2sim.get_roomba_init()
        obj_particle = self.real2sim.get_obj_particle()
        self.cfg = cfg.ENV
        self.cf = cfg
        self.primitives = Primitives(cfg.PRIMITIVES, roomba_init)
        self.shapes = Shapes(cfg.SHAPES, obj_particle)
        self.init_particles, self.particle_colors = self.shapes.get()

        cfg.SIMULATOR.defrost()
        self.n_particles = cfg.SIMULATOR.n_particles = len(self.init_particles)

        self.simulator = MPMSimulator(cfg.SIMULATOR, self.primitives)
        self.renderer = Renderer(cfg.RENDERER, self.primitives)

        if nn:
            self.nn = MLP(self.simulator, self.primitives, (256, 256))

        if loss:
            self.loss = Loss(cfg.ENV.loss, self.simulator)
        else:
            self.loss = None
        self._is_copy = True
        self.observation_space = Box(-np.inf, np.inf, (3, 3))
        self.action_space = Box(-1, 1, (3,))

    def set_copy(self, is_copy: bool):
        self._is_copy= is_copy
    

    def initialize(self):
        # initialize all taichi variable according to configurations..
        self.primitives.initialize()
        self.simulator.initialize()
        self.renderer.initialize()
        if self.loss:
            self.loss.initialize()
            self.renderer.set_target_density(self.loss.target_density.to_numpy()/self.simulator.p_mass)

        # call set_state instead of reset..
        self.simulator.reset(self.init_particles)
        if self.loss:
            self.loss.clear()
    
    def reset_new_env(self):
        from .primitive import Primitives
        from .shapes import Shapes
        from .mpm_simulator import MPMSimulator
        from .renderer import Renderer
        from tasks.multi_bc_rope.multi_bc_rope_real2sim import multi_bc_rope_real2sim
        real2sim = multi_bc_rope_real2sim(self.args.create_grid_mass, self.args.real_demo, self.args.rope_seed, self.args.task_version, self.args.picture_name)
        # initialize all taichi variable according to configurations..
        roomba_init = real2sim.get_roomba_init()
        obj_particle = real2sim.get_obj_particle()
        self.primitives = Primitives(self.cf.PRIMITIVES, roomba_init)
        self.shapes = Shapes(self.cf.SHAPES, obj_particle)
        self.simulator = MPMSimulator(self.cf.SIMULATOR, self.primitives)
        self.renderer = Renderer(self.cf.RENDERER, self.primitives)
        self.init_particles, self.particle_colors = self.shapes.get()
        self.primitives.initialize()
        self.simulator.initialize()
        self.renderer.initialize()
        if self.loss:
            self.loss.initialize()
            self.renderer.set_target_density(self.loss.target_density.to_numpy()/self.simulator.p_mass)

        # call set_state instead of reset..
        self.simulator.reset(self.init_particles)
        if self.loss:
            self.loss.clear()
    
    
    def initialize_update_target(self, target_density_numpy, target_density_path):
        # initialize all taichi variable according to configurations..
        self.primitives.initialize()
        self.simulator.initialize()
        self.renderer.initialize()
        if self.loss:
            self.loss.initialize(target_density_path)
            self.renderer.set_target_density(target_density_numpy/self.simulator.p_mass)

        # call set_state instead of reset..
        self.simulator.reset(self.init_particles)
        if self.loss:
            self.loss.clear()


    def render(self, mode='human', **kwargs):
        assert self._is_copy, "The environment must be in the copy mode for render ..."
        if self.n_particles > 0:
            x = self.simulator.get_x(0)
            self.renderer.set_particles(x, self.particle_colors)
        img = self.renderer.render_frame(shape=1, primitive=1, **kwargs)
        img = np.uint8(img.clip(0, 1) * 255)

        if mode == 'human':
            cv2.imshow('x', img[..., ::-1])
            cv2.waitKey(1)
        elif mode == 'plt':
            import matplotlib.pyplot as plt
            plt.imshow(img)
            plt.show()
        else:
            return img

    def step(self, action=None):
        if action is not None:
            action = np.array(action)
        self.simulator.step(is_copy=self._is_copy, action=action)

    def compute_loss(self):
        assert self.loss is not None
        if self._is_copy:
            self.loss.clear()
            return self.loss.compute_loss(0)
        else:
            return self.loss.compute_loss(self.simulator.cur)
    
    def create_grid_mass(self):
        x = self.simulator.get_grid_mass(0)
        np.save(f'/root/roomba_hack/pbm/plb/envs/assets/{self.args.env_name}.npy', x)

    def get_state(self):
        assert self.simulator.cur == 0
        return {
            'state': self.simulator.get_state(0),
            'softness': self.primitives.get_softness(),
            'is_copy': self._is_copy
        }

    def set_state(self, state, softness, is_copy):
        self.simulator.cur = 0
        self.simulator.set_state(0, state)
        self.primitives.set_softness(softness)
        self._is_copy = is_copy
        if self.loss:
            self.loss.reset()
            self.loss.clear()