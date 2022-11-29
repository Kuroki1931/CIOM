import numpy as np
import copy

COLORS = [
    (127 << 16) + 127,
    (127 << 8),
    127,
    127 << 16,
]


class Shapes:
    # make shapes from the configuration
    def __init__(self, cfg, obj_particle):
        self.objects = []
        self.colors = []

        self.dim = 3

        state = np.random.get_state()
        np.random.seed(0) #fix seed 0
        # self.add_object(p, color, init_rot=init_rot)
        for i in cfg:
            kwargs = {key: eval(val) if isinstance(val, str) else val for key, val in i.items() if key!='shape'}
            print(kwargs)
            if i['shape'] == 'box':
                self.add_box(obj_particle, **kwargs)
            elif i['shape'] == 'sphere':
                self.add_sphere(**kwargs)
            else:
                raise NotImplementedError(f"Shape {i['shape']} is not supported!")
        np.random.set_state(state)

    def get_n_particles(self, volume):
        return max(int(volume/0.2**3) * 10000, 1)

    def add_object(self, particles, color=None, init_rot=None):
        self.objects.append(particles[:,:self.dim])
        if color is None or isinstance(color, int):
            tmp = COLORS[len(self.objects)-1] if color is None else color
            color = np.zeros(len(particles), np.int32)
            color[:] = tmp
        self.colors.append(color)

    def add_box(self, p, init_pos, width, n_particles=10000, color=None, init_rot=None):
        self.add_object(p, color, init_rot=init_rot)

    def get(self):
        assert len(self.objects) > 0, "please add at least one shape into the scene"
        return np.concatenate(self.objects), np.concatenate(self.colors)
