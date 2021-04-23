import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path


class PendulumEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, g=9.8):
        self.max_speed = 6
        self.max_torque = 0.45      # kg * m
        self.dt = .05               # s
        self.g = g                  # m/s/s
        self.m = 0.3                # kg
        self.l = .3048              # m
        self.viewer = None
        self.bin_list = {
            "states": (180, 7),
            "actions": 5
        }

        self.state_space = spaces.Box(
            low=-np.array([np.pi, self.max_speed], dtype=np.float32),
            high=np.array([np.pi, self.max_speed], dtype=np.float32),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.array([1., 1., self.max_speed], dtype=np.float32),
            high=np.array([1., 1., self.max_speed], dtype=np.float32),
            dtype=np.float32
        )

        self.state_space_discrete = self.get_discrete_state_space()
        self.action_space_discrete = self.get_discrete_action_space()

        self.np_random = None
        self.state = None
        self.last_u = None

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_discrete_state_values(self, state):

        return np.linspace(self.state_space.low[state], self.state_space.high[state], self.bin_list.get("states")[state])

    def get_discrete_action_values(self, action=0):

        # return np.linspace(self.action_space.low[action], self.action_space.high[action], self.bin_list.get("actions"))
        return np.array([-0.45, -0.15, 0, 0.15, 0.45])

    def get_discrete_state_space(self):
        # set up empty grid based on discretization

        state_ranges = []
        for lo, hi, bins in zip(self.state_space.low, self.state_space.high, self.bin_list.get("states")):
            state_ranges.append(np.linspace(lo, hi, bins))

        state_space_discrete = np.array([(x, y) for x in state_ranges[0] for y in state_ranges[1]])
        return state_space_discrete

    def get_discrete_action_space(self):

        # action_space_discrete = list(np.linspace(self.action_space.low[0], self.action_space.high[0], self.bin_list.get("actions")))
        # return action_space_discrete
        return self.get_discrete_action_values()

    def get_state_index(self, state):

        x_idx = np.sum(np.abs(self.state_space_discrete - state), axis=1).argmin()

        return x_idx

    def get_action_index(self, action):

        return np.abs(self.action_space_discrete - action).argmin()

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}

    def reset(self):
        low = np.array([5/6*np.pi, -.2])
        high = np.array([7/6*np.pi, .2])
        self.state = self.np_random.uniform(low=low, high=high)
        # self.state = np.array([np.pi, 0])
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__),
                              "venv/lib/python3.8/site-packages/gym/envs/classic_control/assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return ((x+np.pi) % (2*np.pi)) - np.pi
