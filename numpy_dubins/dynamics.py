import numpy as np
import gym


class Dubins(gym.Env):
    """
    Dubins environment
    """

    def __init__(self, theta_limit):
        super(Dubins, self).__init__()

        self.state_dim = 3
        self.action_dim = 1

        self.dt = 0.05

        self.theta_limit = theta_limit

        # Current state
        self.velocity = 0.7

        self.goal = np.array([1.5, 1.5])

        self.full_controls = False

    def get_jacobian(self, obs, action):
        action_clip = np.array(action)
        action_clip[0] = np.clip(
            action[0], -1 * self.theta_limit, self.theta_limit)

        Ac = np.array([[0, 0, -self.velocity * np.sin(obs[2])],
                      [0, 0, self.velocity * np.cos(obs[2])],
                      [0, 0, 0],
                       ])

        Bc = np.array([[0],
                       [0],
                       [1]])

        return np.eye(3) + self.dt * Ac, self.dt * Bc, Ac, Bc

    def deriv(self, obs, action):
        xdot = self.velocity * np.cos(obs[2])
        ydot = self.velocity * np.sin(obs[2])
        thetadot = action[0]

        return np.array([xdot, ydot, thetadot])

    def step(self, obs, action=np.zeros((1, ))):
        assert action.shape[0] == self.action_dim

        # Magnitude of actions
        action_clip = np.array(action)
        action_clip[0] = np.clip(
            action[0], -1 * self.theta_limit, self.theta_limit)

        # RK4 integration step
        k1 = self.deriv(obs, action_clip)
        k2 = self.deriv(obs + k1 * self.dt * 0.5, action_clip)
        k3 = self.deriv(obs + k2 * self.dt * 0.5, action_clip)
        k4 = self.deriv(obs + k3 * self.dt, action_clip)

        new_obs = obs + ((k1 + 2 * k2 + 2 * k3 + k4) * self.dt) / 6.0
        return new_obs, action_clip

    def reset(self):
        # Reset from initial state distribution
        obs = np.zeros((self.state_dim, ))
        obs[0] = -3.0
        obs[1] = 0.0
        obs[2] = np.pi / 2.0
        return np.array(obs)
