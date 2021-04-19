import numpy as np
import config
import simplePendulum


def main():
    # env = gym.make('Pendulum-v0')

    params = config.get_parameters()
    env = simplePendulum.PendulumEnv()
    env.reset()

    # print_environment_bounds(env)

    u_next = 1
    for _ in range(200):

        # z1, r, _, _ = env.step([u_next])

        env.render()
        # # z1, r1, _, _ = env.step(env.action_space.sample())
        z1, r1, _, _ = env.step([u_next])
        #

        #
        # print("z: {}\tr: {}, u: {}".format(z1, r1, env.last_u))

    env.close()


def sarsa(env, Q, params):

    alpha = params.alpha
    epsilon = params.epsilon
    gamma = params.gamma
    iterations = params.iterations

    x0 = observe(env)
    u0 = policy(env, Q, x0)
    x0_idx = env.get_state_index(x0)
    u0_idx = env.get_action_index(u0)
    r = reward(x0, u0)

    for _ in range(0, iterations):

        x1 = observe(env)
        u1 = policy(Q, env, x1, epsilon)

        step(env, u1)

        x1_idx = env.get_state_index(x1)
        u1_idx = env.get_action_index(u1)

        Q[(x0_idx, u0_idx)] += alpha * (r + gamma * Q[(x1_idx, u1_idx)] - Q[(x0_idx, u0_idx)])

        x0_idx = x1_idx
        u0_idx = u1_idx
        r = reward(x0, u0)

        # TODO: delay until end of next step

    return np.copy(Q)


def observe(env):
    return env.state


def step(env, u):
    env.step(u)


def policy(env, Q, x, epsilon=0.1):

    if x[2] < 0:
        u = -0.05
    else:
        u = 0.05

    return u


def reward(x, u):
    return angle_normalize(x[0]) ** 2 + .1 * x[1] ** 2 + .001 * (u ** 2)


def angle_normalize(x):
    return ((x+np.pi) % (2*np.pi)) - np.pi


def print_environment_bounds(env):
    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.low)
    print(env.action_space.sample())


if __name__ == "__main__":
    main()
