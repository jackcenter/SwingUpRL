import matplotlib.pyplot as plt
import numpy as np
import config
import simplePendulum

# TODO: add default action = 0
# TODO: add documentation


def main():

    cfg = config.get_configuration()
    params = config.get_training_parameters()
    env = simplePendulum.PendulumEnv()

    env.reset()

    if cfg.get("load_Q"):
        Q = np.load('Q_new.npy')

    else:
        Q = get_initial_Q(env)

    print(Q)
    plot_policy(env, Q, params.get("default_action_index"))

    if cfg.get("train"):
        for i in range(100):
            if not i%100:
                print(i)

            Q = sarsa(env, Q, params)
            env.reset()

        np.save('Q_new.npy', Q)

    plot_policy(env, Q, params.get("default_action_index"))
    run_policy(env, Q)

    env.close()


def get_initial_Q(env):

    x = env.state_space_discrete
    u = env.action_space_discrete

    Q = np.zeros((len(x), len(u)))

    return Q


def run_policy(env, Q):

    env.reset()

    for _ in range(400):
        env.render()
        x = observe(env)
        x_idx = env.get_state_index(x)
        u_idx = policy(env, Q, x_idx, 0.0)
        u1 = env.action_space_discrete[u_idx]

        step(env, u1)


def sarsa(env, Q, params):

    alpha = params.get("alpha")
    epsilon = params.get("epsilon")
    gamma = params.get("gamma")
    iterations = params.get("iterations")
    ud_idx = params.get("default_action_idx")

    x0 = observe(env)
    x0_idx = env.get_state_index(x0)
    u0_idx = policy(env, Q, x0_idx, ud_idx, epsilon)
    u0 = env.action_space_discrete[u0_idx]
    r = reward(x0, u0)

    for _ in range(0, iterations):

        x1 = observe(env)
        x1_idx = env.get_state_index(x1)

        u1_idx = policy(env, Q, x1_idx, ud_idx, epsilon)

        u1 = env.action_space_discrete[u1_idx]
        step(env, u1)

        Q[(x0_idx, u0_idx)] += alpha * (r + gamma * Q[(x1_idx, u1_idx)] - Q[(x0_idx, u0_idx)])

        x0_idx = x1_idx
        u0_idx = u1_idx
        r = reward(x1, u1)

        # TODO: delay until end of next step

    return np.copy(Q)


def sarsa_lambda(env, Q, N, params):

    alpha = params.get("alpha")
    epsilon = params.get("epsilon")
    gamma = params.get("gamma")
    iterations = params.get("iterations")
    lamb = params.get("lambda")
    ud_idx = params.get("default_action_idx")

    x0 = observe(env)
    x0_idx = env.get_state_index(x0)
    u0_idx = policy(env, Q, x0_idx, ud_idx, epsilon)
    u0 = env.action_space_discrete[u0_idx]
    r = reward(x0, u0)

    for _ in range(0, iterations):

        N[(x0_idx[0], x0_idx[1], u0_idx)] += 1
        x1 = observe(env)
        x1_idx = env.get_state_index(x1)

        u1_idx = policy(env, Q, x1, ud_idx, epsilon)
        u1 = env.action_space_discrete[u1_idx]

        step(env, u1)

        delta = alpha * (r + gamma * Q[(x1_idx[0], x1_idx[1], u1_idx)] - Q[(x0_idx[0], x0_idx[1], u0_idx)])

        for x_idx in env.state_space_discrete:
            for u_idx in env.action_space_discrete:

                Q[(x_idx, u_idx)] += alpha*delta*N[(x_idx[0], u0_idx)]
                N[(x_idx, u_idx)] *= gamma*lamb

        x0 = x1
        x0_idx = x1_idx
        u0 = u1
        u0_idx = u1_idx
        r = reward(x0, u0)

        # TODO: delay until end of next step

    return np.copy(Q)


def observe(env):
    return env.state


def step(env, u):
    env.step([u])


def policy(env, Q, x_idx, ud_idx, epsilon=0.0):

    if np.random.rand() < epsilon:
        u_idx = np.random.randint(len(env.action_space_discrete))

    else:
        max_val = Q[x_idx][ud_idx]
        max_idx = ud_idx

        for u_idx in range(0, len(env.action_space_discrete)):
            val = Q[x_idx][u_idx]

            if val > max_val:
                max_val = val
                max_idx = u_idx

        u_idx = max_idx

    return u_idx


def reward(x, u):
    return -(angle_normalize(x[0]) ** 2 + .1 * x[1] ** 2 + .001 * (u ** 2))


def angle_normalize(x):
    return ((x+np.pi) % (2*np.pi)) - np.pi


def print_environment_bounds(env):
    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.low)
    print(env.action_space.sample())


def plot_policy(env, Q, ud_idx):

    x_list = env.state_space_discrete
    x0_values = env.get_discrete_state_values(0)
    x1_values = env.get_discrete_state_values(1)
    u_values = env.get_discrete_action_values()
    u_idx_list = [u.argmax() if sum(u) else ud_idx for u in Q]
    u_list = [u_values[idx] for idx in u_idx_list]

    actions = np.zeros((len(x0_values), len(x1_values)))
    for (x0, x1), u in zip(x_list, u_list):
        x = np.where(x0_values == x0)[0][0]
        y = np.where(x1_values == x1)[0][0]
        actions[x][y] = u

    plt.imshow(actions, origin='lower')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
