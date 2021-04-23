import matplotlib.pyplot as plt
import numpy as np
import config
import simplePendulum

# TODO: add documentation


def main():

    cfg = config.get_configuration()
    params = config.get_training_parameters()
    env = simplePendulum.PendulumEnv()

    params.update({"default_action_index": np.where(env.get_discrete_action_space() == 0)[0][0]})

    # run_sarsa(env, cfg, params)
    run_sarsa_lambda(env, cfg, params)

    plot_policy(env, "sarsa_lambda", params)
    run_policy(env, "sarsa_lambda", params)


def run_sarsa(env, cfg, params):
    if cfg.get("load_Q"):
        Q = np.load('data/Q_sarsa.npy')
        episodes = np.load('data/E_sarsa.npy')
    else:
        Q = get_initial_Q(env)
        episodes = 0

    if cfg.get("train"):
        for i in range(params.get("episodes")):
            if not i % 100:
                print(i)

            env.reset()
            Q = sarsa(env, Q, params)
            episodes += 1

        np.save('data/Q_sarsa.npy', Q)
        np.save('data/E_sarsa.npy', episodes)

    env.close()
    print(episodes)


def run_sarsa_lambda(env, cfg, params):
    if cfg.get("load_Q"):
        Q = np.load('data/Q_sarsaLambda.npy')
        N = np.load('data/N_sarsaLambda.npy')
        episodes = np.load('data/E_sarsaLambda.npy')
    else:
        Q = get_initial_Q(env)
        N = np.zeros(np.shape(Q))
        episodes = 0

    if cfg.get("train"):
        for i in range(params.get("episodes")):
            if not i % 100:
                print(i)

            env.reset()
            Q, N = sarsa_lambda(env, Q, N, params)
            episodes += 1

        np.save('data/Q_sarsaLambda.npy', Q)
        np.save('data/N_sarsaLambda.npy', N)
        np.save('data/E_sarsaLambda.npy', episodes)

    env.close()
    print(episodes)


def get_initial_Q(env):

    x = env.get_discrete_state_space()
    u = env.get_discrete_action_space()

    Q = np.zeros((len(x), len(u)))

    return Q


def plot_policy(env, algorithm, params):

    if algorithm == "sarsa_lambda":
        Q = np.load('data/Q_sarsaLambda.npy')

    ud_idx = params.get("default_action_index")

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


def run_policy(env, algorithm, params):

    if algorithm == "sarsa_lambda":
        Q = np.load('data/Q_sarsaLambda.npy')

    ud_idx = params.get("default_action_index")

    env.reset()
    for _ in range(400):
        env.render()
        x = observe(env)
        x_idx = env.get_state_index(x)
        u_idx = policy(env, Q, x_idx, ud_idx)
        u1 = env.action_space_discrete[u_idx]

        step(env, u1)


def sarsa(env, Q, params):

    alpha = params.get("alpha")
    epsilon = params.get("epsilon")
    gamma = params.get("gamma")
    iterations = params.get("iterations")
    ud_idx = params.get("default_action_index")

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
    ud_idx = params.get("default_action_index")

    x0 = observe(env)
    x0_idx = env.get_state_index(x0)
    u0_idx = policy(env, Q, x0_idx, ud_idx, epsilon)
    u0 = env.action_space_discrete[u0_idx]
    r = reward(x0, u0)

    for _ in range(0, iterations):

        N[(x0_idx, u0_idx)] += 1
        x1 = observe(env)
        x1_idx = env.get_state_index(x1)

        u1_idx = policy(env, Q, x1_idx, ud_idx, epsilon)
        u1 = env.action_space_discrete[u1_idx]

        step(env, u1)

        delta = r + gamma * Q[(x1_idx, u1_idx)] - Q[(x0_idx, u0_idx)]

        Q += alpha*delta*N
        N *= gamma*lamb

        x0 = x1
        x0_idx = x1_idx
        u0 = u1
        u0_idx = u1_idx
        r = reward(x0, u0)

        # TODO: delay until end of next step

    return np.copy(Q), np.copy(N)


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


if __name__ == "__main__":
    main()
