def get_configuration():
    config = {
        "load_Q": True,
        "train": True
    }

    return config


def get_training_parameters():
    params = {
        "alpha": 0.2,
        "epsilon": 0.1,
        "gamma": 0.9,
        "iterations": 400,
        "default_action_index": None,
        "lambda": 0.9,
        "episodes": 900
    }

    return params


def get_single_run_parameters():
    params = {
        "alpha": 0.1,
        "epsilon": 0.0,
        "gamma": 0.9,
        "iterations": 1
    }

    return params

