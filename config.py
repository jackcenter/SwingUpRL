def get_configuration():
    config = {
        "load_Q": False,
        "train": True
    }

    return config


def get_training_parameters():
    params = {
        "alpha": 0.1,
        "epsilon": 0.1,
        "gamma": 0.9,
        "iterations": 400,
        "default_action_index": 2
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

