from base_configs.rl import get_config as _get_config
from utils import copy_config_and_replace


def get_config():
    config = _get_config()

    config.save_dir = "logs/"
    config.num_train_steps = 1_000_000
    config.replay_buffer_capacity = 100_000
    config.num_seed_steps = 1000
    config.num_eval_episodes = 10
    config.log_frequency = 200

    return config
