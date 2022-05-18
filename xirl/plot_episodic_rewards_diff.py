"""
Plot the difference between env reward and learned reward.
The reward is calculated in episodic domain.

Note: when using this script, we must copy the following files to an empty folder and run.
`train.monitor.csv`
'env_reward'
`learned_reward`
"""
import os
import argparse
from train_policy import ENV_REWARD_FILE_NAME, LEARNED_REWARD_FILE_NAME
import numpy as np
import matplotlib.pyplot as plt
from baselines.common import plot_util as pu


def parse_args():
    parser = argparse.ArgumentParser(description='Parse arguments in plotting environment reward and learned reward')
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--downsample_rate', default=10, type=int)

    config = parser.parse_args()
    return config


def get_rewards(file):
    rewards = []
    with open(file) as f:
        raw_rewards = f.readlines()
        for rew in raw_rewards:
            rewards.append(float(rew))

    f.close()
    return np.array(rewards)


def init(rewards: np.ndarray):
    return rewards


def normalize(rewards: np.ndarray):
    mean = rewards.mean()
    std = rewards.std()
    return (rewards - mean) / std


def normalize_01(rewards: np.ndarray):
    min = rewards.min()
    max = rewards.max()
    return (rewards - min) / (max - min)


if __name__ == '__main__':
    norm_fn = normalize_01

    config = parse_args()
    env_reward_file = os.path.join(config.log_dir, ENV_REWARD_FILE_NAME)
    learned_reward_file = os.path.join(config.log_dir, LEARNED_REWARD_FILE_NAME)

    train_result = pu.load_results(config.log_dir)[0]
    episodic_env_rewards = list(train_result.monitor.r)
    episode_lengths = list(train_result.monitor.l)

    # env_rewards = get_rewards(env_reward_file)
    step_learned_rewards = get_rewards(learned_reward_file)
    episodic_learned_rewards = []
    index = 0
    for length in episode_lengths:
        episodic_learned_rewards.append(step_learned_rewards[index:index + length].sum())
        index += length

    episodic_env_rewards = np.array(episodic_env_rewards)
    episodic_learned_rewards = np.array(episodic_learned_rewards)
    indices = np.arange(0, len(episode_lengths), step=config.downsample_rate)

    x = np.arange(0, len(indices), step=1)
    y1, y2 = norm_fn(episodic_env_rewards[indices]), norm_fn(episodic_learned_rewards[indices])

    plt.plot(x, y1, label='env_reward')
    plt.plot(x, y2, label='learned_reward')

    plt.legend()
    plt.show()
