"""
Plot the difference between env reward and learned reward.
The reward is calculated in step domain.
"""
import os
import argparse
from train_policy import ENV_REWARD_FILE_NAME, LEARNED_REWARD_FILE_NAME
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Parse arguments in plotting environment reward and learned reward')
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--n_samples', default=100, type=int)

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
    norm_fn = normalize

    config = parse_args()
    env_reward_file = os.path.join(config.log_dir, ENV_REWARD_FILE_NAME)
    learned_reward_file = os.path.join(config.log_dir, LEARNED_REWARD_FILE_NAME)

    env_rewards = get_rewards(env_reward_file)
    learned_rewards = get_rewards(learned_reward_file)
    length = min(len(env_rewards), len(learned_rewards))

    indices = np.arange(0, length, step=1)
    sampled_indices = np.random.choice(indices, size=config.n_samples, replace=False)

    x = np.arange(0, len(sampled_indices), step=1)
    y1, y2 = norm_fn(env_rewards[sampled_indices]), norm_fn(learned_rewards[sampled_indices])

    plt.plot(x, y1, label='env_reward')
    plt.plot(x, y2, label='learned_reward')
    # plt.scatter(x, y1, label='env_reward')
    # plt.scatter(x, y2, label='learned_reward')

    plt.legend()
    plt.show()
