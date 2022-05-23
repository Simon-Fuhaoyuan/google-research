from baselines.common import plot_util as pu
# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import argparse
import wandb


MAX_STEPS = 300000


def parse_args():
    parser = argparse.ArgumentParser(description='Parse arguments in plotting RL training process')
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--title', default=None, type=str)
    parser.add_argument('--x_label', default='Steps', type=str)
    parser.add_argument('--y_label', default='Success Rate', choices=['Reward', 'Success Rate'], type=str)
    parser.add_argument('--save_path', default=None, type=bool)

    config = parser.parse_args()
    return config


def xy_success_rate_fn(r):
    x = np.cumsum(r.monitor.l)
    y = pu.smooth(r.monitor.is_success, radius=50)

    if MAX_STEPS < 0 or x.max() < MAX_STEPS:
        return x, y

    x_within_range = x[x<MAX_STEPS]
    x = np.append(x_within_range, [MAX_STEPS])
    y = y[:x.shape[0]]

    return x, y


if __name__ == '__main__':
    wandb.init(project="RoboTube Experiments", entity="simon-fuhaoyuan")
    config = parse_args()
    log_dir = config.log_dir

    # The gray background with axis meshes
    seaborn.set()

    results = pu.load_results(log_dir, enable_progress=False, verbose=True)

    if config.title == None:
        config.title = results[0].dirname.split('/')[-2].split('_')[0]

    x_label = config.x_label
    y_label = config.y_label

    if y_label == 'Reward':
        xy_fn = pu.default_xy_fn
    else:
        xy_fn = xy_success_rate_fn

    group_fn = lambda result : result.dirname.split('/')[-2].split('_')[0]

    f, _, data = pu.plot_results(
        results,
        xy_fn=xy_fn,
        split_fn=lambda _: '',
        group_fn=group_fn,
        average_group=True,
        shaded_std=True,
        shaded_err=False,
        xlabel=x_label,
        ylabel=y_label,
        legend_outside=False
    )

    xs, ys, y_stds = data[0], data[1], data[2]
    data_wandb = [[x, y, y_std] for (x, y, y_std) in zip(xs, ys, y_stds)]
    wandb.log({"tcn_exp_table": wandb.Table(data=data_wandb, columns=["Steps", "Success Rate", "Success Rate Std"])})
