"""RoboTube: Train a policy with the sparse environment reward."""

import subprocess
from absl import app
from absl import flags
from absl import logging
from utils import get_time_str
import os
import pyrfuniverse.assets as assets


# pylint: disable=logging-fstring-interpolation

FLAGS = flags.FLAGS
flags.DEFINE_string("env", None, "Which environment to train.")
flags.DEFINE_string("device", "cuda:0", "The compute device.")
flags.DEFINE_list("seeds", [0], "List specifying the seeds to run.")
flags.DEFINE_string("reward_type", "distance_to_goal", "goal_classifier or distance_to_goal")
flags.DEFINE_bool("parallel", False, "Whether run program in parallel version")


def main(_):
    # Map the embodiment to the x-MAGICAL env name.
    env_name = FLAGS.env
    CONFIG_PATH = "configs/robotube/drawer_closing_w_learned_reward.py"

    PRETRAINED_PATH = assets.join_path('RoboTube/DrawerClosing/XIRL_toydata_4000iter')

    # Generate a unique experiment name.
    experiment_name = env_name + '_' + 'xirl' + '_' + get_time_str()
    logging.info("Experiment name: %s", experiment_name)

    if not FLAGS.parallel:
        for seed in FLAGS.seeds:
            command = "python train_policy.py"
            command += " --experiment_name " + experiment_name
            command += " --env_name " + f"{env_name}"
            command += " --config " + f"{CONFIG_PATH}"
            command += " --seed " + f"{seed}"
            command += " --device " + f"{FLAGS.device}"
            command += " --config.reward_wrapper.pretrained_path " + f"{PRETRAINED_PATH}"
            command += " --config.reward_wrapper.type " + f"{FLAGS.reward_type}"
            os.system(command)

    else:
        # Execute each seed in parallel.
        procs = []
        for seed in FLAGS.seeds:
            procs.append(
                subprocess.Popen([  # pylint: disable=consider-using-with
                    "python",
                    "train_policy.py",
                    "--experiment_name",
                    experiment_name,
                    "--env_name",
                    f"{env_name}",
                    "--config",
                    f"{CONFIG_PATH}",
                    "--seed",
                    f"{seed}",
                    "--device",
                    f"{FLAGS.device}",
                    "--config.reward_wrapper.pretrained_path",
                    f"{PRETRAINED_PATH}",
                    "--config.reward_wrapper.type",
                    f"{FLAGS.reward_type}"
                ]))

        # Wait for each seed to terminate.
        for p in procs:
            p.wait()


if __name__ == "__main__":
    flags.mark_flag_as_required("env")
    app.run(main)
