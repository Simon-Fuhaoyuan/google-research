"""RoboTube: Train a policy with the sparse environment reward."""


import subprocess
from absl import app
from absl import flags
from absl import logging
from torchkit.experiment import string_from_kwargs
from torchkit.experiment import unique_id
from utils import get_time_str
import os
import re


# pylint: disable=logging-fstring-interpolation

FLAGS = flags.FLAGS
CONFIG_PATH = "configs/robotube"

flags.DEFINE_string("env", None, "Which environment to train.")
flags.DEFINE_string("device", "cuda:0", "The compute device.")
flags.DEFINE_list("seeds", [0], "List specifying the seeds to run.")
flags.DEFINE_bool("parallel", False, "Whether run program in parallel version")


def main(_):
    # Map the embodiment to the x-MAGICAL env name.
    env_name = FLAGS.env

    # Generate a unique experiment name.
    experiment_name = env_name + '_env_' + get_time_str()
    logging.info("Experiment name: %s", experiment_name)

    # Parse env name and corresponding config file
    env_name_split_list = re.findall('[A-Z][^A-Z]*', env_name)
    config_file = env_name_split_list[0].lower() + '_' + env_name_split_list[1].lower() + '.py'
    config_file = os.path.join(CONFIG_PATH, config_file)

    if not FLAGS.parallel:
        for seed in FLAGS.seeds:
            command = "python train_policy.py"
            command += " --experiment_name " + experiment_name
            command += " --env_name " + f"{env_name}"
            command += " --config " + f"{config_file}"
            command += " --seed " + f"{seed}"
            command += " --device " + f"{FLAGS.device}"
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
                    f"{config_file}",
                    "--seed",
                    f"{seed}",
                    "--device",
                    f"{FLAGS.device}",
                ]))

        # Wait for each seed to terminate.
        for p in procs:
            p.wait()


if __name__ == "__main__":
    flags.mark_flag_as_required("env")
    app.run(main)
