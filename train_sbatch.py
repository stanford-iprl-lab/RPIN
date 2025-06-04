import argparse
import os
import os.path
import sys
import json
import numpy as np
import time
from datetime import date, datetime

base_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(base_path, ".."))
sys.path.insert(0, os.path.join(base_path, "../.."))

import configs.user_config as uc

"""
Script to launch slurm training jobs

USAGE:
python launch_train_sbatch.py --exp_param_file <path/to/exp_param.yaml> <--launch USE TO LAUNCH JOBS>
"""

def main(args):

    today_date = date.today().strftime("%m-%d-%y")
    timestamp = datetime.now().time().strftime("%H%M%S")

    exp_name = os.path.basename(args.exp_param_file).split(".")[0]
    job_name = "train_{}_{}-{}".format(exp_name, today_date, timestamp)
    exec_cmd = f"python3 train.py --cfg {args.exp_param_file} --output {job_name}"

    if args.run_local:
        print("Executing command locally: {}".format(exec_cmd))
        os.system(exec_cmd)
        exit(0)
    else:
        # Format sbatch command
        template_fname = "slurm/templates/train_template.sbatch"

        with open(template_fname, "r") as sbatch_template:
            filename = os.path.join(
                uc.SLURM_SCRIPT_DIR, "{}.sbatch".format(job_name)
            )
            print("Writing file: ", filename)
            with open(filename, "w") as templated_file:
                for line in sbatch_template:
                    if "{{SET_JOB_NAME}}" in line:
                        templated_file.write(
                            "#SBATCH --job-name=" + job_name + "\n"
                        )
                    elif "{{SET_OUTPUT_PATHS}}" in line:
                        templated_file.write(
                            "#SBATCH --output=" + uc.SLURM_OUTPUT_DIR + "/%j.out\n"
                        )
                        templated_file.write(
                            "#SBATCH --error=" + uc.SLURM_OUTPUT_DIR + "/%j.err\n"
                        )
                    elif "{{SET_EMAIL}}" in line:
                        templated_file.write(
                            "#SBATCH --mail-user=" + uc.SLURM_EMAIL + "\n"
                        )
                    elif "{{CONDA_AND_CD_TO_ROOT}}" in line:
                        templated_file.write(
                            "source {} {}\n".format(
                                uc.CONDA_BIN_PATH, uc.CONDA_ENV_PATH
                            )
                        )
                        templated_file.write("cd {}\n".format(uc.ROOT_DIR))
                    elif "{{CMD}}" in line:
                        templated_file.write(exec_cmd)
                        print("Executing command: {}".format(exec_cmd))
                    else:
                        templated_file.write(line)

        if args.launch:
            cmd = "sbatch " + filename
            print("Launching sbatch job: ", cmd)
            os.system(cmd)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Use arg to launch sbatch jobs. Otherwise, will just write files, for testing",
    )
    parser.add_argument(
        "--exp_param_file", help="Path to exp param file defining hparam_list"
    )
    parser.add_argument("--run_local", action="store_true", help="Run job locally")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)