import os
from argparse import ArgumentParser

import yaml
import re
import tempfile
import torch
import logging
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

args = None
script_to_run = None
config_file = None
parameters = {}
run_configs = []
run_number = 0

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_parameter_value(key, value):
    """
    Get the parameter value from the parameter grid based on the selected index or value.

    Args:
        key (str): The parameter key.
        value: The selected index or actual value.

    Returns:
        value: The parameter value.
    """
    if len(parameters[key]) > 2 or (isinstance(parameters[key], list)
                                    and isinstance(parameters[key][0], str)):
        value = parameters[key][round(value)]

    return value


def is_float(num):
    """
   Check if a string can be converted to a floating-point number.

   Args:
       num (str): The string to check.

   Returns:
       bool: True if the string can be converted to a float, False otherwise.
   """
    try:
        float(num)
        return True
    except ValueError:
        return False


def is_int(num):
    """
    Check if a string can be converted to an integer.

    Args:
        num (str): The string to check.

    Returns:
        bool: True if the string can be converted to an integer, False otherwise.
    """
    try:
        int(num)
        return True
    except ValueError:
        return False


def get_last_version():
    """
    Get the path to the last version directory in the specified root directory.

    Returns:
        str or None: The path to the last version directory or None if not found.
    """
    items = os.scandir(os.path.join(args.default_root_dir, args.logs_dir))

    # Filter out only the directories from the list
    max_mtime = 0
    last_directory_version = None

    for item in items:
        if not os.path.isdir(item.path):
            continue

        item_mtime = item.stat().st_mtime
        if max_mtime < item_mtime:
            max_mtime = item_mtime
            last_directory_version = item.path

    return last_directory_version


def get_best_val_loss_from_ckpt(path):
    """
    Get the best validation loss from the checkpoint files in a directory.

    Args:
        path (str): The path to the directory containing checkpoint files.

    Returns:
        float: The best validation loss.
    """

    # Get only the checkpoints
    checkpoints = [ckpt for ckpt in os.listdir(path) if ckpt.endswith(".ckpt")]

    best_val_loss = float('inf')
    pattern = re.compile(r'val_loss=([\d.]+)')

    for ckpt in checkpoints:
        # 1. Try extracting val_loss from the filename
        match = pattern.search(ckpt)
        if match:
            try:
                val_loss = float(match.group(1).rstrip("."))  # Remove trailing "."
                best_val_loss = min(best_val_loss, val_loss)
                continue  # f found in the filename, skip loading the file
            except ValueError as e:
                logger.warning(f"Failed to convert val_loss from filename '{ckpt}': {e}")

        # 2.1 If not found in filename, try loading the checkpoint file
        try:
            checkpoint_data = torch.load(os.path.join(path, ckpt), map_location="cpu")
        except FileNotFoundError:
            logger.error(f"Checkpoint not found: {os.path.join(path, ckpt)}")
            continue
        except RuntimeError as e:
            logger.error(f"Error loading checkpoint {ckpt}: {e}")
            continue

        # 2.2 Extract val_loss from callbacks
        try:
            val_loss = float("inf")
            # val_loss = checkpoint_data["callbacks"] \
            # ["EarlyStopping{'monitor': 'val/loss', 'mode': 'min'}"]["best_score"]
            for key in checkpoint_data.get("callbacks", {}):
                if "EarlyStopping" in key:
                    val_loss = checkpoint_data["callbacks"][key].get("best_score", float("inf"))
                    break
            if isinstance(val_loss, torch.Tensor):
                val_loss = val_loss.item()
            if not isinstance(val_loss, (int, float)):
                raise TypeError(f"Unexpected type for 'best_score': {type(val_loss)}")

            best_val_loss = min(best_val_loss, val_loss)
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Error extracting 'best_score' from {ckpt}: {e}")

    return best_val_loss


# Evaluation function for Bayesian Optimization
def evaluate_model(**kwargs):
    """
    Evaluates a model using a temporary config file with Bayesian-optimized hyperparameters.

    Args:
        **kwargs (dict): Optimized hyperparameters.

    Returns:
        float: The negative of the best validation loss, suitable for maximization in optimization tasks.
    """
    # with open("config.yaml", "r") as f:
    #     config = yaml.safe_load(f)
    #
    # try:
    #     config["model"]["hidden_channels"] = int(kwargs["hidden_channels"])
    #     config["model"]["num_layers"] = int(kwargs["num_layers"])
    #     config["data"]["batch_size"] = int(kwargs["batch_size"])
    #
    #     config["optimizer"] = {
    #         "class_path": "torch.optim.AdamW",
    #         "init_args": {"lr": float(kwargs["learning_rate"])}
    #     }
    # except (KeyError, ValueError, TypeError) as e:
    #     logger.error(f"Error updating config.yaml with Bayesian Optimization parameters: {e}")
    #     return float("inf")
    #
    # with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as temp_config:
    #     yaml.dump(config, temp_config)
    #     temp_config_path = temp_config.name
    #
    # logger.info(f"Running training with config: {temp_config_path}")

    run_parameters_args = f""
    for key, param_value in kwargs.items():
        run_parameters_args += f"--{key} {get_parameter_value(key, param_value)} "

    run_parameters_args = run_parameters_args.rstrip()

    logger.info(f"running {script_to_run} with config {config_file} and parameters: {run_parameters_args}")

    os.system(
        f"python train.py fit --config {config_file} {run_parameters_args} --trainer.default_root_dir "
        f"{args.default_root_dir} --data.root {args.data_dir}"
    )
    # navigate to the trainer's default root dir, get the latest version, find the checkpoint and pick the best val_loss

    last_version = get_last_version()
    if last_version is None:
        raise Exception("Last directory version not found!")

    checkpoint_dir = os.path.join(args.default_root_dir, last_version, "checkpoints")

    # Return the negative accuracy to maximize (Bayesian Optimization expects a maximization problem)
    return -get_best_val_loss_from_ckpt(checkpoint_dir)


def get_hierarchy_keys(data, pbounds, current_key=""):
    for key, value in data.items():
        new_key = f"{current_key}.{key}" if current_key else key

        if isinstance(value, dict):
            get_hierarchy_keys(value, pbounds, new_key)
        else:
            values = [int(v) if is_int(v) else float(v) if is_float(v) else v for v in value.split(",")]

            if len(values) > 2:
                pbounds[new_key] = [0, len(values) - 1]
            else:
                # if there is a string, handle the params as indexes
                if isinstance(values[0], str):
                    if len(values) == 1:
                        pbounds[new_key] = [0, 0]
                    else:
                        pbounds[new_key] = [i for i in range(len(values))]
                else:
                    pbounds[new_key] = values
            parameters[new_key] = values


# Define the hyperparameter search space for Bayesian Optimization
def hyper_search_space(grid_file: str):
    """
    Define the hyperparameter search space for Bayesian Optimization based on a YAML grid file.

    Args:
        grid_file (str): Path to the YAML grid file specifying hyperparameter ranges.

    Returns:
        dict: A dictionary of hyperparameter bounds (pbounds) for Bayesian Optimization.
    """
    pbounds = {}

    with open(grid_file, "r") as f:
        data = yaml.safe_load(f)

        global script_to_run, parameters, config_file

        script_to_run = data["script"]
        config_file = data["config_file"]

        attr_keys = data["attr_keys"]

        get_hierarchy_keys(attr_keys, pbounds)

    return pbounds


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--default_root_dir", type=str, default=os.getcwd())
    parser.add_argument("--logs_dir", type=str, default="lightning_logs")
    parser.add_argument("--grid_file", type=str, default="hyperparam_grid.yaml")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--n_iter", type=int, default=10)
    parser.add_argument("--n_init_points", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--bayesian_runs_export_file", type=str, default="bayesian_runs.json")
    parser.add_argument("--bayesian_runs_import_file", type=str, default=None)

    args = parser.parse_args()

    if not args.bayesian_runs_export_file or not args.bayesian_runs_export_file.endswith(".json"):
        raise Exception("Provide a valid JSON file for `bayesian_runs_export_file` parameter")

    if args.bayesian_runs_import_file and not args.bayesian_runs_import_file.endswith(".json"):
        raise Exception("Provide a valid JSON file for `bayesian_runs_import_file` parameter")

    optimizer = BayesianOptimization(f=evaluate_model,
                                     pbounds=hyper_search_space(args.grid_file),
                                     verbose=2,
                                     random_state=args.random_state)

    json_logger = JSONLogger(path=args.bayesian_runs_export_file)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, json_logger)

    if args.bayesian_runs_import_file:
        load_logs(optimizer, logs=[args.bayesian_runs_import_file])
    else:
        optimizer.maximize(init_points=args.n_init_points, n_iter=args.n_iter)

    logger.info("Best result: {}; f(x) = {}.".format(optimizer.max["params"], optimizer.max["target"]))
