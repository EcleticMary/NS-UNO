import argparse
import glob
import os
import pickle
from pathlib import Path

import torch
from torch import nn


# --- Device configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"PyTorch: {torch.__version__}")

if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")


# --- Command-line arguments ---
parser = argparse.ArgumentParser(description="Train Normalizing Flow Model")

parser.add_argument(
    "--lambda_penalty",
    type=float,
    default=0.7,
    help="Regularization weight for monotonicity penalty",
)

parser.add_argument(
    "--name",
    type=str,
    default="_no_name_",
    help="Name used to save output files",
)

parser.add_argument("--epochs", type=int, default=2000)

parser.add_argument(
    "--sigma_mode",
    type=str,
    default="per_eos",
    help="Mode for setting noise levels: 'per_eos' or 'constant'",
)

parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3)

parser.add_argument(
    "--description",
    type=str,
    default="No description provided",
)

parser.add_argument(
    "--both_data",
    action="store_true",
    help="Train on both polytropic and GP datasets",
)

args_cli = parser.parse_args()


# --- Default parameters ---
params = {
    "seed": 12,
    "activation": nn.ELU(),
    "num_epochs": args_cli.epochs,
    "batch_size": args_cli.batch_size * 3,
    "learning_rate": args_cli.lr,
    "sigma_mode": args_cli.sigma_mode,
    "log_interval": 2,
    "no": 1,                   # Number of times input vector is repeated
    "context": 128,            # Context dimension
    "dim": 20,                 # Input dimension
    "num_flows": 16,           # Number of sub-flows
    "mhidden_features": 120,   # Neurons in each hidden layer
    "num_layers_block": 3,     # Number of ResNet blocks
    "lambda_penalty": args_cli.lambda_penalty,
    "rand": True,              # Add noise to dataset
    "Lambda": False,           # Use tidal deformability
    "std_M": 0.1,
    "std_R": 0.3,
    "std_L": 1.0,
    "sigmoid_transform": False,
    "load_weights": False,
    "Nmin": 5,
    "Nmax": 30,
    "Ndefault": 30,
    "changing_N": True,        # Sample a random number of observations for each system
    "Nsamples": 300,
    "set_name_lw": "_p_1",
    "set_name": args_cli.name,
    "description": args_cli.description,
    "both_datasets": args_cli.both_data,
}


# --- Derived parameters ---
params["batch_size"] *= params["no"]

if params["Lambda"]:
    params["context"] *= 2


print("Parameters:")
for key, value in params.items():
    print(f"  {key}: {value}")


# --- Paths ---
# By default, the code expects the data folders to be located
# in the same directory as this script. Environment variables can
# be used to override these paths.

BASE_DIR = Path(__file__).resolve().parent
DATA_POLY_DIR = Path(os.environ.get("CNF4_DATA_POLY", BASE_DIR / "data"))
DATA_GORDA_DIR = Path(os.environ.get("CNF4_DATA_GORDA", BASE_DIR / "data_gorda"))
OUTPUT_DIR = Path(os.environ.get("CNF4_OUTPUT_DIR", BASE_DIR / "model_test"))

print("BASE_DIR:", BASE_DIR)
print("DATA_POLY_DIR:", DATA_POLY_DIR)
print("DATA_GORDA_DIR:", DATA_GORDA_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --- Check whether output name already exists ---
PATH = glob.glob(str(OUTPUT_DIR / f"*_{params['set_name']}*"))
print("path is:", PATH)

if PATH:
    print("Path already exists, please change:", params["set_name"])
    new_name = input("New name: ")

    if new_name != "same":
        params["set_name"] = new_name


# --- Save parameters ---
param_path = OUTPUT_DIR / f"parameters_{params['set_name']}.pkl"

with open(param_path, "wb") as f:
    pickle.dump(params, f)


# --- Dictionary with attribute-style access ---
class dotdict(dict):
    """A dictionary that allows dot notation access to its keys."""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


args = dotdict(params)