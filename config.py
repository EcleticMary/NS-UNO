import torch
from pathlib import Path
import glob
import datetime, os
from torch import nn
import argparse
import pickle

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
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3)
args_cli = parser.parse_args()

# --- Default parameters ---
params = {
    "seed": 12,
    "activation": nn.ELU(),
    "num_epochs": args_cli.epochs,
    "batch_size": args_cli.batch_size,
    "learning_rate": args_cli.lr,
    "log_interval": 2,
    "no": 1,                   # number of times input vector is repeated
    "context": 256,            # context dimension
    "dim": 20,                 # input dimension
    "num_flows": 16,           # number of sub-flows
    "mhidden_features": 120,   # neurons in each hidden layer
    "num_layers_block": 5,     # number of ResNet blocks
    "lambda_penalty": args_cli.lambda_penalty,
    "rand": True,              # add noise in dataset
    "Lambda": False,           # use tidal deformability
    "std_M": 0.1,
    "std_R": 0.3,
    "std_L": 1.0,
    "sigmoid_transform": False,
    "load_weights": False,
    "Nmin": 5,
    "Nmax": 40,
    "Nsamples": 300,
    "set_name_lw": "_p_1",
    "set_name": args_cli.name,
    "definition": (
        "First try with CNF_3 with rho per observation random, "
        "without restricting intervals, now with new method for masking."
    ),
}

# --- Derived parameters ---
params["batch_size"] *= params["no"]

if params["Lambda"]:
    params["context"] *= 2

print("Parameters:")
for key, value in params.items():
    print(f"  {key}: {value}")

""" This works perfectly at any machine because it consideres you have a folder data in the same directory of the code, 
and it will create a folder model_test to save the models, if you want to change this just change the path below"""

BASE_DIR = Path(__file__).resolve().parent
DATA_POLY_DIR = Path(os.environ.get("CNF4_DATA_POLY", BASE_DIR / "data"))
DATA_GORDA_DIR = Path(os.environ.get("CNF4_DATA_GORDA", BASE_DIR / "data_gorda"))
OUTPUT_DIR = Path(os.environ.get("CNF4_OUTPUT_DIR", BASE_DIR / "model_test"))

print("BASE_DIR:", BASE_DIR)
print("DATA_POLY_DIR:", DATA_POLY_DIR)
print("DATA_GORDA_DIR:", DATA_GORDA_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATH = glob.glob(str(OUTPUT_DIR / f"*_{params['set_name']}*"))
print("path is:", PATH)

if PATH:
    print("path already exists, please change:", params["set_name"])
    new_name = input("new name")
    if new_name != "same":
        params["set_name"] = new_name

param_path = OUTPUT_DIR / f"parameters_{params['set_name']}.pkl"

with open(param_path, "wb") as f:
    pickle.dump(params, f)


class dotdict(dict):
    """A dictionary that allows dot notation access to its keys."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.__dict__ = self  # Allows attribute-style access

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

args = dotdict(params)