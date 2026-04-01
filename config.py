import torch
import datetime, os
from torch import nn
import argparse
import pickle
import glob

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
print('Pytorch:', torch.__version__)

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

parser = argparse.ArgumentParser(description="Train Normalizing Flow Model")
# you have to run on the terminal python3 NF...py --lambda_penalty 0.4 
parser.add_argument('--lambda_penalty', type=float, default=0.7, help="Regularization weight for monotonicity penalty")
parser.add_argument('--name', type=str, default="_no_name_", help="name of file")
argp = parser.parse_args()
# params = {
#     'seeds': 12,
#     'learning_rate': 1e-4,
#     'activation': nn.ELU(),
#     'num_epochs': 10,
#     'log_interval': 2,
#     'no': 1,  # number of times input vector is repeated
#     'batch_size': 2,
#     'context': 50 ,  # context dimension
#     'dim': 20,  # input dimension
#     'num_flows': 2,  # number of sub-flows
#     'mhidden_features': 2,  # neurons in each hidden layer of the ResNet network
#     'num_layers_block': 3,  # number of ResNet blocks
#     'lambda_penalty': argp.lambda_penalty,  # regularization parameter
#     'rand':False, # noise in dataset
#     'Lambda':False, # if usign tidal deformability
#     "std_M":0.1,
#     "std_R":0.3,
#     "std_L":1,
#     "simgoidtransform":False,
#     "load_weights":False,
#     "set_name_lw":"_p_1", ### ATTENTION if you choose random
# }
params = {
    'seeds': 12,
    'learning_rate': 1e-3,
    'activation': nn.ELU(),
    'num_epochs': 2000,
    'log_interval': 2,
    'no': 1,  # number of times input vector is repeated
    'batch_size': 128,
    'context': 128 ,  # context dimension
    'dim': 20,  # input dimension
    'num_flows': 16,  # number of sub-flows
    'mhidden_features': 120,  # neurons in each hidden layer of the ResNet network
    'num_layers_block': 3,  # number of ResNet blocks
    'lambda_penalty': argp.lambda_penalty,  # regularization parameter
    'rand':True, # noise in dataset
    'Lambda':False, # if usign tidal deformability
    "std_M":0.1,
    "std_R":0.3,
    "std_L":1,
    "simgoidtransform":False,
    "load_weights":False,
    "Nmin":5,
    "Nmax":40,
    "Nsamples":300,
    "set_name_lw":"_p_1", ### ATTENTION if you choose random
}

print('Parameters:', params)
params["batch_size"] *= params["no"]
params["set_name"] = argp.name
params["Definition"]= "First try with CNF_3 with rho per observation random, where i am not restricting intervals, now with new method for masking "

if params["Lambda"]: params["context"]*=2
"""  Tenho que melhorar esta parte dos diretorios para ficar mais correto no futuro"""

# Discoteca=input('are you in discoteca')
Discoteca = os.getenv("DISCO_ANSWER", "yes") 
print('discoteca',Discoteca)
if Discoteca=='yes': 
    DIRDATA_gorda="/data_gorda/"
    """ This is for using marcio dataset"""
    DIRDATA_poly="/localdata/CNF_3/CNF_4/data/"
else: 
    DIRDATA_gorda="/Users/valeria/Documents/Documentos/PhD_1year/Gorda/"
    """ This is for using marcio dataset"""
    DIRDATA_poly="/Users/valeria/Library/CloudStorage/OneDrive-UniversidadedeCoimbra/Documentos/Documentos/PhD_1year/NF_cond/"
# carefull here with the os.getcwd() because this depends on the directory where you are running your file 


DATADIR=os.getcwd()+'/model_test'
print("DATADIR:",DATADIR)

PATH=glob.glob(DATADIR+"*_"+params["set_name"]+"*")
print('path is:',PATH)

# to check if i already had define that name to the file

if PATH:
    print('path already exists, please change:', params["set_name"])
    new_name=input('new name')
    if new_name=='same':
        print('keeping the same name:',params["set_name"])
    else:
        params["set_name"]=new_name
os.makedirs(DATADIR, exist_ok=True)
param_path = os.path.join(DATADIR, "parameters_"+params["set_name"]+".pkl")

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
    # Convert dictionary to dotdict
args = dotdict(params)