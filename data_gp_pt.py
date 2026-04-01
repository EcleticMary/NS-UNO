""" This code was for using pressure but from article of Marcio"""
# We also import few more libraries to plots,
# time measurements, and the creation of the normalizing flow
import numpy as np
import pandas as pd
import sklearn
#import pyarrow
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import glob
import time
import torch.utils.data
from scipy import interpolate
from sklearn.preprocessing import StandardScaler
from config import *


dataname='polytropic'
# --- Data Loading ---
with open(param_path, 'w') as data:
    data.write(str(args))

# Set seeds for reproducibility
torch.manual_seed(args.seeds)
np.random.seed(args.seeds)

DF_=pd.read_csv(DIRDATA_poly+"DF_final.csv")
EOS_=pd.read_csv(DIRDATA_poly+"EOS_final.csv")
MR_=pd.read_csv(DIRDATA_poly+"MR_final.csv")

n=np.linspace(0.13,1.28,20)

eos=EOS_.groupby('Modelo').apply(lambda x: interpolate.interp1d(
        x=x.n,
        y=x.p,
        kind='cubic')(n) )

P=pd.DataFrame(np.array([eos.iloc[j] for j in range(eos.shape[0])]),columns=n)
P['Modelo']=eos.index
train_ID, test_ID = train_test_split(DF_.Modelo, test_size=0.1, random_state=100) 
MR_test=MR_[MR_.Modelo.isin(test_ID)]
MR_train=MR_[MR_.Modelo.isin(train_ID)]
MR_train['Modelo']=MR_train['Modelo'].astype(int).apply(lambda x: f"pt_{str(x)}")
MR_test['Modelo']=MR_test['Modelo'].astype(int).apply(lambda x: f"pt_{str(x)}")
p_train=P[P.Modelo.isin(train_ID)].iloc[:,:-1]
p_test=P[P.Modelo.isin(test_ID)].iloc[:,:-1]
p_train.index=[f"{i}" for i in MR_train['Modelo'].unique()]
p_test.index=[f"{i}" for i in MR_test['Modelo'].unique()]
grouped_MR_train = MR_train.groupby('Modelo')

test_ID=["pt_"+str(test_ID.iloc[j]) for j in range(test_ID.shape[0])]

### esta parte é calulada à parte para o dataset ser criado mais rápido
interpolation_dict = {
    row_id: interpolate.interp1d(
        x=grouped_MR_train.get_group(row_id)['M'],
        y=grouped_MR_train.get_group(row_id)['R'],
        kind='linear'
    )
    for row_id in p_train.index.unique()
}


interpolation_dict2 = {
    row_id: interpolate.interp1d(
        x=grouped_MR_train.get_group(row_id)['M'],
        y=grouped_MR_train.get_group(row_id)['Lambda'],
        kind='linear'
    )
    for row_id in p_train.index.unique()
}


M_max=MR_train.groupby('Modelo').apply(lambda x: x.M.max())


def build_samples(
    ID_list,
    stdM,
    stdR,
    rand=False,
    M_max=M_max,
    interpolation_dict=interpolation_dict,
    N_min=5,
    N_max=40,
    n_draws=300,
    # --- covariance controls ---
    cov_mode="rho",          # "none" | "rho"
    rho_mode="fixed",        # "fixed" | "global_random" | "per_obs_random" | "normal"
    rho=0.0,                 # used if rho_mode="fixed"
    rho_low=-1,            # used if rho_mode in {"global_random","per_obs_random"}
    rho_high=1,
    rho_mean=0.3,            # used if rho_mode="normal"
    rho_sd=0.1,
    rho_clip=0.95,
    seed=None,
):
    """
    Builds samples with optional correlated (M_obs, R_obs) measurement noise.

    Mathematical rule enforced (via rho bounds):
        Cov(M,R) = rho * sigma_M * sigma_R, with -1 <= rho <= 1
    """
    if M_max is None or interpolation_dict is None:
        raise ValueError("You must provide M_max and interpolation_dict.")

    rng = np.random.default_rng(seed)
    samples = []

    for row_id in ID_list:
        # pick a random number of observations
        N = rng.integers(N_min, N_max + 1)

        # sample latent masses and radii
        M = rng.uniform(1.0, float(M_max.loc[row_id]), size=(N, 1))
        R = interpolation_dict[row_id](M)  # (N,1)

        if not rand:
            samples.append(
                dict(row_id=row_id, N=N, M=M, R=R)
            )
            continue

        # per-observation uncertainties (N,1)
        std_M = rng.uniform(0.05, stdM, size=(N, 1))
        std_R = rng.uniform(0.1, stdR, size=(N, 1))

        # draw correlation(s) rho in a safe way
        if cov_mode == "none":
            rho_i = np.zeros((N, 1))
        elif cov_mode == "rho":
            if rho_mode == "fixed":
                rho_i = np.full((N, 1), float(rho))
            elif rho_mode == "global_random":
                rho_g = rng.uniform(rho_low, rho_high)
                rho_i = np.full((N, 1), float(rho_g))
            elif rho_mode == "per_obs_random":
                rho_i = rng.uniform(rho_low, rho_high, size=(N, 1))
            elif rho_mode == "normal":
                rho_i = rng.normal(rho_mean, rho_sd, size=(N, 1))
            else:
                raise ValueError(f"Unknown rho_mode={rho_mode!r}")
        else:
            raise ValueError(f"Unknown cov_mode={cov_mode!r}")

        # hard safety: keep |rho| <= rho_clip (<=1)
        rho_i = np.clip(rho_i, -float(rho_clip), float(rho_clip))

        # ---- correlated sampling via per-observation Cholesky (vectorized) ----
        # Sigma_i = [[sM^2, rho*sM*sR],
        #            [rho*sM*sR, sR^2]]
        sigmaM = std_M
        sigmaR = std_R
        covMR = rho_i * sigmaM * sigmaR  # (N,1)

        # Cholesky factors for each i:
        # a = sigmaM
        # b = cov / a
        # c = sqrt(sigmaR^2 - b^2)
        a = sigmaM
        b = covMR / (a + 1e-12)
        c_sq = sigmaR**2 - b**2
        c = np.sqrt(np.maximum(c_sq, 0.0))

        # standard normals (N, n_draws)
        z1 = rng.normal(size=(N, n_draws))
        z2 = rng.normal(size=(N, n_draws))

        # broadcast means (N,1) + (N,n_draws)
        M_obs = M + a * z1
        R_obs = R + b * z1 + c * z2

        samples.append(
            dict(
                row_id=row_id,
                N=N,
                M=M,
                R=R,
                std_M=std_M,
                std_R=std_R,
                rho=rho_i,
                M_obs=M_obs,
                R_obs=R_obs,
            )
        )

    return samples





def collate_fn(batch):
    B = len(batch)
    S_max = max(len(eos) for eos in batch)
    N_max = max(x.shape[0] for eos in batch for x in eos)
    d_in = 2

    x_padded = torch.zeros(B, S_max, N_max, d_in)
    mask = torch.zeros(B, S_max, N_max, dtype=torch.bool)

    for i, eos in enumerate(batch):
        for j, star in enumerate(eos):
            n = star.shape[0]
            x_padded[i, j, :n] = star
            mask[i, j, :n] = True

    return torch.tensor(x_padded), torch.tensor(mask)#, torch.tensor(np.stack(eos_list))



class Dataset(torch.utils.data.Dataset):
    """Custom Dataset for PyTorch."""
    def __init__(self, eos, tov,mask):
        self.eos = eos
        self.tov = tov
        self.mask = mask

    def __len__(self):
        return self.eos.shape[0]

    def __getitem__(self, index):
        return self.eos[index, :], self.tov[index, :],self.mask[index, :]


def make_batch_from_samples(samples):
    """
    Converts a list of per-star sample dicts into HierarchicalEncoder input.
    Each item in the batch is a list of torch tensors, one per star.
    """
    batch = []
    for s in samples:  # each sample corresponds to one star
        # x_star = np.concatenate([s["M_obs"].reshape(-1, 1), s["R_obs"].reshape(-1, 1)], axis=1)
        x_star=np.concatenate([s["M_obs"][:, :, np.newaxis], s["R_obs"][:, :, np.newaxis]], axis=2)
        x_star = torch.tensor(x_star, dtype=torch.float32)
        batch.append(x_star)
    return batch  # wrapped in outer list for batch size = 1
   
train_ID_, val_ID_ = train_test_split(np.unique(p_train.index.values), test_size=0.1, random_state=11)


x_train = p_train.loc[train_ID_]
x_val =p_train.loc[val_ID_]
x_test = p_test.loc[test_ID]

d=build_samples(train_ID_,0.1,0.3,rand=True),build_samples(val_ID_,0.1,0.3,rand=True)

 #### for GP ######
DIRDATA_gorda="/localdata/CNF_3/CNF_4/data_gorda/"
# Fixing seeds for reproducibility
torch.manual_seed(1)
np.random.seed(1) # como tens uma seed consegues fixar as samples depois
# --- Data Loading ---
with open(DIRDATA_gorda+"EoS_ensemble.pickle", "rb") as f:
    df, n_gp = pickle.load(f)

n_=np.linspace(0.13,1.28,20)

# Set seeds for reproducibility
torch.manual_seed(args.seeds)
np.random.seed(args.seeds)


# Data preprocessing steps:
R = pd.DataFrame(np.array([df.r[j] for j in range(120000)]))
df = df[~R[R < 5].any(axis=1)]  # remove outliers
df = df[df.mmax >= 2]
df.reset_index(drop=True, inplace=True)

# Cs_2 = pd.DataFrame(np.array([df.cs2.iloc[j] for j in range(df.shape[0])]))
R = pd.DataFrame(np.array([df.r.iloc[j] for j in range(df.shape[0])]))
M = pd.DataFrame(np.array([df.m.iloc[j] for j in range(df.shape[0])]))
"""estou a apssar de GeV para MeV a Pressao"""
P = pd.DataFrame(np.array([df.p.iloc[j] for j in range(df.shape[0])]))*(10**3) 
# E = pd.DataFrame(np.array([df.e.iloc[j] for j in range(df.shape[0])]))
L = pd.DataFrame(np.array([df.L.iloc[j] for j in range(df.shape[0])]))


Likeli = df.iloc[:, -4] * df.iloc[:, -3] * df.iloc[:, -2] * df.iloc[:, -1]  # model likelihood
M.dropna(axis=1, inplace=True)
R.dropna(axis=1, inplace=True)
M_max = df.mmax

# Create interpolator for VS values
"""This needs to be here and not after lambda filtering because VS does not have the ID order which is the index"""
interpolator = interpolate.interp1d(
    x=n_gp * 0.16,
    y=P,
    kind='linear',
    fill_value="extrapolate"
)
""" Atention here p small is the interpolatio dataset wich is a numpy array"""
p = interpolator(n)


# Data transformation for M, R, and L
# here i'm taking all the values above the maximum max in every curve
M_2 = M.where((M.diff(axis=1) > 0))
M_2.iloc[:, 0] = M.iloc[:, 0]
M_N = M_2.stack().to_frame().reset_index().drop('level_1', axis=1).rename(columns={'level_0': 'ID', 0: 'M'})
R_N = R.where(M_2.notna()).stack().to_frame().reset_index().drop('level_1', axis=1).rename(columns={'level_0': 'ID', 0: 'R'})
L_N = L.where(M_2.notna()).stack().to_frame().reset_index().drop('level_1', axis=1).rename(columns={'level_0': 'ID', 0: 'L'})

GRo = L_N.groupby('ID').apply(lambda x: (x.L.diff() > 0).any())
R_N = R_N[~R_N.isin({'ID': np.where(GRo == True)[0]})].dropna()
M_N = M_N[~M_N.isin({'ID': np.where(GRo == True)[0]})].dropna()
L_N = L_N[~L_N.isin({'ID': np.where(GRo == True)[0]})].dropna()
M_N['ID'] = M_N['ID'].astype(int).apply(lambda x: f"gp_{str(x)}")
R_N['ID'] = R_N['ID'].astype(int).apply(lambda x: f"gp_{str(x)}")
P.index=[f"gp_{i}" for i in range(len(P))]
P = P.loc[M_N.ID.drop_duplicates().values]



train_IDgp, test_IDgp = train_test_split(P.index, test_size=0.1, random_state=11)
grouped_M = M_N.groupby('ID')
grouped_R = R_N.groupby('ID')
MRL_N=pd.concat([M_N,R_N.R,L_N.L],axis=1)
# MRL_N=MRL_N.sort_values(by=['ID', 'M'])

grouped_MRL_N=MRL_N.groupby('ID')

interpolation_dict2_gp = {
    row_id:  interpolate.interp1d(
        x=grouped_MRL_N.get_group(row_id)['M'],
        y=grouped_MRL_N.get_group(row_id)['L'],
         kind='linear')
    for row_id in train_IDgp
}
# Precompute interpolation dictionary for training
interpolation_dict_gp = {
    row_id: interpolate.interp1d(
        x=grouped_MRL_N.get_group(row_id)['M'],
        y=grouped_MRL_N.get_group(row_id)['R'],
        kind='linear',
    )
    for row_id in train_IDgp
}
max_mass = grouped_M['M'].max().max()
int_ = np.round(np.arange(1.0, max_mass + 0.03, 0.03), 7)

M_max=MR_train.groupby('Modelo').apply(lambda x: x.M.max())
""" P does not have negative values but Because of the interpolation of n, p haves some negative values:"""
train_IDgp=train_IDgp[~train_IDgp.isin(pd.DataFrame(p).iloc[np.where(pd.DataFrame(p)<0)[0]].index)]# isto tem que ser primeiro, se não o p já não tem os valores
p=pd.DataFrame(p).drop(pd.DataFrame(p).iloc[np.where(pd.DataFrame(p)<0)[0]].index)
p.index=[f"gp_{i}" for i in range(len(p))]
# Additional data splits for training and validation
train_ID_gp, val_ID_gp = train_test_split(train_IDgp, test_size=0.1, random_state=11)

x_train_gp = p.loc[train_ID_gp]

x_val_gp = p.loc[val_ID_gp]
x_test_gp = p.loc[test_IDgp]

x_train_gp.columns=n
x_test_gp.columns=n
x_val_gp.columns=n

df=df.loc[np.where(GRo == False)[0]]
df.index=MRL_N.ID.unique()
M_max_gp = df.mmax
# d_gp=build_samples(train_ID_gp,0.1,0.3, rand=True, cov_mode="rho", rho_mode="per_obs_random",interpolation_dict=interpolation_dict_gp, M_max=M_max_gp, N_min=5, N_max=40, n_draws=300,
#                         rho_low=-0.5, rho_high=0.5)

X_train=pd.concat((x_train,x_train_gp))
X_val=pd.concat((x_val,x_val_gp))
X_test=pd.concat((x_test,x_test_gp))
# D=d[0]+d_gp


x_train_s,x_val_s= np.log10(X_train.astype('float32')),np.log10(X_val.astype('float32'))
x_train_s,x_val_s=pd.DataFrame(np.repeat(x_train_s,args.no,axis=0)).values,pd.DataFrame(np.repeat(x_val_s,args.no,axis=0)).values

# training_set = Dataset(eos_new,dataset_creation(0,0)[0])
print(x_train.shape,x_val_s.shape)
print('x_train',x_train_s)
""" to check if dataset is being well created"""
# p_train=p_train.reset_index()

# training_set = Dataset(eos_new,dataset_creation(0,0)[0])
print(x_train_s.shape,x_val_s.shape)
print('x_train',x_train_s)
""" to check if dataset is being well created"""

fig, ax = plt.subplots(figsize=(5,4), ncols=1, nrows=1, sharex=True, sharey='row', layout='constrained')

plt.plot(n,x_train_s[0,],'o')
plt.plot(n,x_train_s[1],'o')
# plt.plot(EOS_[EOS_.Modelo==x_train.Modelo[0]].n,np.log10(EOS_[EOS_.Modelo==x_train.Modelo[0]].p))
# plt.plot(EOS_[EOS_.Modelo==x_train.Modelo[1]].n,np.log10(EOS_[EOS_.Modelo==x_train.Modelo[1]].p))
plt.savefig(DATADIR+f"p_{args.set_name}.pdf")

if args.Lambda:
    fig, ax = plt.subplots(figsize=(5,4), ncols=1, nrows=1, sharex=True, sharey='row', layout='constrained')
    eos=20
    plt.plot(10**d[-2].iloc[eos,-15:],d[-2].iloc[eos,-30:-15],'o')
    plt.plot(MR_train[MR_train["Modelo"].isin([x_train.iloc[eos,0]])].Lambda,MR_train[MR_train["Modelo"].isin([x_train.iloc[eos,0]])].M)

    eos=30
    plt.plot(10**d[-2].iloc[eos,-15:],d[-2].iloc[eos,-30:-15],'o')
    plt.plot(MR_train[MR_train["Modelo"].isin([x_train.iloc[eos,0]])].Lambda,MR_train[MR_train["Modelo"].isin([x_train.iloc[eos,0]])].M)

    plt.xlim(0,2000)
    plt.savefig(DATADIR+f"ML_{args.set_name}.pdf")

""" to check if dataset is being well created"""
M_max_tes=MR_test.groupby('Modelo').apply(lambda x: x.M.max())
grouped_MR_test = MR_test.groupby('Modelo')
interpolation_dict_test = {
    row_id: interpolate.interp1d(
        x=grouped_MR_test.get_group(row_id)['M'],
        y=grouped_MR_test.get_group(row_id)['R'],
        kind='linear')
    for row_id in test_ID
}

test_samples_pt=build_samples(test_ID[:10], stdM=0.1, stdR=0.3, rand=True,M_max=M_max_tes,interpolation_dict=interpolation_dict_test,
                        cov_mode="rho", rho_mode="per_obs_random",
                        rho_low=-0.5, rho_high=0.5)
# print('test_samples',test_samples[0].keys())
batch_test = make_batch_from_samples(test_samples_pt)  # one EOS worth of stars

x_test_sys, mask_test =collate_fn(batch_test)

M_max_gp = df.mmax
grouped_MR_test = MR_test.groupby('Modelo')
interpolation_dict_gp_tes = {
    row_id: interpolate.interp1d(
        x=grouped_MRL_N.get_group(row_id)['M'],
        y=grouped_MRL_N.get_group(row_id)['R'],
        kind='linear',
    )
    for row_id in test_IDgp
}
test_samples_gp=build_samples(test_IDgp[:10], stdM=0.1, stdR=0.3, rand=True,M_max=M_max_gp,interpolation_dict=interpolation_dict_gp_tes,
                        cov_mode="rho", rho_mode="per_obs_random",
                        rho_low=-0.5, rho_high=0.5)
# print('test_samples',test_samples[0].keys())
batch_test = make_batch_from_samples(test_samples_gp)  # one EOS worth of stars

x_test_sys_gp, mask_test_gp =collate_fn(batch_test)