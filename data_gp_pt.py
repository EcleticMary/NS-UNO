import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import interpolate

from config import *


# ---------------------------------------------------------------------------
# Data loading and reproducibility
# ---------------------------------------------------------------------------

dataname = "polytropic"

with open(param_path, "w") as data:
    data.write(str(args))

torch.manual_seed(args.seed)
np.random.seed(args.seed)


# ---------------------------------------------------------------------------
# Sampling functions
# ---------------------------------------------------------------------------

def sample_masses_stratified(N, rng, Mmin=1.0, Mmax=2.1):
    bins = [
        (Mmin, 1.4),
        (1.4, 1.8),
        (1.8, Mmax),
    ]

    nbins = 3
    counts = np.ones(nbins, dtype=int)
    counts += rng.multinomial(N - nbins, [1 / 3, 1 / 3, 1 / 3])

    masses = []

    for count, (a, b) in zip(counts, bins):
        masses.append(rng.uniform(a, b, size=(count, 1)))

    masses = np.concatenate(masses, axis=0)

    # rng.shuffle(masses, axis=0)

    return masses


def build_samples(
    ID_list,
    stdM,
    stdR,
    M_max,
    interpolation_dict,
    sigma_mode=args.sigma_mode,
    rand=False,
    N_min=args.Nmin,
    N_max=args.Nmax,
    N_default=args.Ndefault,
    changing_N=args.changing_N,
    n_draws=args.Nsamples,
    # --- Covariance controls ---
    cov_mode="rho",          # "none" | "rho"
    rho_mode="fixed",        # "fixed" | "global_random" | "per_obs_random" | "normal"
    rho=0.0,                 # Used if rho_mode="fixed"
    rho_low=-1,              # Used if rho_mode is random
    rho_high=1,
    rho_mean=0.3,             # Used if rho_mode="normal"
    rho_sd=0.1,
    rho_clip=0.95,
    seed=None,
):
    """
    Build samples with optional correlated (M_obs, R_obs) measurement noise.

    The covariance is defined as:
        Cov(M, R) = rho * sigma_M * sigma_R

    with -1 <= rho <= 1.
    """

    if M_max is None or interpolation_dict is None:
        raise ValueError("You must provide M_max and interpolation_dict.")

    rng = np.random.default_rng(seed)

    N_range = rng.integers(
        N_min,
        N_max + 1,
        size=(len(ID_list), 1),
    )

    samples = []

    for i, row_id in enumerate(ID_list):

        # Pick a random number of observations
        N = N_range[i][0] if changing_N else N_default

        M = sample_masses_stratified(
            N=N,
            rng=rng,
            Mmin=1.0,
            Mmax=float(M_max.loc[row_id]),
        )

        R = interpolation_dict[row_id](M)

        if not rand:
            samples.append(
                dict(
                    row_id=row_id,
                    N=N,
                    M=M,
                    R=R,
                )
            )
            continue

        # --- Measurement uncertainties ---
        if sigma_mode == "per_obs":
            print("per_obs")

            std_M = rng.uniform(
                0.05,
                stdM,
                size=(N, 1),
            )

            std_R = rng.uniform(
                0.1,
                stdR,
                size=(N, 1),
            )

        elif sigma_mode == "per_eos":
            sigma_M_eos = rng.uniform(0.05, stdM)
            sigma_R_eos = rng.uniform(0.1, stdR)

            std_M = np.full((N, 1), sigma_M_eos)
            std_R = np.full((N, 1), sigma_R_eos)

        # --- Correlation ---
        if cov_mode == "none":
            rho_i = np.zeros((N, 1))

        elif cov_mode == "rho":

            if rho_mode == "fixed":
                rho_i = np.full((N, 1), float(rho))

            elif rho_mode == "global_random":
                rho_g = rng.uniform(rho_low, rho_high)
                rho_i = np.full((N, 1), float(rho_g))

            elif rho_mode == "per_obs_random":
                rho_i = rng.uniform(
                    rho_low,
                    rho_high,
                    size=(N, 1),
                )

            elif rho_mode == "normal":
                rho_i = rng.normal(
                    rho_mean,
                    rho_sd,
                    size=(N, 1),
                )

            else:
                raise ValueError(
                    f"Unknown rho_mode={rho_mode!r}"
                )

        else:
            raise ValueError(
                f"Unknown cov_mode={cov_mode!r}"
            )

        # Keep |rho| <= rho_clip
        rho_i = np.clip(
            rho_i,
            -float(rho_clip),
            float(rho_clip),
        )

        # --- Correlated sampling using Cholesky decomposition ---
        # Sigma_i = [[sigma_M^2, rho*sigma_M*sigma_R],
        #            [rho*sigma_M*sigma_R, sigma_R^2]]

        sigmaM = std_M
        sigmaR = std_R
        covMR = rho_i * sigmaM * sigmaR

        # Cholesky factors:
        # a = sigma_M
        # b = cov / a
        # c = sqrt(sigma_R^2 - b^2)

        a = sigmaM
        b = covMR / (a + 1e-12)
        c_sq = sigmaR**2 - b**2
        c = np.sqrt(np.maximum(c_sq, 0.0))

        # Standard normal samples
        z1 = rng.normal(size=(N, n_draws))
        z2 = rng.normal(size=(N, n_draws))

        # Generate correlated observations
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


# ---------------------------------------------------------------------------
# Batch preparation
# ---------------------------------------------------------------------------

def collate_fn(batch):
    B = len(batch)
    S_max = max(len(eos) for eos in batch)
    N_max = max(
        x.shape[0]
        for eos in batch
        for x in eos
    )

    d_in = 2

    x_padded = torch.zeros(
        B,
        S_max,
        N_max,
        d_in,
    )

    mask = torch.zeros(
        B,
        S_max,
        N_max,
        dtype=torch.bool,
    )

    for i, eos in enumerate(batch):
        for j, star in enumerate(eos):
            n = star.shape[0]

            x_padded[i, j, :n] = star
            mask[i, j, :n] = True

    return x_padded, mask


class Dataset(torch.utils.data.Dataset):
    """Custom Dataset for PyTorch."""

    def __init__(self, eos, tov, mask):
        self.eos = eos
        self.tov = tov
        self.mask = mask

    def __len__(self):
        return self.eos.shape[0]

    def __getitem__(self, index):
        return (
            self.eos[index, :],
            self.tov[index, :],
            self.mask[index, :],
        )


def make_batch_from_samples(samples):
    """
    Convert a list of per-star sample dictionaries into
    HierarchicalEncoder input.
    """

    batch = []

    for sample in samples:
        x_star = np.concatenate(
            [
                sample["M_obs"][:, :, np.newaxis],
                sample["R_obs"][:, :, np.newaxis],
            ],
            axis=2,
        )

        x_star = torch.tensor(
            x_star,
            dtype=torch.float32,
        )

        batch.append(x_star)

    return batch


# ===========================================================================
# POLYTROPIC DATASET
# ===========================================================================

DF_ = pd.read_csv(DATA_POLY_DIR / "DF_final.csv")
EOS_ = pd.read_csv(DATA_POLY_DIR / "EOS_final.csv")
MR_ = pd.read_csv(DATA_POLY_DIR / "MR_final.csv")

n = np.linspace(0.13, 1.28, 20)

eos = EOS_.groupby("Modelo").apply(
    lambda x: interpolate.interp1d(
        x=x.n,
        y=x.p,
        kind="cubic",
    )(n)
)

P_pt = pd.DataFrame(
    np.array([eos.iloc[j] for j in range(eos.shape[0])]),
    columns=n,
)

P_pt["Modelo"] = eos.index

train_ID, test_ID = train_test_split(
    DF_.Modelo,
    test_size=0.1,
    random_state=100,
)

MR_train = MR_[MR_.Modelo.isin(train_ID)].copy()
MR_test = MR_[MR_.Modelo.isin(test_ID)].copy()

MR_train["Modelo"] = MR_train["Modelo"].astype(int).apply(
    lambda x: f"pt_{str(x)}"
)

MR_test["Modelo"] = MR_test["Modelo"].astype(int).apply(
    lambda x: f"pt_{str(x)}"
)

p_train = P_pt[P_pt.Modelo.isin(train_ID)].iloc[:, :-1]
p_test = P_pt[P_pt.Modelo.isin(test_ID)].iloc[:, :-1]

p_train.index = [
    f"{i}" for i in MR_train["Modelo"].unique()
]

p_test.index = [
    f"{i}" for i in MR_test["Modelo"].unique()
]

grouped_MR_train = MR_train.groupby("Modelo")

test_ID = [
    "pt_" + str(test_ID.iloc[j])
    for j in range(test_ID.shape[0])
]


# Precompute interpolation dictionaries
interpolation_dict = {
    row_id: interpolate.interp1d(
        x=grouped_MR_train.get_group(row_id)["M"],
        y=grouped_MR_train.get_group(row_id)["R"],
        kind="linear",
    )
    for row_id in p_train.index.unique()
}

interpolation_dict2 = {
    row_id: interpolate.interp1d(
        x=grouped_MR_train.get_group(row_id)["M"],
        y=grouped_MR_train.get_group(row_id)["Lambda"],
        kind="linear",
    )
    for row_id in p_train.index.unique()
}

M_max_pt = MR_train.groupby("Modelo").apply(
    lambda x: x.M.max()
)


train_ID_, val_ID_ = train_test_split(
    np.unique(p_train.index.values),
    test_size=0.1,
    random_state=11,
)

x_train = p_train.loc[train_ID_]
x_val = p_train.loc[val_ID_]
x_test = p_test.loc[test_ID]


# ===========================================================================
# GP DATASET
# ===========================================================================

with open(DATA_GORDA_DIR / "EoS_ensemble.pickle", "rb") as f:
    df, n_gp = pickle.load(f)


# --- Data preprocessing ---

R = pd.DataFrame(
    np.array([df.r[j] for j in range(120000)])
)

df = df[~R[R < 5].any(axis=1)]
df = df[df.mmax >= 2]
df.reset_index(drop=True, inplace=True)

R = pd.DataFrame(
    np.array([df.r.iloc[j] for j in range(df.shape[0])])
)

M = pd.DataFrame(
    np.array([df.m.iloc[j] for j in range(df.shape[0])])
)

# Convert pressure from GeV to MeV
P_gp = pd.DataFrame(
    np.array([df.p.iloc[j] for j in range(df.shape[0])])
) * 10**3

L = pd.DataFrame(
    np.array([df.L.iloc[j] for j in range(df.shape[0])])
)


M.dropna(axis=1, inplace=True)
R.dropna(axis=1, inplace=True)


# Create pressure interpolator
# This needs to be done before filtering because VS does not
# preserve the same ID ordering as the dataframe index.

interpolator = interpolate.interp1d(
    x=n_gp * 0.16,
    y=P_gp,
    kind="linear",
    fill_value="extrapolate",
)

p = interpolator(n)


# --- Data transformation for M, R, and L ---

# Keep values up to the maximum mass of each curve.
M_2 = M.where(M.diff(axis=1) > 0)
M_2.iloc[:, 0] = M.iloc[:, 0]

M_N = (
    M_2.stack()
    .to_frame()
    .reset_index()
    .drop("level_1", axis=1)
    .rename(columns={"level_0": "ID", 0: "M"})
)

R_N = (
    R.where(M_2.notna())
    .stack()
    .to_frame()
    .reset_index()
    .drop("level_1", axis=1)
    .rename(columns={"level_0": "ID", 0: "R"})
)

L_N = (
    L.where(M_2.notna())
    .stack()
    .to_frame()
    .reset_index()
    .drop("level_1", axis=1)
    .rename(columns={"level_0": "ID", 0: "L"})
)


GRo = L_N.groupby("ID").apply(
    lambda x: (x.L.diff() > 0).any()
)

invalid_IDs = np.where(GRo == True)[0]

R_N = R_N[~R_N.isin({"ID": invalid_IDs})].dropna()
M_N = M_N[~M_N.isin({"ID": invalid_IDs})].dropna()
L_N = L_N[~L_N.isin({"ID": invalid_IDs})].dropna()

M_N["ID"] = M_N["ID"].astype(int).apply(
    lambda x: f"gp_{str(x)}"
)

R_N["ID"] = R_N["ID"].astype(int).apply(
    lambda x: f"gp_{str(x)}"
)

P_gp.index = [
    f"gp_{i}" for i in range(len(P_gp))
]

P_gp = P_gp.loc[
    M_N.ID.drop_duplicates().values
]


# --- GP train/test split ---

train_IDgp, test_IDgp = train_test_split(
    P_gp.index,
    test_size=0.1,
    random_state=11,
)

MRL_N = pd.concat(
    [M_N, R_N.R, L_N.L],
    axis=1,
)

grouped_MRL_N = MRL_N.groupby("ID")


interpolation_dict2_gp = {
    row_id: interpolate.interp1d(
        x=grouped_MRL_N.get_group(row_id)["M"],
        y=grouped_MRL_N.get_group(row_id)["L"],
        kind="linear",
    )
    for row_id in train_IDgp
}


interpolation_dict_gp = {
    row_id: interpolate.interp1d(
        x=grouped_MRL_N.get_group(row_id)["M"],
        y=grouped_MRL_N.get_group(row_id)["R"],
        kind="linear",
    )
    for row_id in train_IDgp
}


# Remove models with negative interpolated pressure.
# This needs to happen before updating the pressure dataframe
# so that the original model IDs are preserved.

negative_pressure_IDs = pd.DataFrame(p).iloc[
    np.where(pd.DataFrame(p) < 0)[0]
].index

train_IDgp = train_IDgp[
    ~train_IDgp.isin(negative_pressure_IDs)
]

p = pd.DataFrame(p).drop(negative_pressure_IDs)

p.index = [
    f"gp_{i}" for i in range(len(p))
]


# --- GP train/validation split ---

train_ID_gp, val_ID_gp = train_test_split(
    train_IDgp,
    test_size=0.1,
    random_state=11,
)

x_train_gp = p.loc[train_ID_gp]
x_val_gp = p.loc[val_ID_gp]
x_test_gp = p.loc[test_IDgp]

x_train_gp.columns = n
x_test_gp.columns = n
x_val_gp.columns = n


df = df.loc[np.where(GRo == False)[0]]
df.index = MRL_N.ID.unique()

M_max_gp = df.mmax


# ===========================================================================
# COMBINE DATASETS
# ===========================================================================

both_datasets = args.both_datasets

if both_datasets:
    X_train = pd.concat((x_train, x_train_gp))
    X_val = pd.concat((x_val, x_val_gp))
    X_test = pd.concat((x_test, x_test_gp))

else:
    # For now, use only the polytropic dataset.
    # This can be changed to use only GP data if needed.
    X_train = x_train
    X_val = x_val
    X_test = x_test


# --- Transform pressure ---

x_train_s = np.log10(X_train.astype("float32"))
x_val_s = np.log10(X_val.astype("float32"))

x_train_s = pd.DataFrame(
    np.repeat(x_train_s, args.no, axis=0)
).values

x_val_s = pd.DataFrame(
    np.repeat(x_val_s, args.no, axis=0)
).values


# ===========================================================================
# DATASET CHECK
# ===========================================================================

fig, ax = plt.subplots(
    figsize=(5, 4),
    ncols=1,
    nrows=1,
    sharex=True,
    sharey="row",
    layout="constrained",
)

plt.plot(n, x_train_s[0], "o")
plt.plot(n, x_train_s[1], "o")

plt.savefig(
    OUTPUT_DIR / f"p_{args.set_name}.pdf"
)


# ===========================================================================
# TEST SAMPLES — POLYTROPIC
# ===========================================================================

M_max_test_pt = MR_test.groupby("Modelo").apply(
    lambda x: x.M.max()
)

grouped_MR_test = MR_test.groupby("Modelo")

interpolation_dict_test = {
    row_id: interpolate.interp1d(
        x=grouped_MR_test.get_group(row_id)["M"],
        y=grouped_MR_test.get_group(row_id)["R"],
        kind="linear",
    )
    for row_id in test_ID
}

test_samples_pt = build_samples(
    test_ID[:10],
    stdM=0.1,
    stdR=0.3,
    rand=True,
    M_max=M_max_test_pt,
    interpolation_dict=interpolation_dict_test,
    cov_mode="rho",
    rho_mode="per_obs_random",
    rho_low=-0.5,
    rho_high=0.5,
)

batch_test = make_batch_from_samples(test_samples_pt)

x_test_sys, mask_test = collate_fn(batch_test)


# ===========================================================================
# TEST SAMPLES — GP
# ===========================================================================

interpolation_dict_gp_test = {
    row_id: interpolate.interp1d(
        x=grouped_MRL_N.get_group(row_id)["M"],
        y=grouped_MRL_N.get_group(row_id)["R"],
        kind="linear",
    )
    for row_id in test_IDgp
}

test_samples_gp = build_samples(
    test_IDgp[:10],
    stdM=0.1,
    stdR=0.3,
    rand=True,
    M_max=M_max_gp,
    interpolation_dict=interpolation_dict_gp_test,
    cov_mode="rho",
    rho_mode="per_obs_random",
    rho_low=-0.5,
    rho_high=0.5,
)

batch_test = make_batch_from_samples(test_samples_gp)

x_test_sys_gp, mask_test_gp = collate_fn(batch_test)