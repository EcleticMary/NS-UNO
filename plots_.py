
import torch
import argparse
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def save_eos_plot_two_datasets(
    epoch,
    model,
    encoder,
    x_test_sys_pt,
    mask_test_pt,
    x_test_sys_gp,
    mask_test_gp,
    pt_eos_index,
    gp_eos_index,true_pressure_pt,true_pressure_gp,
    n,
    dataname,
    outdir,device,
    n_samples=1000
):
    """
    Save a figure with one EOS example from the polytropic dataset
    and one EOS example from the GP dataset.

    Parameters
    ----------
    x_test_sys_pt, mask_test_pt : tensors
        Inputs for encoder for polytropic test set.
    x_test_sys_gp, mask_test_gp : tensors
        Inputs for encoder for GP test set.
    pt_eos_index, gp_eos_index : int
        Which example to plot from each test set.
    test_ID_pt, test_ID_gp : list
        IDs aligned with x_test_sys_pt and x_test_sys_gp.
    interpolation_dict_pt, interpolation_dict_gp : dict
        Maps EOS ID -> interpolation function.
    n : array
        Density grid.
    """

    model.eval()
    encoder.eval()

    with torch.no_grad():
        # Contexts for each dataset
        ctx_test_pt = encoder(x_test_sys_pt.to(device), mask_test_pt.to(device))
        ctx_test_gp = encoder(x_test_sys_gp.to(device), mask_test_gp.to(device))

        # Sample predictive distributions
        pred_samples_pt = model.sample(n_samples, ctx_test_pt[[pt_eos_index]]).cpu().numpy()[0]
        pred_samples_gp = model.sample(n_samples, ctx_test_gp[[gp_eos_index]]).cpu().numpy()[0]

        # Undo log10 transform
        pred_samples_pt = 10 ** pred_samples_pt
        pred_samples_gp = 10 ** pred_samples_gp

    # # True curves from interpolation dictionaries
    # pt_id = test_ID_pt[pt_eos_index]
    # gp_id = test_ID_gp[gp_eos_index]

    # true_pressure_pt = interpolation_dict_pt[pt_id](n)
    # true_pressure_gp = interpolation_dict_gp[gp_id](n)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    color_pred = "#EE9A00"

    # -------- Polytropic --------
    ax = axes[0]
    ax.fill_between(
        n,
        np.percentile(pred_samples_pt, 5, axis=0),
        np.percentile(pred_samples_pt, 95, axis=0),
        color=color_pred,
        alpha=0.2,
        label="90% pred. band"
    )
    ax.plot(
        n,
        np.percentile(pred_samples_pt, 50, axis=0),
        ls="--",
        lw=2,
        color=color_pred,
        label="Pred median"
    )
    ax.plot(n, true_pressure_pt.iloc[pt_eos_index], lw=2, color="black", label="True EOS")
    ax.set_yscale("log")
    ax.set_xlabel("n [fm$^{-3}$]", fontsize=14)
    ax.set_ylabel("p [MeV/fm$^3$]", fontsize=14)
    # ax.set_title(f"Polytropic | ID = {pt_id}", fontsize=14)
    ax.grid(ls="dotted")
    ax.legend()
    ax.set_title(f"PT",fontsize=12)
    # -------- GP --------
    ax = axes[1]
    ax.fill_between(
        n,
        np.percentile(pred_samples_gp, 5, axis=0),
        np.percentile(pred_samples_gp, 95, axis=0),
        color=color_pred,
        alpha=0.2,
        label="90% pred. band"
    )
    ax.plot(
        n,
        np.percentile(pred_samples_gp, 50, axis=0),
        ls="--",
        lw=2,
        color=color_pred,
        label="Pred median"
    )
    ax.plot(n, true_pressure_gp.iloc[gp_eos_index], lw=2, color="black", label="True EOS")
    ax.set_yscale("log")
    ax.set_xlabel("n [fm$^{-3}$]", fontsize=14)
    # ax.set_title(f"GP | ID = {gp_id}", fontsize=14)
    ax.grid(ls="dotted")
    ax.legend()
    ax.set_title(f"GP",fontsize=12)
    fig.suptitle(f"EOS predictions at epoch {epoch}", fontsize=16)
    plt.tight_layout()
    
    plt.title(f"{dataname}",fontsize=12)
    plt.savefig(
        os.path.join(outdir, f"eos_pred_epoch{epoch}_{dataname}.pdf"),
        bbox_inches="tight",
        dpi=200
    )
    plt.close()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
def plot_covariance_ellipse(
    mean,
    cov,
    ax,
    n_std=2.0,
    edgecolor="red",
    facecolor="none",
    linewidth=2,
):
    """
    Plot a covariance ellipse for a 2D Gaussian.

    mean : (2,) array
    cov  : (2,2) covariance matrix
    n_std: number of standard deviations (1, 2, 3)
    """
    # eigenvalues & eigenvectors
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # angle of ellipse (degrees)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # width and height = 2 * n_std * sqrt(eigenvalues)
    width, height = 2 * n_std * np.sqrt(vals)

    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=theta,
        edgecolor=edgecolor,
        facecolor=facecolor,
        linewidth=linewidth,
    )
    ax.add_patch(ellipse)
# ax shape is (3, 6)
# global font size control
LABEL_FS = 14
TITLE_FS = 12
TICK_FS  = 11


def plot_blops(s,dfm,outdir,dataname):
    fig, ax = plt.subplots(1,1, figsize=(4, 4), sharex=True, sharey=True)
    # scatter + ellipse for each observation
    for i in range(s["N"]):
        plt.scatter(
            s["R_obs"][i],
            s["M_obs"][i],
            s=5,
            alpha=0.6,
        )

        cov = np.array([
            [s["std_M"][i, 0]**2,
                s["rho"][i, 0] * s["std_M"][i, 0] * s["std_R"][i, 0]],
            [s["rho"][i, 0] * s["std_M"][i, 0] * s["std_R"][i, 0],
                s["std_R"][i, 0]**2],
        ])

        plot_covariance_ellipse(
            mean=[s["R"][i, 0], s["M"][i, 0]],
            cov=cov[[1, 0]][:, [1, 0]],
            ax=plt.gca(),
            n_std=2,
            edgecolor="black",
            linewidth=1.5,
        )


    
    plt.plot(dfm.R, dfm.M)
    plt.xlim(11,14)
    plt.tight_layout()
    plt.title(f"{dataname}",fontsize=12)
    plt.savefig(
        os.path.join(outdir, f"MR__{dataname}.pdf"),
        bbox_inches="tight",
        dpi=200
    )
    plt.close()

