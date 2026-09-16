import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


# ============================================================================
# EOS prediction plots
# ============================================================================

def save_eos_plot_two_datasets(
    epoch,
    model,
    encoder,
    x_test_sys_pt,
    mask_test_pt,
    x_test_sys_gp,
    mask_test_gp,
    pt_eos_index,
    gp_eos_index,
    true_pressure_pt,
    true_pressure_gp,
    n,
    dataname,
    outdir,
    device,
    n_samples=1000,
):
    """
    Save EOS predictions for one example from the PT dataset
    and one example from the GP dataset.

    Parameters
    ----------
    epoch : int
        Current training epoch.

    model : torch.nn.Module
        Trained normalizing-flow model.

    encoder : torch.nn.Module
        Observation encoder.

    x_test_sys_pt, mask_test_pt : torch.Tensor
        PT test-set inputs for the encoder.

    x_test_sys_gp, mask_test_gp : torch.Tensor
        GP test-set inputs for the encoder.

    pt_eos_index, gp_eos_index : int
        Indices of the EOS examples to plot.

    true_pressure_pt, true_pressure_gp : array-like
        True pressure curves.

    n : array-like
        Density grid.

    dataname : str
        Name used for the output file.

    outdir : str or Path
        Output directory.

    device : torch.device
        Device used for inference.

    n_samples : int, optional
        Number of samples drawn from the posterior.
    """

    model.eval()
    encoder.eval()

    with torch.no_grad():

        # Encode test-set observations.
        context_pt = encoder(
            x_test_sys_pt.to(device),
            mask_test_pt.to(device),
        )

        context_gp = encoder(
            x_test_sys_gp.to(device),
            mask_test_gp.to(device),
        )

        # Sample from the posterior.
        pred_samples_pt = (
            model.sample(
                n_samples,
                context_pt[[pt_eos_index]],
            )
            .cpu()
            .numpy()[0]
        )

        pred_samples_gp = (
            model.sample(
                n_samples,
                context_gp[[gp_eos_index]],
            )
            .cpu()
            .numpy()[0]
        )

        # Undo the log10 pressure transformation.
        pred_samples_pt = 10 ** pred_samples_pt
        pred_samples_gp = 10 ** pred_samples_gp

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        sharey=True,
    )

    color_pred = "#EE9A00"

    # ------------------------------------------------------------------
    # Polytropic dataset
    # ------------------------------------------------------------------

    ax = axes[0]

    ax.fill_between(
        n,
        np.percentile(pred_samples_pt, 5, axis=0),
        np.percentile(pred_samples_pt, 95, axis=0),
        color=color_pred,
        alpha=0.2,
        label="90% pred. band",
    )

    ax.plot(
        n,
        np.percentile(pred_samples_pt, 50, axis=0),
        ls="--",
        lw=2,
        color=color_pred,
        label="Pred median",
    )

    ax.plot(
        n,
        true_pressure_pt.iloc[pt_eos_index],
        lw=2,
        color="black",
        label="True EOS",
    )

    ax.set_yscale("log")
    ax.set_xlabel("n [fm$^{-3}$]", fontsize=14)
    ax.set_ylabel("p [MeV/fm$^3$]", fontsize=14)
    ax.set_title("PT", fontsize=12)
    ax.grid(ls="dotted")
    ax.legend()

    # ------------------------------------------------------------------
    # Gaussian-process dataset
    # ------------------------------------------------------------------

    ax = axes[1]

    ax.fill_between(
        n,
        np.percentile(pred_samples_gp, 5, axis=0),
        np.percentile(pred_samples_gp, 95, axis=0),
        color=color_pred,
        alpha=0.2,
        label="90% pred. band",
    )

    ax.plot(
        n,
        np.percentile(pred_samples_gp, 50, axis=0),
        ls="--",
        lw=2,
        color=color_pred,
        label="Pred median",
    )

    ax.plot(
        n,
        true_pressure_gp.iloc[gp_eos_index],
        lw=2,
        color="black",
        label="True EOS",
    )

    ax.set_yscale("log")
    ax.set_xlabel("n [fm$^{-3}$]", fontsize=14)
    ax.set_title("GP", fontsize=12)
    ax.grid(ls="dotted")
    ax.legend()

    fig.suptitle(
        f"EOS predictions at epoch {epoch}",
        fontsize=16,
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            outdir,
            f"eos_pred_epoch{epoch}_{dataname}.pdf",
        ),
        bbox_inches="tight",
        dpi=200,
    )

    plt.close()


# ============================================================================
# Covariance ellipse
# ============================================================================

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

    Parameters
    ----------
    mean : array-like, shape (2,)
        Mean of the Gaussian.

    cov : array-like, shape (2, 2)
        Covariance matrix.

    ax : matplotlib.axes.Axes
        Axis on which to draw the ellipse.

    n_std : float, optional
        Number of standard deviations represented by the ellipse.
    """

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    order = eigenvalues.argsort()[::-1]

    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    theta = np.degrees(
        np.arctan2(
            *eigenvectors[:, 0][::-1]
        )
    )

    width, height = (
        2 * n_std * np.sqrt(eigenvalues)
    )

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


# ============================================================================
# Mass-radius observations
# ============================================================================

def plot_blops(
    s,
    dfm,
    outdir,
    dataname,
):
    """
    Plot mass-radius observations with their covariance ellipses.
    """

    fig, ax = plt.subplots(
        figsize=(4, 4),
    )

    for i in range(s["N"]):

        ax.scatter(
            s["R_obs"][i],
            s["M_obs"][i],
            s=5,
            alpha=0.6,
        )

        cov = np.array(
            [
                [
                    s["std_M"][i, 0] ** 2,
                    s["rho"][i, 0]
                    * s["std_M"][i, 0]
                    * s["std_R"][i, 0],
                ],
                [
                    s["rho"][i, 0]
                    * s["std_M"][i, 0]
                    * s["std_R"][i, 0],
                    s["std_R"][i, 0] ** 2,
                ],
            ]
        )

        plot_covariance_ellipse(
            mean=[
                s["R"][i, 0],
                s["M"][i, 0],
            ],
            cov=cov[[1, 0]][:, [1, 0]],
            ax=ax,
            n_std=2,
            edgecolor="black",
            linewidth=1.5,
        )

    ax.plot(
        dfm.R,
        dfm.M,
    )

    ax.set_xlim(11, 14)
    ax.set_title(
        dataname,
        fontsize=12,
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            outdir,
            f"MR__{dataname}.pdf",
        ),
        bbox_inches="tight",
        dpi=200,
    )

    plt.close()