"""
Main training script for the normalizing-flow model.

The data-loading and preprocessing are handled by data_gp_pt.py.
"""

import glob
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data_gp_pt import *
from model import model, flow_loss, encoder
from plots_ import save_eos_plot_two_datasets, plot_blops


# -------------------------------------------------------------------------
# Training setup
# -------------------------------------------------------------------------

time0 = time.time()

print("Data:", dataname)

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(encoder.parameters()),
    lr=args.learning_rate,
    weight_decay=1e-5,
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=800,
    gamma=0.1,
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------------
# Checkpoint loading
# -------------------------------------------------------------------------

S = 0
valid_loss_min = np.inf
epoch_0 = 1

if args.load_weights:
    paths = glob.glob(
        str(OUTPUT_DIR / f"saved_model_{args.set_name}_*.pth")
    )

    if not paths:
        raise FileNotFoundError(
            f"No saved model found for {args.set_name} in {OUTPUT_DIR}"
        )

    PATH = paths[0]
    print("Loading model from:", PATH)

    checkpoint = torch.load(
        PATH,
        map_location=torch.device("cpu"),
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model"])
    encoder.load_state_dict(checkpoint["encoder"])

    epoch_0 = checkpoint["epoch"] + 1

    print(f"Loaded model from {PATH}, starting at epoch {epoch_0}")


# -------------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------------

for epoch in range(epoch_0, args.num_epochs + epoch_0):

    model.train()
    encoder.train()

    # -------------------------------------------------------------
    # Build training and validation samples
    # -------------------------------------------------------------

    if both_datasets:

        train_gp = build_samples(
            train_ID_gp[:],
            0.1,
            0.3,
            rand=True,
            cov_mode="rho",
            rho_mode="per_obs_random",
            interpolation_dict=interpolation_dict_gp,
            M_max=M_max_gp,
            rho_low=-0.5,
            rho_high=0.5,
        )

        train_pt = build_samples(
            train_ID_[:],
            0.1,
            0.3,
            rand=True,
            cov_mode="rho",
            rho_mode="per_obs_random",
            interpolation_dict=interpolation_dict,
            M_max=M_max_pt,
            rho_low=-0.5,
            rho_high=0.5,
        )

        train_samples = train_pt + train_gp

        val_gp = build_samples(
            val_ID_gp,
            0.1,
            0.3,
            rand=True,
            cov_mode="rho",
            rho_mode="per_obs_random",
            interpolation_dict=interpolation_dict_gp,
            M_max=M_max_gp,
            rho_low=-0.5,
            rho_high=0.5,
        )

        val_pt = build_samples(
            val_ID_,
            0.1,
            0.3,
            rand=True,
            cov_mode="rho",
            rho_mode="per_obs_random",
            interpolation_dict=interpolation_dict,
            M_max=M_max_pt,
            rho_low=-0.5,
            rho_high=0.5,
        )

        val_samples = val_pt + val_gp

    else:

        train_samples = build_samples(
            train_ID_[:],
            0.1,
            0.3,
            rand=True,
            cov_mode="rho",
            rho_mode="per_obs_random",
            interpolation_dict=interpolation_dict,
            M_max=M_max_pt,
            rho_low=-0.5,
            rho_high=0.5,
        )

        val_samples = build_samples(
            val_ID_[:],
            0.1,
            0.3,
            rand=True,
            cov_mode="rho",
            rho_mode="per_obs_random",
            interpolation_dict=interpolation_dict,
            M_max=M_max_pt,
            rho_low=-0.5,
            rho_high=0.5,
        )

    print(
        "Train samples:",
        x_train_s[:].shape,
        len(train_samples),
        train_samples[0]["M"].shape,
        train_samples[1]["M"].shape,
        "| Validation samples:",
        len(val_samples),
    )

    # -------------------------------------------------------------
    # Create datasets and data loaders
    # -------------------------------------------------------------

    train_samples = make_batch_from_samples(train_samples)
    val_samples = make_batch_from_samples(val_samples)

    train_data, train_mask = collate_fn(train_samples)
    val_data, val_mask = collate_fn(val_samples)

    training_set = Dataset(
        x_train_s[:],
        train_data,
        train_mask,
    )

    validation_set = Dataset(
        x_val_s[:],
        val_data,
        val_mask,
    )

    train_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Validation data should not be shuffled.
    val_loader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # -------------------------------------------------------------
    # Training
    # -------------------------------------------------------------

    running_loss = 0.0
    penalty = 0.0

    for batch_idx, (parameters, data, mask) in enumerate(train_loader):

        optimizer.zero_grad()

        context = encoder(
            data.to(device),
            mask.to(device),
        )

        loss, penalty_batch = flow_loss(
            parameters.to(device),
            context,
            model,
            lambda_penalty=args.lambda_penalty,
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        penalty += penalty_batch.item()

    scheduler.step()

    # -------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------

    with torch.no_grad():

        model.eval()
        encoder.eval()

        val_loss = sum(
            flow_loss(
                parameters.to(device),
                encoder(
                    data.to(device),
                    mask.to(device),
                ),
                model,
                lambda_penalty=args.lambda_penalty,
            )[0].item()
            for parameters, data, mask in val_loader
        ) / len(val_loader)

    train_loss = running_loss / len(train_loader)
    mean_penalty = penalty / len(train_loader)

    print(
        f"Epoch {epoch}: "
        f"loss={train_loss:.6f}, "
        f"val_loss={val_loss:.6f}, "
        f"penalty={mean_penalty:.6f}, "
        f"LR={optimizer.param_groups[0]['lr']:.2e}"
    )

    # -------------------------------------------------------------
    # Save loss history
    # -------------------------------------------------------------

    loss_file = OUTPUT_DIR / f"loss_{args.set_name}.csv"

    pd.DataFrame(
        [[epoch, train_loss, val_loss, mean_penalty]],
        columns=["epoch", "loss", "val_loss", "penalty"],
    ).to_csv(
        loss_file,
        mode="a",
        index=False,
        header=not loss_file.exists(),
    )

    # -------------------------------------------------------------
    # Save model if validation loss improves
    # -------------------------------------------------------------

    if epoch > 50:

        network_learned = val_loss < valid_loss_min

        if network_learned:

            print(
                f"Validation Loss Decreased "
                f"({valid_loss_min:.6f} ---> {val_loss:.6f}) "
                f"\t Saving the model"
            )

            if S != 0:
                os.remove(
                    OUTPUT_DIR / f"saved_model_{args.set_name}_{S}.pth"
                )

            S = epoch
            valid_loss_min = val_loss

            checkpoint_path = (
                OUTPUT_DIR / f"saved_model_{args.set_name}_{epoch}.pth"
            )

            checkpoint = {
                "model": model.state_dict(),
                "encoder": encoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "valid_loss_min": valid_loss_min,
            }

            torch.save(checkpoint, checkpoint_path)

    # -------------------------------------------------------------
    # Save training/validation loss plot
    # -------------------------------------------------------------

    if epoch % 10 == 0:

        plt.figure(figsize=(6, 4))

        df_loss = pd.read_csv(loss_file)

        plt.plot(
            df_loss["loss"],
            label="Train Loss",
            color="blue",
        )

        plt.plot(
            df_loss["val_loss"],
            label="Val Loss",
            color="orange",
        )

        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.title("Training / Validation Loss", fontsize=16)
        plt.grid(ls="dotted")
        plt.legend()

        plt.savefig(
            OUTPUT_DIR / f"loss_curve_{args.set_name}.pdf",
            bbox_inches="tight",
            dpi=200,
        )

        plt.close()

    # -------------------------------------------------------------
    # Save EOS plots
    # -------------------------------------------------------------

    if (epoch - 1) % 50 == 0:

        save_eos_plot_two_datasets(
            epoch,
            model,
            encoder,
            x_test_sys,
            mask_test,
            x_test_sys_gp,
            mask_test_gp,
            pt_eos_index=9,
            gp_eos_index=9,
            true_pressure_pt=x_test,
            true_pressure_gp=x_test_gp,
            n=n,
            dataname=args.set_name,
            outdir=OUTPUT_DIR,
            device=device,
        )

        plot_blops(
            test_samples_gp[9],
            MRL_N[
                MRL_N["ID"].isin([test_samples_gp[9]["row_id"]])
            ],
            dataname=args.set_name + "_gp",
            outdir=OUTPUT_DIR,
        )

        plot_blops(
            test_samples_pt[9],
            MR_test[
                MR_test["Modelo"].isin([x_test.index[9]])
            ],
            dataname=args.set_name + "_pt",
            outdir=OUTPUT_DIR,
        )


# -------------------------------------------------------------------------
# Save final run information
# -------------------------------------------------------------------------

params["time"] = (time.time() - time0) / 60

print(f"Training time: {params['time']:.2f} min")

with open(param_path, "wb") as f:
    pickle.dump(params, f)