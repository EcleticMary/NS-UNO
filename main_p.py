
"""
When using the polytropic dataset we use data_utils_p in file main_p.py and model.py if using GP
is data_utils_p2
"""

import time
time0 = time.time()
from data_gp_pt import *
from model_2 import model, flow_loss, encoder
from plots_ import save_eos_plot_two_datasets,plot_blops
print('data is i main : ',dataname)

# Optimization of the normalizing flow

# We have to choose the optimization method and parameters to optimize
# optimizer = torch.optim.Adam(model.parameters()+encoder.parameters(), lr=args.learning_rate,weight_decay=1e-5)
optimizer = torch.optim.Adam(list(model.parameters()) + list(encoder.parameters()), lr=args.learning_rate,weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=800,   # every 1000 epochs
    gamma=0.1        # multiply LR by 0.5
)
os.makedirs(OUTPUT_DIR, exist_ok=True)




S=0
valid_loss_min = np.inf
epoch_0=1

if args.load_weights:
    paths = glob.glob(str(OUTPUT_DIR / f"saved_model_{args.set_name}_*.pth"))
    # PATH=glob.glob(f"./model_test/saved_model_{args.set_name}_*.pth")[0]
    if not paths:
        raise FileNotFoundError(f"No saved model found for {args.set_name} in {OUTPUT_DIR}")

    PATH = paths[0]
    print("PATH of load_weights", PATH)

    checkpoint = torch.load(PATH, map_location=torch.device("cpu"), weights_only=True)
    model.load_state_dict(checkpoint["model"])
    encoder.load_state_dict(checkpoint["encoder"])
    epoch_0 = checkpoint["epoch"] + epoch_0

    print(f"Loaded model from {PATH}, starting at epoch {epoch_0}")


for epoch in range(epoch_0,   args.num_epochs + epoch_0):
    model.train()
    encoder.train()
    # d=dataset_creation(train_ID_[:], val_ID_[:])
    if both_datasets:
        d_gp=build_samples(train_ID_gp[:],0.1,0.3, rand=True, cov_mode="rho", rho_mode="per_obs_random",interpolation_dict=interpolation_dict_gp, M_max=M_max_gp,
                            rho_low=-0.5, rho_high=0.5)
        d_pt=build_samples(train_ID_[:],0.1,0.3, rand=True, cov_mode="rho", rho_mode="per_obs_random",interpolation_dict=interpolation_dict, M_max=M_max_pt,
                            rho_low=-0.5, rho_high=0.5)
        train_samples=d_pt+d_gp
        d_gp=build_samples(val_ID_gp,0.1,0.3, rand=True, cov_mode="rho", rho_mode="per_obs_random",interpolation_dict=interpolation_dict_gp, M_max=M_max_gp,
                            rho_low=-0.5, rho_high=0.5)
        d_pt=build_samples(val_ID_,0.1,0.3, rand=True, cov_mode="rho", rho_mode="per_obs_random",interpolation_dict=interpolation_dict, M_max=M_max_pt,rho_low=-0.5, rho_high=0.5)
        
        val_samples=d_pt+d_gp
    else:
        train_samples=build_samples(train_ID_[:],0.1,0.3, rand=True, cov_mode="rho", rho_mode="per_obs_random",interpolation_dict=interpolation_dict, M_max=M_max_pt,
                            rho_low=-0.5, rho_high=0.5)
        val_samples=build_samples(val_ID_[:],0.1,0.3, rand=True, cov_mode="rho", rho_mode="per_obs_random",interpolation_dict=interpolation_dict, M_max=M_max_pt,
                            rho_low=-0.5, rho_high=0.5)
    print('train_samples',x_train_s[:].shape,len(train_samples),train_samples[0]['M'].shape,train_samples[1]['M'].shape,'val_samples',len(val_samples))
    train_samples = make_batch_from_samples(train_samples)
    val_samples   = make_batch_from_samples(val_samples)
    training_set = Dataset(x_train_s[:],collate_fn(train_samples)[0],collate_fn(train_samples)[1])
    validation_set = Dataset(x_val_s[:],collate_fn(val_samples)[0],collate_fn(val_samples)[1])
    
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
# aqui é aconselhado a não se fazer um shuffle da validação
    val_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
    penalty=0.0
    running_loss = 0.0
    for batch_idx, (paramtes,data,mask) in enumerate(train_loader):
    #   print('params',torch.isnan(data.to(device)).any(),torch.isnan(data.to(device)).any())
    #   print('batch_idx',batch_idx,'data',paramtes,'data',data)
      optimizer.zero_grad()
      ctx = encoder(data.to(device), mask.to(device))   
    #   print('context',ctx,ctx.shape,paramtes.shape)
      loss,penalt = flow_loss(paramtes.to(device),ctx, model, lambda_penalty=args.lambda_penalty)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
      penalty += penalt.item()
    scheduler.step()

    with torch.no_grad():
      model.eval()
      encoder.eval()
      val_loss = sum( flow_loss(paramtes.to(device),encoder(data.to(device), mask.to(device)).to(device), model, lambda_penalty=args.lambda_penalty)[0].item() for paramtes, data,mask in val_loader ) / len(val_loader)
    # print('val_loss',val_loss)
    print(f"Epoch {epoch}: loss ",running_loss/len(train_loader), val_loss,f"loss_{args.set_name}.csv",penalty,"LR:", optimizer.param_groups[0]["lr"])
    # pd.DataFrame([[epoch,running_loss/len(train_loader), val_loss,penalty/len(train_loader)]], columns=['epoch','loss', 'val_loss','penalty']).to_csv(os.path.join(OUTPUT_DIR, f"loss_{args.set_name}.csv"), mode='a', index=False, header=(epoch == 1))
    loss_file = OUTPUT_DIR / f"loss_{args.set_name}.csv"
    pd.DataFrame(
        [[epoch, running_loss/len(train_loader), val_loss, penalty/len(train_loader)]],
        columns=["epoch", "loss", "val_loss", "penalty"]
    ).to_csv(loss_file, mode="a", index=False, header=not loss_file.exists())
    # Save model if validation loss decreases
    if epoch >50:
        network_learned = val_loss < valid_loss_min
        # if network_learned and valid_loss_min - batch_loss/len(val_loader) > 0.001:
        if network_learned:
            print(f'Validation Loss Decreased ({ valid_loss_min:.6f} ---> {val_loss:.6f}) \t Saving The Model')

            if S != 0:
                os.remove(os.path.join(OUTPUT_DIR, f"saved_model_{args.set_name}_{S}.pth"))
            S = epoch   
            valid_loss_min =  val_loss
            checkpoint_path = os.path.join(OUTPUT_DIR, f"saved_model_{args.set_name}_{epoch}.pth")
            checkpoint = {
                "model": model.state_dict(),
                "encoder": encoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "valid_loss_min": valid_loss_min,
            }
            torch.save(checkpoint, checkpoint_path)

            # torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"saved_model_{args.set_name}_{epoch}.pth"))
    # Print total training time
    # -------------------------
    # SAVE RUNNING LOSS PLOT
    # -------------------------
    if epoch % 10 == 0:  # Save every 10 epochs
        plt.figure(figsize=(6,4))
        df_loss = pd.read_csv(os.path.join(OUTPUT_DIR, f"loss_{args.set_name}.csv"))
        plt.plot( df_loss["loss"], label="Train Loss", color="blue")
        plt.plot( df_loss["val_loss"], label="Val Loss", color="orange")

        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.title("Training / Validation Loss", fontsize=16)
        plt.grid(ls="dotted")
        plt.legend()

        plt.savefig(os.path.join(OUTPUT_DIR, f"loss_curve_{args.set_name}.pdf"),
                    bbox_inches="tight", dpi=200)
        plt.close()
    # -------------------------
    # SAVE EOS PLOT (OPTIONAL)
    # -------------------------
    if (epoch-1) % 50  == 0:
        # Build test-set context (only ONCE outside loop if possible)
        ctx_test = encoder(x_test_sys.to(device), mask_test.to(device))

        # Save EOS plot using EOS index 9 (or choose any)
        save_eos_plot_two_datasets(
            epoch,
            model, encoder,
            x_test_sys, mask_test,
            x_test_sys_gp, mask_test_gp,pt_eos_index=9,gp_eos_index=9,true_pressure_pt=x_test,true_pressure_gp=x_test_gp,
            n=n, dataname=args.set_name,outdir=OUTPUT_DIR,device=device)
        plot_blops(test_samples_gp[9], MRL_N[MRL_N['ID'].isin([test_samples_gp[9]['row_id']])],dataname=args.set_name+"_gp",outdir=OUTPUT_DIR)
        plot_blops(test_samples_pt[9], MR_test[MR_test["Modelo"].isin([x_test.index[9]])],dataname=args.set_name+"_pt",outdir=OUTPUT_DIR)

params['time'] = (time.time() - time0) / 60
print(f'Training: {(time.time() - time0) / 60:.2f} min')
with open(param_path, "wb") as f:
    pickle.dump(params, f)
