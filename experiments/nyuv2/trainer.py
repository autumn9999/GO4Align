import logging
import wandb
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange

from experiments.nyuv2.data import NYUv2
from experiments.nyuv2.models import SegNet, SegNetMtan
from experiments.nyuv2.utils import ConfMatrix, delta_fn, depth_error, normal_error
from experiments.utils import (
    common_parser,
    extract_weight_method_parameters_from_args,
    get_device,
    set_logger,
    set_seed,
    str2bool,
    list_of_float,
)
from methods.weight_methods import WeightMethods

#-----------------
import pdb
import os
import time
import sys
#-----------------

set_logger()

#-----------------
def log_string(file_out, out_str, print_out=True):
    file_out.write(out_str+'\n')
    file_out.flush()
    if print_out:
        print(out_str)
# -----------------

def calc_loss(x_pred, x_output, task_type):
    device = x_pred.device

    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type == "semantic":
        # semantic loss: depth-wise cross entropy
        loss = F.nll_loss(x_pred, x_output, ignore_index=-1)

    if task_type == "depth":
        # depth loss: l1 norm
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(
            binary_mask, as_tuple=False
        ).size(0)

    if task_type == "normal":
        # normal loss: dot product
        loss = 1 - torch.sum((x_pred * x_output) * binary_mask) / torch.nonzero(
            binary_mask, as_tuple=False
        ).size(0)

    return loss


def main(args, file_out, file_weight_out, file_loss_before, file_loss_after, path, lr, bs, device):
    # ----
    # Nets
    # ---
    model = dict(segnet=SegNet(), mtan=SegNetMtan())[args.model]
    model = model.to(device)

    # dataset and dataloaders
    log_str = (
        "Applying data augmentation on NYUv2."
        if args.apply_augmentation
        else "Standard training strategy without data augmentation."
    )
    logging.info(log_str)

    nyuv2_train_set = NYUv2(
        root=path.as_posix(), train=True, augmentation=args.apply_augmentation
    )
    nyuv2_test_set = NYUv2(root=path.as_posix(), train=False)

    train_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_train_set, batch_size=bs, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_test_set, batch_size=bs, shuffle=False
    )

    # weight method
    weight_methods_parameters = extract_weight_method_parameters_from_args(args)
    weight_method = WeightMethods(args.method, n_tasks=3, device=device, **weight_methods_parameters[args.method])

    # optimizer
    optimizer = torch.optim.Adam(
        [
            dict(params=model.parameters(), lr=lr),
            dict(params=weight_method.parameters(), lr=args.method_params_lr),
        ],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    epochs = args.n_epochs
    epoch_iter = trange(epochs)
    train_batch = len(train_loader)
    test_batch = len(test_loader)
    avg_cost = np.zeros([epochs, 24], dtype=np.float32)
    # -----------------
    avg_overall_performance = np.zeros([epochs, 2], dtype=np.float32)
    # -----------------
    custom_step = -1
    conf_mat = ConfMatrix(model.segnet.class_nb)

    # training
    for epoch in epoch_iter:
        # ----
        start = time.time()
        # ----
        cost = np.zeros(24, dtype=np.float32)

        for j, batch in enumerate(train_loader):
            #-----------------
            # pdb.set_trace()
            if args.debugging:
                if j == 2:
                    break
            # -----------------
            custom_step += 1
            model.train()
            optimizer.zero_grad()

            train_data, train_label, train_depth, train_normal = batch
            train_data, train_label = train_data.to(device), train_label.long().to(
                device
            )
            train_depth, train_normal = train_depth.to(device), train_normal.to(device)

            train_pred, features = model(train_data, return_representation=True)

            losses = torch.stack(
                (
                    calc_loss(train_pred[0], train_label, "semantic"),
                    calc_loss(train_pred[1], train_depth, "depth"),
                    calc_loss(train_pred[2], train_normal, "normal"),
                )
            )

            # pdb.set_trace()
            loss, extra_outputs = weight_method.backward(
                losses=losses,
                shared_parameters=list(model.shared_parameters()),
                task_specific_parameters=list(model.task_specific_parameters()),
                last_shared_parameters=list(model.last_shared_parameters()),
                representation=features,
            )
            # pdb.set_trace()

            optimizer.step()

            #----------updated from FAMO
            if "famo" in args.method:
                with torch.no_grad():
                    train_pred = model(train_data, return_representation=False)
                    new_losses = torch.stack(
                        (
                            calc_loss(train_pred[0], train_label, "semantic"),
                            calc_loss(train_pred[1], train_depth, "depth"),
                            calc_loss(train_pred[2], train_normal, "normal"),
                        )
                    )
                    weight_method.method.update(new_losses.detach())
            #----------

            # accumulate label prediction for every pixel in training images
            conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())

            cost[0] = losses[0].item()
            cost[3] = losses[1].item()
            cost[4], cost[5] = depth_error(train_pred[1], train_depth)
            cost[6] = losses[2].item()
            cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(train_pred[2], train_normal)
            avg_cost[epoch, :12] += cost[:12] / train_batch

            epoch_iter.set_description(
                f"[{epoch+1}  {j+1}/{train_batch}] semantic loss: {losses[0].item():.3f}, "
                f"depth loss: {losses[1].item():.3f}, "
                f"normal loss: {losses[2].item():.3f}"
            )

        # scheduler
        scheduler.step()

        # compute mIoU and acc
        avg_cost[epoch, 1:3] = conf_mat.get_metrics()

        # todo: move evaluate to function?
        # evaluating test data
        model.eval()
        conf_mat = ConfMatrix(model.segnet.class_nb)
        with torch.no_grad():  # operations inside don't track history
            test_dataset = iter(test_loader)
            for k in range(test_batch):
                # -----------------
                # pdb.set_trace()
                if args.debugging:
                    if k == 2:
                        break
                # -----------------
                test_data, test_label, test_depth, test_normal = next(test_dataset) #test_dataset.next()#.next is deprecated
                test_data, test_label = test_data.to(device), test_label.long().to(
                    device
                )
                test_depth, test_normal = test_depth.to(device), test_normal.to(device)

                test_pred = model(test_data)
                test_loss = torch.stack(
                    (
                        calc_loss(test_pred[0], test_label, "semantic"),
                        calc_loss(test_pred[1], test_depth, "depth"),
                        calc_loss(test_pred[2], test_normal, "normal"),
                    )
                )

                conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())

                cost[12] = test_loss[0].item()
                cost[15] = test_loss[1].item()
                cost[16], cost[17] = depth_error(test_pred[1], test_depth)
                cost[18] = test_loss[2].item()
                cost[19], cost[20], cost[21], cost[22], cost[23] = normal_error(
                    test_pred[2], test_normal
                )
                avg_cost[epoch, 12:] += cost[12:] / test_batch
            # compute mIoU and acc
            avg_cost[epoch, 13:15] = conf_mat.get_metrics()

            # Test Delta_m
            test_delta_m = delta_fn(
                avg_cost[epoch, [13, 14, 16, 17, 19, 20, 21, 22, 23]]
            )

            # -----------------
            avg_overall_performance[epoch, 0] = test_delta_m
            avg_overall_performance[epoch, 1] = avg_overall_performance[:epoch+1, 0].min()
            # -----------------

            # print results
            # print(
            #     f"LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR "
            #     f"| NORMAL_LOSS MEAN MED <11.25 <22.5 <30 | ∆m (test)"
            # )
            # print("")

            # ----
            end = time.time()
            log_string(file_out, 'Epoch: {:04d} | Epoch time {:.4f}'.format(epoch, end - start))
            # if args.method in {"famo"}:
            log_string(file_weight_out, 'Epoch {:04d} | {}'.format(epoch, extra_outputs["weights"]), False)
            log_string(file_loss_before, 'Epoch {:04d} | {}'.format(epoch, losses.detach().cpu()), False)
            log_string(file_loss_after, 'Epoch {:04d} | {}'.format(epoch, loss.detach().cpu()), False)
            # ----

            log_string(file_out, f"Epoch: {epoch:04d} | TRAIN: "
                f"{avg_cost[epoch, 0]:.4f} {avg_cost[epoch, 1]*100:.2f} {avg_cost[epoch, 2]*100:.2f} | "
                f"{avg_cost[epoch, 3]:.4f} {avg_cost[epoch, 4]:.4f} {avg_cost[epoch, 5]:.4f} | "
                f"{avg_cost[epoch, 6]:.4f} {avg_cost[epoch, 7]:.2f} {avg_cost[epoch, 8]:.2f} {avg_cost[epoch, 9]*100:.2f} {avg_cost[epoch, 10]*100:.2f} {avg_cost[epoch, 11]*100:.2f} || "
                f"TEST: "
                f"{avg_cost[epoch, 12]:.4f} {avg_cost[epoch, 13]*100:.2f} {avg_cost[epoch, 14]*100:.2f} | "
                f"{avg_cost[epoch, 15]:.4f} {avg_cost[epoch, 16]:.4f} {avg_cost[epoch, 17]:.4f} | "
                f"{avg_cost[epoch, 18]:.4f} {avg_cost[epoch, 19]:.2f} {avg_cost[epoch, 20]:.2f} {avg_cost[epoch, 21]*100:.2f} {avg_cost[epoch, 22]*100:.2f} {avg_cost[epoch, 23]*100:.2f} "
                f"| {test_delta_m:.3f} {avg_overall_performance[epoch, -1]:.3f}"
            )

            if wandb.run is not None:
                wandb.log({"Train Semantic Loss": avg_cost[epoch, 0]}, step=epoch)
                wandb.log({"Train Mean IoU": avg_cost[epoch, 1]}, step=epoch)
                wandb.log({"Train Pixel Accuracy": avg_cost[epoch, 2]}, step=epoch)
                wandb.log({"Train Depth Loss": avg_cost[epoch, 3]}, step=epoch)
                wandb.log({"Train Absolute Error": avg_cost[epoch, 4]}, step=epoch)
                wandb.log({"Train Relative Error": avg_cost[epoch, 5]}, step=epoch)
                wandb.log({"Train Normal Loss": avg_cost[epoch, 6]}, step=epoch)
                wandb.log({"Train Loss Mean": avg_cost[epoch, 7]}, step=epoch)
                wandb.log({"Train Loss Med": avg_cost[epoch, 8]}, step=epoch)
                wandb.log({"Train Loss <11.25": avg_cost[epoch, 9]}, step=epoch)
                wandb.log({"Train Loss <22.5": avg_cost[epoch, 10]}, step=epoch)
                wandb.log({"Train Loss <30": avg_cost[epoch, 11]}, step=epoch)

                wandb.log({"Test Semantic Loss": avg_cost[epoch, 12]}, step=epoch)
                wandb.log({"Test Mean IoU": avg_cost[epoch, 13]}, step=epoch)
                wandb.log({"Test Pixel Accuracy": avg_cost[epoch, 14]}, step=epoch)
                wandb.log({"Test Depth Loss": avg_cost[epoch, 15]}, step=epoch)
                wandb.log({"Test Absolute Error": avg_cost[epoch, 16]}, step=epoch)
                wandb.log({"Test Relative Error": avg_cost[epoch, 17]}, step=epoch)
                wandb.log({"Test Normal Loss": avg_cost[epoch, 18]}, step=epoch)
                wandb.log({"Test Loss Mean": avg_cost[epoch, 19]}, step=epoch)
                wandb.log({"Test Loss Med": avg_cost[epoch, 20]}, step=epoch)
                wandb.log({"Test Loss <11.25": avg_cost[epoch, 21]}, step=epoch)
                wandb.log({"Test Loss <22.5": avg_cost[epoch, 22]}, step=epoch)
                wandb.log({"Test Loss <30": avg_cost[epoch, 23]}, step=epoch)
                wandb.log({"Test ∆m": test_delta_m}, step=epoch)

                # #-------------------------------------------
                # wandb.log({"Weight_seg": extra_outputs["weights"][0].item()}, step=epoch)
                # wandb.log({"Weight_depth": extra_outputs["weights"][1].item()}, step=epoch)
                # wandb.log({"Weight_normal": extra_outputs["weights"][2].item()}, step=epoch)
                # # -------------------------------------------

    # -----------------
    # final output
    log_string(file_out, f"FORMAT: MEAN_IOU PIX_ACC | ABS_ERR REL_ERR | MEAN MED <11.25 <22.5 <30 | ∆m (test)")
    log_string(file_out,
        f"Last_10_TEST: "
        f"{avg_cost[-10:, 13].mean()*100:.2f} {avg_cost[-10:, 14].mean()*100:.2f} "
        f"{avg_cost[-10:, 16].mean():.4f} {avg_cost[-10:, 17].mean():.4f} "
        f"{avg_cost[-10:, 19].mean():.2f} {avg_cost[-10:, 20].mean():.2f} {avg_cost[-10:, 21].mean()*100:.2f} {avg_cost[-10:, 22].mean()*100:.2f} {avg_cost[-10:, 23].mean()*100:.2f} "
        f"{avg_overall_performance[-10:, 0].mean():.3f}"
    )
    # -----------------

    # pdb.set_trace()

if __name__ == "__main__":
    parser = ArgumentParser("NYUv2", parents=[common_parser])
    parser.set_defaults(
        data_path = "../../../../../../../dataset/SIMO/nyuv2",
        lr=1e-4,
        n_epochs=200,
        batch_size=2,
    )

    parser.add_argument("--model", type=str, default="mtan", choices=["segnet", "mtan"], help="model type")
    parser.add_argument("--apply-augmentation", type=str2bool, default=True, help="data augmentations")
    # parser.add_argument("--wandb_project", type=str, default="nashmtl_nyuv2", help="Name of Weights & Biases Project.")
    # parser.add_argument("--wandb_entity", type=str, default="jia-yi9999", help="Name of Weights & Biases Entity.")
    parser.add_argument("--wandb_project", type=str, default=None, help="Name of Weights & Biases Project.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Name of Weights & Biases Entity.")
    # -----------------
    parser.add_argument('--debugging', action='store_true', help='with debugging')
    parser.add_argument('--log_name', default="logs_segnet_mtan/log", type=str, help='log name')
    parser.add_argument('--robust_step_size', default=0.0001, type=float, help='for our method')
    parser.add_argument('--task_weights', default="0.1,0.1,0.8", type=list_of_float, help='for group')
    parser.add_argument('--num_groups', default=1, type=int, help='number of groups')
    # -----------------
    args = parser.parse_args()

    # set seed
    set_seed(args.seed)
    #-----------------
    if args.method == "go4align":
        log_name = args.log_name + '_Method={}_Ngroups={}_Seed={}'.format(args.method, args.num_groups, args.seed)
    elif args.method == "group":
        log_name = args.log_name + '_Method={}_{}_{}_{}_Seed={}'.format(args.method, args.task_weights[0], args.task_weights[1], args.task_weights[2], args.seed)
    else:
        log_name = args.log_name + '_Method={}_Seed={}'.format(args.method, args.seed)

    os.system("mkdir -p " + log_name)
    file_out = open(log_name + "/train_log.txt", "w")
    file_weight_out = open(log_name + "/weight_log.txt", "w")
    file_loss_before = open(log_name + "/loss_before.txt", "w")
    file_loss_after = open(log_name+ "/loss_after.txt", "w")

    os.system("mkdir -p " + log_name + '/files')
    os.system('cp %s %s' % ('*.py', os.path.join(log_name, 'files')))
    print(get_device(gpus=args.gpu))
    #--------------------

    if args.wandb_project is not None:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=log_name, config=args)

    device = get_device(gpus=args.gpu)
    main(args, file_out, file_weight_out, file_loss_before, file_loss_after, path=args.data_path, lr=args.lr, bs=args.batch_size, device=device)

    if wandb.run is not None:
        wandb.finish()

    # -----------------
    st = ' '
    log_string(file_out, st.join(sys.argv))
    file_out.close()
    #--------------------