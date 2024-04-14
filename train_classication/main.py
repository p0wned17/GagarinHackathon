import argparse
import os
import os.path as osp
import random
import sys

import numpy as np
import torch
import utils
import wandb

from accelerate import Accelerator
from dataloader import get_dataloaders


from model import ClassificationModel
from tqdm import tqdm
from train import train, validation, test

from accelerate import DistributedDataParallelKwargs
from loss import FocalLossWithSmoothing

wandb.init(project="gagarin_hack_angle")


def main(args: argparse.Namespace) -> None:
    """
    Run train process of classification model
    :param args: all parameters necessary for launch
    :return: None
    """

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision="fp16", kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    seed = 228
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    outdir = osp.join("./experiments", "angle")
    print("Savedir: {}".format(outdir))
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    os.chmod(outdir, 0o777)

    print("Load model...")
    model = ClassificationModel(num_classes=4)

    model.to(device, memory_format=torch.channels_last)

    print("Prepare training params...")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=7e-5,
        weight_decay=0.011,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=40,
        eta_min=1e-9,
        last_epoch=-1,
    )
    train_dataloader, val_dataloader = get_dataloaders()
    (
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        scheduler,
    ) = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        scheduler,
    )

    print("Done.")

    train_epoch = tqdm(range(40), dynamic_ncols=True, desc="Epochs", position=0)

    best_f1 = 0
    criterion = FocalLossWithSmoothing(num_classes=4, gamma=1.5, lb_smooth=0.1)

    for epoch in train_epoch:
        train(
            model,
            criterion,
            accelerator,
            train_dataloader,
            optimizer,
            epoch,
        )

        # if accelerator.is_main_process:
        precision, recall, f1, val_loss = validation(
            model,
            criterion,
            val_dataloader,
            epoch,
        )

        wandb.log(
            {"f1_score": f1, "precision": precision, "recall": recall, "loss": val_loss}
        )

        print(
            {"f1_score": f1, "precision": precision, "recall": recall, "loss": val_loss}
        )

        saved_model = accelerator.unwrap_model(model)
        if f1 > best_f1:
            best_f1 = f1
            epoch_avg_acc = f"{f1:.4f}"

            utils.save_checkpoint(
                saved_model,
                optimizer,
                scheduler,
                epoch,
                outdir,
                epoch_avg_acc,
            )

        scheduler.step()

    best_model_path = f"{outdir}/best.pt"
    checkpoint = torch.load(
        best_model_path,
        map_location="cpu",
    )["state_dict"]
    model.load_state_dict(checkpoint)

    test(model, val_dataloader)
    wandb.finish()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    # parser.add_argument("--cfg", type=str, help="Path to config file.")
    # parser.add_argument("--chkp", type=str, default=None,
    #                     help="Path to checkpoint file.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
