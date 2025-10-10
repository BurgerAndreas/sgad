# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import os
import sys

import traceback
from pathlib import Path

import sgad.utils.distributed_mode as distributed_mode
import hydra
import numpy as np

import pytorch_warmup as warmup
import torch
import torch.backends.cudnn as cudnn

import torch_geometric
from sgad.components.clipper import Clipper, Clipper1d

from sgad.components.datasets import get_homogeneous_dataset
from sgad.components.sample_buffer import BatchBuffer
from sgad.components.sampler import (
    populate_buffer_from_loader,
    populate_buffer_from_loader_rdkit,
)
from sgad.components.sde import (
    ControlledGraphSDE,
    ControlledGraphTorsionSDE,
)
from sgad.eval_loop import evaluation
from sgad.train_loop import train_one_epoch

from omegaconf import OmegaConf
from tqdm import tqdm


cudnn.benchmark = True


@hydra.main(config_path="configs", config_name="train.yaml", version_base="1.1")
def main(cfg):
    try:
        print("Found {} CUDA devices.".format(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(
                "{} \t Memory: {:.2f}GB".format(
                    props.name, props.total_memory / (1024**3)
                )
            )

        print(dict(os.environ))
        distributed_mode.init_distributed_mode(cfg)

        print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
        print(str(cfg))
        if distributed_mode.is_main_process():
            args_filepath = Path("cfg.yaml")
            print(f"Saving cfg to {args_filepath}")
            with open("config.yaml", "w") as fout:
                print(OmegaConf.to_yaml(cfg), file=fout)
            with open("env.json", "w") as fout:
                print(json.dumps(dict(os.environ)), file=fout)

        device = cfg.device  # "cuda"

        # fix the seed for reproducibility
        seed = cfg.seed + distributed_mode.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)

        print("Initializing buffer")
        buffer = BatchBuffer(cfg.buffer_size)

        print("Initializing model")
        noise_schedule = hydra.utils.instantiate(cfg.noise_schedule)
        energy_model = hydra.utils.instantiate(cfg.energy)(
            tau=cfg.tau, alpha=cfg.alpha, device=device
        )

        # # THIS MUST BE DONE AFTER LOADING THE ENERGY MODEL!!
        # if cfg.learn_torsions:
        #     torch.set_default_dtype(torch.float64)

        controller = hydra.utils.instantiate(cfg.controller)
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Check for existing checkpoints
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
        start_epoch = 0

        # checkpoint_path = str(Path(os.getcwd()).parent.parent.parent / "2025.01.09" / "024615" / "0" / "checkpoints" / "checkpoint_latest.pt")
        checkpoint = None
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            # map_location = {"cuda:%d" % 0: "cuda:%d" % distributed_mode.get_rank()}
            checkpoint = torch.load(checkpoint_path)  # , map_location=map_location)
            controller.load_state_dict(checkpoint["controller_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
        else:
            if cfg.init_model is not None:
                print(f"Loading initial weights from {cfg.init_model}")
                checkpoint = torch.load(cfg.init_model)
                controller.load_state_dict(
                    torch.load(cfg.init_model, weights_only=False)[
                        "controller_state_dict"
                    ]
                )

        # Note: Not wrapping this in a DDP since we don't differentiate through SDE simulation.
        sde = ControlledGraphSDE(
            controller, noise_schedule, use_AM_SDE=cfg.use_AM_SDE
        ).to(device)

        if cfg.distributed:
            controller = torch.nn.parallel.DistributedDataParallel(
                controller, device_ids=[cfg.gpu], find_unused_parameters=True
            )

        lr_schedule = None
        optimizer = torch.optim.Adam(list(sde.parameters()), lr=cfg.lr)
        if checkpoint is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        warmup_period = (
            cfg.warmup_period * cfg.num_batches_per_epoch
            if cfg.warmup_period > 0
            else 1
        )
        warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period)
        print("Initializing data loader")

        world_size = distributed_mode.get_world_size()
        global_rank = distributed_mode.get_rank()

        eval_sample_dataset = get_homogeneous_dataset(
            cfg.eval_smiles,
            energy_model,
            duplicate=cfg.num_eval_samples,
        )
        # eval only on main process
        eval_sample_loader = torch_geometric.loader.DataLoader(
            dataset=eval_sample_dataset,
            batch_size=cfg.batch_size,
        )

        train_sample_dataset = hydra.utils.instantiate(cfg.dataset)(
            energy_model=energy_model,
            duplicate=(1 if cfg.amortized else world_size),
        )

        train_sample_loader = torch_geometric.loader.DataLoader(
            dataset=train_sample_dataset,
            batch_size=1,
            sampler=torch.utils.data.DistributedSampler(
                train_sample_dataset,
                num_replicas=world_size,
                rank=global_rank,
                shuffle=True,
            ),
        )

        n_init_batches = int(cfg.num_init_samples // cfg.num_samples_per_epoch)
        n_batches_per_epoch = int(cfg.num_samples_per_epoch // cfg.batch_size)
        clipper = Clipper(cfg.clip_scores, cfg.max_score_norm)

        print(f"Starting from {cfg.start_epoch}/{cfg.num_epochs} epochs")
        pbar = tqdm(range(start_epoch, cfg.num_epochs))
        for epoch in pbar:
            if (
                epoch == start_epoch
            ):  # should we reinitialize buffer randomly like this if resuming?
                if cfg.pretrain_epochs > 0:
                    buffer.add(
                        *populate_buffer_from_loader_rdkit(
                            energy_model,
                            train_sample_loader,
                            sde,
                            n_init_batches,
                            cfg.batch_size,
                            device,
                            duplicates=cfg.duplicates,
                        ),
                    )
                else:
                    buffer.add(
                        *populate_buffer_from_loader(
                            energy_model,
                            train_sample_loader,
                            sde,
                            n_init_batches,
                            cfg.batch_size,
                            device,
                            duplicates=cfg.duplicates,
                            nfe=cfg.train_nfe,
                            controlled=False,
                            discretization_scheme=cfg.discretization_scheme,
                        )
                    )
            else:
                if epoch < cfg.pretrain_epochs:
                    buffer.add(
                        *populate_buffer_from_loader_rdkit(
                            energy_model,
                            train_sample_loader,
                            sde,
                            n_batches_per_epoch,
                            cfg.batch_size,
                            device,
                            duplicates=cfg.duplicates,
                        ),
                    )
                else:
                    buffer.add(
                        *populate_buffer_from_loader(
                            energy_model,
                            train_sample_loader,
                            sde,
                            n_batches_per_epoch,
                            cfg.batch_size,
                            device,
                            duplicates=cfg.duplicates,
                            nfe=cfg.train_nfe,
                            discretization_scheme=cfg.discretization_scheme,
                        )
                    )
            train_dataloader = buffer.get_data_loader(cfg.num_batches_per_epoch)

            train_dict = train_one_epoch(
                controller,
                noise_schedule,
                clipper,
                train_dataloader,
                optimizer,
                warmup_scheduler,
                lr_schedule,
                device,
                cfg,
                pretrain_mode=(epoch < cfg.pretrain_epochs),
            )
            if epoch % cfg.eval_freq == 0 or epoch == cfg.num_epochs - 1:
                if distributed_mode.is_main_process():
                    try:
                        eval_dict = evaluation(
                            sde,
                            energy_model,
                            eval_sample_loader,
                            noise_schedule,
                            energy_model.atomic_numbers,
                            global_rank,
                            device,
                            cfg,
                        )
                        eval_dict["energy_vis"].save("test_im.png")
                        print("saving checkpoint ... ")
                        if cfg.distributed:
                            state = {
                                "controller_state_dict": controller.module.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "epoch": epoch,
                            }
                        else:
                            state = {
                                "controller_state_dict": controller.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "epoch": epoch,
                            }
                        torch.save(state, "checkpoints/checkpoint_{}.pt".format(epoch))
                        torch.save(state, "checkpoints/checkpoint_latest.pt")
                        mode = (
                            "pretrain"
                            if epoch < cfg.pretrain_epochs
                            else "adjoint sampling"
                        )
                        pbar.set_description(
                            "mode: {}, train loss: {:.2f}, eval soc loss: {:.2f}".format(
                                mode, train_dict["loss"], eval_dict["soc_loss"]
                            )
                        )
                    except Exception as e:  # noqa: F841
                        # Log exception but don't stop training.
                        print(traceback.format_exc())
                        print(traceback.format_exc(), file=sys.stderr)

    except Exception as e:
        # This way we have the full traceback in the log.  otherwise Hydra
        # will handle the exception and store only the error in a pkl file
        print(traceback.format_exc())
        print(traceback.format_exc(), file=sys.stderr)
        raise e


if __name__ == "__main__":
    main()
