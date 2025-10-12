# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import os
import sys
import copy

import traceback
from pathlib import Path

import sgad.utils.distributed_mode as distributed_mode
import hydra
import numpy as np

import pytorch_warmup as warmup
import torch
import torch.backends.cudnn as cudnn

import torch_geometric
from torch_geometric.loader import DataLoader as TGDataLoader

from sgad.components.clipper import Clipper

# from sgad.components.datasets import create_t1x_dataset
from sgad.utils.t1x_dataloader import T1xTGDataloader
from sgad.components.sample_buffer import BatchBuffer
from sgad.components.sampler import (
    populate_buffer_with_samples_and_energy_gradients,
)
from sgad.components.sde import (
    ControlledGraphSDE,
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
            controller,
            noise_schedule,
            use_adjointmatching_sde=cfg.use_adjointmatching_sde,
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

        # Create eval dataset from T1x
        if cfg.overfit_single_sample:
            dataloader = T1xTGDataloader(
                hdf5_file=cfg.dataset.datapath,
                datasplit="train",
                indices=None,
                which="ts",
            )
            eval_dataset = [next(dataloader)] * cfg.batch_size
        else:
            dataloader = T1xTGDataloader(
                hdf5_file=cfg.dataset.datapath,
                datasplit="val",
                indices=None,
                which="ts",
            )
            eval_dataset = [
                molecule
                for i, molecule in enumerate(dataloader)
                if i < cfg.num_eval_samples
            ]
        # eval only on main process
        eval_sample_loader = TGDataLoader(
            eval_dataset, batch_size=cfg.batch_size, shuffle=False
        )

        if cfg.overfit_single_sample:
            train_dataset = [copy.deepcopy(_) for _ in eval_dataset]
        else:
            train_dataset = [
                molecule
                for i, molecule in enumerate(dataloader)
                if i < cfg.num_samples_per_epoch
            ]
        train_sample_loader = torch_geometric.loader.DataLoader(
            dataset=train_dataset,
            batch_size=1,
            sampler=torch.utils.data.DistributedSampler(
                train_dataset,
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
            n_batches = (
                n_init_batches if (epoch == start_epoch) else n_batches_per_epoch
            )
            controlled = not (epoch == start_epoch)
            # if we are resuming training, should we reinitialize the buffer randomly like this?
            if epoch < cfg.pretrain_epochs:
                mode = "pretrain"
                # during pretraining, we use T1x molecular structures directly
                buffer.add(
                    *populate_buffer_with_samples_and_energy_gradients(
                        energy_model=energy_model,
                        sample_loader=train_sample_loader,
                        sde=sde,
                        n_batches=n_batches,
                        batch_size=cfg.batch_size,
                        device=device,
                        duplicates=cfg.duplicates,
                        nfe=cfg.train_nfe,
                        controlled=False,  # Use T1x structures directly without SDE
                        discretization_scheme=cfg.discretization_scheme,
                    )
                )
            else:
                # during adjoint sampling, we use the SDE to generate training data
                mode = "adjoint sampling"
                buffer.add(
                    *populate_buffer_with_samples_and_energy_gradients(
                        energy_model=energy_model,
                        sample_loader=train_sample_loader,
                        sde=sde,
                        n_batches=n_batches,
                        batch_size=cfg.batch_size,
                        device=device,
                        duplicates=cfg.duplicates,
                        nfe=cfg.train_nfe,
                        controlled=controlled,
                        discretization_scheme=cfg.discretization_scheme,
                    )
                )
            train_dataloader = buffer.get_data_loader(cfg.num_batches_per_epoch)

            train_dict = train_one_epoch(
                controller=controller,
                noise_schedule=noise_schedule,
                clipper=clipper,
                train_dataloader=train_dataloader,
                optimizer=optimizer,
                warmup_scheduler=warmup_scheduler,
                lr_schedule=lr_schedule,
                device=device,
                cfg=cfg,
                pretrain_mode=(epoch < cfg.pretrain_epochs),
            )
            if epoch % cfg.eval_freq == 0 or epoch == cfg.num_epochs - 1:
                if distributed_mode.is_main_process():
                    try:
                        eval_dict = evaluation(
                            sde=sde,
                            energy_model=energy_model,
                            eval_sample_loader=eval_sample_loader,
                            noise_schedule=noise_schedule,
                            atomic_numbers=energy_model.atomic_numbers,
                            rank=global_rank,
                            device=device,
                            cfg=cfg,
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
                        pbar.set_description(
                            "mode: {}, train loss: {:.2f}, eval soc loss: {:.2f}, TS ratio: {:.2f} ({}/{})".format(
                                mode,
                                train_dict["loss"],
                                eval_dict["soc_loss"],
                                eval_dict["transition_state_ratio"],
                                eval_dict["num_transition_states"],
                                len(eval_dict["frequency_analyses"]),
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
