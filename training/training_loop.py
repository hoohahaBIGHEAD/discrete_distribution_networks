# discrete_distribution_networks/training/training_loop.py

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

import boxx

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc

# [Added] WandB & Logging Imports
import wandb
from collections import defaultdict

# ----------------------------------------------------------------------------


def training_loop(
    run_dir=".",  # Output directory.
    dataset_kwargs={},  # Options for training set.
    data_loader_kwargs={},  # Options for torch.utils.data.DataLoader.
    network_kwargs={},  # Options for model and preconditioning.
    loss_kwargs={},  # Options for loss function.
    optimizer_kwargs={},  # Options for optimizer.
    augment_kwargs=None,  # Options for augmentation pipeline, None = disable.
    seed=0,  # Global random seed.
    batch_size=512,  # Total batch size for one training iteration.
    batch_gpu=None,  # Limit batch size per GPU, None = no limit.
    total_kimg=200000,  # Training duration, measured in thousands of training images.
    ema_halflife_kimg=500,  # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio=0.05,  # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg=10000,  # Learning rate ramp-up duration.
    loss_scaling=1,  # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick=50,  # Interval of progress prints.
    snapshot_ticks=50,  # How often to save network snapshots, None = disable.
    state_dump_ticks=500,  # How often to dump training state, None = disable.
    resume_pkl=None,  # Start from the given network snapshot, None = random initialization.
    resume_state_dump=None,  # Start from the given training state, None = reset training state.
    resume_kimg=0,  # Start from the given training progress.
    cudnn_benchmark=True,  # Enable torch.backends.cudnn.benchmark?
    device=torch.device("cuda"),
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0("Loading dataset...")
    dataset_obj = dnnlib.util.construct_class_by_name(
        **dataset_kwargs
    )  # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(
        dataset=dataset_obj,
        rank=dist.get_rank(),
        num_replicas=dist.get_world_size(),
        seed=seed,
    )
    dataset_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=dataset_obj,
            sampler=dataset_sampler,
            batch_size=batch_gpu,
            **data_loader_kwargs,
        )
    )

    # Construct network.
    dist.print0("Constructing network...")
    interface_kwargs = dict(
        img_resolution=dataset_obj.resolution,
        img_channels=dataset_obj.num_channels,
        label_dim=dataset_obj.label_dim,
    )
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs).to(
        device
    )  # subclass of torch.nn.Module

    # boxx.g()/0
    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros(
                [batch_gpu, net.img_channels, net.img_resolution, net.img_resolution],
                device=device,
            )
            sigma = torch.ones([batch_gpu], device=device)
            labels = torch.zeros([batch_gpu, net.label_dim], device=device)
            misc.print_module_summary(
                net.eval(), [images, sigma, labels], max_nesting=2
            )
    net.train().requires_grad_(True)
    # Setup optimizer.
    dist.print0("Setting up optimizer...")
    loss_fn = dnnlib.util.construct_class_by_name(
        **loss_kwargs
    )  # training.loss.(VP|VE|EDM)Loss
    optimizer = dnnlib.util.construct_class_by_name(
        params=net.parameters(), **optimizer_kwargs
    )  # subclass of torch.optim.Optimizer
    augment_pipe = (
        dnnlib.util.construct_class_by_name(**augment_kwargs)
        if augment_kwargs is not None
        else None
    )  # training.augment.AugmentPipe
    ddp = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[dist.get_rank()], broadcast_buffers=False
    )
    if ema_halflife_kimg:
        ema = copy.deepcopy(net).eval().requires_grad_(False)
    else:
        ema = net

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier()  # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier()  # other ranks follow
        misc.copy_params_and_buffers(
            src_module=data["ema"], dst_module=net, require_all=False
        )
        misc.copy_params_and_buffers(
            src_module=data["ema"], dst_module=ema, require_all=False
        )
        # print(net.model.block_8x8_2.ddo.sdd)
        # print(data["ema"].model.block_8x8_2.ddo.sdd)
        # boxx.g()/0
        del data  # conserve memory
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device("cpu"))
        misc.copy_params_and_buffers(
            src_module=data["net"], dst_module=net, require_all=True
        )
        optimizer.load_state_dict(data["optimizer_state"])
        del data  # conserve memory

    # Train.
    dist.print0(f"Training for {total_kimg} kimg...")
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None

    # [Added] WandB & Logging Initialization
    global_step = 0
    layer_idx_buffers = defaultdict(list)

    while True:
        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                images = images.to(device).to(torch.float32) / 127.5 - 1
                labels = labels.to(device)
                
                # [Modified] Unpack loss and dict
                loss, d = loss_fn(
                    net=ddp, images=images, labels=labels, augment_pipe=augment_pipe
                )
                
                training_stats.report("Loss/loss", loss)
                loss_ = loss.sum().mul(loss_scaling / batch_gpu_total)
                # TODO: loss_ up to 1044.4198
                loss_.backward()
                # boxx.g()/0

        # Update weights.
        for g in optimizer.param_groups:
            g["lr"] = optimizer_kwargs["lr"] * min(
                cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1
            )
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(
                    param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad
                )
        optimizer.step()
        
        # Try split all (SNP Control)
        from sddn import DiscreteDistributionOutput
        if getattr(DiscreteDistributionOutput, "use_snp", True):
            DiscreteDistributionOutput.try_split_all(optimizer)

        # [Added] WandB Logging (Rank 0 only)
        if dist.get_rank() == 0:
            log_dict = {
                # [Log] Match Author's 'loss' (scaled loss used for backward)
                "train/total_loss": loss_.item(), 
                # [Log] Match Author's 'mean' (per-pixel mean loss)
                "train/mean_pixel_loss": loss.sum().item() / images.numel(),
                "kimg": cur_nimg / 1000,
                "lr": optimizer.param_groups[0]['lr']
            }

            if isinstance(d, dict):
                # Component Losses
                if "log_attract" in d and d["log_attract"]: 
                    log_dict["train/loss_attract"] = torch.stack(d["log_attract"]).mean().item()
                if "log_ortho" in d and d["log_ortho"]: 
                    log_dict["train/loss_ortho"] = torch.stack(d["log_ortho"]).mean().item()
                if "log_weak" in d and d["log_weak"]: 
                    log_dict["train/loss_weak"] = torch.stack(d["log_weak"]).mean().item()

                # Collect Indices
                if "layer_idx_k" in d:
                    for layer_idx, idx_tensor in d["layer_idx_k"].items():
                        layer_idx_buffers[layer_idx].append(idx_tensor.flatten().detach().cpu())

            # Periodic Logging (Every 50 steps)
            if global_step % 50 == 0:
                for layer_idx, buffers in layer_idx_buffers.items():
                    if buffers:
                        all_indices = torch.cat(buffers).numpy()
                        # Assuming K=64 or similar, dynamic binning
                        max_val = int(all_indices.max())
                        bins = np.arange(max_val + 2)
                        counts = np.bincount(all_indices, minlength=max_val + 1)
                        
                        # 1. Histogram
                        np_hist = (counts, bins)
                        log_dict[f"layers/{layer_idx}/hist"] = wandb.Histogram(np_histogram=np_hist)
                        
                        # 2. Perplexity (Reset based on window)
                        total_count = counts.sum()
                        if total_count > 0:
                            probs = counts / total_count
                            probs = probs[probs > 0]
                            entropy = -np.sum(probs * np.log(probs))
                            perplexity = np.exp(entropy)
                        else:
                            perplexity = 0.0
                        
                        # log_dict[f"layers/{layer_idx}/entropy"] = entropy # (Disabled)
                        log_dict[f"layers/{layer_idx}/perplexity"] = perplexity

                # Clear buffers for next window
                layer_idx_buffers.clear()

                # Log Splits (Cumulative)
                for i, ddo in enumerate(DiscreteDistributionOutput.inits):
                    log_dict[f"layers/{i}/splits"] = len(ddo.sdd.split_iters)

            wandb.log(log_dict)
            global_step += 1

        # Update EMA.
        if ema_halflife_kimg:
            ema_halflife_nimg = ema_halflife_kimg * 1000
            if ema_rampup_ratio is not None:
                ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
            ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
            for p_ema, p_net in zip(ema.parameters(), net.parameters()):
                p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))
        # else:
        #     ema = ema.to("cpu")

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = cur_nimg >= total_kimg * 1000
        if (
            (not done)
            and (cur_tick != 0)
            and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000)
        ):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [
            f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):.1f}"
            + f"/{total_kimg}({round(cur_nimg / 1e3/total_kimg*100, 1)}%)\t"
        ]
        fields += [
            f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"
        ]
        fields += [
            f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"
        ]
        fields += [
            f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"
        ]
        fields += [
            f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"
        ]
        fields += [
            f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"
        ]
        fields += [
            f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"
        ]
        fields += [
            f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"
        ]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(
            boxx.cf.desc
            + ": "
            + " ".join(fields)
            + f" loss {round(loss_.tolist(),3)}"
            + f"/mean {boxx.strnum(loss.sum().tolist()/images.numel())}"
        )

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0("Aborting...")

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            torch.distributed.barrier()
            # if not ema_halflife_kimg:
            #     # TODO 适配和打开 EMA 让 split 支持 EMA
            #     ema = copy.deepcopy(net).eval().requires_grad_(False)
            data = dict(
                ema=ema,
                loss_fn=loss_fn,
                augment_pipe=augment_pipe,
                dataset_kwargs=dict(dataset_kwargs),
            )
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value  # conserve memory
            if dist.get_rank() == 0:
                with open(
                    os.path.join(run_dir, f"shot-{cur_nimg//1000:06d}.pkl"),
                    "wb",
                ) as f:
                    pickle.dump(data, f)
            del data  # conserve memory

        # Save full dump of the training state.
        if (
            (state_dump_ticks is not None)
            and (done or cur_tick % state_dump_ticks == 0)
            and cur_tick != 0
            and dist.get_rank() == 0
        ):
            torch.save(
                dict(net=net, optimizer_state=optimizer.state_dict()),
                os.path.join(run_dir, f"training-state-{cur_nimg//1000:06d}.pt"),
            )
        if (
            boxx.timegap(3600 * 2, "save_training-state")
            and cur_tick != 0
            and dist.get_rank() == 0
        ):
            torch.save(
                dict(net=net, optimizer_state=optimizer.state_dict()),
                os.path.join(run_dir, f"training-state-last.pt"),
            )

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, "stats.jsonl"), "at")
            stats_jsonl.write(
                json.dumps(
                    dict(
                        training_stats.default_collector.as_dict(),
                        timestamp=time.time(),
                    )
                )
                + "\n"
            )
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if boxx.cf.debug and cur_nimg >= 12:
            boxx.g()
            done = True
        if done:
            break
    # Done.
    dist.print0()
    dist.print0("Exiting...")


# ----------------------------------------------------------------------------