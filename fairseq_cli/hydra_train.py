#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from fairseq import distributed_utils, metrics
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.initialize import add_defaults, hydra_init
from fairseq.dataclass.utils import omegaconf_no_object_check
from fairseq.utils import reset_logging
from fairseq_cli.train import main as pre_main

logger = logging.getLogger("fairseq_cli.hydra_train")


@hydra.main(config_path=os.path.join("..", "fairseq", "config"), config_name="config")
def hydra_main(cfg: FairseqConfig) -> float:
    
    # HACK [DEBUG]
    # I want every run to create a new folder with outputs
    # modified to make each run save its checkpoints in a different folder 
    if True:
        import time
        t = time.localtime()
        new_output_path = os.path.join(os.getcwd(), "pre_train_outputs", time.strftime("%Y_%m_%d_%H_%M_%S", t))
        logger.info(f"[DEBUG] new_output_path: {new_output_path}")
        os.makedirs(new_output_path, exist_ok=True)
        os.chdir(new_output_path)
    
    _hydra_main(cfg)


def _hydra_main(cfg: FairseqConfig, **kwargs) -> float:

    logger.info(f"[DEBUG] Entered _hydra_main()")

    # TODO understand what is this doing (config defaults)
    # (This function adds default values that are stored in dataclasses that hydra doesn't know about)
    add_defaults(cfg)
    #logger.info(f"[DEBUG] cfg: {cfg}")

    # TODO understand what is this doing (logging)
    if cfg.common.reset_logging:
        #logger.info(f"[DEBUG] Entered 1.1")
        reset_logging()  # Hydra hijacks logging, fix that
    else:
        #logger.info(f"[DEBUG] Entered 1.2")
        # check if directly called or called through hydra_main
        if HydraConfig.initialized():
            #logger.info(f"[DEBUG] Entered 1.2.1")
            with open_dict(cfg):
                # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
                cfg.job_logging_cfg = OmegaConf.to_container(
                    HydraConfig.get().job_logging, resolve=True
                )

    # TODO understand what is this doing (?)
    with omegaconf_no_object_check():
        cfg = OmegaConf.create(
            OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
        )
    OmegaConf.set_struct(cfg, True)

    # TODO understand what is this doing (distributed?)
    try:
        #logger.info(f"[DEBUG] Entered 2.1")
        if cfg.common.profile:
            #logger.info(f"[DEBUG] Entered 2.1.1")
            with torch.cuda.profiler.profile():
                with torch.autograd.profiler.emit_nvtx():
                    logger.info(f"[DEBUG] DISTRIBUTED with torch.autograd.profiler.emit_nvtx()")
                    distributed_utils.call_main(cfg, pre_main, **kwargs)
        else:
            #logger.info(f"[DEBUG] Entered 2.1.2")
            logger.info(f"[DEBUG] Entered DISTRIBUTED else")
            distributed_utils.call_main(cfg, pre_main, **kwargs)
    except BaseException as e:
        #logger.info(f"[DEBUG] Entered 2.2")
        if not cfg.common.suppress_crashes:
            #logger.info(f"[DEBUG] Entered 2.2.1")
            raise
        else:
            #logger.info(f"[DEBUG] Entered 2.2.2")
            logger.error("Crashed! " + str(e))

    # TODO understand what is this doing
    # get best val and return - useful for sweepers
    try:
        #logger.info(f"[DEBUG] Entered 3.1")
        best_val = metrics.get_smoothed_value(
            "valid", cfg.checkpoint.best_checkpoint_metric
        )
    except:
        #logger.info(f"[DEBUG] Entered 3.2")
        best_val = None

    if best_val is None:
        best_val = float("inf")

    logger.info(f"[DEBUG] best_val = {best_val}")
    logger.info("[DEBUG] Exited _hydra_main()")

    return best_val


def cli_main():

    logger.info("[DEBUG] Entered cli_main()")

    try:
        from hydra._internal.utils import get_args
        cfg_name = get_args().config_name or "config"
        logger.info(f"[DEBUG] cfg_name: {cfg_name}")
    except:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "config"

    # TODO understand what is this doing
    hydra_init(cfg_name)
    hydra_main()

    logger.info("[DEBUG] Exited cli_main()")


if __name__ == "__main__":
    cli_main()
