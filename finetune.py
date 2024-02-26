#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""A one-line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""

import sys
import os
from datetime import timedelta
import torch
from mpi4py import MPI
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))
from transformers import HfArgumentParser

from transformers import TrainingArguments
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass, field

from src.lmflow.args import (
    ModelArguments,
    DatasetArguments,
    AutoArguments,
)

from src.lmflow.datasets.dataset import Dataset
from src.lmflow.models.auto_model import AutoModel
from src.lmflow.pipeline.auto_pipeline import AutoPipeline


def main():
    master_addr = os.environ["MASTER_ADDR"]
    print("Master address from main:", master_addr)
    master_port = "29500"

    # default_pg_timeout = timedelta(minutes=1)

    def setup_distributed_env(init_method=None, rank=0, world_size=16):
        comm = MPI.COMM_WORLD
        world_size = comm.Get_size()
        world_rank = rank = comm.Get_rank()
        # world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        # world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        backend = None
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(world_rank)
        os.environ['LOCAL_RANK'] = "0"  # str(world_rank % 8)
        print("initialization parameters:", init_method, backend, rank, world_size)
        torch.distributed.init_process_group(backend,
                                             # timeout = default_pg_timeout,
                                             init_method=init_method,
                                             rank=rank,
                                             world_size=world_size)

        using_mpi = torch.distributed.get_backend() == 'mpi'
        print("using_mpi=", using_mpi)

    setup_distributed_env()

    pipeline_name = "finetuner"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

    parser = HfArgumentParser((ModelArguments, DatasetArguments, PipelineArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, pipeline_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, pipeline_args = parser.parse_args_into_dataclasses()

    # Initialization
    finetuner = AutoPipeline.get_pipeline(
        pipeline_name=pipeline_name,
        model_args=model_args,
        data_args=data_args,
        pipeline_args=pipeline_args,
    )
    dataset = Dataset(data_args)
    model = AutoModel.get_model(model_args)
    # Finetuning
    tuned_model = finetuner.tune(model=model, dataset=dataset)


if __name__ == '__main__':
    main()
