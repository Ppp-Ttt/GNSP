import argparse
import copy
import logging
import os
import sys

import json
from random import random
import time
from datetime import datetime
import clip
import torch

from . import utils
from .args import parse_arguments
from .models import evaluate, evaluate_fc, evaluate_wise_ft, finetune, finetune_fc, finetune_icarl, finetune_gift, finetune_new
from .models.finetune_gnsp import finetune_gnsp_for_mtil
from .models.modeling import create_image_classifier

import argparse
import json
import torch

# with open('project_config.json', 'r') as f:
#     project_config = json.load(f)
#
# SYN_DATA_LOCATION = project_config['SYN_DATA_LOCATION']
# CL_DATA_LOCATION = project_config['CL_DATA_LOCATION']
# MTIL_DATA_LOCATION = project_config['MTIL_DATA_LOCATION']

def merge(model_0, model_1, alpha=0.95):
    key_name = [k for k, v in model_0.named_parameters()]
    for i, (param_q, param_k) in enumerate(zip(model_0.parameters(), model_1.parameters())):
        param_k.data = param_k.data * alpha + param_q.data * (1 - alpha)
    return model_1

def mtil_main(args):

    logger = logging.getLogger("gnsp." + __name__)
    utils.seed_all(args.seed)

    if args.eval_only:
        torch.cuda.set_device(args.devices[0])
        model, tra_preprocess, val_preprocess = clip.load(args.model, jit=False)
        if args.load:
            utils.torch_load(model, args.load, device=args.devices[0])
            logger.info(f"loading ckp from {args.load}")
        top1_list, _ = evaluate(model, args, val_preprocess, None)
    else:
        top1_list = finetune_gnsp_for_mtil(args)

    return top1_list
