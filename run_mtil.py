import logging
import os
import sys
import torch
import matplotlib.pyplot as plt
import json
import numpy as np

from src.main import mtil_main, mtil_main
from src.model_load_test import model_load
from src.args import parse_arguments
from src.models import evaluate
from src.logger import setup_logger
from src.models.get_data_matrix import get_data_matrix


if __name__ == '__main__':
    args = parse_arguments()
    # args.cfg = './configs/mtil_order_I.json'
    try:
        with open(args.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except:
        print(f"Failed to load config file!")

    args.gnsp_layers = config["gnsp_layers"]

    devices = config["devices"]
    args.gnsp_selected_layers = config["gnsp_selected_layers"]
    exp_name = config["exp_name"]

    datasets = config["train_dataset"]
    results = []
    for i in range(1, len(datasets)):

        args.train_dataset = datasets[i]
        args.log_path = f'log/{exp_name}/{args.train_dataset}.txt'
        args.lr = config["learning_rate"][i]
        args.ls = config["label_smoothing"]
        args.batch_size = config["batch_size"]
        args.iterations = config["iterations"][i]
        args.map = config["map"]
        args.distill = config["distill"]
        args.method = 'GNSP'
        args.train_loss = 'cross_entropy'
        args.save = f'./checkpoint/{exp_name}'
        args.devices = devices
        args.awc = False
        if i > 0:
            args.load = f"./checkpoint/{exp_name}/{datasets[i - 1]}_final.pth"
        else:
            args.load = None
        args.ref_model = config["ref_model"]
        args.ref_dataset = config["ref_dataset"]
        args.ref_data_nums = config["ref_data_nums"]

        args.gnsp = i>0 
        args.gnsp_rho = config["gnsp_rho"]
        if i > 0:
            args.pt_source_dir = f"./gram_matrix/{exp_name}/{datasets[i - 1]}.pt"
        else:
            args.pt_source_dir = None
        args.eval_interval = None  # eval per 500 iter
        args.eval_datasets = datasets
        args.save_interval = None
        args.eval_in_train = True

        setup_logger(logger_name='gnsp', save_path=args.log_path)
        logger = logging.getLogger("gnsp.main")
        for k, v in vars(args).items():
            logger.info(f"{k}: {v}")

        # start
        top1_list = mtil_main(args)
        results.append(top1_list)

        if i == len(datasets) - 1:
            logger.info(f"{exp_name} over!")
            break


        args.load = f"./checkpoint/{exp_name}/{datasets[i]}_final.pth"

        args.pt_save_dir = f"./gram_matrix/{exp_name}/{datasets[i]}.pt"

        logger.info("="*20+" save gram_matrix "+"="*20)
        logger.info(f"ckp load from: {args.load}")
        logger.info(f"pt_source_dir: {args.pt_source_dir}")
        logger.info(f"pt_save_dir: {args.pt_save_dir}")
        logger.info("="*60)

        get_data_matrix(args)

    setup_logger(logger_name='gift', save_path=f'log/{exp_name}/results.txt')
    logger = logging.getLogger("gift.main")

    for list in results:
        logger.info(f"top1_list: {np.round(list, decimals=2).tolist()}, mean: {np.mean(list):.2f}")

