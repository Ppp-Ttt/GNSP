import copy
import os
import pickle
import random
import argparse

import clip.clip as clip

import logging
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from src import datasets, utils
import torch
import numpy as np


def get_data_matrix(args):

    devices = args.devices
    torch.cuda.set_device(devices[0])

    model, train_preprocess, val_preprocess = clip.load(args.model, jit=False)

    # load model
    if args.load is not None:
        utils.torch_load(model, args.load, device=devices[0])
        # require grads (not that useful)
        model.requires_grad_(False)


    # prepare training dataset
    dataset_class = getattr(datasets, args.train_dataset)
    if args.train_dataset == "ImageNet1K":
        dataset = dataset_class(
            train_preprocess,
            location='/home/liuyuyang/ptt/GIFT_CL/data/ImageNet-1K/ILSVRC/Data/CLS-LOC/train',
            synset_map_file=os.path.expanduser('/home/liuyuyang/ptt/GIFT_CL/data/ImageNet-1K/LOC_synset_mapping.txt'),
            batch_size=16,
            image_nums=args.image_nums,
            seed=41,
            num_workers=8,
        )
    else:
        dataset = dataset_class(
            train_preprocess,
            location=args.data_location,
            batch_size=16,
            batch_size_eval=args.batch_size_eval,
            num_workers=8,
        )

    # number of iterations
    num_batches = len(dataset.train_loader)
    total_iterations = num_batches

    model = model.cuda()

    data_iter = iter(dataset.train_loader)
    ffn_M0 = torch.zeros((12,768,768)).cuda()
    ffn_M1 = torch.zeros((12,3072,3072)).cuda()
    proj_M = torch.zeros((768,768)).cuda()
    for iteration in tqdm(range(1, total_iterations+1)):
        if args.train_dataset == "ImageNet1K":
            batch = next(data_iter)
            images = batch["images"].cuda(device=devices[0])
        else:
            images, labels = next(data_iter)
            images = images.cuda(device=devices[0])

        # -- get image embedding --
        with torch.no_grad():
            ffn_x0, ffn_x1, proj_x = model.get_image_matrix(images)

        ffn_x0 = ffn_x0.reshape(ffn_x0.size(0), ffn_x0.size(1)*ffn_x0.size(2), ffn_x0.size(3))
        ffn_x0t = ffn_x0.transpose(1, 2)

        ffn_x1 = ffn_x1.reshape(ffn_x1.size(0), ffn_x1.size(1) * ffn_x1.size(2), ffn_x1.size(3))
        ffn_x1t = ffn_x1.transpose(1, 2)

        # proj_x = proj_x.view(proj_x.size(0) * proj_x.size(1), proj_x.size(2))
        proj_xt = proj_x.transpose(0,1)

        ffn_m0 = torch.einsum('bij,bjk->bik', ffn_x0t, ffn_x0)
        ffn_m1 = torch.einsum('bij,bjk->bik', ffn_x1t, ffn_x1)
        proj_m = torch.einsum('ij,jk->ik', proj_xt, proj_x)

        ffn_M0 = ffn_M0 + ffn_m0 / torch.norm(ffn_m0, p=2, dim=(1, 2), keepdim=True) # (12, 768, 768)
        ffn_M1 = ffn_M1 + ffn_m1 / torch.norm(ffn_m1, p=2, dim=(1, 2), keepdim=True) # (12,3072,3072)
        proj_M = proj_M + proj_m / torch.norm(proj_m, p=2, dim=(0, 1), keepdim=True)


    if args.pt_source_dir:
        M_dict = torch.load(args.pt_source_dir)
        for i in range(12):
            M_dict[f'visual.transformer.resblocks.{i}.mlp.c_fc.weight'] = \
                M_dict[f'visual.transformer.resblocks.{i}.mlp.c_fc.weight']\
                + \
                ffn_M0[i].cpu()/torch.norm(ffn_M0[i].cpu(), p=2, dim=(0, 1), keepdim=True)

            M_dict[f'visual.transformer.resblocks.{i}.mlp.c_proj.weight'] = \
                M_dict[f'visual.transformer.resblocks.{i}.mlp.c_proj.weight']\
                + \
                ffn_M1[i].cpu()/torch.norm(ffn_M1[i].cpu(), p=2, dim=(0, 1), keepdim=True)

        M_dict['visual.proj'] = M_dict['visual.proj']/torch.norm(M_dict['visual.proj'], p=2, dim=(0, 1), keepdim=True) + proj_M.cpu()/torch.norm(proj_M.cpu(), p=2, dim=(0, 1), keepdim=True)

    else:
        M_dict = {}
        for i in range(12):
            M_dict[f'visual.transformer.resblocks.{i}.mlp.c_fc.weight'] = ffn_M0[i].cpu()/torch.norm(ffn_M0[i].cpu(), p=2, dim=(0, 1), keepdim=True)
            M_dict[f'visual.transformer.resblocks.{i}.mlp.c_proj.weight'] = ffn_M1[i].cpu()/torch.norm(ffn_M1[i].cpu(), p=2, dim=(0, 1), keepdim=True)

        M_dict['visual.proj'] = proj_M.cpu()/torch.norm(proj_M.cpu(), p=2, dim=(0, 1), keepdim=True)

    os.makedirs(os.path.dirname(args.pt_save_dir), exist_ok=True)
    torch.save(M_dict, args.pt_save_dir)
    return M_dict


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.pt_source_dir = None
    args.pt_save_dir = "/home/liuyuyang/ptt/GIFT_CL/data/data_matrix/ImageNet100k.pt"
    args.devices = [6]
    args.model = 'ViT-B/16'
    args.load = None
    args.train_dataset = "ImageNet1K"
    args.image_nums = 1000000

    print(f"Start get data matrix on {args.train_dataset}")
    print(f"load from {args.load}")

    M_dict = get_data_matrix(args)

    for k ,v in M_dict.items():
        print(f"{k}, shape {v.shape}")
    print(f"data matrix has saved to {args.pt_save_dir}")






