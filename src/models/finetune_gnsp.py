import copy
import os
import pickle
import random
import time
from ast import increment_lineno
from datetime import timedelta

from torch.utils.data import DataLoader

import clip.clip as clip

import logging
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import json

from .. import datasets, templates, utils
from .evaluation import evaluate, zeroshot_classifier
from .helpers import merge_we, wise_we, moving_avg, l2_loss, distillation, prepare_ref_dataset, ewc_loss, kl_divergence
import torch
import numpy as np

from continuum import ClassIncremental
from continuum.datasets import CIFAR100
from torchvision.transforms import transforms
from collections import defaultdict

def get_data_matrix(dataloader, model, pt_source_dir, pt_save_dir):
    device = next(model.parameters()).device
    # prepare training dataset
    # number of iterations

    num_batches = len(dataloader)
    total_iterations = num_batches

    model = model.cuda()

    data_iter = iter(dataloader)
    ffn_M0 = torch.zeros((12,768,768)).cuda()
    ffn_M1 = torch.zeros((12,3072,3072)).cuda()
    proj_M = torch.zeros((768,768)).cuda()
    for iteration in tqdm(range(1, total_iterations+1)):

        images, labels,_ = next(data_iter)
        images = images.cuda(device=device)

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


    if pt_source_dir:
        M_dict = torch.load(pt_source_dir)
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

    os.makedirs(os.path.dirname(pt_save_dir), exist_ok=True)
    torch.save(M_dict, pt_save_dir)
    return M_dict

def evaluate_for_cil(model,val_loader, task_id, current_class_names):
    logger = logging.getLogger("gnsp." + __name__)

    with open('./configs/CIFAR100.json', 'r') as f:
        cifar100_cfg = json.load(f)

    device = next(model.parameters()).device

    eval_iter = iter(val_loader)
    class_names_to_eval = current_class_names
    #[task_id*10:task_id*10+10]
    logger.info(f"class to eval: {class_names_to_eval}")
    text_tokens = clip.tokenize([cifar100_cfg["prompt_template"].format(c) for c in class_names_to_eval]).cuda()

    embeddings = model(None, text_tokens)
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    logit_scale = model.logit_scale

    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    for batch in tqdm(eval_iter):
        images, labels, _ = batch

        images, labels = images.cuda(device=device), labels.cuda(device=device)

        # -- get image embedding --
        out = model(images, None)
        out = out / out.norm(dim=-1, keepdim=True)

        logits_per_image = logit_scale.exp() * out @ embeddings.t()

        preds = logits_per_image.argmax(dim=-1)

        # 遍历每个样本
        for pred, label in zip(preds, labels):
            label = label.item()
            class_total[label] += 1
            if pred.item() == label:
                class_correct[label] += 1

    # 计算每个类的准确率
    per_class_acc = {}
    acc_all = 0
    for label in class_total:
        correct = class_correct[label]
        total = class_total[label]
        acc = correct / total if total > 0 else 0.0
        per_class_acc[label] = acc
        acc_all += acc
    avg = acc_all/len(class_total)

    # 打印每类准确率
    for label, acc in sorted(per_class_acc.items()):
        print(f"Class {label}: Accuracy = {acc:.4f}")
    logger.info(f"AVG: {avg:.4f}")
    return avg

def finetune_gnsp_for_mtil(args):
    logger = logging.getLogger("gnsp."+__name__)
    logger.info(f"Prepare training for dataset: {args.train_dataset}")

    devices = args.devices
    torch.cuda.set_device(devices[0])

    model, train_preprocess, val_preprocess = clip.load(args.model, jit=False)

    # load model
    if args.load is not None:
        logger.info(f"Loading model from {args.load}")
        utils.torch_load(model, args.load, device=devices[0])
    else:
        pass

    for name, param in model.named_parameters():
        if name != 'logit_scale':
            param.requires_grad = True

    # prepare model for ensemble (not used now)
    # if args.we_wise or (args.wise_merge and args.wise_ft_model != "zeroshot"): # False
    #     logger.info("Using WiSE-FT with Loaded Model")
    #     model_fix, train_preprocess, val_preprocess = clip.load(args.model, jit=False)
    #     if args.load is not None:
    #         utils.torch_load(model_fix, args.load)
    # if args.we or args.moving_avg or args.we_wise: # False
    #     logger.info("Averaging training")
    #     if args.moving_avg and args.mv_avg_model == "zeroshot":  # mv+zeroshot
    #         we_model, _, _ = clip.load(args.model, jit=False)
    #         we_model.cuda()
    #         we_n = 0
    #     else:  # we; mv+m; mv+t; we_wise
    #         we_model = copy.deepcopy(model)
    #         we_n = 0
    #         we_model.cuda()

    # if args.l2 > 0: # False
    #     logger.info(f"L2 norm:{args.l2}")
    #     l2_model = copy.deepcopy(model)  # L2 model is the initial model
    #     l2_model.cuda()

    # prepare training dataset
    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(
        train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
        num_workers=8,
    )

    # prepare template for training dataset
    if args.template is not None:
        template = getattr(templates, args.template)[0]
    else:
        template = dataset.template

    texts = [template(x) for x in dataset.classnames]
    texts = clip.tokenize(texts).cuda()

    # number of iterations
    num_batches = len(dataset.train_loader)
    if args.epochs is not None:
        total_iterations = args.epochs * num_batches
    else:
        total_iterations = args.iterations
    if args.eval_every_epoch:
        eval_iterations = num_batches
    else:
        eval_iterations = args.eval_interval

    save_iterations = args.save_interval

    logger.info(f"Total iterations:{total_iterations}, {num_batches}iters/epoch")

    # prepare for null-space gradient projection matirx P
    if args.gnsp:
        logger.info(f"gnsp_rho: {args.gnsp_rho}")
        selected_layer_names = [args.gnsp_layers[i] for i in args.gnsp_selected_layers]

        for name, param in model.named_parameters():
            # In CLIP ViT, FFN layers are in the mlp module of each transformer block
            # FFN parameter names:
            # 'module.visual.transformer.resblocks.0.mlp.c_fc.weight' ,shape:(3072,768) in
            # 'module.visual.transformer.resblocks.0.mlp.c_proj.weight' ,shape:(768,3072) out
            if name in selected_layer_names:
                param.requires_grad = True
                logger.debug(f"parameters to update: {name}")
            # elif name == 'positional_embedding':
            #     param.requires_grad = True
            # elif name =='text_projection':
            #     param.requires_grad = True
            # elif name == 'token_embedding.weight':
            #     param.requires_grad = True
            # elif name == 'ln_final.weight':
            #     param.requires_grad = True
            # elif name == 'ln_final.bias':
            #     param.requires_grad = True
            else:
                param.requires_grad = False

        # for name,param in model.transformer.named_parameters():
        #     param.requires_grad = True


        if args.pt_source_dir is not None:
            logger.info(f"loading data matrix file from {args.pt_source_dir}")
            M_dict = torch.load(args.pt_source_dir)
            for selected_layer_name in selected_layer_names:
                if selected_layer_name not in M_dict.keys():
                    assert False, f"selected layer {selected_layer_name}is not in M_dict!"
            proj_matrix_dict = {}

            def find_k_binary(S, ratio=0.05):
                total_sum = torch.sum(S)
                if total_sum == 0 or len(S) == 0:
                    return -1

                left, right = 0, len(S)
                while left < right:
                    mid = (left + right) // 2
                    tail_sum = torch.sum(S[mid:])
                    if tail_sum / total_sum < ratio:
                        right = mid
                    else:
                        left = mid + 1
                return left

            for name, M in M_dict.items():
                if name not in selected_layer_names:
                    continue
                M = M.cuda()
                U, S, V = torch.svd(M, some=False)  # some=False即保留零空间基底, svd分解后有: M = U @ torch.diag(S) @ V.t()

                # threshold = 0.1 * torch.mean(S)
                # threshold = 100 * torch.min(S)

                rank = find_k_binary(S, args.gnsp_rho)
                # rank = torch.sum(S > threshold)
                # rank = int(S.size(0) * 0.5)
                # rank = -128

                logger.debug(f"rank(P): {S.size(0) - rank} for {name}, S[{rank}] = {S[rank]}")
                null_space_basis = V[:, rank:]
                P = null_space_basis @ null_space_basis.t()
                proj_matrix_dict[name] = P
            del M_dict
        else:
            proj_matrix_dict = None
            logger.info(f"args.pt_source_dir is None! don't project")

    # params to tune
    frozen_params = ['logit_scale']
    params = [v for k, v in model.named_parameters() if k not in frozen_params]
    if args.gnsp:
        params = [v for k, v in model.named_parameters() if v.requires_grad]

    # optimizer
    optimizer = torch.optim.AdamW(
        params, lr=args.lr, weight_decay=args.wd, betas=(0.9, args.beta2)
    )
    scheduler = utils.cosine_lr(
        optimizer, args.lr, args.warmup_length, total_iterations
    )

    logit_scale = model.logit_scale

    logger.info(f"Using GPUs: {devices}")
    model = torch.nn.DataParallel(model, device_ids=devices)
    model = model.cuda()

    # 准备ref model
    if args.distill > 0 or args.map > 0:

        logger.info(f"map: {args.map}, distill: {args.distill}")
        ref_model, _, test_preprocess = clip.load(args.model, jit=False)

        ref_dataset, ref_texts = prepare_ref_dataset(args, test_preprocess)
        ref_iter = iter(ref_dataset.train_loader)

        if args.ref_dataset == "Current":
            img_nums = len(ref_dataset.train_dataset)
        else:
            img_nums = len(ref_dataset)
        if args.ref_model == "load" and args.load is not None:
            logger.info(f"ref Model: {args.load}, ref dataset: {args.ref_dataset}, nums: {img_nums}")
            utils.torch_load(ref_model, args.load, device=devices[0])
        elif "merge" in args.ref_model:
            # e.g. merge:0.2
            merge_ratio = float(args.ref_model.split(":")[1])
            logger.info(f"ref Model: {args.ref_model}, ref dataset: {args.ref_dataset}, nums: {img_nums}")
            for param_q, param_k in zip(ref_model.parameters(), model.module.parameters()):
                param_q.data = param_q.data * merge_ratio + param_k.data * (1 - merge_ratio)
        else:
            logger.info(f"ref Model: [Initial CLIP], ref dataset: {args.ref_dataset}, nums: {img_nums}")

        ref_model = torch.nn.DataParallel(ref_model, device_ids=devices)
        ref_model = ref_model.cuda()
        ref_model.eval()


    logger.info(f"Start training on {args.train_dataset}")

    start_time = time.time()
    iteration_threshold = 1
    for iteration in range(total_iterations):

        if save_iterations is not None and iteration % save_iterations == 0 and iteration >= iteration_threshold:
            path = os.path.join(args.save, f"{args.train_dataset}_{iteration}.pth")
            utils.torch_save(model.module, path)
            logger.info(f"Checkpoint saved to {path}")

        if eval_iterations is not None and iteration % eval_iterations == 0 and iteration >= iteration_threshold:
            logger.info(f"Evaluating for iteration {iteration}")
            top1_list, _ = evaluate(model.module, args, val_preprocess)
            logger.info(f"top1_list: {np.round(top1_list, decimals=2).tolist()}")
            logger.info(f"top1_mean: {np.mean(top1_list):.2f}")

        if iteration % num_batches == 0:
            data_iter = iter(dataset.train_loader)

        model.train()
        scheduler(iteration)

        try:
            images, labels = next(data_iter)
        except:
            data_iter = iter(dataset.train_loader)
            images, labels = next(data_iter)

        images, labels = images.cuda(device=devices[0]), labels.cuda(device=devices[0])

        # -- get image embedding --
        out = model(images, None)
        out = out / out.norm(dim=-1, keepdim=True)

        loss_dict = {}  # record all parts of losses

        # -- get text embedding --
        embeddings = model(None, texts)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        logits_per_image = logit_scale.exp() * out @ embeddings.t()

        ce_loss = F.cross_entropy(logits_per_image, labels, label_smoothing=args.ls)
        loss = ce_loss
        loss_dict['ce_loss'] = ce_loss.item()

        batch_acc = (logits_per_image.argmax(dim=-1) == labels).float().mean().item()
        ref_batch_acc = None

        del images, labels

        # -- distillation and map --
        if args.distill > 0 or args.map > 0:
            try:
                ref_batch = next(ref_iter)
            except:
                ref_iter = iter(ref_dataset.train_loader)
                ref_batch = next(ref_iter)

            if args.ref_dataset == 'Current':
                ref_images, ref_labels = ref_batch[0].cuda(device=devices[0]), ref_batch[1].cuda(device=devices[0])
            else:
                ref_images, ref_labels = ref_batch["images"].cuda(device=devices[0]), ref_batch["labels"].cuda(device=devices[0])

            if args.ref_dataset == 'ImageNet1K':
                batch_ref_texts = ref_batch["texts"]
            else:
                batch_ref_texts = [ref_texts[x] for x in ref_labels]
            batch_ref_texts = clip.tokenize(batch_ref_texts).cuda(device=devices[0])

            with torch.no_grad():
                ref_text_embeddings = ref_model(None, batch_ref_texts)
                ref_text_embeddings = ref_text_embeddings / ref_text_embeddings.norm(
                    dim=-1, keepdim=True
                )
                # -- get ref image embedding --
                ref_image_embeddings = ref_model(ref_images, None)
                ref_image_embeddings = ref_image_embeddings / ref_image_embeddings.norm(dim=-1, keepdim=True)
                logits_ref = logit_scale.exp() * ref_image_embeddings @ ref_text_embeddings.t()

            ref_image_embeddings_now = model(ref_images, None)
            ref_image_embeddings_now  = ref_image_embeddings_now  / ref_image_embeddings_now .norm(
                dim=-1, keepdim=True
            )

            ref_text_embeddings_now = model(None, batch_ref_texts)
            ref_text_embeddings_now = ref_text_embeddings_now / ref_text_embeddings_now.norm(
                dim=-1, keepdim=True
            )

            if args.image_only:
                logits_current = logit_scale.exp() * ref_image_embeddings_now @ ref_text_embeddings.t()
            elif args.text_only:
                logits_current = logit_scale.exp() * ref_image_embeddings @ ref_text_embeddings_now.t()
            else:
                logits_current = logit_scale.exp() * ref_image_embeddings_now @ ref_text_embeddings_now.t()

            # accuracy on ref batch
            batch_ref_labels = torch.arange(ref_images.shape[0]).cuda()
            ref_batch_acc = (logits_current.argmax(dim=-1) == batch_ref_labels).float().mean().item()

            ref_labels_repeated = ref_labels.view(1, -1).repeat(ref_images.shape[0], 1)
            ref_equal_labels = (ref_labels_repeated == ref_labels.view(-1, 1)).type(torch.float)
            batch_ref_labels = ref_equal_labels / torch.sum(ref_equal_labels, dim=1).view(-1, 1).cuda()

            if args.distill > 0:
                distill_loss = 0
                # if args.image_loss:
                if args.feature_mse:  # another form of KD under mild assumption
                    # image_distill_loss = args.distill * F.mse_loss(logits_ref, logits_current)
                    image_distill_loss = args.distill * F.mse_loss(ref_image_embeddings, ref_image_embeddings_now)
                elif args.kl_div:
                    image_distill_loss = args.distill * kl_divergence(logits_ref, logits_current, T=args.T)
                else:
                    image_distill_loss = args.distill * distillation(logits_ref, logits_current, T=args.T)
                loss_dict['image_distill_loss'] = image_distill_loss.item()
                distill_loss += image_distill_loss
                # if args.text_loss:
                if args.feature_mse:
                    # text_distill_loss = args.distill * F.mse_loss(logits_ref.t(), logits_current.t())
                    text_distill_loss = args.distill * F.mse_loss(ref_text_embeddings, ref_text_embeddings_now)
                elif args.kl_div:
                    text_distill_loss = args.distill * kl_divergence(logits_ref.t(), logits_current.t(), T=args.T)
                else:
                    text_distill_loss = args.distill * distillation(logits_ref.t(), logits_current.t(), T=args.T)
                loss_dict['text_distill_loss'] = text_distill_loss.item()
                distill_loss += text_distill_loss
                loss += distill_loss

            if args.map > 0:
                map_loss = 0
                # if args.image_loss:
                image_map_loss = args.map * F.cross_entropy(logits_current, batch_ref_labels, label_smoothing=args.ls)
                loss_dict['image_map_loss'] = image_map_loss.item()
                map_loss += image_map_loss
                # if args.text_loss:
                text_map_loss = args.map * F.cross_entropy(logits_current.t(), batch_ref_labels,
                                                           label_smoothing=args.ls)
                loss_dict['text_map_loss'] = text_map_loss.item()
                map_loss += text_map_loss
                loss += map_loss

            # elastic weight consolidation
            if args.awc and (args.static_awc == 0 or iteration < args.static_awc):
                optimizer.zero_grad()

                if args.map > 0 and args.distill > 0:
                    (distill_loss + map_loss).backward(retain_graph=True)
                elif args.map == 0:
                    distill_loss.backward(retain_graph=True)
                elif args.distill == 0:
                    map_loss.backward(retain_graph=True)

                active_params = {n: p for n, p in model.named_parameters()}

                if args.static_awc != 0 and iteration > 0:
                    t = iteration
                    for n, p in active_params.items():
                        if p.grad is not None and 'logit_scale' not in n:
                            fisher_temp = p.grad.clone() ** 2
                            fisher_temp /= fisher_temp.mean()
                        else:
                            fisher_temp = torch.zeros_like(p) + 1e-6
                        _precision_matrices[n] = _precision_matrices[n].cuda()
                        _precision_matrices[n] = t / (t + 1) * _precision_matrices[n] + 1 / (t + 1) * fisher_temp
                        _precision_matrices[n] = _precision_matrices[n].detach()
                else:
                    _precision_matrices = {}
                    for n, p in active_params.items():
                        if p.grad is not None and 'logit_scale' not in n:
                            _precision_matrices[n] = p.grad.clone() ** 2
                            _precision_matrices[n] /= _precision_matrices[n].mean()
                        else:
                            _precision_matrices[n] = torch.zeros_like(p) + 1e-6

                        _precision_matrices[n] = _precision_matrices[n].detach()

            del ref_batch

        # if args.l2 > 0:
        #     if args.awc:
        #         wc_loss = args.l2 * ewc_loss(model, l2_model, _precision_matrices)
        #     else:
        #         wc_loss = l2_loss(model, l2_model)
        #         loss_dict['l2_dis'] = wc_loss.item()
        #         wc_loss = args.l2 * wc_loss
        #     loss += wc_loss
        #     loss_dict['wc_loss'] = wc_loss.item()

        formatted_loss_dict = {key: f'{value:.4f}' for key, value in loss_dict.items()}
        # progress_bar.set_postfix(loss=loss.item(), batch_acc=batch_acc, ref_batch_acc=ref_batch_acc, loss_dict=formatted_loss_dict)
        if (iteration+1) % 50 == 0:
            elapsed_time = time.time() - start_time
            avg_time_per_iter = elapsed_time / (iteration + 1)  # +1 防止除以0

            remaining_iters = total_iterations - iteration - 1
            remaining_seconds = remaining_iters * avg_time_per_iter

            eta = timedelta(seconds=int(remaining_seconds))

            # logger.info(
            #     f"Iteration {iteration + 1}, ETA: {eta}, loss={loss.item():.2f} batch_acc={batch_acc:.3f}, ref_batch_acc={ref_batch_acc:.3f}, loss_dict={formatted_loss_dict}")
            logger.info(
                f"Iteration {iteration + 1}, ETA: {eta}, loss={loss.item():.2f} batch_acc={batch_acc:.3f}")
        # update
        optimizer.zero_grad()
        loss.backward()
        if args.gnsp and proj_matrix_dict is not None:

            def get_param_and_field(model, s):
                parts = s.split('.')
                # 获取最后一个字段名（例如 'weight'）
                field = parts[-1]
                # 获取前面的对象路径（例如 'mlp_proj'）
                module_path = parts[:-1]
                # 递归获取子模块
                obj = model
                for attr in module_path:
                    obj = getattr(obj, attr)
                return obj, field

            for selected_layer_name in selected_layer_names:
                param_obj, param_field = get_param_and_field(model.module, selected_layer_name)
                param = getattr(param_obj, param_field)
                g = param.grad
                if selected_layer_name == 'visual.proj':
                    g = proj_matrix_dict[selected_layer_name] @ g
                else:
                    g = g @ proj_matrix_dict[selected_layer_name].t()
                param.grad = g.clone()

        optimizer.step()




    # Saving model
    if args.save is not None:
        to_save_model = model.module  # 单GPU时直接保存model
        path = os.path.join(args.save, f"{args.train_dataset}_final.pth")
        utils.torch_save(to_save_model, path)
        logger.info(f"Saved model to {path}")

    if eval_iterations is not None or args.eval_in_train:
        logger.info(f"Evaluating for final")
        to_eval_model = model.module
        top1_list, _ = evaluate(to_eval_model, args, val_preprocess)
        logger.info(f"top1_list: {np.round(top1_list, decimals=2).tolist()}")
        logger.info(f"top1_mean: {np.mean(top1_list):.2f}")

def finetune_gnsp_for_cil(args):
    logger = logging.getLogger("gnsp."+__name__)
    logger.info(f"Prepare training for dataset: {args.train_dataset}")

    devices = args.devices
    torch.cuda.set_device(devices[0])

    model, train_preprocess, val_preprocess = clip.load(args.model, jit=False)

    # load model
    if args.load is not None:
        logger.info(f"Loading model from {args.load}")
        utils.torch_load(model, args.load, device=devices[0])

    for name, param in model.named_parameters():
        if name != 'logit_scale':
            param.requires_grad = True

    # prepare training dataset
    increment = args.increment
    initial_increment = args.initial_increment
    if args.train_dataset == "CIFAR100":
        with open('./configs/CIFAR100.json', 'r') as f:
            cifar100_cfg = json.load(f)

        cifar100_dataset = CIFAR100(data_path="./data/mtil", download=True, train=True)
        scenario = ClassIncremental(
            cifar100_dataset,
            increment=increment,
            initial_increment=initial_increment,
            class_order=cifar100_cfg["class_order"],
            transformations=train_preprocess.transforms
        )
        class_names = cifar100_dataset.dataset.classes
        template = cifar100_cfg["prompt_template"]

        cifar100_val_dataset = CIFAR100(data_path="./data/mtil", download=True, train=False)
        scenario_val = ClassIncremental(
            cifar100_val_dataset,
            increment=increment,
            initial_increment=initial_increment,
            class_order=cifar100_cfg["class_order"],
            transformations=val_preprocess.transforms
        )

    avg_list = []
    load_ckpt = None
    pt_source_dir = None
    pt_save_dir = None
    start_task_id = 0
    if start_task_id > 0:
        load_ckpt = os.path.join(args.save, f"step{start_task_id-1}_final.pth")
        pt_save_dir = f"./data/data_matrix/{args.exp_name}/step{start_task_id-1}.pt"
    for task_id, taskset in enumerate(scenario,start=start_task_id):
        train_loader = DataLoader(scenario[task_id:task_id+1], batch_size=args.batch_size, shuffle=True, num_workers=4)
        train_iter = iter(train_loader)

        current_class_names = [class_names[class_id] for class_id in cifar100_cfg["class_order"]]
        text_tokens = clip.tokenize([template.format(c) for c in current_class_names]).cuda()

        logger.info(f"task_id: {task_id}")
        logger.info(f"current_class: {len(current_class_names)}, names: {current_class_names}")

        # prepare for null-space gradient projection matirx P
        if args.gnsp:
            logger.info(f"gnsp_rho: {args.gnsp_rho}")
            selected_layer_names = [args.model_layer_names[i] for i in args.selected_layers]

            for name, param in model.named_parameters():
                if name in selected_layer_names:
                    param.requires_grad = True
                    logger.debug(f"parameters to update: {name}")
                else:
                    param.requires_grad = False

            if pt_source_dir is not None:
                logger.info(f"loading data matrix file from {pt_source_dir}")
                M_dict = torch.load(pt_source_dir)
                for selected_layer_name in selected_layer_names:
                    if selected_layer_name not in M_dict.keys():
                        assert False, f"selected layer {selected_layer_name}is not in M_dict!"
                proj_matrix_dict = {}

                def find_k_binary(S, ratio=0.05):
                    total_sum = torch.sum(S)
                    if total_sum == 0 or len(S) == 0:
                        return -1

                    left, right = 0, len(S)
                    while left < right:
                        mid = (left + right) // 2
                        tail_sum = torch.sum(S[mid:])
                        if tail_sum / total_sum < ratio:
                            right = mid
                        else:
                            left = mid + 1
                    return left

                for name, M in M_dict.items():
                    if name not in selected_layer_names:
                        continue
                    M = M.cuda()
                    U, S, V = torch.svd(M, some=False)  # some=False即保留零空间基底, svd分解后有: M = U @ torch.diag(S) @ V.t()

                    # threshold = 0.1 * torch.mean(S)
                    # threshold = 100 * torch.min(S)

                    rank = find_k_binary(S, args.gnsp_rho)
                    # rank = torch.sum(S > threshold)
                    # rank = int(S.size(0) * 0.5)
                    # rank = -128

                    logger.debug(f"rank(P): {S.size(0) - rank} for {name}, S[{rank}] = {S[rank]}")
                    null_space_basis = V[:, rank:]
                    P = null_space_basis @ null_space_basis.t()
                    proj_matrix_dict[name] = P
                del M_dict
            else:
                proj_matrix_dict = None
                logger.info(f"args.pt_source_dir is None! don't project")

        # params to tune
        model, train_preprocess, val_preprocess = clip.load(args.model, jit=False)
        if load_ckpt is not None:
            logger.info(f"Loading model from {load_ckpt}")
            utils.torch_load(model, load_ckpt, device=devices[0])
        # optimizer
        logit_scale = model.logit_scale
        frozen_params = ['logit_scale']
        params = [v for k, v in model.named_parameters() if k not in frozen_params]
        logger.info(f"Using GPUs: {devices}")

        if args.gnsp:
            params = [v for k, v in model.named_parameters() if v.requires_grad]

        optimizer = torch.optim.AdamW(
            params, lr=args.lr, weight_decay=args.wd, betas=(0.9, args.beta2)
        )
        scheduler = utils.cosine_lr(
            optimizer, args.lr, args.warmup_length, args.iterations
        )
        model = torch.nn.DataParallel(model, device_ids=devices)
        model = model.cuda()

        start_time = time.time()
        iteration_threshold = 1
        total_iterations = args.iterations
        for iteration in tqdm(range(total_iterations)):
            break
            model.train()
            scheduler(iteration)
        #
            try:
                images, labels, _ = next(train_iter)
            except:
                train_iter = iter(train_loader)
                images, labels, _ = next(train_iter)
        #
            images, labels = images.cuda(device=devices[0]), labels.cuda(device=devices[0])
        #
            # -- get image embedding --
            out = model(images, None)
            out = out / out.norm(dim=-1, keepdim=True)

            loss_dict = {}  # record all parts of losses

            # -- get text embedding --
            embeddings = model(None, text_tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            logits_per_image = logit_scale.exp() * out @ embeddings.t()

            ce_loss = F.cross_entropy(logits_per_image, labels, label_smoothing=args.ls)
            loss = ce_loss
            loss_dict['ce_loss'] = ce_loss.item()

            batch_acc = (logits_per_image.argmax(dim=-1) == labels).float().mean().item()

            optimizer.zero_grad()
            loss.backward()
            if args.gnsp and proj_matrix_dict is not None:

                def get_param_and_field(model, s):
                    parts = s.split('.')
                    # 获取最后一个字段名（例如 'weight'）
                    field = parts[-1]
                    # 获取前面的对象路径（例如 'mlp_proj'）
                    module_path = parts[:-1]
                    # 递归获取子模块
                    obj = model
                    for attr in module_path:
                        obj = getattr(obj, attr)
                    return obj, field

                for selected_layer_name in selected_layer_names:
                    param_obj, param_field = get_param_and_field(model.module, selected_layer_name)
                    param = getattr(param_obj, param_field)
                    g = param.grad
                    if selected_layer_name == 'visual.proj':
                        g = proj_matrix_dict[selected_layer_name] @ g
                    else:
                        g = g @ proj_matrix_dict[selected_layer_name].t()
                    param.grad = g.clone()

            optimizer.step()

        # # Saving model
        to_save_model = model.module  # 单GPU时直接保存model
        path = os.path.join(args.save, f"step{task_id}_final.pth")
        # utils.torch_save(to_save_model, path)
        logger.info(f"Saved model to {path}")
        load_ckpt = path

        val_loader = DataLoader(scenario_val[:task_id+1], batch_size=8, shuffle=False, num_workers=0)
        logger.info(f"Evaluating for step{task_id}")
        to_eval_model = model.module
        avg = evaluate_for_cil(to_eval_model, val_loader, task_id, current_class_names[:task_id*increment+initial_increment])
        avg_list.append(avg)

        pt_save_dir = f"./data/data_matrix/{args.exp_name}/step{task_id}.pt"
        # get_data_matrix_train_loader = DataLoader(scenario[task_id:task_id+1], batch_size=64, shuffle=False, num_workers=4)
        # get_data_matrix(get_data_matrix_train_loader, to_eval_model, pt_source_dir, pt_save_dir)
        pt_source_dir = None
        # del train_loader, train_iter, val_loader, get_data_matrix_train_loader
        del train_loader, train_iter, val_loader
        torch.cuda.empty_cache()
        import gc; gc.collect()

    logger.info(f"avg:{sum(avg_list)/len(avg_list)}")
    logger.info(f"last: {avg_list[-1]}")
    logger.info(avg_list)
