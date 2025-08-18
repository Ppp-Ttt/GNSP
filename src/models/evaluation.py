import clip

import torch
from tqdm import tqdm

from src import datasets
from src.datasets.common import get_dataloader, maybe_dictionarize
import json

with open('/home/liuyuyang/ptt/GIFT_CL/project_config.json', 'r') as f:
    project_config = json.load(f)

SYN_DATA_LOCATION = project_config['SYN_DATA_LOCATION']
CL_DATA_LOCATION = project_config['CL_DATA_LOCATION']
MTIL_DATA_LOCATION = project_config['MTIL_DATA_LOCATION']


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


@torch.no_grad()
def zeroshot_classifier(classnames, templates, model, is_zero_shot):
    if not isinstance(templates, list):
        templates = [templates]
    elif not is_zero_shot:
        templates = [templates[0]]
    zeroshot_weights = []

    for classname in classnames:
        texts = [template(classname) for template in templates]  # format with class
        texts = clip.tokenize(texts).cuda()  # tokenize
        class_embeddings = model.encode_text(texts)  # embed with text encoder
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


@torch.no_grad()
def zeroshot_eval(model, loader, zeroshot_weights):
    top1, top5, n = 0.0, 0.0, 0.0
    for _, data in enumerate(tqdm(loader)):
        data = maybe_dictionarize(data)
        images = data["images"].cuda()
        target = data["labels"].cuda()

        # predict
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ zeroshot_weights

        # measure accuracy
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += images.size(0)

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100
    return top1, top5


def eval_single_dataset(image_classifier, dataset, args, is_zero_shot=False):
    model = image_classifier
    input_key = "images"
    image_enc = None

    model.eval()

    zeroshot_weights = zeroshot_classifier(
        dataset.classnames, dataset.templates, model, is_zero_shot
    )

    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=image_enc
    )

    top1, top5 = zeroshot_eval(model, dataloader, zeroshot_weights)

    print(f"Top-1 accuracy: {top1:.2f}")
    # print(f"Top-5 accuracy: {top5:.2f}")
    return top1, top5


def evaluate(image_classifier, args, val_preprocess, tra_preprocess=None):
    if args.eval_datasets is None:
        return
    top1_list, top5_list = [], []
    for _, dataset_name in enumerate(args.eval_datasets):

        is_zero_shot = True

        if args.load is not None and dataset_name != 'ImageNet':
            train_on_dataset = args.load.split('/')[-1].split('.')[0]
            for name in args.eval_datasets:
                if name in args.load:
                    train_on_dataset = name
                    break
            if train_on_dataset in args.eval_datasets:
                is_zero_shot = args.eval_datasets.index(train_on_dataset) < args.eval_datasets.index(dataset_name)

        print("Evaluating on", dataset_name, ", zero-shot" if is_zero_shot else "")

        if (not is_zero_shot) and args.eval_only and (tra_preprocess is not None):
            preprocess = tra_preprocess
        else:
            preprocess = val_preprocess

        if dataset_name == 'ImageNet':
            dataset_class = getattr(datasets, dataset_name)
            dataset = dataset_class(
                preprocess,
                location=CL_DATA_LOCATION,
                batch_size=args.batch_size,
                batch_size_eval=args.batch_size_eval,
            )
        # TODO: add synthetic datasets
        else:
            dataset_class = getattr(datasets, dataset_name)
            dataset = dataset_class(
                preprocess,
                location=args.data_location,
                batch_size=args.batch_size,
                batch_size_eval=args.batch_size_eval,
            )

        top1, top5 = eval_single_dataset(image_classifier, dataset, args, is_zero_shot=is_zero_shot)
        top1_list.append(top1)
        top5_list.append(top5)
    return top1_list, top5_list