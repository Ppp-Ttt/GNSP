import clip.clip as clip
from .args import parse_arguments
from .models import evaluate, evaluate_fc, evaluate_wise_ft, finetune, finetune_fc, finetune_icarl, finetune_gift

import json

with open('project_config.json', 'r') as f:
    project_config = json.load(f)

SYN_DATA_LOCATION = project_config['SYN_DATA_LOCATION']
CL_DATA_LOCATION = project_config['CL_DATA_LOCATION']
MTIL_DATA_LOCATION = project_config['MTIL_DATA_LOCATION']

def model_load():

    args = parse_arguments()
    args.train_mode = 'whole'
    # 选择数据集
    # ['Aircraft', 'Caltech101', 'CIFAR100', 'DTD', 'EuroSAT',
    # 'Flowers', 'Food', 'MNIST', 'OxfordPet', 'StanfordCars', 'SUN397']
    args.train_dataset = 'Aircraft'
    args.log_path = f'log/model_load_test.txt'
    args.lr = 1e-5 # 后续为1e-5
    args.ls = 0.2
    args.iterations = 1000
    args.l2 = 1
    args.ita = 0.25
    args.distill = 1
    args.method = 'GIFT'
    args.train_loss = 'cross_entropy'
    args.save = './checkpoint/exp_mtil_order_I'
    args.devices = [2]
    args.awc = True
    # args.load = "./checkpoint/exp_mtil_order_I/MNIST.pth"
    args.ref_model = 'load'
    args.ref_dataset = 'SyntheticDataset_A'

    # mtil_main(args)
    model, train_preprocess, val_preprocess = clip.load(args.model, jit=False)