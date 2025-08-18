import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader

class CocoCaptions(Dataset):
    def __init__(self,
                 preprocess=None,
                 location=os.path.expanduser('./data/CocoCaptions'),
                 batch_size=64,
                 batch_size_eval=64,
                 image_nums=1000,
                 seed=42,
                 num_workers=8,
                 shuffle=False):
        self.preprocess = preprocess
        self.location = os.path.expanduser('./data/CocoCaptions/val2014')
        self.batch_size = batch_size
        self.eval_batch_size = batch_size_eval
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.image_nums = image_nums
        self.seed = seed

        self.name = "CocoCaptions"
        self.dataset = datasets.CocoCaptions(
            root=self.location,
            annFile=os.path.join('./data/CocoCaptions/', "annotations/captions_val2014.json"),
            transform=self.preprocess
        )

    # def load_synset_map(self):
    #     """Load mapping from synset ID (e.g., n01440764) to class name."""
    #     class_map = {}
    #     with open(self.synset_map_file, 'r') as f:
    #         for line in f:
    #             synset_id, class_names = line.strip().split(' ', 1)
    #             # Take the first class name (before comma) for simplicity
    #             class_name = class_names.split(',')[0].replace('_', ' ')
    #             class_map[synset_id] = class_name
    #     return class_map
    #
    # def is_black(self, image_path):
    #     """Check if an image is completely black."""
    #     try:
    #         image = Image.open(image_path).convert('RGB')
    #         array = np.array(image)
    #         return np.all(array == 0)
    #     except:
    #         return True  # Skip corrupted images

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    # def populate_train(self):
    #     self.train_loader = DataLoader(
    #         self,
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #         shuffle=self.shuffle,
    #         # pin_memory=True
    #     )
    #
    # def populate_test(self):
    #     self.test_loader = DataLoader(
    #         self,
    #         batch_size=self.eval_batch_size,
    #         num_workers=self.num_workers,
    #         shuffle=False,
    #         # pin_memory=True
    #     )
