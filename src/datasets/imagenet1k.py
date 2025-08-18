import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class ImageNet1K(Dataset):
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('./data/ImageNet-1K/ILSVRC/Data/CLS-LOC/train'),
                 synset_map_file=os.path.expanduser('./data/ImageNet-1K/LOC_synset_mapping.txt'),
                 batch_size=64,
                 batch_size_eval=64,
                 image_nums=1000,
                 seed=42,
                 num_workers=8,
                 shuffle=False):
        self.preprocess = preprocess
        self.location = location
        self.synset_map_file = synset_map_file
        self.batch_size = batch_size
        self.eval_batch_size = batch_size_eval
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.image_nums = image_nums
        self.seed = seed

        # Load synset to class name mapping
        self.class_map = self.load_synset_map()

        # Collect all image files from subfolders
        self.image_files = []
        self.class_ids = []
        for class_id in os.listdir(self.location):
            class_path = os.path.join(self.location, class_id)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_file)
                    self.image_files.append(img_path)
                    self.class_ids.append(class_id)

        # Limit the number of images if specified
        if self.image_nums is not None:
            if self.seed is not None:
                rng = np.random.RandomState(self.seed)  # 创建 RandomState 实例
                indices = rng.choice(len(self.image_files), self.image_nums, replace=False)
            else:
                indices = np.random.choice(len(self.image_files), self.image_nums, replace=False)
            self.image_files = [self.image_files[i] for i in indices]
            self.class_ids = [self.class_ids[i] for i in indices]

        # Create prompt dictionary: map image path to text description
        self.prompt_dict = {}
        for img_path, class_id in zip(self.image_files, self.class_ids):
            class_name = self.class_map.get(class_id, 'unknown')
            # Use 'an' for vowel-starting words, 'a' otherwise
            article = 'an' if class_name[0].lower() in 'aeiou' else 'a'
            self.prompt_dict[img_path] = f"a photo of {article} {class_name}"

        # Create label dictionary: map image path to index of unique prompts
        self.all_prompts = list(set(self.prompt_dict.values()))
        self.label_dict = {img_path: self.all_prompts.index(prompt)
                           for img_path, prompt in self.prompt_dict.items()}

        self.populate_train()
        # self.populate_test()

    def load_synset_map(self):
        """Load mapping from synset ID (e.g., n01440764) to class name."""
        class_map = {}
        with open(self.synset_map_file, 'r') as f:
            for line in f:
                synset_id, class_names = line.strip().split('\t', 1)
                # Take the first class name (before comma) for simplicity
                class_name = class_names.split(',')[0].replace('_', ' ')
                class_map[synset_id] = class_name
        return class_map

    def is_black(self, image_path):
        """Check if an image is completely black."""
        try:
            image = Image.open(image_path).convert('RGB')
            array = np.array(image)
            return np.all(array == 0)
        except:
            return True  # Skip corrupted images

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')

        if self.preprocess:
            image = self.preprocess(image)

        return {
            'images': image,
            'labels': self.label_dict[image_path],
            'texts': self.prompt_dict[image_path]
        }

    def populate_train(self):
        self.train_loader = DataLoader(
            self,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            # pin_memory=True
        )

    def populate_test(self):
        self.test_loader = DataLoader(
            self,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            # pin_memory=True
        )