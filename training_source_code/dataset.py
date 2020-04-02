import os

import numpy as np
from PIL import Image
from jdit.dataset import DataLoadersFactory
from torch.utils.data import Dataset, random_split
from torchvision import transforms


class _RadonNoisedDataset(Dataset):
    def __init__(self, transform, root):
        self.transform = transforms.Compose(transform)
        names = os.listdir(r"%s/original" % root)
        self.paths_input = [os.path.join(r"%s/noised" % root, name.replace("png", "npy")) for name in names]
        self.paths_groundtruth = [os.path.join(r"%s/original" % root, name) for name in names]
        self.total = len(self.paths_input)

    def __len__(self):
        return len(self.paths_input)

    def __getitem__(self, i):
        path_input = self.paths_input[i]
        path_output = self.paths_groundtruth[i]
        # with Image.open(path_input) as img_input:
        #     img_input = img_input.convert("L")
        img_input = Image.fromarray(np.load(path_input).astype(float))
        with Image.open(path_output) as img_groundtruth:
            img_groundtruth = img_groundtruth.convert("L")
        img_input = self.transform(img_input)
        img_groundtruth = self.transform(img_groundtruth)

        return img_input, img_groundtruth


class RadonNoisedDatasets(DataLoadersFactory):
    def __init__(self, root, train_dir, batch_size=32, num_workers=-1, valid_size=100, shuffle=True):
        self.valid_size = valid_size
        self.train_dir = train_dir
        super(RadonNoisedDatasets, self).__init__(root, batch_size, num_workers=num_workers, shuffle=shuffle,
                                                  subdata_size=valid_size)

    def build_datasets(self):
        randon_dataset = _RadonNoisedDataset(self.transform,
                                             root=os.path.join(self.root, self.train_dir))
        self.dataset_train, self.dataset_valid = random_split(randon_dataset,
                                                              [randon_dataset.total - self.valid_size, self.valid_size])
        print("train size:%d    valid size:%d" % (randon_dataset.total - self.valid_size, self.valid_size))
        self.dataset_test = _RadonNoisedDataset(self.transform,
                                                root=self.root + "/test")

    def build_transforms(self, resize=256):
        self.transform = [
            transforms.ToTensor(),  # (H x W x C)=> (C x H x W)
            # transforms.Normalize([0.], [255.])
        ]
