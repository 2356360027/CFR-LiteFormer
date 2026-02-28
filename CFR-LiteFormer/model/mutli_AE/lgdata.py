import numpy
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import os


class MyCustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = os.listdir(root_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 假设数据集中的文件是图像，使用PIL加载图像
        image_path = os.path.join(self.root_dir, self.data[idx])
        image = Image.open(image_path).convert("RGB")

        transform = transforms.Compose([

            transforms.ToTensor()
        ])
        image = transform(image)

        return image

