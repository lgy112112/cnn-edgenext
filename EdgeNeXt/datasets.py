import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch

class PairedDataset(Dataset):
    def __init__(self, csv_file, methods=['CWT', 'STFT'], snr='20', transform=None):
        self.data = pd.read_csv(csv_file)
        
        # 过滤数据
        self.data = self.data[(self.data['method'].isin(methods)) & (self.data['SNR'] == int(snr))]
        
        # 提取目标类别并进行标签编码
        self.targets = self.data['target'].unique()
        self.target_to_idx = {target: idx for idx, target in enumerate(self.targets)}
        self.data['label'] = self.data['target'].map(self.target_to_idx)
        
        # 提取图像名称（用于匹配）
        self.data['image_name'] = self.data['image_path'].apply(lambda x: os.path.basename(x))
        
        # 创建匹配的图像对
        self.paired_data = []
        image_names = self.data['image_name'].unique()
        for image_name in image_names:
            group = self.data[self.data['image_name'] == image_name]
            if len(group) == len(methods):  # 确保两个方法都有对应的图像
                method_images = {}
                label = None
                for _, row in group.iterrows():
                    method_images[row['method']] = row['image_path']
                    label = row['label']
                self.paired_data.append({
                    'image_name': image_name,
                    'images': method_images,
                    'label': label
                })
        
        self.transform = transform

    def __len__(self):
        return len(self.paired_data)

    def __getitem__(self, idx):
        sample = self.paired_data[idx]
        images = {}
        for method in sample['images']:
            image_path = sample['images'][method]
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images[method] = image
        label = sample['label']
        return images, label

# 定义图像的转换
transform = transforms.Compose([
    # transforms.Resize((128, 128)),
    transforms.ToTensor(),
])