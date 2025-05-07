import os
import torch
from torch.utils.data import Dataset, DataLoader
import pydicom
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

class DICOMDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.patient_paths = []
        self.labels = []

        # 上位フォルダ（normal, abnormal）をラベルとして設定
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                for patient_folder in os.listdir(label_dir):
                    patient_folder_path = os.path.join(label_dir, patient_folder)
                    if os.path.isdir(patient_folder_path):
                        self.patient_paths.append(patient_folder_path)
                        self.labels.append(label)  # normalやabnormalのフォルダ名をラベルとして追加

        # 'normal' を 0 に、'abnormal' を 1 にラベル付け
        self.label_to_idx = {'normal': 0, 'abnormal': 1}
        self.labels = [self.label_to_idx[label] for label in self.labels]  # ラベルを数値に変換

    def __len__(self):
        return len(self.patient_paths)

    def __getitem__(self, idx):
        folder_path = self.patient_paths[idx]
        dicom_files = []

        # フォルダ内のすべてのDICOM画像を収集し、imagenumberでソート
        for img_name in os.listdir(folder_path):
            if img_name.lower().endswith('.dcm'):
                img_path = os.path.join(folder_path, img_name)
                dicom_files.append(img_path)

        dicom_files.sort(key=lambda x: pydicom.dcmread(x).InstanceNumber)

        images = []
        for img_path in dicom_files:
            dicom = pydicom.dcmread(img_path)
            image = dicom.pixel_array

            # 画像を正規化
            image[image <= 0] = 0
            image = image / 4095
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # チャネル次元を追加

            if self.transform:
                image = self.transform(image)

            images.append(image)

        images = torch.stack(images)  # ソート済みのすべての画像をテンソルとしてスタック

        label = self.labels[idx]  # 対応するラベルを取得
        
        return images, label

'''    
#if __name__ == '__main__':
 # データセットの定義
data_dir = 'D:/workspace_torch/AnoViT/Dataset_Lung_cancer'

data_transforms = {
    'train': transforms.Compose([]),
    'val': transforms.Compose([]),
    'test': transforms.Compose([]),
 }

# データセットの初期化
image_datasets = {}
for x in ['train', 'val', 'test']:
    dataset = DICOMDataset(root_dir=os.path.join(data_dir, x), transform=data_transforms[x])
    if len(dataset) > 0:  # データセットが空でないことを確認
        image_datasets[x] = dataset

# データローダーの定義
dataloaders = {}
for x in image_datasets:
    dataloaders[x] = DataLoader(image_datasets[x], batch_size=1, shuffle=True, num_workers=4)

dataset_sizes = {x: len(image_datasets[x]) for x in image_datasets}
'''