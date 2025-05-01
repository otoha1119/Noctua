import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
from dataloader import DICOMDataset
from Encoder import VitEncoder
from Decoder import Decoder
from statistics import mean
def load_model(encoder_path, decoder_path, device):
    encoder = VitEncoder().to(device)
    decoder = Decoder().to(device)

    checkpoint = torch.load(encoder_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    encoder.eval()
    decoder.eval()

    return encoder, decoder

def calculate_metrics(labels, scores, threshold):
    # Calculate ROC AUC
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Calculate accuracy
    predictions = [1 if scores > threshold else 0 for scores in scores]
    accuracy = accuracy_score(labels, predictions)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

    # Calculate sensitivity and specificity
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return roc_auc, accuracy, sensitivity, specificity

def test_model(encoder, decoder, dataloader, device, threshold):
    all_labels = []
    all_scores = []
    maxmse=0
   
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(dataloader, desc="Testing")):
            images = images.to(device)
            labels = labels.to(device)

            b, s, c, h, w = images.size()
            images = images.view(-1, c, h, w)
            encoded = encoder(images)
            decoded = decoder(encoded)
            decoded = decoded.view(b, s, c, h, w)

            # 各スライスごとにMSEを計算
            mse = nn.MSELoss(reduction='none')
            reconstruction_error = mse(decoded, images.view(b, s, c, h, w))
            reconstruction_error = reconstruction_error.mean(dim=[2, 3, 4]).cpu().numpy()

            # 患者ごとのスライスの中で最大MSEを取得
            max_mse = reconstruction_error.max()
            
            maxmse=max_mse if max_mse > maxmse else maxmse #デバッグ用，全スライスの中で、もっとも高い
            # 最大MSEが閾値を超える場合は異常と判定
            patient_score = 1 if max_mse > threshold else 0

            all_labels.extend(labels.cpu().numpy())
            all_scores.append(patient_score)

    # 結果の評価
    roc_auc, accuracy, sensitivity, specificity = calculate_metrics(all_labels, all_scores, 0.5)
    print('MAXMSE=',maxmse)
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")

    return roc_auc, accuracy, sensitivity, specificity



def main():
    # Data and model paths
    data_dir = 'D:/workspace_torch/AnoViT/Dataset_Lung_cancer/test'
    data_transform = transforms.Compose([])

    # Initialize dataset and dataloader
    test_dataset = DICOMDataset(root_dir=data_dir, transform=data_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_path = "D:/workspace_torch/AnoViT/result0911/model6_0001.pth"  # Path to the saved encoder model
    decoder_path = "D:/workspace_torch/AnoViT/result0911/model6_0001.pth"   # Path to the saved decoder model

    encoder, decoder = load_model(encoder_path, decoder_path, device)

    # Set threshold for anomaly detection using MSE
    threshold = 0.0022 # Example threshold, this may need to be adjusted based on the data

    # Run the test model
    roc_auc, accuracy, sensitivity, specificity = test_model(encoder, decoder, test_loader, device, threshold)
    print('threshold:',threshold)
    print('Testing complete')

if __name__ == '__main__':
    main()
