# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import roc_curve, auc, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import pandas as pd  # 追加: pandas をインポート

from dataloader import DICOMDataset
from Encoder import VitEncoder
from Decoder import Decoder
from ViTInputlayer import VitInputLayer
# from AnoViT2D import AnoViT2D  # AnoViT2Dクラスを別ファイルに分割している場合

class AnoViT2D(nn.Module):
    def __init__(self, encoder, decoder):
        super(AnoViT2D, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# デバイスの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

####################################################
# シード値の固定（再現性が必要な場合のみ使用）
####################################################
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

####################################################
# データセットとデータローダーを準備する関数
####################################################
def prepare_dataloaders(data_dir, batch_size=1, num_workers=1):
    """
    データセットとデータローダーを準備する。

    Parameters:
    -----------
    data_dir : str
        データが格納されたディレクトリのパス。
    batch_size : int
        バッチサイズ。
    num_workers : int
        DataLoaderのワーカー数。

    Returns:
    --------
    dataloaders : dict
        'train' と 'val' の DataLoader を含む辞書。
    dataset_sizes : dict
        'train' と 'val' のデータセットのサイズを含む辞書。
    """
    data_transforms = {
        'train': None,  # 必要に応じてトランスフォームを追加
        'val': None
    }

    image_datasets = {
        'train': DICOMDataset(root_dir=os.path.join(data_dir, 'train'), transform=data_transforms['train']),
        'val': DICOMDataset(root_dir=os.path.join(data_dir, 'val'), transform=data_transforms['val'])
    }

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    return dataloaders, dataset_sizes

####################################################
# 損失関数の定義
####################################################
def get_criterion():
    return nn.MSELoss()

####################################################
# モデルとオプティマイザーの準備
####################################################
def prepare_model_and_optimizer():
    """
    モデルとオプティマイザーを準備する。

    Returns:
    --------
    model : nn.Module
        AnoViT2Dモデル。
    optimizer : torch.optim.Optimizer
        Adamオプティマイザー。
    """
    input_layer = VitInputLayer(
        in_channels=1,
        emb_dim=64,
        num_patch_row=8,
        image_size=128
    )

    encoder = VitEncoder(
        in_channels=1,
        emb_dim=64,
        num_patch_row=8,
        image_size=128,
        num_blocks=8,
        dropout=0.2,
        head=16
    )

    decoder = Decoder(
        emb_dim=64,
        image_size=128,
        num_patch_row=8
    )
    
    model = AnoViT2D(encoder, decoder).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3) # 学習率の設定

    return model, optimizer

####################################################
# 画像正規化のためのヘルパー関数
####################################################
def normalize_image(img):
    """
    画像を[0, 1]範囲に正規化する。
    """
    img_min, img_max = img.min(), img.max()
    denom = (img_max - img_min) + 1e-8
    return (img - img_min) / denom

####################################################
# 再構成結果スライスとヒートマップを保存する関数
####################################################
def save_slices_and_heatmaps(outputs, inputs, sample_id, recon_dir, heatmap_dir):
    """
    再構成画像および対応するヒートマップをスライスごとに保存する。

    Parameters:
    -----------
    outputs   : (B=1, C=1, H, W) の再構成画像テンソル
    inputs    : (B=1, C=1, H, W) の元画像テンソル
    sample_id : 保存ファイル名につけるサンプルIDなど
    recon_dir : 再構成画像を保存するベースディレクトリ
    heatmap_dir : ヒートマップを保存するベースディレクトリ
    """
    # テンソルを NumPy に変換
    recon_output = outputs.cpu().detach().numpy()[0, 0]  # shape: (H, W)
    input_image = inputs.cpu().detach().numpy()[0, 0]    # shape: (H, W)

    # フォルダを作成
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(heatmap_dir, exist_ok=True)

    # 再構成画像の保存
    slice_recon_norm = np.clip(recon_output, 0, 1)
    slice_recon_scaled = (slice_recon_norm * 255).astype(np.uint8)
    recon_img = Image.fromarray(slice_recon_scaled, mode='L')

    recon_filename = f'{sample_id}_recon.png'
    recon_filepath = os.path.join(recon_dir, recon_filename)
    recon_img.save(recon_filepath)

    # ヒートマップの生成・保存
    diff_map = recon_output - input_image  # 差分
    heatmap_output = normalize_image(diff_map)  # 0~1に正規化

    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(heatmap_output, cmap='jet', interpolation='nearest', vmin=-1, vmax=1)
    heatmap_filename = f'{sample_id}_heatmap.png'
    heatmap_filepath = os.path.join(heatmap_dir, heatmap_filename)
    plt.savefig(heatmap_filepath, bbox_inches='tight', pad_inches=0)
    plt.close()

####################################################
# ROC曲線の描画・保存
####################################################
def plot_roc_curve(fpr, tpr, roc_auc, save_dir, epoch):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - Epoch {epoch+1}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, f'roc_epoch_{epoch+1}.png'))
    plt.close()

####################################################
# 混同行列の描画・保存
####################################################
def plot_confusion_matrix(cm, save_dir, epoch):
    os.makedirs(save_dir, exist_ok=True)
    if cm.shape != (2, 2):
        print('[Warning] Confusion matrix is not 2x2, skip plotting.')
        return

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred:0', 'Pred:1'],
                yticklabels=['GT:0', 'GT:1'])
    plt.title(f'Confusion Matrix - Epoch {epoch+1}')
    cm_filename = os.path.join(save_dir, f'cm_epoch_{epoch+1}.png')
    plt.savefig(cm_filename)
    plt.close()

####################################################
# 異常度スコア分布のプロット・保存
####################################################
def plot_score_distribution(all_labels, all_scores, save_dir, epoch):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame({'label': all_labels, 'score': all_scores})

    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='score', hue='label', kde=True, bins=30, element='step', stat="density")
    plt.title(f'Distribution of Anomaly Scores by Label - Epoch {epoch+1}')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.legend(title='Label', labels=['Normal (0)', 'Anomaly (1)'])

    save_path = os.path.join(save_dir, f'score_distribution_epoch_{epoch+1}.png')
    plt.savefig(save_path)
    plt.close()
    print(f'[Info] Score distribution plot saved at: {save_path}')

####################################################
# トレーニングと検証のループ
####################################################
def train_model(
    model,
    criterion,
    optimizer,
    dataloaders,
    dataset_sizes,
    num_epochs=50,
    patience=50,
    roc_dir='roc_curves',
    cm_dir='confusion_matrix',
    recon_dir='reconstructed_images',
    heatmap_dir='heatmaps'
):
    """
    モデルのトレーニングと検証を行う。

    Parameters:
    -----------
    model : nn.Module
        トレーニングするモデル。
    criterion : nn.Module
        損失関数。
    optimizer : torch.optim.Optimizer
        オプティマイザー。
    dataloaders : dict
        'train' と 'val' の DataLoader を含む辞書。
    dataset_sizes : dict
        'train' と 'val' のデータセットのサイズを含む辞書。
    num_epochs : int
        トレーニングするエポック数。
    patience : int
        早期終了のための忍耐期間。
    roc_dir : str
        ROC曲線を保存するディレクトリ。
    cm_dir : str
        混同行列を保存するディレクトリ。
    recon_dir : str
        再構成画像を保存するディレクトリ。
    heatmap_dir : str
        ヒートマップを保存するディレクトリ。

    Returns:
    --------
    model : nn.Module
        最良の検証損失を持つモデル。
    history : dict
        トレーニングと検証の損失履歴。
    best_epoch : int
        最良の検証損失を達成したエポック番号。
    """
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = -1
    history = {'train_loss': [], 'val_loss': []}

    # カラーマップを一度だけ初期化
    colormap = plt.get_cmap('jet')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            # ROC曲線とAUCのためのリスト（検証フェーズのみ）
            all_labels = []
            all_scores = []

            # 最終エポックの検証フェーズでのみ再構成画像を保存するためのフラグとカウンターを初期化
            if phase == 'val' and epoch == num_epochs - 1:
                normal_saved = False
                abnormal_saved = False
                patient_counter = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f'{phase} Progress'):
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    # 入力形状を調整（バッチサイズとシーケンス長を考慮）
                    b, s, c, h, w = inputs.size()
                    inputs_reshaped = inputs.view(-1, c, h, w)
                    outputs = model(inputs_reshaped)
                    outputs = outputs.view(b, s, c, h, w)
                    inputs = inputs.view(b, s, c, h, w)

                    # 損失の計算
                    loss = criterion(outputs, inputs)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                if phase == 'val':
                    recon_error = loss.item()
                    all_labels.append(labels.item())
                    all_scores.append(recon_error)

                    # 最終エポックの検証フェーズでのみ再構成画像とヒートマップを保存
                    if epoch == num_epochs - 1:
                        patient_counter += 1
                        label = labels.item()
                        save_dir_recon = None
                        save_dir_heatmap = None

                        if label == 0 and not normal_saved:
                            save_dir_recon = os.path.join(recon_dir, 'normal')
                            save_dir_heatmap = os.path.join(heatmap_dir, 'heatmap_normal')
                            normal_saved = True
                        elif label == 1 and not abnormal_saved:
                            save_dir_recon = os.path.join(recon_dir, 'abnormal')
                            save_dir_heatmap = os.path.join(heatmap_dir, 'heatmap_abnormal')
                            abnormal_saved = True

                        if save_dir_recon and save_dir_heatmap:
                            sample_id = f'patient_{patient_counter}'
                            save_slices_and_heatmaps(
                                outputs[0, 0],  # (C, H, W) -> (H, W)
                                inputs[0, 0],
                                sample_id,
                                recon_dir=save_dir_recon,
                                heatmap_dir=save_dir_heatmap
                            )

            epoch_loss = running_loss / dataset_sizes[phase]
            history[f'{phase}_loss'].append(epoch_loss)
            print(f'{phase} Loss: {epoch_loss:.4f}')

            if phase == 'val':
                # ROC曲線とAUCの計算
                fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
                roc_auc = auc(fpr, tpr)
                print(f'Validation AUC: {roc_auc:.4f}')

                # ROC曲線のプロット
                plot_roc_curve(fpr, tpr, roc_auc, roc_dir, epoch)

                # 閾値の選択（YoudenのJ統計量を最大化する閾値）
                youden_index = tpr - fpr
                optimal_idx = np.argmax(youden_index)
                optimal_threshold = thresholds[optimal_idx]
                print(f'Optimal Threshold: {optimal_threshold:.4f}')

                # 予測ラベルの計算
                pred_labels = [1 if score >= optimal_threshold else 0 for score in all_scores]

                # 混同行列の計算
                if len(set(all_labels)) > 1:
                    tn, fp, fn, tp = confusion_matrix(all_labels, pred_labels).ravel()
                else:
                    tn, fp, fn, tp = 0, 0, 0, 0
                    print("ラベルに一種類しか存在しません。混同行列は定義されていません。")

                # 評価指標の計算
                accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
                sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

                print(f'Accuracy: {accuracy:.4f}')
                print(f'Sensitivity (Recall): {sensitivity:.4f}')
                print(f'Specificity: {specificity:.4f}')

                # 混同行列のプロット
                cm = confusion_matrix(all_labels, pred_labels)
                plot_confusion_matrix(cm, cm_dir, epoch)

                # スコア分布のプロット
                plot_score_distribution(all_labels, all_scores, save_dir=roc_dir, epoch=epoch)

                # 早期終了のチェック
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    patience_counter = 0
                    best_epoch = epoch + 1
                    # モデルの保存
                    torch.save(model.state_dict(), 'best_model_AnoViT.pth')
                    print('Best model updated and saved.')
                else:
                    patience_counter += 1
                    print(f'No improvement in validation loss for {patience_counter} epoch(s).')
                    if patience_counter >= patience:
                        print('Early stopping triggered.')
                        return model, history, best_epoch

        print('-----------------------------------------------------')
    
    print('Training completed without early stopping.')
    return model, history, best_epoch

####################################################
# トレーニング履歴のプロット・保存
####################################################
def plot_loss_curve(history, save_path='loss_curve.png'):
    plt.figure()
    plt.plot(range(1, len(history['train_loss'])+1), history['train_loss'], label='Training Loss')
    plt.plot(range(1, len(history['val_loss'])+1), history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f'[Info] Loss curve saved as {save_path}')

####################################################
# メイン処理
####################################################
def main():
    # ======== 必要に応じてシード固定 ========
    set_seed(42)

    # ======== データセットとデータローダーの準備 ========
    data_dir = r'D:\workspace_torch\data\Dataset_800_60_100'
    dataloaders, dataset_sizes = prepare_dataloaders(data_dir, batch_size=1, num_workers=1)
    print(f'[Info] Loaded train and validation datasets: {dataset_sizes}')

    # ======== モデルとオプティマイザーの準備 ========
    model, optimizer = prepare_model_and_optimizer()
    print('[Info] Model and optimizer are initialized.')

    # ======== 損失関数の準備 ========
    criterion = get_criterion()

    # ======== 結果保存用ディレクトリの設定 ========
    roc_dir = r'D:\workspace_torch\model\AnoViT\2D_roc_curves0217'
    cm_dir = r'D:\workspace_torch\model\AnoViT\2D_confusion_matrices0217'
    os.makedirs(roc_dir, exist_ok=True)
    os.makedirs(cm_dir, exist_ok=True)

    # 再構成画像とヒートマップの保存先ディレクトリ（サブディレクトリを作成）
    recon_dir = r'D:\workspace_torch\model\AnoViT\2D_reconstructed_images0217'
    heatmap_dir = r'D:\workspace_torch\model\AnoViT\2D_heatmaps0217'
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(heatmap_dir, exist_ok=True)

    # ======== トレーニングの実行 ========
    model, history, best_epoch = train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        num_epochs=50,
        patience=50,
        roc_dir=roc_dir,
        cm_dir=cm_dir,
        recon_dir=recon_dir,
        heatmap_dir=heatmap_dir
    )

    print('Training completed.')

    # ======== 損失曲線のプロットと保存 ========
    plot_loss_curve(history, save_path=r'D:\workspace_torch\model\AnoViT\2D_loss_curve0217.png')

    # 最良のエポックを表示
    print(f'Best validation loss achieved at epoch {best_epoch}')

if __name__ == '__main__':
    main()
