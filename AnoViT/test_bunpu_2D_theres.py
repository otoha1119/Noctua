# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import roc_curve, auc, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import pandas as pd

# ============
# 必要なモジュール・クラスをインポート
# パスを環境に合わせて修正してください
# ============
from dataloader import DICOMDataset
from Encoder import VitEncoder
from Decoder import Decoder
from ViTInputlayer import VitInputLayer
# from AnoViT2D import AnoViT2D  # AnoViT2Dクラスをインポート（後述）

class AnoViT2D(nn.Module):
    def __init__(self, encoder, decoder):
        super(AnoViT2D, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

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
# テスト用のデータセットをロードする関数の例
# （dataloader.py の DICOMDataset 実装内容に応じて調整）
####################################################
def load_test_dataset(test_data_dir):
    """
    テスト用DataLoaderを用意する。
    - test_data_dir: テスト用のデータが格納されたディレクトリを指定
    """
    # transform が不要なら None。必要なら適宜定義してください。
    transform = None
    test_dataset = DICOMDataset(root_dir=test_data_dir, transform=transform)

    # DataLoader: バッチサイズ1(=患者or検査1つごと)を想定
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    return test_dataset, test_loader

####################################################
# シンプルなMSEベースの損失関数（AnoViT2Dの再構成誤差など）
####################################################
def compute_patient_mse_loss(recon_x, x):
    """
    再構成画像と元画像のスライスごとのMSEを計算し、その最大値を返す。
    (B=1, s, C, H, W) 形状を前提として計算
    """
    # 再構成誤差を計算: (B, s, C, H, W)
    mse = F.mse_loss(recon_x, x, reduction='none')  # (B, s, C, H, W)
    mse = mse.view(mse.size(0), mse.size(1), -1)  # (B, s, C*H*W)
    mse_per_slice = mse.mean(dim=2)  # (B, s)
    max_mse_per_patient, _ = mse_per_slice.max(dim=1)  # (B,)
    return max_mse_per_patient

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
    ただし、AnoViT2Dは2Dモデルなので、各スライスごとに保存します。

    Parameters:
    -----------
    outputs   : (B=1, s, C, H, W) の再構成画像テンソル
    inputs    : (B=1, s, C, H, W) の元画像テンソル
    sample_id : 保存ファイル名につけるサンプルIDなど
    recon_dir : 再構成画像を保存するベースディレクトリ
    heatmap_dir : ヒートマップを保存するベースディレクトリ
    """
    # テンソルを NumPy に変換
    recon_outputs = outputs.cpu().detach().numpy()[0]  # shape: (s, C, H, W)
    input_images = inputs.cpu().detach().numpy()[0]    # shape: (s, C, H, W)
    num_slices = recon_outputs.shape[0]

    # フォルダを作成
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(heatmap_dir, exist_ok=True)

    for slice_idx in range(num_slices):
        # ========================
        # 再構成画像のスライスを保存
        # ========================
        slice_recon = recon_outputs[slice_idx, 0]  # (H, W)
        #slice_recon_norm = np.clip(slice_recon, 0, 1)
        slice_recon_norm = (slice_recon-slice_recon.min())/(slice_recon.max()-slice_recon.min())
        slice_recon_scaled = (slice_recon_norm * 255).astype(np.uint8)
        recon_img = Image.fromarray(slice_recon_scaled, mode='L')

        recon_filename = f'{sample_id}_slice_{slice_idx+1:03d}.png'
        recon_filepath = os.path.join(recon_dir, recon_filename)
        recon_img.save(recon_filepath)

        # ========================
        # ヒートマップの生成・保存
        # ========================
        slice_input = input_images[slice_idx, 0]  # (H, W)
        diff_map = slice_recon - slice_input  # 差分(単純に引き算)
        heatmap_output = normalize_image(diff_map)  # 0~1に正規化

        plt.figure(figsize=(5, 5))
        plt.axis('off')
        plt.imshow(heatmap_output, cmap='jet', interpolation='nearest', vmin=0, vmax=1)
        heatmap_filename = f'{sample_id}_slice_{slice_idx+1:03d}_heatmap.png'
        heatmap_filepath = os.path.join(heatmap_dir, heatmap_filename)
        plt.savefig(heatmap_filepath, bbox_inches='tight', pad_inches=0)
        plt.close()

####################################################
# テストデータで推論し、再構成誤差などのスコアを収集 & 画像保存
####################################################
def inference_and_collect_scores(
    model,
    dataloader,
    recon_save_root=None,
    heatmap_save_root=None
):
    """
    テストデータローダーを用いて、各サンプル(患者)の:
      1) GTラベル(all_labels)
      2) 再構成誤差(all_scores)
    を推論・取得する。
    異常度スコアは各患者のスライスの中で最も高いMSEとする。

    さらに、引数で指定があれば、各サンプルの再構成結果とヒートマップを保存する。
    """

    model.eval()
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(tqdm(dataloader, desc='Testing')):
            labels = labels.to(device)
            inputs = inputs.to(device)

            # (B=1, s, C, H, W) 形状を前提
            if inputs.dim() == 4:
                inputs = inputs.unsqueeze(0)  # (B=1, s, C, H, W)

            # 推論: AnoViT2Dの再構成出力
            outputs = model(inputs.view(-1, inputs.size(2), inputs.size(3), inputs.size(4)))  # フラット化して2D画像として処理
            outputs = outputs.view(inputs.size(0), inputs.size(1), outputs.size(1), outputs.size(2), outputs.size(3))  # (B, s, C, H, W)

            # スコア(MSE)を計算
            loss_val = compute_patient_mse_loss(outputs, inputs).item()

            # リストに格納
            all_labels.append(labels.item())   # 0 or 1想定
            all_scores.append(loss_val)

            # ========= 再構成結果・ヒートマップを保存 (任意) =========
            if recon_save_root is not None and heatmap_save_root is not None:
                # サンプルごとの保存フォルダを作る
                sample_id = f'test_{idx+1:03d}'
                sample_recon_dir = os.path.join(recon_save_root, sample_id)
                sample_heatmap_dir = os.path.join(heatmap_save_root, sample_id)

                save_slices_and_heatmaps(
                    outputs,
                    inputs,
                    sample_id=sample_id,
                    recon_dir=sample_recon_dir,
                    heatmap_dir=sample_heatmap_dir
                )

    return all_labels, all_scores

####################################################
# 閾値を探索して目標感度に近づける関数
####################################################
def find_threshold_for_target_sensitivity(all_labels, all_scores, target_sensitivity=0.6, tolerance=0.01):
    """
    目標とする感度に最も近い閾値を探索する。

    Parameters:
    -----------
    all_labels: List[int]
        実際のラベル（0または1）。
    all_scores: List[float]
        異常度スコア。
    target_sensitivity: float
        目標とする感度（0~1）。
    tolerance: float
        許容誤差。

    Returns:
    --------
    best_threshold: float
        目標感度に最も近い閾値。
    achieved_sensitivity: float
        閾値により達成された感度。
    """
    # スコアをソートし重複を除去
    unique_scores = sorted(list(set(all_scores)))
    
    best_threshold = unique_scores[0]
    min_diff = float('inf')
    achieved_sensitivity = 0.0

    for threshold in unique_scores:
        pred_labels = [1 if s >= threshold else 0 for s in all_scores]
        cm = confusion_matrix(all_labels, pred_labels)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
        else:
            sensitivity = 0.0
        
        diff = abs(sensitivity - target_sensitivity)
        if diff < min_diff:
            min_diff = diff
            best_threshold = threshold
            achieved_sensitivity = sensitivity
        
        if min_diff <= tolerance:
            break  # 許容誤差内に収まったら終了

    return best_threshold, achieved_sensitivity

####################################################
# ROC-AUCや混同行列などの指標をまとめて計算
####################################################
def compute_metrics(all_labels, all_scores, threshold=None):
    """
    各種評価指標を計算する。
    - ROC曲線, AUC
    - YoudenのJ統計量による最適閾値（thresholdがNoneの場合）
    - 指定閾値による混同行列, Accuracy, Sensitivity, Specificity（thresholdが指定された場合）

    Parameters:
    -----------
    all_labels: List[int]
        実際のラベル。
    all_scores: List[float]
        異常度スコア。
    threshold: float, optional
        使用する閾値。指定しない場合はYoudenのJ統計量に基づく閾値を使用。

    Returns:
    --------
    metrics: dict
        計算された評価指標。
    """
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)

    if threshold is None:
        # YoudenのJ統計量 (tpr - fpr) を最大化する閾値
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
        optimal_threshold = thresholds[optimal_idx]

        # 最適閾値で予測ラベルを作成
        pred_labels = [1 if s >= optimal_threshold else 0 for s in all_scores]
    else:
        # 指定閾値で予測ラベルを作成
        optimal_threshold = threshold
        pred_labels = [1 if s >= threshold else 0 for s in all_scores]

    # 混同行列 (2クラス)
    cm = confusion_matrix(all_labels, pred_labels)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

    return {
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'optimal_threshold': optimal_threshold,
        'cm': cm,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity
    }

####################################################
# ROC曲線の描画・保存
####################################################
def plot_roc_curve(fpr, tpr, roc_auc, save_path, prefix=''):
    os.makedirs(save_path, exist_ok=True)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve {prefix}')
    plt.legend(loc="lower right")

    roc_filename = os.path.join(save_path, f'roc_curve_{prefix}.png')
    plt.savefig(roc_filename)
    plt.close()
    print(f'[Info] ROC曲線が保存されました: {roc_filename}')

####################################################
# 混同行列のヒートマップ描画・保存
####################################################
def plot_confusion_matrix(cm, save_path, prefix=''):
    """
    cm : shape=(2,2) のみ対象
    """
    os.makedirs(save_path, exist_ok=True)
    if cm.shape != (2, 2):
        print('[Warning] 混同行列が2x2ではありません。プロットをスキップします。')
        return

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred:0', 'Pred:1'],
                yticklabels=['GT:0', 'GT:1'])
    plt.title(f'Confusion Matrix {prefix}')

    cm_filename = os.path.join(save_path, f'cm_matrix_{prefix}.png')
    plt.savefig(cm_filename)
    plt.close()
    print(f'[Info] 混同行列が保存されました: {cm_filename}')

####################################################
# ラベル別の異常度スコア分布をプロット・保存する関数
####################################################
def plot_score_distribution(all_labels, all_scores, save_dir, filename='score_distribution.png'):
    """
    ラベルごとの異常度スコアの分布をヒストグラムでプロットし、画像として保存する。
    
    Parameters:
    - all_labels: List[int]
        ラベル（例: 0=正常, 1=異常）
    - all_scores: List[float]
        異常度スコア
    - save_dir: str
        プロット画像を保存するディレクトリ
    - filename: str
        保存するファイル名
    """
    os.makedirs(save_dir, exist_ok=True)

    # DataFrameを作成してラベル別にスコアをまとめる
    df = pd.DataFrame({'label': all_labels, 'score': all_scores})

    plt.figure(figsize=(10, 6))
    
    # カラーパレットの定義: 0 -> blue, 1 -> orange
    palette = {0: 'blue', 1: 'orange'}
    
    sns.histplot(
        data=df,
        x='score',
        hue='label',
        palette=palette,
        kde=False,
        bins=50,
        element='step',
        stat="density"
    )
    
    plt.title('Distribution of Anomaly Scores by Label')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    
    # Update legend labels to match the colors
    plt.legend(title='Label', labels=['Abnormal', 'normal'])
    
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f'[Info] Score distribution plot saved at: {save_path}')
####################################################
# メイン処理 (テスト専用)
####################################################
def main():
    # ======== 必要に応じてシード固定 ========
    set_seed(42)

    # ======== パス設定（環境に合わせて変更） ========
    test_data_dir = r'D:\workspace_torch\data\Dataset_800_60_100\test'           # テスト用データがあるディレクトリ
    model_path = r'D:\workspace_torch\best_model_AnoViT.pth'      # 学習時に保存したモデル
    saved_threshold = 0.0009  # 学習時に算出した最適閾値

    # 結果保存用ディレクトリ
    result_root = r'D:\workspace_torch\4モデル比較用\AnoViT\AnoViT2D_test_results3'
    roc_dir = os.path.join(result_root, 'roc_curves')
    cm_dir = os.path.join(result_root, 'confusion_matrix')
    distribution_dir = os.path.join(result_root, 'score_distribution')  # スコア分布の保存先

    # 再構成結果の保存先 (任意) - None にすれば保存しない
    recon_save_root = os.path.join(result_root, 'recon_images')
    heatmap_save_root = os.path.join(result_root, 'heatmaps')

    # ======== データセットの読み込み ========
    test_dataset, test_loader = load_test_dataset(test_data_dir)
    print(f'[Info] テストデータセットが読み込まれました: {len(test_dataset)} サンプル')

    # ======== モデルの定義 & ロード ========
    # モデルの定義
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
   
    model = AnoViT2D(encoder, decoder).to(device)  # モデルをデバイスに移動

    # モデルのロード
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f'[Info] モデルがロードされました: {model_path}')

    # ======== テストデータで推論 & 画像保存 ========
    all_labels, all_scores = inference_and_collect_scores(
        model,
        test_loader,
        # recon_save_root=None,     # None にすれば保存しない
        # heatmap_save_root=None  
        recon_save_root=recon_save_root,     # 任意で設定
        heatmap_save_root=heatmap_save_root  # 任意で設定
    )

    # ======== Youden閾値に基づく評価指標計算 ========
    metrics_youden = compute_metrics(all_labels, all_scores)
    fpr_youden, tpr_youden, roc_auc_youden = metrics_youden['fpr'], metrics_youden['tpr'], metrics_youden['roc_auc']
    cm_youden = metrics_youden['cm']
    accuracy_youden = metrics_youden['accuracy']
    sensitivity_youden = metrics_youden['sensitivity']
    specificity_youden = metrics_youden['specificity']
    computed_opt_thr = metrics_youden['optimal_threshold']

    # --- [1] Youden閾値での結果 ---
    print('==================== テスト結果 (Youden) ====================')
    print(f'[ROC-AUC]: {roc_auc_youden:.4f}')
    print(f'[最適閾値 (Youden) ]: {computed_opt_thr:.4f}')
    print(f'[Accuracy   ]: {accuracy_youden:.4f}')
    print(f'[Sensitivity]: {sensitivity_youden:.4f}')
    print(f'[Specificity]: {specificity_youden:.4f}')
    print('===============================================================')

    # ======== 学習時の閾値での評価指標計算 ========
    metrics_saved_thr = compute_metrics(all_labels, all_scores, threshold=saved_threshold)
    cm_saved_thr = metrics_saved_thr['cm']
    accuracy_saved = metrics_saved_thr['accuracy']
    sensitivity_saved = metrics_saved_thr['sensitivity']
    specificity_saved = metrics_saved_thr['specificity']

    print('==================== テスト結果 (保存された閾値) ================')
    print(f'[保存された閾値 ]: {saved_threshold:.4f}')
    print(f'[Accuracy   ]: {accuracy_saved:.4f}')
    print(f'[Sensitivity]: {sensitivity_saved:.4f}')
    print(f'[Specificity]: {specificity_saved:.4f}')
    print('===============================================================')

    # ======== 目標感度に基づく閾値探索 ========
    target_sensitivity = 0.55  # 目標感度
    found_threshold, achieved_sensitivity = find_threshold_for_target_sensitivity(
        all_labels,
        all_scores,
        target_sensitivity=target_sensitivity,
        tolerance=0.005  # 例: 0.005の許容誤差
    )
    print('==================== テスト結果 (目標感度) ============')
    print(f'[目標感度]: {target_sensitivity:.3f}')
    print(f'[探索された閾値]: {found_threshold:.4f}')
    print(f'[達成された感度]: {achieved_sensitivity:.4f}')
    print('=============================================================')

    # ======== 目標感度に基づく評価指標計算 ========
    metrics_target = compute_metrics(all_labels, all_scores, threshold=found_threshold)
    cm_target = metrics_target['cm']
    accuracy_target = metrics_target['accuracy']
    sensitivity_target = metrics_target['sensitivity']
    specificity_target = metrics_target['specificity']

    print('==================== テスト結果 (探索された閾値) ================')
    print(f'[探索された閾値]: {found_threshold:.4f}')
    print(f'[Accuracy   ]: {accuracy_target:.4f}')
    print(f'[Sensitivity]: {sensitivity_target:.4f}')
    print(f'[Specificity]: {specificity_target:.4f}')
    print('===============================================================')

    # ======== ROC曲線と混同行列の保存 ========
    # ROC曲線 (Youdenベース)
    plot_roc_curve(fpr_youden, tpr_youden, roc_auc_youden, roc_dir, prefix='Youden')

    # 混同行列 (Youdenベース)
    plot_confusion_matrix(cm_youden, cm_dir, prefix='Youden')

    # 混同行列 (Saved Thr)
    if cm_saved_thr.shape == (2, 2):
        plot_confusion_matrix(cm_saved_thr, cm_dir, prefix=f'SavedThr_{saved_threshold:.4f}')
    else:
        print('[Warning] cm_saved_thr の形状が (2,2) ではありません。プロットをスキップします。')

    # 混同行列 (Found Threshold)
    if cm_target.shape == (2, 2):
        plot_confusion_matrix(cm_target, cm_dir, prefix=f'FoundThr_{found_threshold:.4f}')
    else:
        print('[Warning] cm_target の形状が (2,2) ではありません。プロットをスキップします。')

    # ======== ラベル別の異常度スコア分布のプロット・保存 ========
    plot_score_distribution(all_labels, all_scores, distribution_dir)

    # ======== 結果をCSVに保存 ========
    metrics_df = pd.DataFrame({
        'Method': ['Youden', 'Saved_Thr', 'Target_Sensitivity'],
        'Threshold': [computed_opt_thr, saved_threshold, found_threshold],
        'Accuracy': [accuracy_youden, accuracy_saved, accuracy_target],
        'Sensitivity': [sensitivity_youden, sensitivity_saved, sensitivity_target],
        'Specificity': [specificity_youden, specificity_saved, specificity_target]
    })

    metrics_csv_path = os.path.join(result_root, 'evaluation_metrics.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f'[Info] 評価指標がCSVに保存されました: {metrics_csv_path}')

if __name__ == '__main__':
    main()
