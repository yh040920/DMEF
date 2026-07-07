import numpy as np
import cv2
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import nibabel
import os
from model_MoE_plus import feature_Netin_tp4_multr_score


class VisualizationModel(nn.Module):
    def __init__(self, model_path, device, use_moe="MoE", top_k=2, dropout=0.0):
        super(VisualizationModel, self).__init__()
        self.model = feature_Netin_tp4_multr_score(dropout=dropout, use_moe=use_moe, top_k=top_k)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model = self.model.to(device)
        self.model.eval()
        self.ta_features = {}
        print(f"Load model from: {model_path}")

    def forward(self, x):
        return self.model(x)

    def forward_with_ta_features(self, x):
        features_x = self.model.feature_extractor(x)

        if self.model.use_moe == "MoE":
            shared_pooled = self.model.shared_pool(features_x)
            shared_pooled = shared_pooled.view(shared_pooled.shape[0], -1)

        features_x1_locat = self.model.locat1(features_x)
        map1 = self.model.ca1(features_x1_locat)
        ta1 = self.model.de1(features_x1_locat, map1) * features_x1_locat
        ta1.retain_grad()
        self.ta_features['ta1'] = ta1
        features_x1 = self.model.pool1(ta1)
        features_x1 = features_x1.view(features_x1.shape[0], -1)
        features_x1 = self.model.expert_fc1(features_x1)

        features_x2_locat = self.model.locat2(features_x)
        map2 = self.model.ca2(features_x2_locat)
        ta2 = self.model.de2(features_x2_locat, map2) * features_x2_locat
        ta2.retain_grad()
        self.ta_features['ta2'] = ta2
        features_x2 = self.model.pool2(ta2)
        features_x2 = features_x2.view(features_x2.shape[0], -1)
        features_x2 = self.model.expert_fc2(features_x2)

        features_x3_locat = self.model.locat3(features_x)
        map3 = self.model.ca3(features_x3_locat)
        ta3 = self.model.de3(features_x3_locat, map3) * features_x3_locat
        ta3.retain_grad()
        self.ta_features['ta3'] = ta3
        features_x3 = self.model.pool3(ta3)
        features_x3 = features_x3.view(features_x3.shape[0], -1)
        features_x3 = self.model.expert_fc3(features_x3)

        features_x4_locat = self.model.locat4(features_x)
        map4 = self.model.ca4(features_x4_locat)
        ta4 = self.model.de4(features_x4_locat, map4) * features_x4_locat
        ta4.retain_grad()
        self.ta_features['ta4'] = ta4
        features_x4 = self.model.pool4(ta4)
        features_x4 = features_x4.view(features_x4.shape[0], -1)
        features_x4 = self.model.expert_fc4(features_x4)

        logits1 = self.model.classifier1(features_x1)
        logits2 = self.model.classifier2(features_x2)
        logits3 = self.model.classifier3(features_x3)
        logits4 = self.model.classifier4(features_x4)

        if self.model.use_moe == "MoE":
            expert_features = [features_x1, features_x2, features_x3, features_x4]
            
            shared_pooled_norm = self.model.moe_router.router_norm(shared_pooled)
            clean_logits = self.model.moe_router.gate(shared_pooled_norm)
            gate_weights = F.softmax(clean_logits, dim=-1)
            
            top_k_weights, top_k_indices = torch.topk(gate_weights, self.model.moe_router.top_k, dim=-1)
            final_weights = torch.zeros_like(gate_weights)
            final_weights.scatter_(1, top_k_indices, top_k_weights)
            final_weights = final_weights / (final_weights.sum(dim=-1, keepdim=True) + 1e-8)
            
            fused_feature = torch.zeros_like(expert_features[0])
            for i in range(self.model.moe_router.num_experts):
                weight = final_weights[:, i].unsqueeze(1)
                fused_feature = fused_feature + weight * expert_features[i]
            
            fused_logit = self.model.moe_router.fusion_classifier(fused_feature)
            
            diversity_metrics = self.model._compute_diversity_metrics(expert_features)
            return fused_logit, logits1, logits2, logits3, logits4, gate_weights, top_k_indices, diversity_metrics

        return None

    def get_ta_features(self):
        return self.ta_features


def compute_grad_cam(features, grads):
    """
    计算Grad-CAM
    Args:
        features: [B, C, D, H, W] - 特征图
        grads: [B, C, D, H, W] - 梯度
    Returns:
        cam: [B, D, H, W] - Grad-CAM热力图
    """
    alpha_k = grads.mean(dim=(0, 2, 3, 4), keepdim=True)
    cam = (features * alpha_k).sum(dim=1)
    cam = F.relu(cam)
    return cam


def process_single_image(MRI_path, class_label, data_type, test_model, device, args, save_dir):
    """
    处理单个MRI图像并生成Grad-CAM可视化
    使用TA之后的特征，根据top-k router权重组合
    """
    print(f"Processing: {MRI_path}")

    MRI = nibabel.load(MRI_path)
    MRI_array = MRI.get_fdata()
    print('MRI_array shape:', MRI_array.shape, '(D, H, W)')
    MRI_array = MRI_array.astype('float32')

    threshold = 0.1
    MRI_array[MRI_array < threshold] = 0

    img_fla = np.array(MRI_array).flatten()
    index = np.argwhere(img_fla == 0)
    img_median = np.median(np.delete(img_fla, index))
    MRI_array = MRI_array / img_median

    MRI_tensor = torch.FloatTensor(MRI_array).unsqueeze(0).unsqueeze(0)
    print('origin MRI shape: ', MRI_tensor.shape)

    MRI_tensor = MRI_tensor.to(device)
    MRI_tensor.requires_grad_(True)

    output1, logits1, logits2, logits3, logits4, gate_weights, top_k_indices, diversity_metrics = test_model.forward_with_ta_features(MRI_tensor)

    pred_class = output1.argmax(dim=1)
    pred_class_value = pred_class.item()
    pred_class_name = 'SMCI' if pred_class_value == 0 else 'PMCI'
    print('Predicted class:', pred_class_value, f'({pred_class_name})')
    print('gate_weights:', gate_weights)
    print('top_k_indices:', top_k_indices)

    test_model.zero_grad()

    ta_features = test_model.get_ta_features()
    ta1 = ta_features['ta1']
    ta2 = ta_features['ta2']
    ta3 = ta_features['ta3']
    ta4 = ta_features['ta4']

    output1[0, pred_class].backward()

    grads1 = ta1.grad
    grads2 = ta2.grad
    grads3 = ta3.grad
    grads4 = ta4.grad

    print('ta1 shape:', ta1.shape)
    print('grads1 shape:', grads1.shape)

    cam1 = compute_grad_cam(ta1, grads1)
    cam2 = compute_grad_cam(ta2, grads2)
    cam3 = compute_grad_cam(ta3, grads3)
    cam4 = compute_grad_cam(ta4, grads4)

    print('cam1 shape:', cam1.shape)

    top_k_idx = top_k_indices[0]
    top_k_weights = gate_weights[0]

    print(f'Using top-{len(top_k_idx)} experts: {top_k_idx.tolist()}')
    print(f'Top-k weights: {top_k_weights[top_k_idx].tolist()}')

    cam_fused = torch.zeros_like(cam1)
    for idx in top_k_idx:
        expert_idx = idx.item()
        weight = top_k_weights[expert_idx]
        cam_fused += eval(f'cam{expert_idx + 1}') * weight

    cam = cam_fused[0]
    cam = cam.unsqueeze(0).unsqueeze(0)
    print('cam shape before resize:', cam.shape)

    cam_resized = F.interpolate(cam, size=MRI_array.shape, mode='trilinear', align_corners=False)
    cam_resized = cam_resized.squeeze(0).squeeze(0)
    cam = cam_resized.detach().cpu().numpy()

    capi = np.maximum(cam, 0)
    heatmap = (capi - capi.min()) / (capi.max() - capi.min() + 1e-8)
    print('heatmap shape:', heatmap.shape)

    f, axarr = plt.subplots(3, 2, figsize=(12, 12))

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1, wspace=0.1, hspace=0.1)

    D, H, W = MRI_array.shape

    axial_slice_count = args.axial_slice if hasattr(args, 'axial_slice') else 55
    sagittal_slice_count = args.sagittal_slice if hasattr(args, 'sagittal_slice') else 70
    coronal_slice_count = args.coronal_slice if hasattr(args, 'coronal_slice') else 65

    print(f'Image shape: D={D}, H={H}, W={W}')
    print(f'Axial slice index: {axial_slice_count}')
    print(f'Sagittal slice index: {sagittal_slice_count}')
    print(f'Coronal slice index: {coronal_slice_count}')

    sagittal_ct_img = np.squeeze(MRI_array[sagittal_slice_count, :, :])
    sagittal_grad_cmap_img = np.squeeze(heatmap[sagittal_slice_count, :, :])

    axial_ct_img = np.squeeze(MRI_array[:, :, axial_slice_count])
    axial_grad_cmap_img = np.squeeze(heatmap[:, :, axial_slice_count])

    coronal_ct_img = np.squeeze(MRI_array[:, coronal_slice_count, :])
    coronal_grad_cmap_img = np.squeeze(heatmap[:, coronal_slice_count, :])

    img_plot = axarr[0, 0].imshow(np.rot90(axial_ct_img, 1), cmap='gray')
    axarr[0, 0].axis('off')
    axarr[0, 0].set_title('axial MRI', fontsize=22)

    axial_overlay = cv2.addWeighted(axial_ct_img, 0.3, axial_grad_cmap_img, 0.6, 0)

    img_plot = axarr[0, 1].imshow(np.rot90(axial_overlay, 1), cmap='jet', vmin=0.2)
    axarr[0, 1].axis('off')
    axarr[0, 1].set_title('Overlay', fontsize=22)

    img_plot = axarr[1, 0].imshow(np.rot90(sagittal_ct_img, 1), cmap='gray')
    axarr[1, 0].axis('off')
    axarr[1, 0].set_title('sagittal MRI', fontsize=22)

    sagittal_overlay = cv2.addWeighted(sagittal_ct_img, 0.3, sagittal_grad_cmap_img, 0.6, 0)

    img_plot = axarr[1, 1].imshow(np.rot90(sagittal_overlay, 1), cmap='jet', vmin=0.2)
    axarr[1, 1].axis('off')
    axarr[1, 1].set_title('Overlay', fontsize=22)

    img_plot = axarr[2, 0].imshow(np.rot90(coronal_ct_img, 1), cmap='gray')
    axarr[2, 0].axis('off')
    axarr[2, 0].set_title('coronal MRI', fontsize=22)

    coronal_overlay = cv2.addWeighted(coronal_ct_img, 0.3, coronal_grad_cmap_img, 0.6, 0)

    img_plot = axarr[2, 1].imshow(np.rot90(coronal_overlay, 1), cmap='jet', vmin=0.2)
    axarr[2, 1].axis('off')
    axarr[2, 1].set_title('Overlay', fontsize=22)

    class_save_dir = os.path.join(save_dir, data_type, class_label)
    os.makedirs(class_save_dir, exist_ok=True)

    filename = os.path.basename(MRI_path).replace('.nii.gz', '')
    save_path = os.path.join(class_save_dir, f'grad_CAM_{class_label}_{filename}_pred_{pred_class_name}_gradcam.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved to: {save_path}\n")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    args = type('Args', (), {
        'axial_slice': 55,
        'sagittal_slice': 70,
        'coronal_slice': 65
    })()

    # ========== 参数配置区域 - 请填写以下参数 ==========
    
    # 模型路径
    model_path = r"/3251903008/yyh/result_MoE_SMCI&PMCI/MoE_finetune_20260613_153106_data_dataArgument_top2_bl0.05_dl0.01_detach/models/3DNet4l_1_tp4_sifeature_clonefs_1bt1_l43_1_epoch_58_d1_det.pt"
    
    # MoE类型: "MoE", "concat", "CTrans"
    # use_moe = "MoE"
    use_moe = "MoE"
    
    # top_k参数 (仅当use_moe="MoE"时有效)
    # top_k = 2
    top_k = 2
    # dropout参数
    dropout = 0.0

    # 保存可视化结果的目录
    save_dir = r"/3251903008/yyh/result_MoE_SMCI&PMCI/MoE_finetune_20260613_153106_data_dataArgument_top2_bl0.05_dl0.01_detach"
    
    # 数据目录 (包含train/val/test子目录)
    data_dir = r"/3251903008/yyh/pMCIsMCI/split_dataset_4"

    # 文件过滤: 指定要处理的文件列表 (None = 处理所有文件)
    # 示例: file_filter = ['subject_001.nii.gz', 'subject_002.nii.gz']
    # 示例: file_filter = None (处理所有文件)
    file_filter = None
    
    # 文件名前缀过滤: 指定要处理的文件名前缀列表 (None = 处理所有文件)
    # 示例: file_prefix_filter = ['141S1137', '116S4732'] (只处理以这些前缀开头的文件)
    # 示例: file_prefix_filter = None (处理所有文件)
    file_prefix_filter = None
    # file_prefix_filter = None
    
    # ========== 参数配置区域结束 ==========
    
    # 切片索引参数 (根据需要调整)
    args.axial_slice = 55
    args.sagittal_slice = 72
    args.coronal_slice = 65

    test_model = VisualizationModel(model_path, device, use_moe=use_moe, top_k=top_k, dropout=dropout)

    classes = ['PMCI', 'SMCI']
    data_types = ['train', 'val', 'test']

    for data_type in data_types:
        for class_label in classes:
            class_dir = os.path.join(data_dir, data_type, class_label)
            if not os.path.exists(class_dir):
                print(f"Directory not found: {class_dir}")
                continue

            print(f"\n{'='*50}")
            print(f"Processing: {data_type}/{class_label}")
            print(f"{'='*50}\n")

            nii_files = [f for f in os.listdir(class_dir) if f.endswith('.nii.gz')]
            
            if file_filter is not None:
                nii_files = [f for f in nii_files if f in file_filter]
                print(f"Filtered to {len(nii_files)} files based on file filter")
            elif file_prefix_filter is not None:
                nii_files = [f for f in nii_files if any(f.startswith(prefix) for prefix in file_prefix_filter)]
                print(f"Filtered to {len(nii_files)} files based on prefix filter")
            else:
                print(f"Found {len(nii_files)} files in {class_dir}")

            for nii_file in nii_files:
                MRI_path = os.path.join(class_dir, nii_file)
                try:
                    process_single_image(MRI_path, class_label, data_type, test_model, device, args, save_dir)
                except Exception as e:
                    print(f"Error processing {MRI_path}: {str(e)}")
                    continue


if __name__ == "__main__":
    main()
