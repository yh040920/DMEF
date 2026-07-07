import numpy as np
import cv2
import SimpleITK as sitk
import torch
import scipy.ndimage as ndimage
from skimage.transform import resize
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import nibabel
import torch.nn.functional as F
from scipy.ndimage import zoom
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from cnn3dend import feature_Netin_tp4_multr_class, feature_Netin_tp4_multr_class_score_attention_baseline
from config import get_args
import os
import torch.nn as nn

show_dir = "/3251903006/study/result/run_20251209_134818_js_divergence_loss_data_argument"
base_save_dir = "/3251903006/study/modelt1"
data_dir = "/3251903006/study/data"
model_path2 = os.path.join(show_dir, 'models/3DNet4l_1_tp4_sifeature_clonefs_1bt1_l43_1_epoch_120_d1_det.pt')

# show_dir = "/3251903006/study/result/20251209_020901_pretrian_score_js_divergence_loss_data"
# base_save_dir = "/3251903006/study/modelt1"
# data_dir = "/3251903006/study/data"
# model_path2 = os.path.join(show_dir, 'models/3DNet4l_1_tp4_sifeature_clonefs_1bt1_l43_1_epoch_62_d1_det.pt')

model_name = os.path.basename(show_dir)
save_dir = os.path.join(base_save_dir, model_name+'_single')

class tesst_model(nn.Module):
    def __init__(self, device):
        super(tesst_model, self).__init__()
        self.s = feature_Netin_tp4_multr_class()
        self.s.load_state_dict(torch.load(model_path2, map_location=device))
        self.s = self.s.to(device)
        print("load model from: ", model_path2)

    def forward(self, x):
        features_x = self.s.feature_extractor(x)
        
        features_x1 = self.s.locat1(features_x)
        map1 = self.s.ca1(features_x1)
        features_x1_noPooling = self.s.de1(features_x1, map1) * features_x1
        features_x1 = self.s.pool1(features_x1_noPooling)
        features_x1 = features_x1.view(features_x1.shape[0], -1)
        
        features_x2 = self.s.locat2(features_x)
        map2 = self.s.ca2(features_x2)
        features_x2_noPooling = self.s.de2(features_x2, map2) * features_x2
        features_x2 = self.s.pool2(features_x2_noPooling)
        features_x2 = features_x2.view(features_x2.shape[0], -1)
        
        features_x3 = self.s.locat3(features_x)
        map3 = self.s.ca3(features_x3)
        features_x3_noPooling = self.s.de3(features_x3, map3) * features_x3
        features_x3 = self.s.pool3(features_x3_noPooling)
        features_x3 = features_x3.view(features_x3.shape[0], -1)
        
        features_x4 = self.s.locat4(features_x)
        map4 = self.s.ca4(features_x4)  
        features_x4_noPooling = self.s.de4(features_x4, map4) * features_x4
        features_x4 = self.s.pool4(features_x4_noPooling)
        features_x4 = features_x4.view(features_x4.shape[0], -1)
        
        batch_size = features_x1.size(0)
        cls_token = self.s.cls_token.expand(batch_size, -1, -1)
        
        cls_feature, attn_weights = self.s.fusion_block(cls_token, [features_x1, features_x2, features_x3, features_x4])
        cls_feature = cls_feature.squeeze(1)
        logits1 = self.s.classifier1(cls_feature)
        
        return logits1, features_x, features_x1_noPooling, features_x2_noPooling, features_x3_noPooling, features_x4_noPooling, attn_weights

def compute_grad_cam(features, grads):
    """
    计算Grad-CAM
    Args:
        features: [B, C, D, H, W] - 特征图
        grads: [B, C, D, H, W] - 梯度
    Returns:
        cam: [B, D, H, W] - Grad-CAM热力图
    """
    alpha_k = grads.abs().mean(dim=(0, 2, 3, 4), keepdim=True)  # [1, C, 1, 1, 1]
    cam = (features * alpha_k).sum(dim=1)  # [B, D, H, W]
    cam = F.relu(cam)
    return cam

def process_single_image(MRI_path, class_label, data_type, test_model, device, args):
    """
    处理单个MRI图像并生成Grad-CAM可视化
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

    test_model.eval()

    MRI_tensor.requires_grad_(True)

    logits1, features_x, features_x1_noPooling, features_x2_noPooling, features_x3_noPooling, features_x4_noPooling, attn_weights = test_model(MRI_tensor)

    print('features_x1_noPooling shape:', features_x1_noPooling.shape)
    print('attn_weights shape:', attn_weights.shape)

    features_x1_noPooling.retain_grad()
    features_x2_noPooling.retain_grad()
    features_x3_noPooling.retain_grad()
    features_x4_noPooling.retain_grad()

    pred_class = logits1.argmax(dim=1)
    pred_class_value = pred_class.item()
    pred_class_name = 'CN' if pred_class_value == 0 else 'AD'
    print('Predicted class:', pred_class_value, f'({pred_class_name})')

    test_model.zero_grad()

    logits1[0, pred_class].backward()

    grads1 = features_x1_noPooling.grad
    grads2 = features_x2_noPooling.grad
    grads3 = features_x3_noPooling.grad
    grads4 = features_x4_noPooling.grad

    print('grads1 shape:', grads1.shape)

    cam1 = compute_grad_cam(features_x1_noPooling, grads1)
    cam2 = compute_grad_cam(features_x2_noPooling, grads2)
    cam3 = compute_grad_cam(features_x3_noPooling, grads3)
    cam4 = compute_grad_cam(features_x4_noPooling, grads4)

    print('cam1 shape:', cam1.shape)

    attn_weights_normalized = F.softmax(attn_weights.squeeze(1), dim=1)
    print('attn_weights_normalized:', attn_weights_normalized)

    cam_fused = (cam1 * attn_weights_normalized[:, 0:1, None, None] + 
                 cam2 * attn_weights_normalized[:, 1:2, None, None] + 
                 cam3 * attn_weights_normalized[:, 2:3, None, None] + 
                 cam4 * attn_weights_normalized[:, 3:4, None, None])

    print('cam_fused shape:', cam_fused.shape)

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
    axial_slice_count = 55
    sagittal_slice_count = 70
    coronal_slice_count = 65

    # axial_slice_count = 50
    # sagittal_slice_count = 65
    # coronal_slice_count = 60

    # axial_slice_count = 60
    # sagittal_slice_count = 75
    # coronal_slice_count = 70

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
    save_path = os.path.join(class_save_dir, f'grad_CAM4_{class_label}_{filename}_pred_{pred_class_name}_gradcam.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved to: {save_path}\n")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    args = get_args()
    DROPOUT = 0

    test_model = tesst_model(device)

    classes = ['AD', 'CN']
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
            print(f"Found {len(nii_files)} files in {class_dir}")
            
            for nii_file in nii_files:
                # ['116S4732', '128S0216', '131S5138', '141S0852', '116S4010', '128S1242', '153S4139', '941S4100']
                # ["011S4912","014S0328","020S0213","035S4783","053S1044","094S1402","006S4449","007S0070","020S0883","021S4254","029S0843"]
                if nii_file.split('\\')[-1].split('_')[0] in ['141S1137']:
                    
                    MRI_path = os.path.join(class_dir, nii_file)
                    try:
                        process_single_image(MRI_path, class_label, data_type, test_model, device, args)
                    except Exception as e:
                        print(f"Error processing {MRI_path}: {str(e)}")
                        continue

if __name__ == "__main__":
    main()
