import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel
import os
from cnn3dend import feature_Netin_tp4_multr_class_score_attention_baseline
from model_MoE_plus import feature_Netin_tp4_multr_score, feature_Netin_tp4_multr_score_simpleTA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity  # 新增导入


class BaselineFeatureExtractor(nn.Module):
    def __init__(self, model_path, device, dropout=0.0):
        super(BaselineFeatureExtractor, self).__init__()
        self.model = feature_Netin_tp4_multr_class_score_attention_baseline(dropout=dropout)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model = self.model.to(device)
        self.model.eval()
        print(f"Load baseline model from: {model_path}")

    def extract_feature(self, x):
        features_x = self.model.feature_extractor(x)
        features_x1 = self.model.locat1(features_x)
        map1 = self.model.ca1(features_x1)
        features_x1 = self.model.de1(features_x1, map1) * features_x1
        features_x1 = self.model.pool1(features_x1)
        features_x = features_x1.view(features_x1.shape[0], -1)
        logits = self.model.classifier1(features_x)
        return features_x, logits


class MoEPlusFeatureExtractor(nn.Module):
    def __init__(self, model_path, device, use_moe="MoE", top_k=2, dropout=0.0):
        super(MoEPlusFeatureExtractor, self).__init__()
        self.model = feature_Netin_tp4_multr_score(dropout=dropout, use_moe=use_moe, top_k=top_k)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model = self.model.to(device)
        self.model.eval()
        self.use_moe = use_moe
        print(f"Load MoE_plus model from: {model_path}")

    def extract_feature(self, x):
        features_x = self.model.feature_extractor(x)
        
        if self.use_moe == "MoE":
            shared_pooled = self.model.shared_pool(features_x)
            shared_pooled = shared_pooled.view(shared_pooled.shape[0], -1)

        features_x1 = self.model.locat1(features_x)
        map1 = self.model.ca1(features_x1)
        features_x1 = self.model.de1(features_x1, map1) * features_x1
        features_x1 = self.model.pool1(features_x1)
        features_x1 = features_x1.view(features_x1.shape[0], -1)
        features_x1 = self.model.expert_fc1(features_x1)

        features_x2 = self.model.locat2(features_x)
        map2 = self.model.ca2(features_x2)
        features_x2 = self.model.de2(features_x2, map2) * features_x2
        features_x2 = self.model.pool2(features_x2)
        features_x2 = features_x2.view(features_x2.shape[0], -1)
        features_x2 = self.model.expert_fc2(features_x2)

        features_x3 = self.model.locat3(features_x)
        map3 = self.model.ca3(features_x3)
        features_x3 = self.model.de3(features_x3, map3) * features_x3
        features_x3 = self.model.pool3(features_x3)
        features_x3 = features_x3.view(features_x3.shape[0], -1)
        features_x3 = self.model.expert_fc3(features_x3)

        features_x4 = self.model.locat4(features_x)
        map4 = self.model.ca4(features_x4)
        features_x4 = self.model.de4(features_x4, map4) * features_x4
        features_x4 = self.model.pool4(features_x4)
        features_x4 = features_x4.view(features_x4.shape[0], -1)
        features_x4 = self.model.expert_fc4(features_x4)

        expert_features = [features_x1, features_x2, features_x3, features_x4]
        
        if self.use_moe == "MoE":
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
            return fused_feature, fused_logit
            
        elif self.use_moe == "concat":
            features = torch.cat(expert_features, dim=1)
            fused_logit = self.model.cls(features)
            return features, fused_logit
            
        return None, None


class SimpleTAFeatureExtractor(nn.Module):
    def __init__(self, model_path, device, top_k=4, dropout=0.0):
        super(SimpleTAFeatureExtractor, self).__init__()
        self.model = feature_Netin_tp4_multr_score_simpleTA(dropout=dropout, use_moe=True, top_k=top_k)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model = self.model.to(device)
        self.model.eval()
        print(f"Load simpleTA model from: {model_path}")

    def extract_feature(self, x):
        features_x = self.model.feature_extractor(x)
        features_x = self.model.locat1(features_x)
        map1 = self.model.ca1(features_x)
        features_x = self.model.de1(features_x, map1) * features_x
        
        shared_pooled = self.model.shared_pool(features_x)
        shared_pooled = shared_pooled.view(shared_pooled.shape[0], -1)

        features_x1 = self.model.pool1(features_x)
        features_x1 = features_x1.view(features_x1.shape[0], -1)
        features_x1 = self.model.expert_fc1(features_x1)

        features_x2 = self.model.pool2(features_x)
        features_x2 = features_x2.view(features_x2.shape[0], -1)
        features_x2 = self.model.expert_fc2(features_x2)

        features_x3 = self.model.pool3(features_x)
        features_x3 = features_x3.view(features_x3.shape[0], -1)
        features_x3 = self.model.expert_fc3(features_x3)

        features_x4 = self.model.pool4(features_x)
        features_x4 = features_x4.view(features_x4.shape[0], -1)
        features_x4 = self.model.expert_fc4(features_x4)

        expert_features = [features_x1, features_x2, features_x3, features_x4]
        fused_logit, gate_weights, top_k_indices = self.model.moe_router(shared_pooled, expert_features)
        
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
        
        return fused_feature, fused_logit


def load_and_preprocess_mri(MRI_path):
    MRI = nibabel.load(MRI_path)
    MRI_array = MRI.get_fdata()
    MRI_array = MRI_array.astype('float32')

    threshold = 0.1
    MRI_array[MRI_array < threshold] = 0

    img_fla = np.array(MRI_array).flatten()
    index = np.argwhere(img_fla == 0)
    img_median = np.median(np.delete(img_fla, index))
    MRI_array = MRI_array / img_median

    MRI_tensor = torch.FloatTensor(MRI_array).unsqueeze(0).unsqueeze(0)
    return MRI_tensor


def extract_features_from_model(extractor, data_dir, device, class_order=['AD', 'CN']):
    features_list = []
    labels_list = []
    sample_names = []
    
    for class_idx, class_label in enumerate(class_order):
        class_dir = os.path.join(data_dir, 'test', class_label)
        if not os.path.exists(class_dir):
            print(f"Directory not found: {class_dir}")
            continue
            
        nii_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.nii.gz')])
        print(f"Processing {len(nii_files)} files from {class_label}")
        
        for nii_file in nii_files:
            MRI_path = os.path.join(class_dir, nii_file)
            try:
                MRI_tensor = load_and_preprocess_mri(MRI_path)
                MRI_tensor = MRI_tensor.to(device)
                
                with torch.no_grad():
                    feature, logits = extractor.extract_feature(MRI_tensor)
                
                features_list.append(feature.cpu().numpy().flatten())
                labels_list.append(class_idx)
                sample_names.append(nii_file)
                
            except Exception as e:
                print(f"Error processing {MRI_path}: {str(e)}")
                continue
    
    features = np.array(features_list)
    labels = np.array(labels_list)
    
    return features, labels, sample_names


def compute_correlation_matrix(features, labels):
    # 使用余弦相似度替代皮尔逊相关系数
    corr_matrix = cosine_similarity(features)
    return corr_matrix


def plot_correlation_matrix(corr_matrix, labels, model_name, save_path):
    n_samples = corr_matrix.shape[0]
    
    n_ad = np.sum(labels == 0)
    n_nc = np.sum(labels == 1)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    mask = None
    
    sns.heatmap(corr_matrix, 
                cmap='coolwarm',
                center=0,
                vmin=-1, vmax=1,
                square=True,
                cbar_kws={'shrink': 0.8, 'label': 'Cosine Similarity'},  # 修改颜色条标签
                ax=ax)
    
    if n_ad > 0 and n_nc > 0:
        ax.axhline(y=n_ad, color='black', linewidth=2)
        ax.axvline(x=n_ad, color='black', linewidth=2)
        
        ax.text(n_ad/2, -5, 'AD', ha='center', fontsize=14, fontweight='bold')
        ax.text(n_ad + n_nc/2, -5, 'NC', ha='center', fontsize=14, fontweight='bold')
        ax.text(-5, n_ad/2, 'AD', ha='center', va='center', fontsize=14, fontweight='bold', rotation=90)
        ax.text(-5, n_ad + n_nc/2, 'NC', ha='center', va='center', fontsize=14, fontweight='bold', rotation=90)
    
    ax.set_title(f'Sample-wise Cosine Similarity Matrix - {model_name}\n(AD group first, then NC group.)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Sample Index', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved cosine similarity matrix to: {save_path}")


def plot_all_models_combined(all_corr_matrices, all_labels, model_names, save_path):
    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, corr_matrix) in enumerate(zip(model_names, [all_corr_matrices[m] for m in model_names])):
        labels = all_labels[model_name]
        ax = axes[idx]
        
        n_ad = np.sum(labels == 0)
        n_nc = np.sum(labels == 1)
        
        sns.heatmap(corr_matrix, 
                    cmap='coolwarm',
                    center=0,
                    vmin=-1, vmax=1,
                    square=True,
                    cbar=True,
                    ax=ax,
                    xticklabels=False,
                    yticklabels=False)
        
        if n_ad > 0 and n_nc > 0:
            ax.axhline(y=n_ad, color='black', linewidth=1.5)
            ax.axvline(x=n_ad, color='black', linewidth=1.5)
        
        ax.set_title(f'{model_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    fig.suptitle('Sample-wise Cosine Similarity Matrices (AD group first, then NC group.)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved combined cosine similarity matrix to: {save_path}")


def compute_intra_inter_class_correlation(corr_matrix, labels):
    n_ad = np.sum(labels == 0)
    n_nc = np.sum(labels == 1)
    
    ad_indices = np.where(labels == 0)[0]
    nc_indices = np.where(labels == 1)[0]
    
    intra_ad_corrs = []
    for i in range(len(ad_indices)):
        for j in range(i+1, len(ad_indices)):
            intra_ad_corrs.append(corr_matrix[ad_indices[i], ad_indices[j]])
    
    intra_nc_corrs = []
    for i in range(len(nc_indices)):
        for j in range(i+1, len(nc_indices)):
            intra_nc_corrs.append(corr_matrix[nc_indices[i], nc_indices[j]])
    
    inter_corrs = []
    for i in ad_indices:
        for j in nc_indices:
            inter_corrs.append(corr_matrix[i, j])
    
    results = {
        'intra_AD_mean': np.mean(intra_ad_corrs) if intra_ad_corrs else 0,
        'intra_AD_std': np.std(intra_ad_corrs) if intra_ad_corrs else 0,
        'intra_NC_mean': np.mean(intra_nc_corrs) if intra_nc_corrs else 0,
        'intra_NC_std': np.std(intra_nc_corrs) if intra_nc_corrs else 0,
        'inter_mean': np.mean(inter_corrs) if inter_corrs else 0,
        'inter_std': np.std(inter_corrs) if inter_corrs else 0,
    }
    
    return results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    data_dir = r"/3251903008/yyh/image1"
    save_dir = r"/3251903008/yyh/imagescore_code/feature_sim_analysis"
    os.makedirs(save_dir, exist_ok=True)
    
    model_configs = {
        'ACNN': {
            'path': r"/3251903008/yyh/result_MoE_plus/baseline_20260311_085507_TACNN3D/models/3DNet4l_1_tp4_sifeature_clonefs_1bt1_l43_1_epoch_65_d1_det.pt",
            'type': 'baseline',
            'feature_dim': 128
        },
        'CACNN': {
            'path': r"/3251903008/yyh/result_MoE_plus/a_20260311_133241_image1_dataArgument_concat_noMoE/models/3DNet4l_1_tp4_sifeature_clonefs_1bt1_l43_1_epoch_49_d1_det.pt",
            'type': 'moe',
            'use_moe': 'concat',
            'feature_dim': 512
        },
        'OURS-simple': {
            'path': r"/3251903008/yyh/result_MoE_plus/MoE_20260309_062202_image1_dataArgument_topall_bl0.05_dl0.01_detach_simpleTA/model/3DNet4l_1_tp4_sifeature_clonefs_1bt1_l43_1_epoch_66_d1_det.pt",
            'type': 'simpleTA',
            'top_k': 4,
            'feature_dim': 128
        },
        'OURS': {
            'path': r"/3251903008/yyh/result_MoE_plus/MoE_20260303_141037_image1_dataArgument_top2_bl0.05_dl0.01_detach/models/3DNet4l_1_tp4_sifeature_clonefs_1bt1_l43_1_epoch_90_d1_det.pt",
            'type': 'moe',
            'use_moe': 'MoE',
            'top_k': 2,
            'feature_dim': 128
        }
    }
    
    all_features = {}
    all_labels = {}
    all_corr_results = {}
    all_corr_matrices = {}
    
    for model_name, config in model_configs.items():
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"{'='*60}")
        
        if config['type'] == 'baseline':
            extractor = BaselineFeatureExtractor(config['path'], device, dropout=0.0)
        elif config['type'] == 'moe':
            use_moe = config.get('use_moe', 'MoE')
            top_k = config.get('top_k', None)
            extractor = MoEPlusFeatureExtractor(config['path'], device, use_moe=use_moe, top_k=top_k, dropout=0.0)
        elif config['type'] == 'simpleTA':
            top_k = config.get('top_k', 4)
            extractor = SimpleTAFeatureExtractor(config['path'], device, top_k=top_k, dropout=0.0)
        
        features, labels, sample_names = extract_features_from_model(extractor, data_dir, device)
        all_features[model_name] = features
        all_labels[model_name] = labels
        
        print(f"Extracted features shape: {features.shape}")
        print(f"Number of AD samples: {np.sum(labels == 0)}")
        print(f"Number of NC samples: {np.sum(labels == 1)}")
        
        print("Computing cosine similarity matrix...")
        corr_matrix = compute_correlation_matrix(features, labels)
        all_corr_matrices[model_name] = corr_matrix
        
        corr_results = compute_intra_inter_class_correlation(corr_matrix, labels)
        all_corr_results[model_name] = corr_results
        
        print(f"Intra-AD similarity: {corr_results['intra_AD_mean']:.4f} ± {corr_results['intra_AD_std']:.4f}")
        print(f"Intra-NC similarity: {corr_results['intra_NC_mean']:.4f} ± {corr_results['intra_NC_std']:.4f}")
        print(f"Inter-class similarity: {corr_results['inter_mean']:.4f} ± {corr_results['inter_std']:.4f}")
        
        save_path = os.path.join(save_dir, f'{model_name}_cosine_similarity_matrix.png')
        plot_correlation_matrix(corr_matrix, labels, model_name, save_path)
        
        np.save(os.path.join(save_dir, f'{model_name}_features.npy'), features)
        np.save(os.path.join(save_dir, f'{model_name}_labels.npy'), labels)
        np.save(os.path.join(save_dir, f'{model_name}_cos_sim_matrix.npy'), corr_matrix)
        
        del extractor
        torch.cuda.empty_cache()
    
    model_order = ['ACNN', 'CACNN', 'OURS-simple', 'OURS']
    combined_save_path = os.path.join(save_dir, 'all_models_cosine_similarity_matrix_combined.png')
    plot_all_models_combined(all_corr_matrices, all_labels, model_order, combined_save_path)
    
    print(f"\n{'='*60}")
    print("Summary of Cosine Similarity Analysis")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'Intra-AD':<20} {'Intra-NC':<20} {'Inter-class':<20}")
    print("-" * 75)
    for model_name, results in all_corr_results.items():
        intra_ad = f"{results['intra_AD_mean']:.4f} ± {results['intra_AD_std']:.4f}"
        intra_nc = f"{results['intra_NC_mean']:.4f} ± {results['intra_NC_std']:.4f}"
        inter = f"{results['inter_mean']:.4f} ± {results['inter_std']:.4f}"
        print(f"{model_name:<15} {intra_ad:<20} {intra_nc:<20} {inter:<20}")
    
    print(f"\nAll results saved to: {save_dir}")


if __name__ == "__main__":
    main()