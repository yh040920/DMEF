import gc
from re import A
import nibabel
from torch.utils.data import Dataset, DataLoader
import os
import torch
from config import get_args
import numpy as np
import nibabel
import pandas as pd
import torchio as tio

def get_effective_ad_augmentation():
    """
    【实战推荐】针对 AD 分类任务的有效增强方案
    平衡了多样性与解剖结构的真实性
    """
    
    # 1. 几何变换组 (必选)
    # 目的：打破模型对"位置"和"大小"的过拟合，强迫其看"结构"
    spatial_transforms = tio.Compose([
        # 左右翻转：直接让训练数据量 x2，这是提升效果最明显的单一手段
        tio.RandomFlip(axes=('LR',), p=0.5),

        # 仿射变换：中等幅度
        tio.RandomAffine(
            scales=(0.9, 1.1),       # 缩放 ±10%：模拟不同人的头颅大小差异
            degrees=7,               # 旋转 ±7度：模拟摆位不正，7度是安全阈值
            translation=5,           # 平移 ±5体素：防止模型死记硬背特定坐标
            isotropic=True,          # 保持长宽比，防止大脑变形
            default_pad_value=0,     # 边缘补0
            image_interpolation='linear', # 保持纹理平滑
            p=0.75                   # 75% 的概率执行，保留 25% 的原始位置
        )
    ])

    # 2. 物理/强度变换组 (提升泛化性关键)
    # 目的：模拟不同机器、不同参数扫描出来的效果
    intensity_transforms = tio.OneOf({
        # 偏置场：模拟 MRI 常见的磁场不均匀导致的亮度渐变 (非常重要)
        tio.RandomBiasField(coefficients=0.3): 0.5,
        
        # Gamma 变换：非线性地改变对比度
        # 能够模拟有的图像更"灰"，有的更"黑白分明"
        tio.RandomGamma(log_gamma=(-0.3, 0.3)): 0.3,
        
        # 运动伪影：模拟轻微头动
        # 虽然计算慢一点，但对 AD (老年人) 数据非常真实
        tio.RandomMotion(degrees=3, translation=3, num_transforms=2): 0.2,
    }, p=0.5) # 50% 的概率执行其中一种，防止图像太烂

    # 3. 噪声组 (正则化)
    # 目的：防止模型关注过于微小的纹理细节
    noise_transform = tio.RandomNoise(std=(0, 0.03), p=0.25)

    # 4. 最终组合
    transform = tio.Compose([
        spatial_transforms,
        intensity_transforms,
        noise_transform,
        # 【兜底】因为增强可能导致数值越界，最后强制切回 [0, 1]
        tio.Lambda(lambda x: torch.clamp(x, 0, 1))
    ])
    
    return transform

# 定义一个自定义数据集类，继承自PyTorch的Dataset类
class DataSet(Dataset):
    def __init__(self, root_path, dir, excel_file, compute_stats=True, norm_type='minmax', use_augmentation=False):
        # 使用指定的根路径、目录和Excel文件路径初始化数据集
        self.root_path = root_path
        self.dir = dir
        self.image_path = os.path.join(self.root_path, dir)
        self.use_augmentation = use_augmentation
        
        # 如果启用数据增强，获取增强变换
        if self.use_augmentation:
            self.transform = get_effective_ad_augmentation()
        self.images = os.listdir(self.image_path)  # 获取指定目录中的图像文件名列表
        self.scores_df = pd.read_excel(excel_file)  # 从提供的Excel文件中读取分数数据
        self.compute_stats = compute_stats
        
        # 初始化评分统计量字典
        self.stats = {
            'ADAS13': {'min': None, 'max': None, 'mean': None, 'std': None},
            'ADASQ4': {'min': None, 'max': None, 'mean': None, 'std': None},
            'MMSE': {'min': None, 'max': None, 'mean': None, 'std': None},
            'RAVLT_immediate': {'min': None, 'max': None, 'mean': None, 'std': None}
        }
        
        # 如果需要计算统计量，在初始化时计算
        if self.compute_stats:
            self._compute_scores_statistics()
            # 打印统计信息
            self._print_statistics()
        
        # 设置归一化类型
        self.norm_type = norm_type
    
    def _compute_scores_statistics(self):
        """计算所有评分指标的统计量，处理缺失值"""
        # 收集所有图像对应的评分数据，过滤掉nan值
        adas13_values = []
        adasq4_values = []
        mmse_values = []
        ravlt_values = []
        
        # 记录缺失值数量
        missing_counts = {
            'ADAS13': 0,
            'ADASQ4': 0,
            'MMSE': 0,
            'RAVLT_immediate': 0
        }
        
        total_samples = 0
        
        for image_index in self.images:
            try:
                image_id = image_index[0:8]
                scores_row = self.scores_df[self.scores_df['PTID'] == image_id].iloc[0]
                total_samples += 1
                
                # 检测并记录缺失值
                if pd.notna(scores_row['ADAS13']):
                    adas13_values.append(scores_row['ADAS13'])
                else:
                    missing_counts['ADAS13'] += 1
                    
                if pd.notna(scores_row['ADASQ4']):
                    adasq4_values.append(scores_row['ADASQ4'])
                else:
                    missing_counts['ADASQ4'] += 1
                    
                if pd.notna(scores_row['MMSE']):
                    mmse_values.append(scores_row['MMSE'])
                else:
                    missing_counts['MMSE'] += 1
                    
                if pd.notna(scores_row['RAVLT_immediate']):
                    ravlt_values.append(scores_row['RAVLT_immediate'])
                else:
                    missing_counts['RAVLT_immediate'] += 1
                    
            except (IndexError, KeyError):
                # 忽略找不到对应数据的情况
                continue
        
        # 打印缺失值统计
        print(f"\n缺失值统计 ({self.dir}):")
        for score_name, count in missing_counts.items():
            if total_samples > 0:
                percentage = (count / total_samples) * 100
                print(f"{score_name}: {count}个缺失值 ({percentage:.1f}%)")
        
        # 计算统计量，使用np.nanmin、np.nanmax等处理可能存在的nan值
        self.stats['ADAS13']['min'] = np.nanmin(adas13_values) if adas13_values else 0
        self.stats['ADAS13']['max'] = np.nanmax(adas13_values) if adas13_values else 1
        self.stats['ADAS13']['mean'] = np.nanmean(adas13_values) if adas13_values else 0
        self.stats['ADAS13']['std'] = np.nanstd(adas13_values) if adas13_values else 1
        
        self.stats['ADASQ4']['min'] = np.nanmin(adasq4_values) if adasq4_values else 0
        self.stats['ADASQ4']['max'] = np.nanmax(adasq4_values) if adasq4_values else 1
        self.stats['ADASQ4']['mean'] = np.nanmean(adasq4_values) if adasq4_values else 0
        self.stats['ADASQ4']['std'] = np.nanstd(adasq4_values) if adasq4_values else 1
        
        self.stats['MMSE']['min'] = np.nanmin(mmse_values) if mmse_values else 0
        self.stats['MMSE']['max'] = np.nanmax(mmse_values) if mmse_values else 1
        self.stats['MMSE']['mean'] = np.nanmean(mmse_values) if mmse_values else 0
        self.stats['MMSE']['std'] = np.nanstd(mmse_values) if mmse_values else 1
        
        self.stats['RAVLT_immediate']['min'] = np.nanmin(ravlt_values) if ravlt_values else 0
        self.stats['RAVLT_immediate']['max'] = np.nanmax(ravlt_values) if ravlt_values else 1
        self.stats['RAVLT_immediate']['mean'] = np.nanmean(ravlt_values) if ravlt_values else 0
        self.stats['RAVLT_immediate']['std'] = np.nanstd(ravlt_values) if ravlt_values else 1
    
    def _print_statistics(self):
        """打印评分指标的统计信息"""
        print(f"Directory: {self.dir}")
        for score_name, stats_dict in self.stats.items():
            print(f"{score_name}: min={stats_dict['min']:.4f}, max={stats_dict['max']:.4f}, mean={stats_dict['mean']:.4f}, std={stats_dict['std']:.4f}")
        print()
    
    def normalize_score(self, score, score_name, norm_type='minmax'):
        """
        对单个评分进行归一化处理，增强防御性处理
        
        Args:
            score: 原始评分值
            score_name: 评分名称 ('ADAS13', 'ADASQ4', 'MMSE', 'RAVLT_immediate')
            norm_type: 归一化类型 ('minmax' 或 'zscore')
            
        Returns:
            归一化后的评分值
        """
        # 首先检查score是否为nan，如果是则返回0
        if pd.isna(score):
            return 0.0
        
        # 检查是否有有效的统计量
        if score_name not in self.stats or self.stats[score_name]['max'] is None:
            return 0.0
        
        try:
            if norm_type == 'minmax':
                # Min-Max归一化: (x - min) / (max - min)
                min_val = self.stats[score_name]['min']
                max_val = self.stats[score_name]['max']
                
                # 避免除以零和nan的情况
                if not pd.isna(max_val) and not pd.isna(min_val) and max_val - min_val > 0:
                    result = (score - min_val) / (max_val - min_val)
                    # 确保结果不是nan或无穷大
                    if pd.isna(result) or not np.isfinite(result):
                        return 0.0
                    return result
                else:
                    return 0.0  # 如果统计量无效，返回0
            
            elif norm_type == 'zscore':
                # Z-score归一化: (x - mean) / std
                mean_val = self.stats[score_name]['mean']
                std_val = self.stats[score_name]['std']
                
                # 避免除以零和nan的情况
                if not pd.isna(mean_val) and not pd.isna(std_val) and std_val > 0:
                    result = (score - mean_val) / std_val
                    # 确保结果不是nan或无穷大
                    if pd.isna(result) or not np.isfinite(result):
                        return 0.0
                    return result
                else:
                    return 0.0  # 如果统计量无效，返回0
            
            else:
                # 如果指定了未知的归一化类型，返回0
                return 0.0
        except Exception as e:
            # 捕获任何异常，确保不会返回nan或导致程序崩溃
            print(f"归一化过程中出错: {e}")
            return 0.0
    
    def set_normalization_stats(self, stats_dict):
        """
        从外部设置归一化统计量，用于验证集和测试集使用与训练集相同的统计量
        
        Args:
            stats_dict: 包含所有评分指标统计量的字典
        """
        self.stats = stats_dict
        self.compute_stats = False  # 设置后不再重新计算统计量

    def __getitem__(self, index):
        label = 0  # 初始化标签为0
        image_index = self.images[index]  # 获取给定索引处的图像文件名
        img_path = os.path.join(self.image_path, image_index)  # 创建图像文件的完整路径
        img = nibabel.load(img_path).get_fdata().astype('float32')  # 使用nibabel加载图像数据
        normalization = 'minmax'  # 选择归一化方法（minmax或median）

        # 根据选择的方法对图像数据进行归一化
        if normalization == 'minmax':
            img_max = img.max()
            img = img / img_max
        elif normalization == 'median':
            img_fla = np.array(img).flatten()
            index = np.argwhere(img_fla == 0)
            img_median = np.median(np.delete(img_fla, index))
            img = img / img_median
        
        # 如果启用数据增强，应用增强变换
        if self.use_augmentation:
            # 将numpy数组转换为torch.Tensor并添加通道维度
            img_tensor = torch.from_numpy(img).unsqueeze(0)
            # 创建torchio.Subject对象
            subject = tio.Subject(image=tio.ScalarImage(tensor=img_tensor))
            # 应用变换
            transformed = self.transform(subject)
            # 提取变换后的图像数据
            img = transformed['image'].data.numpy()[0]  # 移除通道维度
        
        img = np.expand_dims(img, axis=0)  # 添加通道维度
        # 根据目录设置标签值
        if self.dir == 'AD/':
            label = 1
        elif self.dir == 'CN/':
            label = 0
        elif self.dir == 'PMCI/':
            label = 2
        elif self.dir == 'SMCI/':
            label = 2

        # 从图像文件名中提取图像ID，并获取相应行的分数数据
        image_id = image_index[0:8]
        scores_row = self.scores_df[self.scores_df['PTID'] == image_id].iloc[0]

        # 获取原始评分值并使用均值填补缺失值
        # 先检查是否有有效的统计量，如果没有则使用默认值
        adas13_mean = self.stats['ADAS13']['mean'] if self.stats['ADAS13']['mean'] is not None else 0
        adasq4_mean = self.stats['ADASQ4']['mean'] if self.stats['ADASQ4']['mean'] is not None else 0
        mmse_mean = self.stats['MMSE']['mean'] if self.stats['MMSE']['mean'] is not None else 0
        ravlt_mean = self.stats['RAVLT_immediate']['mean'] if self.stats['RAVLT_immediate']['mean'] is not None else 0
        
        # 使用均值填补缺失值
        original_ADAS13 = scores_row['ADAS13'] if pd.notna(scores_row['ADAS13']) else adas13_mean
        original_ADASQ4 = scores_row['ADASQ4'] if pd.notna(scores_row['ADASQ4']) else adasq4_mean
        original_MMSE = scores_row['MMSE'] if pd.notna(scores_row['MMSE']) else mmse_mean
        original_RAVLT_immediate = scores_row['RAVLT_immediate'] if pd.notna(scores_row['RAVLT_immediate']) else ravlt_mean
        
        # 对评分进行归一化
        ADAS13 = self.normalize_score(original_ADAS13, 'ADAS13', self.norm_type)
        ADASQ4 = self.normalize_score(original_ADASQ4, 'ADASQ4', self.norm_type)
        MMSE = self.normalize_score(original_MMSE, 'MMSE', self.norm_type)
        RAVLT_immediate = self.normalize_score(original_RAVLT_immediate, 'RAVLT_immediate', self.norm_type)
        
        # 清理变量以节省内存
        if normalization == 'minmax':
            del img_max
        else:
            del img_fla, index, img_median
        gc.collect()

        return img, ADAS13, ADASQ4, MMSE, RAVLT_immediate, label

    def __len__(self):
        return len(self.images)


def load_data(args, root_path, path1, path2, excel_file, norm_type='minmax'):
    """
    加载数据并返回数据加载器
    
    注意：当用于训练集时，会计算并使用自己的统计量，并且启用数据增强
    当用于验证/测试集时，应该从训练集传递统计量
    
    改进：现在先合并数据集，然后计算整体统计量，确保归一化的准确性
    """
    # 先创建不计算统计量的数据集，启用数据增强
    dataset_AD = DataSet(root_path, path1, excel_file, compute_stats=False, norm_type=norm_type, use_augmentation=args.use_augmentation)
    dataset_CN = DataSet(root_path, path2, excel_file, compute_stats=False, norm_type=norm_type, use_augmentation=args.use_augmentation)
    
    # 计算整体数据集的统计量
    # 收集所有样本的评分数据
    scores_df = pd.read_excel(excel_file)
    
    # 初始化评分统计量字典
    combined_stats = {
        'ADAS13': {'min': None, 'max': None, 'mean': None, 'std': None},
        'ADASQ4': {'min': None, 'max': None, 'mean': None, 'std': None},
        'MMSE': {'min': None, 'max': None, 'mean': None, 'std': None},
        'RAVLT_immediate': {'min': None, 'max': None, 'mean': None, 'std': None}
    }
    
    # 收集所有图像对应的评分数据
    all_adas13 = []
    all_adasq4 = []
    all_mmse = []
    all_ravlt = []
    
    # 获取所有图像文件
    all_images = dataset_AD.images + dataset_CN.images
    
    for image_index in all_images:
        try:
            image_id = image_index[0:8]
            scores_row = scores_df[scores_df['PTID'] == image_id].iloc[0]
            
            # 收集非缺失的评分值
            if pd.notna(scores_row['ADAS13']):
                all_adas13.append(scores_row['ADAS13'])
            if pd.notna(scores_row['ADASQ4']):
                all_adasq4.append(scores_row['ADASQ4'])
            if pd.notna(scores_row['MMSE']):
                all_mmse.append(scores_row['MMSE'])
            if pd.notna(scores_row['RAVLT_immediate']):
                all_ravlt.append(scores_row['RAVLT_immediate'])
                
        except (IndexError, KeyError):
            # 忽略找不到对应数据的情况
            continue
    
    # 计算合并后的统计量
    combined_stats['ADAS13']['min'] = np.nanmin(all_adas13) if all_adas13 else 0
    combined_stats['ADAS13']['max'] = np.nanmax(all_adas13) if all_adas13 else 1
    combined_stats['ADAS13']['mean'] = np.nanmean(all_adas13) if all_adas13 else 0
    combined_stats['ADAS13']['std'] = np.nanstd(all_adas13) if all_adas13 else 1
    
    combined_stats['ADASQ4']['min'] = np.nanmin(all_adasq4) if all_adasq4 else 0
    combined_stats['ADASQ4']['max'] = np.nanmax(all_adasq4) if all_adasq4 else 1
    combined_stats['ADASQ4']['mean'] = np.nanmean(all_adasq4) if all_adasq4 else 0
    combined_stats['ADASQ4']['std'] = np.nanstd(all_adasq4) if all_adasq4 else 1
    
    combined_stats['MMSE']['min'] = np.nanmin(all_mmse) if all_mmse else 0
    combined_stats['MMSE']['max'] = np.nanmax(all_mmse) if all_mmse else 1
    combined_stats['MMSE']['mean'] = np.nanmean(all_mmse) if all_mmse else 0
    combined_stats['MMSE']['std'] = np.nanstd(all_mmse) if all_mmse else 1
    
    combined_stats['RAVLT_immediate']['min'] = np.nanmin(all_ravlt) if all_ravlt else 0
    combined_stats['RAVLT_immediate']['max'] = np.nanmax(all_ravlt) if all_ravlt else 1
    combined_stats['RAVLT_immediate']['mean'] = np.nanmean(all_ravlt) if all_ravlt else 0
    combined_stats['RAVLT_immediate']['std'] = np.nanstd(all_ravlt) if all_ravlt else 1
    
    # 打印合并后的统计信息
    print("合并后的数据集统计量:")
    for score_name, stats_dict in combined_stats.items():
        print(f"{score_name}: min={stats_dict['min']:.4f}, max={stats_dict['max']:.4f}, mean={stats_dict['mean']:.4f}, std={stats_dict['std']:.4f}")
    print()
    
    # 关键修正：将计算好的合并统计量设置到数据集对象中
    # 这样在归一化过程中会使用整体数据集的统计量
    dataset_AD.set_normalization_stats(combined_stats)
    dataset_CN.set_normalization_stats(combined_stats)
    
    # 现在合并数据集
    dataset = dataset_AD + dataset_CN
    
    # 创建数据加载器
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    # 清理内存
    del dataset
    gc.collect()
    
    # 返回数据加载器和合并后的统计信息
    return loader, combined_stats


def load_test_data(args, root_path, path1, path2, excel_file, train_stats, norm_type='minmax'):
    """
    加载测试数据，使用训练集的统计量进行归一化
    
    Args:
        args: 配置参数
        root_path: 数据根路径
        path1, path2: 两个子目录路径
        excel_file: 包含评分数据的Excel文件
        train_stats: 训练集的合并统计量字典
        norm_type: 归一化类型
    """
    # 创建数据集，不计算统计量
    dataset_AD = DataSet(root_path, path1, excel_file, compute_stats=False, norm_type=norm_type)
    dataset_CN = DataSet(root_path, path2, excel_file, compute_stats=False, norm_type=norm_type)
    
    # 设置训练集的统计量（使用合并后的统计量）
    dataset_AD.set_normalization_stats(train_stats)
    dataset_CN.set_normalization_stats(train_stats)
    
    # 合并数据集
    dataset = dataset_AD + dataset_CN
    
    # 创建数据加载器（测试时通常不打乱）
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # 清理内存
    del dataset
    gc.collect()
    
    return loader


if __name__ == '__main__':
    args = get_args()
    
    # 设置归一化类型
    norm_type = 'minmax'  # 可选: 'minmax' 或 'zscore'
    
    # 加载训练数据并获取统计量
    train_loader, train_stats = load_data(args, args.train_root_path, args.AD_dir, args.CN_dir, args.excel_file, norm_type)
    
    # 使用训练集的统计量加载验证数据
    val_loader = load_test_data(args, args.val_root_path, args.AD_dir, args.CN_dir, args.excel_file, train_stats, norm_type)
    
    # 使用训练集的统计量加载测试数据
    test_loader = load_test_data(args, args.test_root_path, args.AD_dir, args.CN_dir, args.excel_file, train_stats, norm_type)

    print(f"数据集大小: 训练={len(train_loader)}, 验证={len(val_loader)}, 测试={len(test_loader)}")
    print(f"使用的归一化方法: {norm_type}")
    print("\n训练集统计量:")
    for score_name, stats_dict in train_stats.items():
        print(f"{score_name}: min={stats_dict['min']:.4f}, max={stats_dict['max']:.4f}, mean={stats_dict['mean']:.4f}, std={stats_dict['std']:.4f}")

    # 测试缺失值填补功能
    print("\n=== 测试缺失值填补功能 ===")
    print("\n训练集数据样例（归一化后）:")
    nan_count_adas13 = 0
    nan_count_adasq4 = 0
    nan_count_mmse = 0
    nan_count_ravlt = 0
    total_samples = 0
    
    for batch in train_loader:
        images, ADAS13, ADASQ4, MMSE, RAVLT_immediate, labels = batch
        
        # 检查是否有nan值存在
        nan_count_adas13 += torch.isnan(ADAS13).sum().item()
        nan_count_adasq4 += torch.isnan(ADASQ4).sum().item()
        nan_count_mmse += torch.isnan(MMSE).sum().item()
        nan_count_ravlt += torch.isnan(RAVLT_immediate).sum().item()
        total_samples += ADAS13.size(0)
        
        print(f"图像形状: {images.shape}")
        print(f"归一化后ADAS13范围: {torch.min(ADAS13):.4f} - {torch.max(ADAS13):.4f}")
        print(f"归一化后ADASQ4范围: {torch.min(ADASQ4):.4f} - {torch.max(ADASQ4):.4f}")
        print(f"归一化后MMSE范围: {torch.min(MMSE):.4f} - {torch.max(MMSE):.4f}")
        print(f"归一化后RAVLT_immediate范围: {torch.min(RAVLT_immediate):.4f} - {torch.max(RAVLT_immediate):.4f}")
        print(f"标签: {labels}")
        break
    
    # 打印nan值检测结果
    print("\n=== 缺失值填补验证结果 ===")
    print(f"ADAS13中的nan值数量: {nan_count_adas13}/{total_samples}")
    print(f"ADASQ4中的nan值数量: {nan_count_adasq4}/{total_samples}")
    print(f"MMSE中的nan值数量: {nan_count_mmse}/{total_samples}")
    print(f"RAVLT_immediate中的nan值数量: {nan_count_ravlt}/{total_samples}")
    
    # 检查是否所有nan值都已被填补
    if nan_count_adas13 == 0 and nan_count_adasq4 == 0 and nan_count_mmse == 0 and nan_count_ravlt == 0:
        print("\n✅ 成功: 所有缺失值都已被正确填补!")
    else:
        print("\n❌ 警告: 仍存在未被填补的缺失值!")
    
    print("\n验证集数据样例（归一化后）:")
    for batch in val_loader:
        images, ADAS13, ADASQ4, MMSE, RAVLT_immediate, labels = batch
        print(f"图像形状: {images.shape}")
        print(f"归一化后ADAS13范围: {torch.min(ADAS13):.4f} - {torch.max(ADAS13):.4f}")
        print(f"归一化后ADASQ4范围: {torch.min(ADASQ4):.4f} - {torch.max(ADASQ4):.4f}")
        print(f"归一化后MMSE范围: {torch.min(MMSE):.4f} - {torch.max(MMSE):.4f}")
        print(f"归一化后RAVLT_immediate范围: {torch.min(RAVLT_immediate):.4f} - {torch.max(RAVLT_immediate):.4f}")
        print(f"标签: {labels}")
        break
    
    print("\n测试集数据样例（归一化后）:")
    for batch in test_loader:
        images, ADAS13, ADASQ4, MMSE, RAVLT_immediate, labels = batch
        print(f"图像形状: {images.shape}")
        print(f"归一化后ADAS13范围: {torch.min(ADAS13):.4f} - {torch.max(ADAS13):.4f}")
        print(f"归一化后ADASQ4范围: {torch.min(ADASQ4):.4f} - {torch.max(ADASQ4):.4f}")
        print(f"归一化后MMSE范围: {torch.min(MMSE):.4f} - {torch.max(MMSE):.4f}")
        print(f"归一化后RAVLT_immediate范围: {torch.min(RAVLT_immediate):.4f} - {torch.max(RAVLT_immediate):.4f}")
        print(f"标签: {labels}")
        break
     