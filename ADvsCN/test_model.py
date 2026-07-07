import torch
import numpy as np
import os
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from dataload_4score import DataSet, load_data, load_test_data
import gc
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
from config import get_args
from model_MoE_plus import feature_Netin_tp4_multr_score
import argparse


def test_model(test_data, ResNet_3D1, device):
    """
    测试模型性能
    
    Args:
        test_data: 测试数据加载器
        ResNet_3D1: 要测试的模型
    """
    ResNet_3D1.eval()
    n_batches1 = len(test_data)
    print('Test a model on the test data...')
    correct = 0
    total = 0
    true_label = []
    data_pre = []
    
    with torch.no_grad():
        for (images, ADAS13, ADASQ4, MMSE, RAVLT_immediate, label) in tqdm(test_data, total = n_batches1):
            images, ADAS13, ADASQ4, MMSE, RAVLT_immediate, label = images.to(device), ADAS13.to(device), ADASQ4.to(device), MMSE.to(device), RAVLT_immediate.to(device), label.to(device)
            images= images.to(torch.float32)
            ADAS13 = ADAS13.to(torch.float32)
            ADASQ4 = ADASQ4.to(torch.float32)
            MMSE = MMSE.to(torch.float32)
            RAVLT_immediate = RAVLT_immediate.to(torch.float32)
            # output1, features_x_img, features_x1_img, features_x2_img, features_x3_img, features_x4_img = ResNet_3D1(images)
            # scores = torch.stack([ADAS13, ADASQ4, MMSE, RAVLT_immediate], dim=1)
            # output1, logits1, logits2, logits3, logits4, gate_weights, top_k_indices, _ = ResNet_3D1(images)
            output1, logits1, logits2, logits3, logits4, gate_weights, top_k_indices, diversity_metrics = ResNet_3D1(images)
            _, predicted = torch.max(output1, 1)
            true_label.extend(list(label.cpu().flatten().numpy()))
            data_pre.extend(list(predicted.cpu().flatten().numpy()))
            total += label.size(0)
            correct += (predicted == label).sum().item()

    TN, FP, FN, TP = confusion_matrix(true_label, data_pre).ravel()
    ACC = 100 * (TP + TN) / (TP + TN + FP + FN)
    SEN = 100 * (TP) / (TP + FN)
    SPE = 100 * (TN) / (TN + FP)
    AUC = 100 * roc_auc_score(true_label, data_pre)
    
    print('The result of test data: \n')
    print('TP:', TP, 'FP:', FP, 'FN:', FN, 'TN:', TN)
    print('ACC: %.4f %%' % ACC)
    print('SEN: %.4f %%' % SEN)
    print('SPE: %.4f %%' % SPE)
    print('AUC: %.4f %%' % AUC)
    
    # 清理内存
    del correct, total, true_label, data_pre, images, label, output1, TN, FP, FN, TP
    gc.collect()
    
    return ACC, SEN, SPE, AUC


def main():
    """
    主函数：加载模型和数据，执行测试
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Model Testing Script')
    parser.add_argument('--model_path', type=str, default='/3251903008/yyh/result_MoE_plus/MoE_20260303_141037_image1_dataArgument_top2_bl0.05_dl0.01_detach/models/3DNet4l_1_tp4_sifeature_clonefs_1bt1_l43_1_epoch_90_d1_det.pt', 
                        help='Path to the trained model file (.pt)')
    parser.add_argument('--norm_type', type=str, default='minmax', 
                        choices=['minmax', 'zscore'],
                        help='Normalization type: minmax or zscore')
    parser.add_argument('--gpu', type=str, default='0,1', 
                        help='GPU ID to use')
    
    # 解析已知参数，其余使用config.py中的默认值 
    known_args, _ = parser.parse_known_args()
    
    # 获取config.py中的默认参数
    args = get_args()
    
    # 更新GPU设置
    args.gpu = known_args.gpu

    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # 创建device对象
    if torch.cuda.is_available():
        device = torch.device(f"cuda:0")
        print(f"使用GPU: {args.gpu}")
    else:
        device = torch.device("cpu")
        print("CUDA不可用，使用CPU")
    
    # 设置随机种子
    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 打印配置信息
    print("===== 测试配置 =====")
    print(f"模型路径: {known_args.model_path}")
    print(f"归一化类型: {known_args.norm_type}")
    print(f"GPU: {args.gpu}")
    print(f"训练数据路径: {args.train_root_path}")
    print(f"测试数据路径: {args.test_root_path}")
    print(f"Excel文件: {args.excel_file}")
    print(f"批大小: {args.batch_size}")
    print("===================")
    
    # 检查模型文件是否存在
    if not os.path.exists(known_args.model_path):
        print(f"错误: 模型文件 {known_args.model_path} 不存在!")
        return
    
    # 加载训练数据并获取统计量（用于归一化）
    print("\n===== 加载数据 =====")
    print("加载训练数据以获取统计量...")
    train_loader, train_stats = load_data(args, args.train_root_path, args.AD_dir, args.CN_dir, args.excel_file, known_args.norm_type)
    
    # 使用训练集的统计量加载测试数据
    print("加载测试数据...")
    test_loader = load_test_data(args, args.test_root_path, args.AD_dir, args.CN_dir, args.excel_file, train_stats, known_args.norm_type)
    
    print(f"测试数据批次数: {len(test_loader)}")
    
    # 初始化模型
    print("\n===== 初始化模型 =====")
    # ResNet_3D1 = feature_Netin_tp4_multr_class().to(device)
    ResNet_3D1 = feature_Netin_tp4_multr_score(use_moe="MoE", top_k=2, dropout=args.dropout).to(device)
    
    # 加载模型权重
    print(f"加载模型权重: {known_args.model_path}")
    try:
        checkpoint = torch.load(known_args.model_path)
        ResNet_3D1.load_state_dict(checkpoint)
        print("模型权重加载成功!")
    except Exception as e:
        print(f"模型权重加载失败: {e}")
        return
    
    # 执行测试
    print("\n===== 开始测试 =====")
    ACC, SEN, SPE, AUC = test_model(test_loader, ResNet_3D1, device)
    
    # 打印最终结果
    print("\n===== 测试结果 =====")
    print(f"准确率 (ACC): {ACC:.4f}%")
    print(f"灵敏度 (SEN): {SEN:.4f}%")
    print(f"特异度 (SPE): {SPE:.4f}%")
    print(f"AUC: {AUC:.4f}%")
    print("===================")
    
    # 保存测试结果到文件
    result_dir = os.path.dirname(known_args.model_path)
    result_file = os.path.join(result_dir, "test_results.txt")
    
    with open(result_file, "w") as f:
        f.write("===== 测试配置 =====\n")
        f.write(f"模型路径: {known_args.model_path}\n")
        f.write(f"归一化类型: {known_args.norm_type}\n")
        f.write(f"GPU: {args.gpu}\n")
        f.write(f"测试数据路径: {args.test_root_path}\n")
        f.write(f"Excel文件: {args.excel_file}\n")
        f.write(f"批大小: {args.batch_size}\n")
        f.write("===================\n\n")
        
        f.write("===== 测试结果 =====\n")
        f.write(f"准确率 (ACC): {ACC:.4f}%\n")
        f.write(f"灵敏度 (SEN): {SEN:.4f}%\n")
        f.write(f"特异度 (SPE): {SPE:.4f}%\n")
        f.write(f"AUC: {AUC:.4f}%\n")
        f.write("===================\n")
    
    print(f"\n测试结果已保存到: {result_file}")


if __name__ == '__main__':
    main()