import os
import sys
import time
from itertools import cycle
import math
import argparse
import numpy as np
from tqdm import tqdm
import gc
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix
from config import get_args

# 训练脚本本地参数，不写入 config.py；先解析并从 sys.argv 中移除，避免 get_args() 报 unknown argument
train_parser = argparse.ArgumentParser(add_help=False)
train_parser.add_argument('--model_type', type=str, choices=['MoE', 'concat', 'CTrans'], default='MoE',
                          help='fusion model type: MoE, concat or CTrans')
train_args, remaining_argv = train_parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining_argv

# 在导入torch之前设置CUDA_VISIBLE_DEVICES
args = get_args()
args.model_type = train_args.model_type
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataload_4score import DataSet, load_data, load_test_data
from dataload_excelremove import load_data_remove
from model_MoE_plus import feature_Netin_tp4_multr_score

def unpack_model_outputs(outputs, model_type):
    if model_type == "MoE":
        output1, logits1, logits2, logits3, logits4, gate_weights, top_k_indices, diversity_metrics = outputs
    else:
        output1, logits1, logits2, logits3, logits4, diversity_metrics = outputs
        gate_weights = None
        top_k_indices = None

    return output1, logits1, logits2, logits3, logits4, gate_weights, top_k_indices, diversity_metrics



def mse_loss(student_logits, teacher_logits):
    """
    MSE between logits (or intermediate features).
    """
    teacher_logits_detach = teacher_logits.detach()
    return F.mse_loss(student_logits, teacher_logits_detach, reduction='mean')


def train_epoch(epoch, ResNet_3D1, train_data, fo, device):
    ResNet_3D1.train()
    n_batches = len(train_data)
    print(n_batches)
    learning_rate = '111'
    # 学习率设置
    if learning_rate == '111':
        if epoch < 30:
            LEARNING_RATE = 0.0001
        elif epoch < 40:
            LEARNING_RATE = 0.00001
        elif epoch < 50:
            LEARNING_RATE = 0.00001
        elif epoch < 60:
            LEARNING_RATE = 0.00001
        else:
            LEARNING_RATE = 0.00001
    else:
        LEARNING_RATE = args.lr / math.pow((1 + 10 * (epoch - 1) / args.nepoch), 0.75)

    optimizer = torch.optim.Adam([
        {'params': ResNet_3D1.parameters(), 'lr': LEARNING_RATE}
    ], lr=0.0001, weight_decay=1e-4)

    total_loss_sum = 0
    total_loss_reg = 0
    total_loss_cls = 0
    total_loss_balance = 0
    total_loss_diversity = 0

    source_correct = 0
    total_label = 0
    criterion_reg_adas13 = nn.MSELoss()
    criterion_reg_adas13 = criterion_reg_adas13.to(torch.float32)
    criterion_reg_adasq4 = nn.MSELoss()
    criterion_reg_adasq4 = criterion_reg_adasq4.to(torch.float32)
    criterion_reg_mmse = nn.MSELoss()
    criterion_reg_mmse = criterion_reg_mmse.to(torch.float32)
    criterion_reg_RAVLT_immediate = nn.MSELoss()
    criterion_reg_RAVLT_immediate = criterion_reg_RAVLT_immediate.to(torch.float32)

    # sinkhorn_loss_fn = Sinkhorn()

    for (images, ADAS13, ADASQ4, MMSE, RAVLT_immediate, label) in tqdm(train_data, total = n_batches):
        images, ADAS13, ADASQ4, MMSE, RAVLT_immediate, label = images.to(device), ADAS13.to(device), ADASQ4.to(device), MMSE.to(device), RAVLT_immediate.to(device), label.to(device)
        images, ADAS13, ADASQ4, MMSE, RAVLT_immediate = images.to(torch.float32),ADAS13.to(torch.float32), ADASQ4.to(torch.float32), MMSE.to(torch.float32), RAVLT_immediate.to(torch.float32)

        output1, logits1, logits2, logits3, logits4, gate_weights, top_k_indices, diversity_metrics = unpack_model_outputs(
            ResNet_3D1(images), args.model_type
        )

        if gate_weights is not None:
            mean_gates = gate_weights.mean(dim=0)  # shape: (4,)
            balance_loss = 4 * torch.sum(mean_gates ** 2)
        else:
            balance_loss = output1.new_tensor(0.0)

        diversity_loss = diversity_metrics['diversity_loss']

        loss_ADAS13 = criterion_reg_adas13(logits1[:,0], ADAS13)
        loss_ADASQ4 = criterion_reg_adasq4(logits2[:,0], ADASQ4)
        loss_MMSE = criterion_reg_mmse(logits3[:,0], MMSE)
        loss_RAVLT_immediate = criterion_reg_RAVLT_immediate(logits4[:,0], RAVLT_immediate)

        loss_fenshu = loss_ADAS13 + loss_ADASQ4 + loss_MMSE + loss_RAVLT_immediate  #回归预测任务的损失

        loss_label = F.cross_entropy(output1, label)

        toal_loss = loss_label + loss_fenshu + 0.05*balance_loss + 0.01*diversity_loss

        _, preds = torch.max(output1, 1)
        source_correct += preds.eq(label.data.view_as(preds)).cpu().sum()
        total_label += label.size(0)
        optimizer.zero_grad()
        toal_loss.backward()
        optimizer.step()
        # torch.cuda.empty_cache()
        total_loss_sum += toal_loss.item()
        total_loss_reg += loss_fenshu.item()
        total_loss_cls += loss_label.item()
        total_loss_balance += balance_loss.item()
        total_loss_diversity += diversity_loss.item()
    acc = source_correct / total_label
    mean_loss = total_loss_sum / n_batches
    mean_loss_reg = total_loss_reg / n_batches
    mean_loss_cls = total_loss_cls / n_batches    
    mean_loss_balance = total_loss_balance / n_batches
    mean_loss_diversity = total_loss_diversity / n_batches
    print(f'Epoch: [{epoch:2d}], '
          f'Loss: {mean_loss:.6f}, '
          f'Loss_reg: {mean_loss_reg:.6f}, '
          f'Loss_cls: {mean_loss_cls:.6f}, '
          f'Loss_balance: {mean_loss_balance:.6f}, '
          f'Loss_diversity: {mean_loss_diversity:.6f}')   
    log_str = 'Epoch: '+str(epoch)\
              +' Loss: '+str(mean_loss)\
              +' train_acc: '+str(acc)+'\n'
    fo.write(log_str)
    del acc, train_data, n_batches
    # gc.collect()

    return mean_loss, source_correct, total_label


def val_model(epoch, val_data, ResNet_3D1, log_best, device):
    ResNet_3D1.eval()
    n_batches1=len(val_data)
    print('Test a model on the val data...')
    correct = 0
    total = 0
    total_loss = 0
    true_label = []
    data_pre = []

    with torch.no_grad():
        for (images,ADAS13, ADASQ4, MMSE, RAVLT_immediate, label) in tqdm(val_data, total = n_batches1):
            images, ADAS13, ADASQ4, MMSE, RAVLT_immediate, label = images.to(device), ADAS13.to(device), ADASQ4.to(device), MMSE.to(device), RAVLT_immediate.to(device), label.to(device)
            images, ADAS13, ADASQ4, MMSE, RAVLT_immediate = images.to(torch.float32),ADAS13.to(torch.float32), ADASQ4.to(torch.float32), MMSE.to(torch.float32), RAVLT_immediate.to(torch.float32)
            output1, logits1, logits2, logits3, logits4, gate_weights, top_k_indices, diversity_metrics = unpack_model_outputs(
                ResNet_3D1(images), args.model_type
            )
            if gate_weights is not None:
                print(gate_weights)
            loss_label = F.cross_entropy(output1, label)
            loss = loss_label 
            total_loss += loss.item()

            _, predicted = torch.max(output1, 1)
            true_label.extend(list(label.cpu().flatten().numpy()))
            data_pre.extend(list(predicted.cpu().flatten().numpy()))
            total += label.size(0)
            correct += (predicted == label).sum().item()

    mean_loss = total_loss / len(val_data)
    TN, FP, FN, TP = confusion_matrix(true_label, data_pre).ravel()
    ACC = 100 * (TP + TN) / (TP + TN + FP + FN)
    SEN = 100 * (TP) / (TP + FN)
    SPE = 100 * (TN) / (TN + FP)
    AUC = 100 * roc_auc_score(true_label, data_pre)
    print('TP:', TP, 'FP:', FP, 'FN:', FN, 'TN:', TN)
    print('ACC: %.4f %%' % ACC)
    print('SEN: %.4f %%' % SEN)
    print('SPE: %.4f %%' % SPE)
    print('AUC: %.4f %%' % AUC)
    log_str = 'Epoch: ' + str(epoch) \
              + '\n' \
              + 'TP: ' + str(TP) + ' TN: ' + str(TN) + ' FP: ' + str(FP) + ' FN: ' + str(FN) \
              + '  ACC:  ' + str(ACC) \
              + '  SEN:  ' + str(SEN) \
              + '  SPE:  ' + str(SPE) \
              + '  AUC:  ' + str(AUC) \
              + '\n'
    log_best.write(log_str)
    del correct, total, true_label, data_pre, images, label, output1, TN, FP, FN, TP
    gc.collect()
    return ACC, SEN, SPE, AUC, mean_loss  #, feature_for_GCN


def test_model(test_data,  ResNet_3D1, device):
    ResNet_3D1.eval()
    n_batches1 = len(test_data)
    print('Test a model on the test data...')
    correct = 0
    total = 0
    true_label = []
    data_pre = []
    with torch.no_grad():
        for (images, _, _, _, _, label) in tqdm(test_data, total = n_batches1):
            images, label = images.to(device), label.to(device)
            images= images.to(torch.float32)
            output1, logits1, logits2, logits3, logits4, gate_weights, top_k_indices, diversity_metrics = unpack_model_outputs(
                ResNet_3D1(images), args.model_type
            )

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
    del correct, total, true_label, data_pre, images, label, output1, TN, FP, FN, TP
    gc.collect()
    # return ACC, SEN, SPE, AUC


if __name__ == '__main__':
    print(vars(args))
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"/3251903008/yyh/result_MoE_SMCI&PMCI/{args.model_type}_finetune_{timestamp}_data_dataArgument_top2_bl0.05_dl0.01_detach"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(f"{result_dir}/models", exist_ok=True)
    print(f"Results will be saved to: {result_dir}")
    
    # 将配置信息保存到文件
    with open(f"{result_dir}/config.txt", "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
    
    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    print(f"CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 当设置了CUDA_VISIBLE_DEVICES时，PyTorch会将可见GPU映射为逻辑设备0
    device = torch.device(args.device)
    print(f"Using device: {device}")
    torch.cuda.manual_seed_all(SEED)

    # 设置归一化类型
    norm_type = 'minmax'  # 可选: 'minmax' 或 'zscore'
    
    # 加载训练数据并获取统计量
    train_loader, train_stats = load_data(args, args.train_root_path, args.PMCI_dir, args.SMCI_dir, args.excel_file, norm_type)
    
    # 使用训练集的统计量加载验证数据
    val_loader = load_test_data(args, args.val_root_path, args.PMCI_dir, args.SMCI_dir, args.excel_file, train_stats, norm_type)
    
    # 使用训练集的统计量加载测试数据
    test_loader = load_test_data(args, args.test_root_path, args.PMCI_dir, args.SMCI_dir, args.excel_file, train_stats, norm_type)

    ResNet_3D1 = feature_Netin_tp4_multr_score(use_moe=args.model_type, top_k=4, dropout=args.dropout).to(device)

    # 加载预训练模型
    if args.model_path:
        print(f"Loading pretrained model from: {args.model_path}")
        pretrained_dict = torch.load(args.model_path, map_location=device)
        model_dict = ResNet_3D1.state_dict()
        
        # 过滤掉不匹配的层
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        
        # 更新模型权重
        model_dict.update(pretrained_dict)
        ResNet_3D1.load_state_dict(model_dict)
        
        loaded_layers = len(pretrained_dict)
        total_layers = len(model_dict)
        print(f"Successfully loaded {loaded_layers}/{total_layers} layers from pretrained model")
    else:
        print("No pretrained model specified, training from scratch")

    train_best_loss = 10000
    val_best_loss = 10000
    train_best_acc = 0
    val_best_acc = 0
    t_SEN = 0
    t_SPE = 0
    t_AUC = 0
    t_precision = 0
    t_f1 = 0

    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    test_acc_all = []
    count = 0

    since = time.time()

    for epoch in range(1, args.nepoch + 1):
        # 训练模型
        fo = open(f"{result_dir}/test.txt", "a")
        log_best = open(f'{result_dir}/log_best.txt', 'a')
        train_loss, train_correct, len_train = train_epoch(epoch, ResNet_3D1, train_loader, fo, device)
        # 保存模型
        print('model saved...')
        torch.save(ResNet_3D1.state_dict(), f'{result_dir}/models/3DNet4l_1_tp4_sifeature_clonefs_1bt1_l43_1_epoch_{epoch}_d1_det.pt')
        # torch.save(ResNet_3D2.state_dict(), f'{result_dir}/models/3DNet4l_2_tp4_sifeature_clonefs_1bt1_l43_1_epoch_{epoch}_d1_det.pt')
        # 打印当前训练损失、最小损失和准确率
        if train_loss < train_best_loss:
            train_best_loss = train_loss
        train_acc = 100. * train_correct / len_train
        if train_acc > train_best_acc:
            train_best_acc = train_acc
        print('current loss: ', train_loss, 'the best loss: ', train_best_loss)
        print(f'train_correct/train_data: {train_correct}/{len_train} accuracy: {train_acc:.2f}%')
    
        # 模型用于验证集并将评价指标写入txt
        ACC, SEN, SPE, AUC, val_loss = val_model(epoch, val_loader,  ResNet_3D1, log_best, device)
        if ACC > val_best_acc:  # if val_loss < val_best_loss   特征保存条件
            target_best_acc = ACC  # val_best_loss = val_loss
            val_best_acc = target_best_acc
            t_SEN = SEN
            t_SPE = SPE
            t_AUC = AUC
            #np.savez('feature_' + str(count), feature_for_GCN.detach().cpu().numpy())
            #count = count+1
    
        log_best.write('The best result:\n')
        log_best.write('ACC:  ' + str(val_best_acc) + '  SEN:  ' + str(t_SEN) + '  SPE:  ' + str(
            t_SPE) + '  AUC:  ' + str(t_AUC) + '\n\n')
    
        print(f'The train acc of this epoch: {train_acc:.2f}%')
        print(f'The best acc: {train_best_acc:.2f}% \n')
        fo.write('train_acc: '+str(train_acc)+' The current total loss: '+str(train_loss)+' The best loss: '+str(train_best_loss)+'\n\n')
    
        # 保存训练结果用于画图
        train_loss_all.append(train_loss)
        train_acc_all.append(train_acc)
        val_loss_all.append(val_loss)
        test_acc_all.append(ACC)
    
        del train_loss, train_correct, len_train, train_acc
        gc.collect()
        fo.close()
        log_best.close()
    
    # 将模型用于测试集
    test_model(test_loader, ResNet_3D1, device)
    
    time_use = time.time() - since
    print("Train and Test complete in {:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))
    
    train_process = pd.DataFrame(
        data={"epoch": range(args.nepoch),
              "train_loss_all": train_loss_all,
              "train_acc_all": train_acc_all,
              "val_loss_all": val_loss_all,
              "test_acc_all": test_acc_all}
    )
    train_process.to_csv(f'{result_dir}/train_process_result.csv')
    
    # 画图
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_process.epoch, train_process.train_loss_all, label="Train loss")
    plt.plot(train_process.epoch, train_process.val_loss_all, label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    
    plt.subplot(1,2,2)
    plt.plot(train_process.epoch, train_process.train_acc_all, label="Train acc")
    plt.plot(train_process.epoch, train_process.test_acc_all, label="Val acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.savefig(f"{result_dir}/accuracy_loss.png")
    #plt.show()
