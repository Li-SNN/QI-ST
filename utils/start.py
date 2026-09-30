# -*- coding: utf-8 -*-
"""
@Author: Pangpd (https://github.com/pangpd/DS-pResNet-HSI)
@UsedBy: Katherine_Cao (https://github.com/Katherine-Cao/HSI_SNN)
"""
import math

import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report, accuracy_score
from torch import nn

from models.baselineSNN_CrossmodalCL import contrastive_loss, supervised_crossmodal_contrastive_loss
from models.baselineSNN_CrossmodalCL_Time import temporal_supervised_crossmodal_contrastive_loss
# from models.baselineSNN_CrossmodalCL_Time import contrastive_loss, supervised_crossmodal_contrastive_loss
from utils import evaluate
import torch
import torch.nn.parallel

from utils.auxiliary import supervised_contrastive_loss, temporal_contrastive_loss
from utils.evaluate import AA_andEachClassAccuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiTimestepContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_hsi, z_lidar):
        """
        z_hsi:   [T, B, D]
        z_lidar: [T, B, D]
        """
        T, B, D = z_hsi.shape
        device = z_hsi.device

        # 1. 调整维度为 [B, T, D] 并展平为 [B * T, D]
        # 展平后，每连续 T 个向量都属于同一个样本
        z_h = F.normalize(z_hsi.permute(1, 0, 2).reshape(B * T, D), dim=-1)
        z_l = F.normalize(z_lidar.permute(1, 0, 2).reshape(B * T, D), dim=-1)

        # 2. 构建多正样本掩码矩阵 pos_mask: [B * T, B * T]
        # 生成标签 [0,...,0, 1,...,1, ..., B-1,...,B-1]，每个标签重复 T 次
        labels = torch.arange(B, device=device).repeat_interleave(T)
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # 同一样本的所有时间步均为 1

        # 3. 计算跨模态相似度矩阵: [B * T, B * T]
        sim_h2l = torch.matmul(z_h, z_l.T) / self.temperature
        sim_l2h = sim_h2l.T

        # 4. 计算 Log-Softmax (分母包含所有 B * T 个特征作为候选)
        log_prob_h2l = F.log_softmax(sim_h2l, dim=1)
        log_prob_l2h = F.log_softmax(sim_l2h, dim=1)

        # 5. 仅在正样本位置提取 log 概率，并对该样本对应的正样本总数求平均
        loss_h2l = -(log_prob_h2l * pos_mask).sum(dim=1) / pos_mask.sum(dim=1)
        loss_l2h = -(log_prob_l2h * pos_mask).sum(dim=1) / pos_mask.sum(dim=1)

        # 6. 计算双向对比损失的平均值
        total_loss = (loss_h2l.mean() + loss_l2h.mean()) / 2.0
        return total_loss

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt

criterion_contrast = MultiTimestepContrastiveLoss(temperature=0.07)

def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    model.train()
    accs = np.ones((len(trainloader))) * -1000.0
    losses = np.ones((len(trainloader))) * -1000.0
    for batch_idx, (hsi,lidar,labels) in enumerate(trainloader):
        hsi = hsi.to(device)
        lidar = lidar.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs,z_hsi_seq,z_lidar_seq = model(hsi,lidar)
        # outputs = model(hsi,lidar)
        # outputs, z_hsi, z_lidar = model(hsi, lidar, mode="joint")
        loss = criterion(outputs, labels)  # CrossEntropyloss
        loss_cl = criterion_contrast(z_hsi_seq, z_lidar_seq)

        loss = loss + loss_cl # Berlin用0.3 MUUFL 0.1
        # loss = loss + 0.1 * loss_cl # Houston2013 0.5 最好，Augsburg 或 0.1.Houston2018用的1.0，Trento 0.5或1.0
        # loss = loss# Houston2013 0.5 最好，Augsburg 0.1.Houston2018用的1.0，Trento 0.5或1.0
        # MUUFL数据集使用0.1

        losses[batch_idx] = loss.item()
        accs[batch_idx] = evaluate.accuracy(outputs.data, labels.data)[0].item()

        loss.backward()
        optimizer.step()

    return np.average(losses), np.average(accs)

def train_baseline(trainloader, model, criterion, optimizer, epoch, use_cuda):
    model.train()
    accs = np.ones((len(trainloader))) * -1000.0
    losses = np.ones((len(trainloader))) * -1000.0
    for batch_idx, (hsi,lidar,labels) in enumerate(trainloader):
        hsi = hsi.to(device)
        lidar = lidar.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(hsi,lidar)
        loss = criterion(outputs, labels)  # CrossEntropyloss

        losses[batch_idx] = loss.item()
        accs[batch_idx] = evaluate.accuracy(outputs.data, labels.data)[0].item()

        loss.backward()
        optimizer.step()

    return np.average(losses), np.average(accs)

def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    # print(matrix)
    # OA, AA_mean, Kappa, AA = cal_results(matrix[1:,1:])
    # print("haha")
    # OA, AA_mean, Kappa, AA = cal_results(matrix[:,1:12])
    OA, AA_mean, Kappa, AA = cal_results(matrix)

    return OA, AA_mean, Kappa, AA

def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA

def test_baseline(testloader, model, criterion, epoch, use_cuda):
    model.eval()
    accs = np.ones((len(testloader))) * -1000.0
    losses = np.ones((len(testloader))) * -1000.0
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    with torch.no_grad():
        for batch_idx, (hsi,lidar,targets) in enumerate(testloader):
            hsi = hsi.to(device)
            lidar = lidar.to(device)
            targets = targets.to(device)

            # outputs, _, _, _ = model(hsi, lidar, boundary)
            outputs = model(hsi,lidar)
            # for name, rate in spike_rates.items():
            #     print(f"{name} average firing rate: {rate:.4f}")

            losses[batch_idx] = criterion(outputs, targets).item()     # CrossEntropyLoss
            loss = criterion(outputs, targets)     # CrossEntropyLoss
            accs[batch_idx] = evaluate.accuracy(outputs.data, targets.data, topk=(1,))[0].item()

            prec1, t, p = accuracy(outputs, targets, topk=(1,))
            n = hsi.shape[0]
            objs.update(loss.data, n)
            top1.update(prec1[0].data, n)
            tar = np.append(tar, t.data.cpu().numpy())
            pre = np.append(pre, p.data.cpu().numpy())

    return np.average(losses), np.average(accs),top1.avg,objs.avg,tar,pre

def test(testloader, model, criterion, epoch, use_cuda):
    model.eval()
    accs = np.ones((len(testloader))) * -1000.0
    losses = np.ones((len(testloader))) * -1000.0
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    with torch.no_grad():
        for batch_idx, (hsi,lidar,targets) in enumerate(testloader):
            hsi = hsi.to(device)
            lidar = lidar.to(device)
            targets = targets.to(device)

            outputs = model(hsi,lidar)

            losses[batch_idx] = criterion(outputs, targets).item()     # CrossEntropyLoss
            loss = criterion(outputs, targets)     # CrossEntropyLoss
            accs[batch_idx] = evaluate.accuracy(outputs.data, targets.data, topk=(1,))[0].item()

            prec1, t, p = accuracy(outputs, targets, topk=(1,))
            n = hsi.shape[0]
            objs.update(loss.data, n)
            top1.update(prec1[0].data, n)
            tar = np.append(tar, t.data.cpu().numpy())
            pre = np.append(pre, p.data.cpu().numpy())

    return np.average(losses), np.average(accs),top1.avg,objs.avg,tar,pre

def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res, target, pred.squeeze()

# def predict(test_loader, model, use_cuda):
#     model.eval()
#     predicted = []
#     with torch.no_grad():
#         for batch_idx, (hsi,lidar, targets) in enumerate(test_loader):
#             hsi = hsi.to(device)
#             lidar = lidar.to(device)
#             targets = targets.to(device)
#             [predicted.append(a) for a in model(hsi,lidar).data.cpu().numpy()]
#     return np.array(predicted)

def predict_CSNN(test_loader, model, use_cuda):
    model.eval()
    predicted = []
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):

            hsi = data[0]
            lidar = data[1]

            if use_cuda:
                hsi = hsi.cuda()
                lidar = lidar.cuda()

            logits = model(hsi, lidar)

            predicted.extend(logits.data.cpu().numpy())
    return np.array(predicted)

def predict(test_loader, model, use_cuda):
    model.eval()
    predicted = []
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):

            hsi = data[0]
            lidar = data[1]

            if use_cuda:
                hsi = hsi.cuda()
                lidar = lidar.cuda()

            logits= model(hsi, lidar)

            predicted.extend(logits.data.cpu().numpy())
    return np.array(predicted)


def adjust_learning_rate(optimizer, epoch, learn_rate):
    lr = learn_rate * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))  # 1-149:0.1，150-200:0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
