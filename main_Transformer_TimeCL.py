# -*- coding: utf-8 -*-
"""
@CreatedDate:   2020/4/27 12:08
@Author: Pangpd(https://github.com/pangpd/DS-pResNet-HSI)
@UsedBy: lyh
"""
import os
import sys
import time

from data.Dataset import getTrentoData,getHouston2018Data,getAugsburgData,getHoustonData,getBerlinData,getMUUFLData

from utils.auxiliary import get_logger, PrototypeContrastiveLoss
from utils.hyper_pytorch import *
from datetime import datetime

import torch
import torch.nn.parallel
import warnings
warnings.filterwarnings('ignore')

from utils.start import test, train, output_metric,test_baseline

from models.QI_STransformer import ComplexMultimodalSpikformer
# from models.QI_TCCL import ComplexMultimodalSpikformer

np.set_printoptions(linewidth=400)
np.set_printoptions(threshold=sys.maxsize)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# -------------------------定义超参数--------------------------
data_path = os.path.join(os.getcwd(), 'data')  # 数据集路径

seed = 1014
epochs = 50

# learn_rate = 0.0005
learn_rate = 0.0085
momentum = 0.9
weight_decay = 0.0001
# train_dataset = "Augsburg"
# train_dataset = "Houston2013"
train_dataset = "Houston2018"
# train_dataset = "Trento"
# train_dataset = "MUUFL"
# train_dataset = "Berlin"

# Dataset.set_random_seed(0)
if train_dataset == "Houston2013":
    image_h = 349
    image_w = 1905
    train_loader, test_loader, trntst_loader, all_loader = getHoustonData(
        hsi_path="data/Houston2013/houston_hsi.mat",
        lidar_path="data/Houston2013/houston_lidar.mat",
        gt_path="data/Houston2013/houston_gt.mat",
        index_path="data/Houston2013/houston_index.mat",
        channels=20,
        windowSize=7,
        batch_size=128,
        num_workers=0)
elif train_dataset == "MUUFL":
    image_h = 166
    image_w = 600
    train_loader, test_loader, trntst_loader, all_loader = getMUUFLData(
        hsi_path='data/MUUFL/muufl_hsi.mat',
        lidar_path='data/MUUFL/muufl_lidar.mat',
        gt_path='data/MUUFL/muufl_gt.mat',
        channels=10,
        windowSize=9,
        batch_size=128,
        num_workers=0,
        train_samples_per_class=[1162,214,344,91,334,23,112,312,69,9,13],
        seed=1014)
elif train_dataset == "Trento":
    image_h = 166
    image_w = 600
    train_loader, test_loader, trntst_loader, all_loader = getTrentoData(
        hsi_path="data/Trento/trento_hsi.mat",
        lidar_path="data/Trento/trento_lidar.mat",
        gt_path="data/Trento/trento_gt.mat",
        index_path="data/Trento/trento_index.mat",
        channels=10,
        windowSize=17,
        batch_size=128,
        num_workers=0)
elif train_dataset == "Augsburg":
    image_h = 166
    image_w = 600
    train_loader, test_loader, trntst_loader, all_loader = getAugsburgData(
        hsi_path="data/Augsburg/augsburg_hsi.mat",
        lidar_path="data/Augsburg/augsburg_sar.mat",
        gt_path="data/Augsburg/augsburg_gt.mat",
        index_path="data/Augsburg/augsburg_index.mat",
        channels=10,
        windowSize=7,
        batch_size=128,
        num_workers=0)
elif train_dataset == "Berlin":
    image_h = 1723
    image_w = 476
    train_loader, test_loader, trntst_loader, all_loader = getBerlinData(
        hsi_path="data/Berlin/berlin_hsi.mat",
        lidar_path="data/Berlin/berlin_sar.mat",
        gt_path="data/Berlin/berlin_gt.mat",
        index_path="data/Berlin/berlin_index.mat",
        channels=10,
        windowSize=9,
        batch_size=128,
        num_workers=0)
else:
    image_h = 1202
    image_w = 4768
    train_loader, test_loader, trntst_loader, all_loader = getHouston2018Data(
        hsi_path="data/Houston2018/houston_hsi.mat",
        lidar_path="data/Houston2018/houston_lidar.mat",
        gt_path="data/Houston2018/houston_gt.mat",
        index_path="data/Houston2018/houston_index.mat",
        channels=20,
        windowSize=15,
        batch_size=128,
        num_workers=0)
iter = 1
def main():
    # ----------------------定义日志格式---------------------------
    time_str = datetime.strftime(datetime.now(), '%m-%d_%H-%M-%S')
    log_path = os.path.join(os.getcwd(), "logs")  # logs目录
    log_dir = os.path.join(log_path, time_str)  # log组根目录

    oa_list = []
    aa_list = []
    kappa_list = []
    each_acc_list = []
    train_time_list = []
    test_time_list = []

    torch.cuda.empty_cache()
    group_log_dir = os.path.join(log_dir, "Experiment_")  # logs组目录
    if not os.path.exists(group_log_dir):
        os.makedirs(group_log_dir)
    group_logger = get_logger(str(iter + 1), group_log_dir)
    random_state = seed + iter
    print('-------------------------------------------Iter %s----------------------------------' % (iter + 1))
    start(group_log_dir, logger=group_logger)

def start(group_log_dir, logger):
    print('进入main.py 中的start方法！')
    use_cuda = True
    # MUUFL
    # model = ComplexMultimodalSpikformer(
    #     hsi_channels=10,
    #     lidar_channels=2,
    #     num_classes=11,
    #     embed_dim=256,  # Houyston2013 192，Augsburg 64, Trento 16，Houston2018 Berlin,128,MUUFL 256 head2
    #     num_heads=2,  # Houston2013 Houston2018 Trento Berlin 1,
    #     depth=1,
    #     time_steps=5  # Augsburg和Houston2013 Houston2013 10,Trento 5，Houston2018 最好15
    # ).to(device)

    # Augsburg
    # model = ComplexMultimodalSpikformer(
    #     hsi_channels=10,
    #     lidar_channels=4,
    #     num_classes=7,
    #     embed_dim=64,  # Houyston2013 192，Augsburg 64, Trento 16，Houston2018 Berlin,128
    #     num_heads=1,  # Houston2013 Houston2018 Trento Berlin 1,
    #     depth=1,
    #     time_steps=10  # Augsburg和Houston2013 Houston2013 10,Trento 5，Houston2018 最好15
    # ).to(device)

    # Houston2018
    model = ComplexMultimodalSpikformer(
        hsi_channels=20,
        lidar_channels=1,
        num_classes=20,
        embed_dim=128, # Houyston2013 128，Augsburg 64, Trento 16，Houston2018 Berlin,128
        num_heads=1, #Houston2013 Houston2018 Trento Berlin 1,
        depth=1,
        time_steps=15 # Augsburg和Houston2013 Houston2013 10,Trento 5，Houston2018 最好15
    ).to(device)

    print(model)
    model =model.cuda()

    # 定义损失函数和优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    # optimizer = torch.optim.Adam(all_parameters, lr=0.0085, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    best_oa = -1
    best_aa = -1
    best_kappa = -1
    best_each_acc = -1
    best_acc = -1
    # 定义两个数组,记录训练损失和验证损失
    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []

    train_start_time = time.time()  # 返回当前的时间戳
    for epoch in range(epochs):
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
        if hasattr(model, 'fusion') and hasattr(
                model.fusion, 'log_and_reset_epoch'
        ):
            model.fusion.log_and_reset_epoch(epoch)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        valid_loss, valid_acc, test_acc1, test_obj, tar_v, pre_v = test_baseline(test_loader, model, criterion, epoch, use_cuda)

        logger.info('Epoch: %03d   Train Loss: %f Train Accuracy: %f   Valid Loss: %f Valid Accuracy: %f' % (
            epoch, train_loss, train_acc, valid_loss, valid_acc))

        OA_TE, AA_TE, Kappa_TE, CA_TE = output_metric(tar_v, pre_v)
        # print("OA: {:.2f} | AA: {:.2f} | Kappa: {:.4f}".format(OA_TE * 100, AA_TE * 100, Kappa_TE))
        # print("CA: ", CA_TE * 100)
        # logger.info('AA: %f, OA: %f, kappa: %f\n ' % (AA_TE * 100, OA_TE * 100, Kappa_TE))
        # logger.info('CA: %s \n' , CA_TE * 100)
        # scheduler.step(train_loss)
        scheduler.step(valid_loss)
        valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_acc)

        # save model
        if valid_acc > best_acc:
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': valid_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, group_log_dir + "/best_model.pth_Trento.tar")
            best_acc = valid_acc
            best_oa = OA_TE * 100
            best_aa = AA_TE * 100
            best_kappa = Kappa_TE
            best_each_acc = CA_TE * 100

    logger.info('best_AA: %f, best_OA: %f, best_kappa: %f\n ' % (best_aa, best_oa, best_kappa))
    logger.info('best_CA: %s \n', best_each_acc)

    train_end_time = time.time()
    checkpoint = torch.load(group_log_dir + "/best_model.pth_Trento.tar")
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    # 测试
    test_start_time = time.time()
    test_loss, test_acc, test_acc1, test_obj, tar_v, pre_v = test_baseline(test_loader, model, criterion, epoch, use_cuda)
    OA_TE, AA_TE, Kappa_TE, CA_TE = output_metric(tar_v, pre_v)
    print("OA: {:.2f} | AA: {:.2f} | Kappa: {:.4f}".format(OA_TE * 100, AA_TE * 100, Kappa_TE))
    logger.info('AA: %f, OA: %f, kappa: %f\n '% (AA_TE* 100, OA_TE * 100, Kappa_TE))
    test_end_time = time.time()
    logger.info("Final:   Loss: %s  Accuracy: %s", test_loss, test_acc)

    train_time = train_end_time - train_start_time
    test_time = test_end_time - test_start_time
    # logger.debug('classification:\n %s\n confusion:\n%s\n ' % (classification, confusion))
    logger.info("Train time:%s , Test time:%s", train_time, test_time)


def adjust_learning_rate(optimizer, epoch, learn_rate):
    lr = learn_rate * (0.1 ** (epoch // 50)) * (0.1 ** (epoch // 225))  # 每隔25个epoch更新学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()