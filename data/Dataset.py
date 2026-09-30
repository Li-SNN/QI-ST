import torch
import random
import numpy as np
import torch.nn as nn
from scipy.io import loadmat
from sklearn.decomposition import PCA
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset


def min_max(x):
    min = np.min(x)
    max = np.max(x)
    return (x - min) / (max - min)

# 设置随机数种子
def set_random_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def applyPCA(X, numComponents):
    """
    apply PCA to the image to reduce dimensionality
  """
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


# 创建 Dataset
class HLDataset(Dataset):

    def __init__(self, hsi, lidar, pos, windowSize, gt=None, transform=None):
        self.pad = (windowSize - 1) // 2
        self.windowSize = windowSize
        self.hsi = np.pad(hsi, ((self.pad, self.pad),
                          (self.pad, self.pad), (0, 0)), mode='reflect')
        # if lidar.ndim ==2:
        #     self.lidar = np.pad(lidar, ((self.pad, self.pad),
        #                             (self.pad, self.pad)), mode='reflect')
        #
        # self.lidar = np.pad(lidar, ((self.pad, self.pad),
        #                             (self.pad, self.pad),(0, 0),), mode='reflect')
        # --- 统一 LiDAR 维度 ---
        if lidar.ndim == 2:
            lidar = np.expand_dims(lidar, axis=2)  # (H,W) → (H,W,1)

        # --- 只 pad 一次 ---
        self.lidar = np.pad(lidar,
                            ((self.pad, self.pad),
                             (self.pad, self.pad),
                             (0, 0)),
                            mode='reflect')
        self.pos = pos
        self.gt = None
        if gt is not None:
            self.gt = gt
        if transform:
            self.transform = transform

    def __getitem__(self, index):
        h, w = self.pos[index, :]
        hsi = self.hsi[h: h + self.windowSize, w: w + self.windowSize]
        lidar = self.lidar[h: h + self.windowSize, w: w + self.windowSize]
        if self.transform:
            hsi = self.transform(hsi).float()
            lidar = self.transform(lidar).float()
        if self.gt is not None:
            gt = torch.tensor(self.gt[h, w] - 1).long()
            # if gt == -1:
            #     print("warning!")
            # return hsi.unsqueeze(0), lidar, gt
            return hsi, lidar, gt
        # return hsi.unsqueeze(0), lidar, h, w
        return hsi, lidar, h, w

    def __len__(self):
        return self.pos.shape[0]

def split_train_test_by_class(gt, train_samples_per_class, seed=0):

    rng = np.random.RandomState(seed)

    classes = np.unique(gt)
    classes = classes[classes != 0]  # 删除未标注类别 0

    train_index = []
    test_index = []

    print("=" * 60)
    print("MUUFL train/test samples")
    print("=" * 60)

    for i, cls in enumerate(classes):

        # 当前类别所有样本的二维坐标
        pos = np.argwhere(gt == cls)

        # 打乱
        rng.shuffle(pos)

        # 获取当前类别需要的训练样本数
        if isinstance(train_samples_per_class, int):
            n_train = train_samples_per_class

        elif isinstance(train_samples_per_class, (list, tuple, np.ndarray)):
            n_train = train_samples_per_class[i]

        elif isinstance(train_samples_per_class, dict):
            n_train = train_samples_per_class[int(cls)]

        else:
            raise TypeError(
                "train_samples_per_class must be int, list, tuple, ndarray or dict"
            )

        # 防止训练样本数超过该类别总样本数
        if n_train >= len(pos):
            raise ValueError(
                f"Class {cls}: train samples ({n_train}) "
                f"must be smaller than total samples ({len(pos)})."
            )

        # 划分
        train_pos = pos[:n_train]
        test_pos = pos[n_train:]

        train_index.append(train_pos)
        test_index.append(test_pos)

        print(
            f"Class {int(cls):2d}: "
            f"Total = {len(pos):5d}, "
            f"Train = {len(train_pos):4d}, "
            f"Test = {len(test_pos):5d}"
        )

    train_index = np.concatenate(train_index, axis=0)
    test_index = np.concatenate(test_index, axis=0)

    print("-" * 60)
    print(f"Total training samples: {len(train_index)}")
    print(f"Total testing samples : {len(test_index)}")
    print("=" * 60)

    return train_index, test_index


# 根据 index 获取数据
def getData(hsi_path, lidar_path, gt_path, index_path, keys, channels, windowSize, batch_size, num_workers):
    '''
    hsi_path: 高光谱数据集路径
    lidar_path: Lidar数据集路径
    gt_path: 真实标签数据集路径
    index_path: 索引数据集路径
    keys: mat 文件的 key
    channels: 降维后的通道数
    windowSize: 每张图片切割后的尺寸
    batch_size: 每个 batch 中的图片数量
    num_workers: 使用几个工作进程进行 Dataloader 的加载
    '''

    # 加载图片数据和坐标位置
    '''
    hsi: 高光谱图像数据
    lidar: Lidar 图像数据
    gt: 真实标签, 0 代表未标注
    train_index: 用于训练的数据索引
    test_index: 用于测试的数据索引
    trntst_index: 用于训练和测试的数据索引，用于对有标签的数据进行可视化
    all_index: 所有数据的索引，包含未标注数据，用于对所有数据进行可视化
    '''
    hsi = loadmat(hsi_path)[keys[0]]
    lidar = loadmat(lidar_path)[keys[1]]

    if hsi_path == "data/Houston2013/houston_hsi.mat":
        lidar = min_max(lidar)

    gt = loadmat(gt_path)[keys[2]]
    train_index = loadmat(index_path)[keys[3]]
    test_index = loadmat(index_path)[keys[4]]
    trntst_index = np.concatenate((train_index, test_index), axis=0)
    all_index = loadmat(index_path)[keys[5]]

    # 使用 PCA 对 HSI 进行降维
    hsi = applyPCA(hsi, channels)

    # 创建 Dataset, 用于生成对应的 Dataloader
    HLtrainset = HLDataset(hsi, lidar, train_index,
                           windowSize, gt, transform=ToTensor())
    HLtestset = HLDataset(hsi, lidar, test_index,
                          windowSize, gt, transform=ToTensor())
    HLtrntstset = HLDataset(hsi, lidar, trntst_index,
                            windowSize, transform=ToTensor())
    HLallset = HLDataset(hsi, lidar, all_index,
                         windowSize, transform=ToTensor())

    # 创建 Dataloader
    '''
    train_loader: 训练集
    test_loader: 测试集 
    trntst_loader: 用于画图，底色为白色，如 Trento 可视化图
    all_loader: 用于画图，底色为非白色，如 Houston 可视化图
    '''

    train_loader = DataLoader(
        HLtrainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(
        HLtestset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    trntst_loader = DataLoader(
        HLtrntstset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    all_loader = DataLoader(
        HLallset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print("Success!")
    return train_loader, test_loader, trntst_loader, all_loader

def getMUUFLData(hsi_path,
                 lidar_path,
                 gt_path,
                 channels,
                 windowSize,
                 batch_size,
                 num_workers,
                 train_samples_per_class,
                 seed=0):


    print("MUUFL!")

    # --------------------------------------------------
    # 1. 加载数据
    # --------------------------------------------------
    hsi = loadmat(hsi_path)['muufl_hsi']
    lidar = loadmat(lidar_path)['muufl_lidar']
    gt = loadmat(gt_path)['muufl_gt']

    print("HSI shape   :", hsi.shape)
    print("LiDAR shape :", lidar.shape)
    print("GT shape    :", gt.shape)

    # --------------------------------------------------
    # 2. 根据类别划分训练集和测试集
    # --------------------------------------------------
    train_index, test_index = split_train_test_by_class(
        gt,
        train_samples_per_class=train_samples_per_class,
        seed=seed
    )

    # 所有有标签样本
    trntst_index = np.concatenate(
        (train_index, test_index),
        axis=0
    )

    # 所有像素，用于生成整幅分类图
    H, W = gt.shape
    row, col = np.meshgrid(
        np.arange(H),
        np.arange(W),
        indexing='ij'
    )

    all_index = np.stack(
        (row.reshape(-1), col.reshape(-1)),
        axis=1
    )

    # --------------------------------------------------
    # 3. HSI PCA
    # --------------------------------------------------
    hsi = applyPCA(hsi, channels)

    # --------------------------------------------------
    # 4. LiDAR 归一化
    # --------------------------------------------------
    lidar = min_max(lidar)

    # --------------------------------------------------
    # 5. 创建 Dataset
    # --------------------------------------------------
    HLtrainset = HLDataset(
        hsi,
        lidar,
        train_index,
        windowSize,
        gt,
        transform=ToTensor()
    )

    HLtestset = HLDataset(
        hsi,
        lidar,
        test_index,
        windowSize,
        gt,
        transform=ToTensor()
    )

    # 有标签像素，用于绘制分类图
    HLtrntstset = HLDataset(
        hsi,
        lidar,
        trntst_index,
        windowSize,
        transform=ToTensor()
    )

    # 所有像素，用于绘制完整分类图
    HLallset = HLDataset(
        hsi,
        lidar,
        all_index,
        windowSize,
        transform=ToTensor()
    )

    # --------------------------------------------------
    # 6. DataLoader
    # --------------------------------------------------
    train_loader = DataLoader(
        HLtrainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        HLtestset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    trntst_loader = DataLoader(
        HLtrntstset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    all_loader = DataLoader(
        HLallset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    print("MUUFL loading success!")

    return train_loader, test_loader, trntst_loader, all_loader

# 获取 Houston 数据集
def getHoustonData(hsi_path, lidar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers):
    '''
    hsi_path: 高光谱数据集路径
    lidar_path: Lidar数据集路径
    gt_path: 真实标签数据集路径
    index_path: 索引数据集路径
    channels: 降维后的通道数
    windowSize: 每张图片切割后的尺寸
    batch_size: 每个 batch 中的图片数量
    num_workers: 使用几个工作进程进行 Dataloader 的加载
    '''

    print("Houston!")

    # Houston mat keys
    keys = ['houston_hsi', 'houston_lidar', 'houston_gt',
            'houston_train', 'houston_test', 'houston_all']

    return getData(hsi_path, lidar_path, gt_path, index_path, keys, channels, windowSize, batch_size, num_workers)

def getBerlinData(hsi_path, lidar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers):
    '''
    hsi_path: 高光谱数据集路径
    lidar_path: Lidar数据集路径
    gt_path: 真实标签数据集路径
    index_path: 索引数据集路径
    channels: 降维后的通道数
    windowSize: 每张图片切割后的尺寸
    batch_size: 每个 batch 中的图片数量
    num_workers: 使用几个工作进程进行 Dataloader 的加载
    '''

    print("Berlin!")

    # Houston mat keys
    keys = ['berlin_hsi', 'berlin_sar', 'berlin_gt',
            'berlin_train', 'berlin_test', 'berlin_all']

    return getData(hsi_path, lidar_path, gt_path, index_path, keys, channels, windowSize, batch_size, num_workers)


def getTrentoData(hsi_path, lidar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers):
    '''
    hsi_path: 高光谱数据集路径
    lidar_path: Lidar数据集路径
    gt_path: 真实标签数据集路径
    index_path: 索引数据集路径
    channels: 降维后的通道数
    windowSize: 每张图片切割后的尺寸
    batch_size: 每个 batch 中的图片数量
    num_workers: 使用几个工作进程进行 Dataloader 的加载
    '''

    print("Trento!")

    # Trento mat keys
    keys = ['trento_hsi', 'trento_lidar', 'trento_gt',
            'trento_train', 'trento_test', 'trento_all']

    return getData(hsi_path, lidar_path, gt_path, index_path, keys, channels, windowSize, batch_size, num_workers)

def getAugsburgData(hsi_path, lidar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers):
    '''
    hsi_path: 高光谱数据集路径
    lidar_path: Lidar数据集路径
    gt_path: 真实标签数据集路径
    index_path: 索引数据集路径
    channels: 降维后的通道数
    windowSize: 每张图片切割后的尺寸
    batch_size: 每个 batch 中的图片数量
    num_workers: 使用几个工作进程进行 Dataloader 的加载
    '''

    print("Augsburg!")

    # Trento mat keys
    keys = ['augsburg_hsi', 'augsburg_sar', 'augsburg_gt',
            'augsburg_train', 'augsburg_test', 'augsburg_all']

    return getData(hsi_path, lidar_path, gt_path, index_path, keys, channels, windowSize, batch_size, num_workers)

def getHouston2018Data(hsi_path, lidar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers):
    '''
    hsi_path: 高光谱数据集路径
    lidar_path: Lidar数据集路径
    gt_path: 真实标签数据集路径
    index_path: 索引数据集路径
    channels: 降维后的通道数
    windowSize: 每张图片切割后的尺寸
    batch_size: 每个 batch 中的图片数量
    num_workers: 使用几个工作进程进行 Dataloader 的加载
    '''

    print("Houston2018!")

    # Trento mat keys
    keys = ['houston_hsi', 'houston_lidar', 'houston_gt',
            'houston_train', 'houston_test', 'houston_all']

    return getData(hsi_path, lidar_path, gt_path, index_path, keys, channels, windowSize, batch_size, num_workers)
