from __future__ import print_function
# 命令选项与参数解析模块
import argparse
# 操作系统相关功能的函数
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
# 数据集构造模块
from pointnet.dataset import ShapeNetDataset, ModelNetDataset
# pointnet网络模块
from pointnet.model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
# 进度条模块
from tqdm import tqdm

# 传入运行参数
parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=0) # 调试的时候最好设置为0，不然可能没法单步调试
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder') # 输出文件夹的名称
parser.add_argument('--model', type=str, default='', help='model path')  # 预训练模型
parser.add_argument('--dataset', type=str, required=True, help="dataset path") #
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

# 解析参数
opt = parser.parse_args()
print(opt)

# 正则表达式？将打印时的test字符变蓝？
blue = lambda x: '\033[94m' + x + '\033[0m'

# 返回固定的随机数种子
opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# 构建dataset
if opt.dataset_type == 'shapenet':
    dataset = ShapeNetDataset(  # 实例化训练集
        root=opt.dataset,
        classification=True,
        npoints=opt.num_points)

    test_dataset = ShapeNetDataset(  # 实例化测试集
        root=opt.dataset,
        classification=True,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
elif opt.dataset_type == 'modelnet40': # 如果下载的数据集是modelenet则运行Modelnet
    dataset = ModelNetDataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split='trainval')

    test_dataset = ModelNetDataset(
        root=opt.dataset,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
else:
    exit('wrong dataset type')


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

print(len(dataset), len(test_dataset)) # 训练样本：12137 测试样本:2874
num_classes = len(dataset.classes) # 16类
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

# 实例化pointnet的网络结构
classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))


optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda() # 网络参数放到cuda上去

num_batch = len(dataset) / opt.batchSize

for epoch in range(opt.nepoch):
    scheduler.step()
    for i, data in enumerate(dataloader, 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1) # 数据维度变为【B,C,N】
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        classifier = classifier.train() #等价于model.train()，
        pred, trans, trans_feat = classifier(points)
        loss = F.nll_loss(pred, target) # 负log对数似然损失，因为网络前向传播最后一层是softmax，本质上就是交叉熵：crossentry_loss = nll_loss + softmax结果
        # 如果采用了feature_transform模块来增强网络的旋转不变性
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1] #.max(1)按照列找到最大值的下标，同时返回的是字典：{最大值，最大值下标}
        correct = pred_choice.eq(target.data).cpu().sum() # 分类正确的数量
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))

        if i % 10 == 0: # 每10个batch进行一次evaluate,而不是训练完一个epoch后就评估整个数据集，效率上更慢
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _, _ = classifier(points)
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            # 打印的是testdata里一个batch的loss和acc
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize)))

    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

total_correct = 0
total_testset = 0
for i,data in tqdm(enumerate(testdataloader, 0)): # 训练完所有epoch后，再整体对test评估一遍
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

print("final accuracy {}".format(total_correct / float(total_testset))) # 这里打印整个testdata的数据

# 参考地址：https://blog.csdn.net/yuanmiyu6522/article/details/121435650?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166330797616800184110423%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166330797616800184110423&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-121435650-null-null.142^v47^pc_rank_34_default_3,201^v3^control&utm_term=pointnet.pytorch&spm=1018.2226.3001.4187