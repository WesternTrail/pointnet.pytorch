from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512) # 而后再进行下采样，提取整体
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9) # 注意是9，而不是16.因为我们要得到的是3x3的变换矩阵
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x))) # 32,3,2500 -> 32,64,2500
        x = F.relu(self.bn2(self.conv2(x))) # - > 32,128,2500
        x = F.relu(self.bn3(self.conv3(x))) # - > 32,1024,2500
        x = torch.max(x, 2, keepdim=True)[0] # -> 32,1024,1
        x = x.view(-1, 1024) # -> 32,1024

        x = F.relu(self.bn4(self.fc1(x)))  # -> 32,512
        x = F.relu(self.bn5(self.fc2(x))) # -> 32,256
        x = self.fc3(x) # -> 32,9
        # iden: 32,9,ariable是一种可以不断变化的变量，符合反向传播，参数更新的属性。
        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden # 这里为什么相加？？我的理解是：iden的参数本身不好训练，加上x后作为一个整体一起练就好训练了？
        x = x.view(-1, 3, 3)   # -> [b,3,3]
        return x


class STNkd(nn.Module): # 这个是feature_transform
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x # 得到[b,64,64]

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)  # 用1x1卷积来代替MLP 1)为了可以不受输入尺寸也就是点云的点的个数的影响 2) 仅仅考虑一个点跨通道的特征，而不是整体 3) 如果使用全连接层提取特征，不满足点云的置换不变性
        self.conv2 = torch.nn.Conv1d(64, 128, 1) # 同时也符合论文提到的对称函数的设计，因为如果用全连接，这里点的输入顺序改变，输出结果也会有所不同。
        self.conv3 = torch.nn.Conv1d(128, 1024, 1) # 而1x1卷积作用于一个点的channel上，之后再取max的结果还是相同的
        self.bn1 = nn.BatchNorm1d(64) # 每经过一个全连接层就用bn层调整所有batch数据到均值为0，方差为1的分布中去
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x) # 得到旋转矩阵
        x = x.transpose(2, 1)  #(B,3,2500) -> (B,2500,3)
        x = torch.bmm(x, trans)  # input transform
        x = x.transpose(2, 1) #(B,2500,3) -> (B,3,2500)
        x = F.relu(self.bn1(self.conv1(x))) # 第一处：mlp 注意论文这里是进行了两次全连接

        if self.feature_transform: # false
            trans_feat = self.fstn(x) # k x k
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x # 保存每个点的局部特征，方便之后做分割
        x = F.relu(self.bn2(self.conv2(x))) # 32,64,2500-> 32,128,2500 注意这里论文则还经过了一个[64x64]的全连接
        x = self.bn3(self.conv3(x)) # 32,128,2500 -> 32,1024,2500
        x = torch.max(x, 2, keepdim=True)[0]  # 32,1024,2500 -> 32,1024,1，max的目的是从所有点中提取特征
        x = x.view(-1, 1024) # 32，1024，1 -> 32,1024，得到global_feature全局特征
        if self.global_feat: # 分类则直接返回
            return x, trans, trans_feat
        else:  # 分割的话与之前的局部特征进行拼接
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts) # 32,1024,1 -> 32,1024,2500 将全局特征调整到与x相同的维度，方便相加
            return torch.cat([x, pointfeat], 1), trans, trans_feat # -> 32,1088,2500

class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)

        # 现在能用普通的全连接层，是因为之前x做了max处理。就算原始输入变更输入顺序，也不会改变global_feature
        x = F.relu(self.bn1(self.fc1(x)))   # 32,1024 -> 32,512
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))     # -> 32,256
        x = self.fc3(x)     # -> 32,16
        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetDenseCls(nn.Module): # pointnet网络
    def __init__(self, k = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x) # x: [16 ,1088 ,2500]
        x = F.relu(self.bn1(self.conv1(x))) # 1088->512
        x = F.relu(self.bn2(self.conv2(x))) # 512->256
        x = F.relu(self.bn3(self.conv3(x))) # 256->128
        x = self.conv4(x)                   # 128 ->4
        x = x.transpose(2,1).contiguous()   # (16,4,2500) -> [16,2500,4]
        x = F.log_softmax(x.view(-1,self.k), dim=-1)  # [40000,4]
        x = x.view(batchsize, n_pts, self.k) # 40000,4
        return x, trans, trans_feat

def feature_transform_regularizer(trans): # 正则化变换矩阵
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

if __name__ == '__main__': # 调试用代码
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())
