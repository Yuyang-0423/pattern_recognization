import torch
import torchvision.models
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import datetime

def main():
    ##### 加载数据
    ##  tensorboard --logdir=task/train
    batchsize = 64
    epoch_num = 500
    class_num = 10
    learn_rate = 1e-4
    layers = [2, 2, 2, 2]
    val_score = 0
    data_time = datetime.date.today()
    model_path = "model_layers_{}.pth".format("_".join(map(str, layers)))
    best_val_score_file = "best_val_score_layers_{}.txt".format("_".join(map(str, layers)))
    if os.path.exists(best_val_score_file):
        with open(best_val_score_file, "r") as file:
            val_score = float(file.read())

    train_set = datasets.FashionMNIST(
        root='FasionMNIST',  # 将数据保存在本地什么位置
        train=True,  # 我们希望数据用于训练集，其中6万张图片用作训练数据，1万张图片用于测试数据
        download=True,  # 如果目录下没有文件，则自动下载
        transform=transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.RandomRotation(15),
            transforms.ToTensor()
        ])  # 我们将数据转为Tensor类型
    )
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchsize, shuffle=True)
    test_set = datasets.FashionMNIST(
        root='FasionMNIST',  # 将数据保存在本地什么位置
        train=False,  # 我们希望数据用于训练集，其中6万张图片用作训练数据，1万张图片用于测试数据
        download=True,  # 如果目录下没有文件，则自动下载
        transform=transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])  # 我们将数据转为Tensor类型
    )
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batchsize, shuffle=True)

    # 损失函数、优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b2 = nn.Sequential(*resnet_block(64, 64, layers[0], first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, layers[1]))
    b4 = nn.Sequential(*resnet_block(128, 256, layers[2]))
    b5 = nn.Sequential(*resnet_block(256, 512, layers[3]))
    net = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(), nn.Linear(512, class_num)).to(device)

    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))  # 加载模型参数

    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=learn_rate)
    writer1 = SummaryWriter('train')
    for epoch in range(epoch_num):
        # train
        # loss_record = []
        # net.train()
        # for x, label in train_loader:
        #     x, label = x.to(device), label.to(device)
        #     y = net(x)
        #     loss = criteon(y, label)
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #     loss_record.append(loss.item())
        # # writer1.add_graph(net,x)
        # writer1.add_scalar('{3}_{2}_Loss'.format(epoch, sum(loss_record),"_".join(map(str, layers)),data_time),sum(loss_record), global_step=epoch)
        # print('{3}_{2}_epoch:{0}, Loss:{1}'.format(epoch, sum(loss_record),"_".join(map(str, layers)),data_time))

        # test
        net.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for x, label in test_loader:
                x, label = x.to(device), label.to(device)
                y = net(x)
                pred = y.argmax(dim=1)

                correct += torch.eq(pred, label).float().sum().item()
                total += x.size(0)
            acc = correct / total
            writer1.add_scalar('{0}_{1}_acc'.format("_".join(map(str, layers)),
                                                       data_time), acc, global_step=epoch)
            print('{3}_{2}_epoch:{0}, acc:{1}'.format(epoch, acc, "_".join(map(str, layers)),
                                                       data_time))

        if acc > val_score:
            torch.save(net.state_dict(), model_path)
            val_score = acc
            with open(best_val_score_file, "w") as file:
                file.write(str(val_score))
        # torch.save(net.state_dict(), model_path)
    writer1.close()

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

if __name__ == '__main__':
        main()