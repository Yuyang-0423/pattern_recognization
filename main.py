import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch import nn
from torch import optim
from torch.nn import functional as F
from tensorboardX import SummaryWriter
def main():
    ##### 加载数据
    batchsize = 64
    epoch_num = 6
    class_num = 2
    learn_rate = 1e-4
    layers = [3, 4, 6, 3]
    # 训练集
    train_dataset = datasets.ImageFolder(root='D:\python\新建文件夹\PR\dataset/train', transform=transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.ToTensor()
    ]))
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)

    # 测试集
    test_dataset = datasets.ImageFolder(root='D:\python\新建文件夹\PR\dataset/test', transform=transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]))
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True)

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

    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=learn_rate)
    writer1 = SummaryWriter('run/train')
    for epoch in range(epoch_num):
        # train
        loss_record = []
        net.train()
        for x, label in train_loader:
            x, label = x.to(device), label.to(device)
            y = net(x)
            loss = criteon(y, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_record.append(loss.item())
        writer1.add_scalar('loss',sum(loss_record), global_step=epoch)
        print('epoch:{0}, Loss:{1}'.format(epoch, sum(loss_record)))

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
            writer1.add_scalar('Acc', acc, global_step=epoch)
            print('epoch:{0}, acc:{1}'.format(epoch, acc))

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