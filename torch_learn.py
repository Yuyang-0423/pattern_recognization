import os

import torch.nn
import torchvision.datasets
from torch import nn, optim
from torch.utils import tensorboard
from torch.utils.data import DataLoader

train_set = torchvision.datasets.CIFAR10(root='cifar10', train=True, transform=torchvision.transforms.Compose([
    torchvision.transforms.RandomRotation(15),
    torchvision.transforms.ToTensor()
]), download=True)
test_set = torchvision.datasets.CIFAR10(root='cifar10', train=False, transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
]), download=True)
train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_data = DataLoader(test_set, batch_size=64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torchvision.models.resnet18(weights=None)
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# module = torch.load('module.pth')
if os.path.exists('model.pth'):
    model.load_state_dict(torch.load('model.pth'))

writer = tensorboard.SummaryWriter('cifar10_writer')

for epoch in range(10):
    model.train()
    loss_record = []
    for img, label in train_data:
        img = img.to(device)
        label = label.to(device)
        result = model(img)
        optimizer.zero_grad()
        loss = loss_fn(label, result)
        loss.backward()
        optimizer.step()
    writer.add_scalar('cifar训练', sum(loss_record), global_step=epoch)
    print('第{}次测试的loss：{}'.format(epoch, sum(loss_record)))

    model.eval()
    with torch.no_grad():
        for img, label in test_data:
            acc = 0
            img = img.to(device)
            label = label.to(device)
            result = model(img)
            acc += torch.eq(result.argmax(1), label).float().sum().item()

        writer.add_scalar('cifar测试', acc, epoch)
        print('第{}次测试的acc：{}'.format(epoch, acc / len(test_data)))

    torch.save(model.state_dict(), 'model_{}.pth'.format(epoch))
    # torch.save(model,'module.pth')
writer.close()
