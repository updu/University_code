import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    # 网络初始化
    def __init__(self):
        super(Net, self).__init__()

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 32, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2)
        self.bn4 = nn.BatchNorm2d(32)

        self.conv5 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(32 * 2 * 2, 256)
        self.drop1 = nn.Dropout2d()
        self.fc2 = nn.Linear(256, 128)
        self.drop2 = nn.Dropout2d()
        self.fc3 = nn.Linear(128, 10)

    # 前向传播
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        x = x.view(-1, 32 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)
        x = F.relu(x)
        return x


# 卷积核可视化
def draw_kernel(net):
    # cnn.cpu()
    print(np.shape(net.conv1.cpu().weight))
    weight = net.conv1.cpu().weight.data.numpy()
    weight = weight - np.min(weight)
    weight = weight / (np.max(weight))
    plt.figure()
    for i, filt in enumerate(weight):
        plt.subplot(8, 8, i + 1)
        filt = filt.transpose(1, 2, 0)
        plt.imshow(filt)
    plt.savefig("./images/kernel.png")
    plt.show()


# 训练函数
def train():
    # 归一化
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # 加载训练数据集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=1)

    net = Net()

    net = net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    lv = []
    for epoch in range(1):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            # 梯度清零
            optimizer.zero_grad()
            # net()为网络模型，通过模型得到输出
            outputs = net(inputs)
            # 计算损失
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 20 == 19:
                print('[%d,%5d] loss:%.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                lv.append(running_loss / 1000)
                running_loss = 0.0
        test(net)

    lv.append(running_loss / 20)

    print('Finished Training')
    # plt.plot(lv, '-g')
    # plt.show()
    draw_kernel(net)


#测试函数
def test(net):
    # 使用gpu训练
    net.cuda()
    # 数据归一化
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # 加载测试集
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=1)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images:%d %%' % (100 * correct / total))


if __name__ == '__main__':
    train()