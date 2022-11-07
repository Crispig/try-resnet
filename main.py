'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from model import *
from utils import progress_bar
from model_self import *
import numpy as np
# 设置随机数种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

setup_seed(0)

# hyper parameter
learning_rate = 0.001

device = 'cuda' if torch.cuda.is_available() else 'cpu'

start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = ResNet18()
net = resnet18()

net = net.to(device)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=learning_rate,
#                       momentum=0.9, weight_decay=5e-4)

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# for name ,param in net.named_parameters():
#     param.requires_grad = True
#     if 'old' in name:
#         param.requires_grad = False

# for name, param in net.named_parameters():
#     if param.requires_grad:
#         print(name,param.size())

optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.00001)

best_train_acc = 0
best_test_acc = 0


old_res_list = []
last_iter_list = []
plane = 64
size = 32

for i in range(8):
    old_res_list.append(torch.zeros(128, plane, int(size), int(size), device=device, requires_grad=False))
    if i %2 != 0:
        plane *= 2
        size /= 2

plane = 64
size = 32
for i in range(8):
    last_iter_list.append(torch.zeros(128, plane, int(size), int(size), device=device, requires_grad=False))
    if i %2 != 0:
        plane *= 2
        size /= 2

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    global best_train_acc

    last_iter_temp = []
    for i in range(len(old_res_list)):
        last_iter_temp.append(old_res_list[i])

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        # if batch_idx = 0
        is_time_res = False
        if inputs.shape[0] != 128:
            is_time_res = False
        outputs,  new_res_list= net(inputs, old_res_list, is_time_res)

        # old_res_list.clear()

        # if inputs.shape[0] == 128:
        #     for i in range(len(new_res_list)):
        #         x = new_res_list[i].clone().detach()*0.05 + last_iter_temp[i]*0.05
        #         last_iter_list[i] += new_res_list[i].clone().detach()
        #         old_res_list.append(x)


        loss = criterion(outputs, targets)
        loss.backward(retain_graph=True)
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if correct/total > best_train_acc:
            best_train_acc = correct/total
        progress_bar(batch_idx, len(trainloader), 'Train Loss: %.3f | Acc: %.3f%% | Best Acc: %.3f%%'
                     % (train_loss/(batch_idx+1), 100.*correct/total, 100.*best_train_acc))


def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    global best_test_acc
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            is_time_res = False
            # old_res_list = []
            outputs, _ = net(inputs, old_res_list, is_time_res)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if correct/total > best_test_acc:
                best_test_acc = correct/total
            progress_bar(batch_idx, len(testloader), 'Test Loss: %.3f | Acc: %.3f%% | Best Acc: %.3f%%'
                     % (test_loss/(batch_idx+1), 100.*correct/total, 100.*best_test_acc))




for epoch in range(start_epoch, start_epoch+50):
    train(epoch)
    # old_res_list.clear()
    # for i in range(len(last_iter_list)):
    #     old_res_list.append(last_iter_list[i] / 80)

    # plane = 64
    # size = 32
    # for i in range(8):
    #     last_iter_list.append(torch.zeros(128, plane, int(size), int(size), device=device, requires_grad=False))
    #     if i %2 != 0:
    #         plane *= 2
    #         size /= 2
    
    test(epoch)
    scheduler.step()