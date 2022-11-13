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
import numpy as np
# random seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

setup_seed(0)

learning_rate = 0.01

device = 'cuda' if torch.cuda.is_available() else 'cpu'

start_epoch = 0

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

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


net = resnet18()

net = net.to(device)

criterion = nn.CrossEntropyLoss()


optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

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

def train(epoch):
    print('\nEpoch: ', epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    global best_train_acc

    # last_iter_temp = []
    # for i in range(len(old_res_list)):
    #     last_iter_temp.append(old_res_list[i])

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        

        is_time_res = False
        if epoch > 40:
            is_time_res = True
        if inputs.shape[0] != 128:
            is_time_res = False
        outputs,  new_res_list= net(inputs, old_res_list, is_time_res)

        if epoch > 40:
            old_res_list.clear()

            if inputs.shape[0] == 128:
                for i in range(len(new_res_list)):
                    # x = new_res_list[i].clone().detach()*0.05 + last_iter_temp[i]*0.05
                    x = new_res_list[i].clone().detach()*0.05
                    # last_iter_list[i] += new_res_list[i].clone().detach()*0.05
                    old_res_list.append(x)


        loss = criterion(outputs, targets)
        loss.backward(retain_graph=True)
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    if correct/total > best_train_acc:
        best_train_acc = correct/total
    print('Train | ', 'Acc : ', correct/total, ' | ', 'Best Acc : ', best_train_acc)
    


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
    print('Test | ', 'Acc : ', correct/total, ' | ', 'Best Acc : ', best_test_acc)
        



for epoch in range(start_epoch, start_epoch+50):
    train(epoch)

    # if epoch > 40:
    #     old_res_list.clear()
    #     for i in range(len(last_iter_list)):
    #         old_res_list.append(last_iter_list[i] / 80)

    #     plane = 64
    #     size = 32
    #     for i in range(8):
    #         last_iter_list.append(torch.zeros(128, plane, int(size), int(size), device=device, requires_grad=False))
    #         if i %2 != 0:
    #             plane *= 2
    #             size /= 2
    if epoch > 40:
        plane = 64
        size = 32

        for i in range(8):
            old_res_list.append(torch.zeros(128, plane, int(size), int(size), device=device, requires_grad=False))
            if i %2 != 0:
                plane *= 2
                size /= 2
    test(epoch)
    scheduler.step()