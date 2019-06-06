from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()

data_dir = '..\\input'
TRAIN = 'train'
VAL = 'val'
TEST = 'test'

# VGG-16 Takes 224x224 images as input, so we resize all of them
# 数据增强和处理--------------------------------------------------
data_transforms = {
    TRAIN: transforms.Compose([
        # 随机反转，随机变换亮度
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
    ]),
    VAL: transforms.Compose([
        transforms.Resize(244),
        transforms.ToTensor(),
    ]),
    TEST: transforms.Compose([
        transforms.Resize(244),
        transforms.ToTensor(),
    ])
}
image_datasets = {
    x: datasets.ImageFolder(
        os.path.join(data_dir, x),
        transform=data_transforms[x]
    )
    for x in [TRAIN, VAL, TEST]
}

dataloaders = {
    TRAIN: torch.utils.data.DataLoader(
        image_datasets[TRAIN], batch_size=20,
        shuffle=True, num_workers=4
    ),
    VAL: torch.utils.data.DataLoader(
        image_datasets[VAL], batch_size=20,
        shuffle=True, num_workers=4
    ),
    TEST: torch.utils.data.DataLoader(
        image_datasets[TEST], batch_size=20,
        shuffle=True, num_workers=4
    )
}


# 评估模型——思路与训练相似，只是不需要反向传播，只要前向传播
def eval_model(vgg, criterion):
    since = time.time()
    avg_loss = 0
    avg_acc = 0
    loss_test = 0
    acc_test = 0

    test_batches = len(dataloaders[TEST])
    print("Evaluating model")
    print('-' * 10)
    # 开始测试
    for i, data in enumerate(dataloaders[TEST]):
        if i % 100 == 0:
            print("\rTest batch {}/{}".format(i, test_batches), end='', flush=True)
        # 关闭反向传播功能
        vgg.train(False)
        vgg.eval()
        # 读取数据
        inputs, labels = data

        if use_gpu:
            inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)
        else:
            inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)
        # 前向传播
        outputs = vgg(inputs)
        # 预测结果
        _, preds = torch.max(outputs.data, 1)
        # 计算损失函数
        loss = criterion(outputs, labels)
        loss_test += loss.item()
        # 计算正确率
        acc_test += torch.sum(preds == labels.data)

        del inputs, labels, outputs, preds
        torch.cuda.empty_cache()
    # 平均损失函数和正确率
    avg_loss = loss_test / dataset_sizes[TEST]
    avg_acc = acc_test.item() / dataset_sizes[TEST]
    # 用时
    elapsed_time = time.time() - since
    print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Avg loss (test): {:.4f}".format(avg_loss))
    print("Avg acc (test): {:.4f}".format(avg_acc))
    print('-' * 10)


# 训练模型
def train_model(vgg, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(vgg.state_dict())
    best_acc = 0.0

    avg_loss = 0
    avg_acc = 0
    avg_loss_val = 0
    avg_acc_val = 0

    train_batches = len(dataloaders[TRAIN])
    val_batches = len(dataloaders[VAL])

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)

        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0

        vgg.train(True)
        for i, data in enumerate(dataloaders[TRAIN]):
            if i % 100 == 0:
                print("\rTraining batch {}/{}".format(i, train_batches / 2), end='', flush=True)

            # 使用一半数据集训练
            if i >= train_batches / 2:
                break
            inputs, labels = data
            print(labels)
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            # 梯度置零
            optimizer.zero_grad()
            # 前向传播
            outputs = vgg(inputs)
            # 取最大值对应的标签作为预测值
            _, preds = torch.max(outputs.data, 1)
            # 计算损失函数
            loss = criterion(outputs, labels)
            # 反向传播求梯度
            loss.backward()
            # 权重更新
            optimizer.step()
            # 求和损失函数
            loss_train += loss.item()
            # 累加正确个数
            acc_train += torch.sum(preds == labels.data)
            # 释放空间
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

        # * 2 as we only used half of the dataset
        # 每个epoch计算平均损失函数和正确率
        avg_loss = loss_train * 2 / dataset_sizes[TRAIN]
        avg_acc = acc_train.item()*2/dataset_sizes[TRAIN]

        vgg.train(False)
        vgg.eval()

        # 每个epoch 评估一下
        for i, data in enumerate(dataloaders[VAL]):
            if i % 100 == 0:
                print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)

            inputs, labels = data
            # 判断是否使用GPU
            if use_gpu:
                inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)
            else:
                inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            outputs = vgg(inputs)
            # 取概率最大的作为预测类
            _, preds = torch.max(outputs.data, 1)
            # 计算损失函数（交叉熵损失函数）
            loss = criterion(outputs, labels)
            loss_val += loss.item()
            # 计算正确的数目
            acc_val += torch.sum(preds == labels.data)
            # 释放空间
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
        # 计算平均损失函数
        avg_loss_val = loss_val / dataset_sizes[VAL]
        # 计算准确率
        avg_acc_val = acc_val.item() / dataset_sizes[VAL]
        # 打印train和val的损失函数和正确率
        print()
        print("Epoch {} result: ".format(epoch))
        print("Avg loss (train): {:.4f}".format(avg_loss))
        print("Avg acc (train): {:.4f}".format(avg_acc))
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        print('-' * 10)
        # 如果正确率高则保存此参数
        if avg_acc_val > best_acc:
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(vgg.state_dict())
        # 学习率更新
        scheduler.step()
    # 计算训练总时间
    elapsed_time = time.time() - since
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))
    # 储存训练好的参数
    vgg.load_state_dict(best_model_wts)
    return vgg

if __name__ == '__main__':
    # 判断是否使用GPU
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using CUDA")
    # 计算数据集大小
    dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, VAL, TEST]}
    for x in [TRAIN, VAL, TEST]:
        print("Loaded {} images under {}".format(dataset_sizes[x], x))
    # 计算数据集的类
    print("Classes: ")
    class_names = image_datasets[TRAIN].classes
    print(image_datasets[TRAIN].classes)

    # 加载模型
    vgg16 = models.vgg16_bn()
    vgg16.load_state_dict(torch.load("../vgg16_bn.pth"))

    for param in vgg16.features.parameters():
        param.require_grad = False

    # 打印一共要分几个类
    num_features = vgg16.classifier[6].in_features
    print('num_features', num_features)

    # 配置最后一层网络
    features = list(vgg16.classifier.children())[:-1]  # 移除网络最后一层
    features.extend([nn.Linear(num_features, len(class_names))])  # 加上7个输出的全连接层
    vgg16.classifier = nn.Sequential(*features)  # 替代之前的classifier
    print(vgg16)

    if use_gpu:
        vgg16.cuda()
    # 配置损失函数
    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    # 使用SGD随机梯度下降法
    optimizer_ft = optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)
    # 学习率衰减
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.7)
    # 训练模型
    vgg16 = train_model(vgg16, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10)
    # 评估模型
    eval_model(vgg16, criterion)
    # 储存模型
    torch.save(vgg16.state_dict(), '2VGG16_v2-OCT_Retina_half_dataset.pth')