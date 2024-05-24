import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import torchvision.models as models
# This is for the progress bar.
from tqdm import tqdm
import seaborn as sns

# 看看label长啥样啊

labels_dataframe = pd.read_csv('../classify-leaves/train.csv')

'''
print(labels_dataframe.head(5))

print(labels_dataframe.describe())
'''


def barw(ax):
    for p in ax.patches:
        val = p.get_width()
        x = p.get_x() + p.get_width()
        y = p.get_y() + p.get_height()
        ax.annotate(round(val, 2), (x, y))


'''
plt.figure(figsize=(15, 30))
ax0 = sns.countplot(y=labels_dataframe['label'], order=labels_dataframe['label'].value_counts().index)
barw(ax0)
plt.show()
'''

# 将label文件排序
leaves_labels = sorted(list(set(labels_dataframe['label'])))
n_classes = len(leaves_labels)  # 类别总数
print(n_classes)

# 将label转成对应数字
class_to_num = dict(zip(leaves_labels, range(n_classes)))
# print(class_to_num)

# 再转换为数字对label，方便最后预测使用
num_to_class = {label: idx for idx, label in class_to_num.items()}


# print(num_to_class)

# 整个dataset出来
class LeavesData(Dataset):
    def __init__(self, csv_path, file_path, mode='train',
                 valid_ratio=0.2, resize_height=256, resize_width=256):
        """
        初始化函数
        :param csv_path: str, CSV文件的路径，其中包含了数据的元信息
        :param file_path: str, 数据文件的路径，可以是单个文件或文件夹
        :param mode: str, 模式，默认为'train'，可选值包括'train'和'test'等，用于指定数据的使用模式
        :param valid_ratio: float, 验证集与训练集的比例，默认为0.2
        :param resize_height: int, 图片高度的缩放大小，默认为256
        :param resize_width: int, 图片宽度的缩放大小，默认为256
        """

        # 需要调整后的照片尺寸， 这里每张图片的大小尺寸不一致
        self.resize_height = resize_height
        self.resize_width = resize_width

        self.file_path = file_path
        self.mode = mode

        # 读取csv文件
        self.data_info = pd.read_csv(csv_path, header=None)  # header=None是去掉表头部分

        # 计算length
        self.data_len = len(self.data_info.index) - 1
        self.train_len = int(self.data_len * (1 - valid_ratio))

        if mode == 'train':
            # 第一列包含图像文件的名称
            self.train_image = np.asarray(self.data_info.iloc[1: self.train_len, 0])  # 将图像名称转换为numpy数组
            # 第二列是图像的label
            self.train_label = np.asarray(self.data_info.iloc[1: self.train_len, 1])
            self.image_arr = self.train_image  # 训练集图像文件名
            self.label_arr = self.train_label  # 训练集标签
        elif mode == 'valid':
            self.valid_image = np.asarray(self.data_info.iloc[self.train_len:, 0])
            self.valid_label = np.asarray(self.data_info.iloc[self.train_len:, 1])
            self.image_arr = self.valid_image
            self.label_arr = self.valid_label
        elif mode == 'test':
            self.test_image = np.asarray(self.data_info.iloc[1:, 0])
            self.image_arr = self.test_image

        self.real_len = len(self.image_arr)  # 实际数据长度

        print('Finished reading the {} set of Leaves Dataset ({} samples found)'.format(mode, self.real_len))

    def __getitem__(self, index):
        # 从 image_arr 中得到索引对应文件名
        single_image_name = self.image_arr[index]

        # 读取图像文件
        img_as_img = Image.open(self.file_path + single_image_name)

        # 设置好需要转换的变量，还可以包括一系列的normalize操作
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
                transforms.ToTensor()
            ])
        else:
            # valid和test不做数据增强
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

        img_as_img = transform(img_as_img)

        if self.mode == 'test':
            return img_as_img
        else:
            label = self.label_arr[index]
            # number label
            number_label = class_to_num[label]
            return img_as_img, number_label

    def __len__(self):
        return self.real_len


train_path = '../classify-leaves/train.csv'
test_path = '../classify-leaves/test.csv'
# csv文件中已经有数据路径了，因此只到上一级目录
img_path = '../classify-leaves/'

train_dataset = LeavesData(train_path, img_path, mode='train')
val_dataset = LeavesData(train_path, img_path, mode='valid')
test_dataset = LeavesData(test_path, img_path, mode='test')

"""
print(train_dataset)
print(val_dataset)
print(test_dataset)
"""

# 定义data_loader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=8,
    shuffle=False
)

val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=8,
    shuffle=False
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=8,
    shuffle=False
)


# 展示数据
def im_convert(tensor):
    """
    将tensor转换为图片
    :param tensor:
    :return:
    """
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2 ,0)
    image = image.clip(0, 1)
    return image


fig = plt.figure(figsize=(20, 12))
columns = 4
rows = 2

dataiter = iter(val_loader)
inputs, classes = next(dataiter)  # 获取第一个批次的数据

for idx in range(columns * rows):
    ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
    ax.set_title(num_to_class[int(classes[idx])])
    plt.imshow(im_convert(inputs[idx]))
plt.show()


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return torch.device(device)


device = get_device()


# 是否要冻住模型的前面一些层
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# resnet34模型
def res_model(num_classes, feature_extract=False, use_pretrained=True):
    model_ft = models.resnet50(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features # 获取最后一层节点个数
    model_ft.fc = nn.Sequential(nn.Dropout(0.6), nn.Linear(num_ftrs, num_classes))
    return model_ft


# 超参数
learning_rate = 3e-6
weight_decay = 1e-3
num_epoch = 30
model_path = './Leaves_model.pth'

model = res_model(176)
model = model.to(device)
model.device = device
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
n_epochs = num_epoch
best_acc = 0.0
for epoch in range(n_epochs):
    # 训练模式
    model.train()
    train_loss = []
    train_accs = []
    for batch in tqdm(train_loader):
        imgs, labels = batch
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        train_loss.append(loss.item())
        train_accs.append(acc)

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    # 验证模式
    model.eval()
    val_loss = []
    val_accs = []
    for batch in tqdm(val_loader):
        imgs, labels = batch
        imgs = imgs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            logits = model(imgs)
            loss = criterion(logits, labels)
            acc = (logits.argmax(dim=-1) == labels).float().mean()
            val_loss.append(loss.item())
            val_accs.append(acc)

    val_loss = sum(val_loss) / len(val_loss)
    val_acc = sum(val_accs) / len(val_accs)
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {val_loss:.5f}, acc = {val_acc:.5f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), model_path)
        print('saving model with acc {:.3f}'.format(best_acc))


saveFileName = './submission.csv'
# 预测
model = res_model(176)
model = model.to(device)
model.load_state_dict(torch.load(model_path))
model.eval()


predictions = []
for batch in tqdm(test_loader):
    imgs = batch
    with torch.no_grad():
        logits = model(imgs.to(device))

    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

preds = []
for i in predictions:
    preds.append(num_to_class[i])

test_data = pd.read_csv(test_path)
test_data['label'] = pd.Series(preds)
submission = pd.concat([test_data['image'], test_data['label']], axis=1)
submission.to_csv(saveFileName, index=False)
print('Done!')