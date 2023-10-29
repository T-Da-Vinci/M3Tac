from numpy.lib import split, tracemalloc_domain
from numpy.lib.function_base import bartlett
from torchvision import models, transforms
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
from torch.utils.data import RandomSampler
from os import listdir
import glob
import numpy as np
import argparse
from PIL import Image
import copy, time
from torchvision.transforms.transforms import RandomHorizontalFlip, RandomRotation, ToPILImage
from tqdm import tqdm
from matplotlib import pyplot
import random
from googlenet import GoogLeNet

def parse_args():
    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument('--data-path', type=str, help='data path')
    parser.add_argument('--data-split', type=float, default=0.8, help='split data')
    parser.add_argument('--batch-size', type=int, default=4, help='batch_size')
    parser.add_argument('--random-seed', type=int, default=123, help='random seed')
    parser.add_argument('--num-workers', type=int, default=0, help='Dataset workers')
    parser.add_argument('--epoch', type=int, default=50, help='training epoch')
    parser.add_argument('--use-rgb', type=int, default=1, help='Use RGB image for training (1/0)')
    parser.add_argument('--use-tactile', type=int, default=1, help='Use tactile image for training (1/0)')

    args = parser.parse_args()
    return args


class GripData(Dataset):
    def __init__(self, is_train=None):
        if is_train:
            self.imgs = glob.glob("./train/texture/*/*.jpg")
            self.labels = self._get_labels(r'./train/texture')
        else:
            self.imgs = glob.glob("./test/texture/*/*.jpg")
            self.labels = self._get_labels(r'./test/texture')
        print(self.labels)
        self.transform = transforms.Compose([
            transforms.Resize([128, 128]),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def _get_labels(self, root):
        keys = listdir(root)
        labels = {}
        '''
        one_hot_map = np.eye(len(keys))
        
        for i, key in enumerate(keys):
            labels[key] = one_hot_map[i]
        '''
        for i, key in enumerate(keys):
            labels[key] = i

        return labels

    def __getitem__(self, index):
        img = self.imgs[index]
        if args.use_tactile:
            path = os.path.split(img)[0]
            if 'train' in img:
                tactile = os.path.join(path.replace('temp', 'texture'), r'{}.jpg'.format(
                    random.randint(1, len(glob.glob(os.path.join(path.replace('temp', 'texture'), r'*'))))))
            elif 'test' in img:
                # tactile = os.path.join(img.replace('visual', 'tactile'))
                tactile = os.path.join(path.replace('temp', 'texture'), r'{}.jpg'.format(
                    random.randint(1, len(glob.glob(os.path.join(path.replace('temp', 'texture'), r'*'))))))
        label = self.labels[os.path.split(img)[0][-1]]
        img = Image.open(img)
        tactile = Image.open(tactile)
        img = self.transform(img)
        tactile = self.transform(tactile)
        fusion = torch.cat((img, tactile), 0)
        return fusion, label

    def __len__(self):
        return len(self.imgs)





class GripperVision(nn.Module):
    def __init__(self, backbone, nclass, use_rgb, use_tactile):
        super().__init__()
        # backbone.conv1.conv = nn.Conv2d(3 * use_rgb + 3 * use_tactile, 64, kernel_size=(7, 7), stride=(2, 2),
        #                                 padding=(3, 3), bias=False)
        self.pre_conv = nn.Conv2d(6, 3, kernel_size=(3, 3))
        self.backbone = backbone
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Linear(100, nclass),
        )

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.backbone(x)
        x = self.mlp(x)
        return x


def split_data(dataset, args):
    indices = list(range(len(dataset)))
    split = int(len(dataset) * args.data_split)
    np.random.seed(args.random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    train_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=train_sampler
    )
    val_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        sampler=val_sampler
    )

    return train_data, val_data


def train_model(net, dataloader_dict, criterion, optimizer, num_epoch):
    since = time.time()
    best_model_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0

    device = "cuda:0"
    net = net.to(device)

    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch + 1, num_epoch))
        print('-' * 20)

        for phase in ['train', 'val']:

            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            for labels, inputs in tqdm(dataloader_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device).long()
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

                epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
                epoch_acc = epoch_corrects.double() / len(dataloader_dict[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(net.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    net.load_state_dict(best_model_wts)
    return net


def trainer(net, loss, optimizer, dataset, args):
    num_epoch = args.epoch
    train_data_length = dataset["train"].__len__() * args.batch_size
    val_data_length = dataset["val"].__len__()

    best_model_wts = copy.deepcopy(net.state_dict())
    best_train_acc = 0.0
    best_val_acc = 0.0
    best_acc = 0.0
    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0
        net.train()  # 確保 model 是在 train model (開啟 Dropout 等...)
        for i, data in enumerate(dataset["train"]):
            optimizer.zero_grad()  # 用 optimizer 將 model 參數的 gradient 歸零
            train_pred = net(data[0].cuda())  # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
            batch_loss = loss(train_pred, data[1].cuda())  # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
            batch_loss.backward()  # 利用 back propagation 算出每個參數的 gradient
            optimizer.step()  # 以 optimizer 用 gradient 更新參數值

            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            train_loss += batch_loss.item()

        net.eval()
        with torch.no_grad():
            for i, data in enumerate(dataset["val"]):
                val_pred = net(data[0].cuda())
                batch_loss = loss(val_pred, data[1].cuda())

                val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
                val_loss += batch_loss.item()

            # 將結果 print 出來
            print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
                  (epoch + 1, num_epoch, time.time() - epoch_start_time, \
                   train_acc / train_data_length, train_loss / train_data_length, val_acc / val_data_length,
                   val_loss / val_data_length))

        if 1.5 * val_acc + train_acc > best_acc:
            best_acc = 1.5 * val_acc + train_acc
            best_model_wts = copy.deepcopy(net.state_dict())
            best_train_acc = train_acc
            best_val_acc = val_acc
    best_train_acc = best_train_acc / train_data_length
    best_val_acc = best_val_acc / val_data_length
    torch.save(best_model_wts, "./val_acc{0:.1f}-train_acc{1:.1f}.pth".format(best_val_acc, best_train_acc))


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    resnet18 = models.googlenet(pretrained=True)
    # resnet18 = GoogLeNet()
    net = GripperVision(resnet18, 15, args.use_rgb, args.use_tactile).cuda()
    # net = Classifier().cuda()
    train_grip_data = GripData(is_train=True)
    test_grip_data = GripData(is_train=False)

    train_data = torch.utils.data.DataLoader(
        train_grip_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=RandomSampler(train_grip_data)
    )
    val_data = torch.utils.data.DataLoader(
        test_grip_data,
        batch_size=1,
        num_workers=args.num_workers,
        sampler=RandomSampler(test_grip_data)
    )
    dataloader_dict = {'train': train_data, 'val': val_data}

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    trainer(net, loss, optimizer, dataloader_dict, args)
    '''
    val = list(val_data)
    img = val[0][0].squeeze(0)
    tpl = transforms.ToPILImage()
    print(img.shape)
    img = tpl(img)
    #img = np.array(img)
    print(type(img))
    pyplot.imshow(img)
    pyplot.show()
    '''

    '''
    train_data = iter(train_data)
    a, b = train_data.next()
    print(b)
    print(a.shape)
    out = net(a.cuda())
    print(out)
    print(out.shape)
    acc = np.sum(np.argmax(out.cpu().data.numpy(), axis=1) == b.numpy())
    print(acc)
    print(train_data.__len__())
    '''
