import time

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib import ticker
from torch.utils.data import DataLoader, Dataset
from network import Classifier
from tqdm import tqdm


class DealDataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self, features_file, labels_file):
        # xy = np.loadtxt(file, delimiter=',', dtype=np.float32)[1:]  # 使用numpy读取数据
        # self.x_data = xy[:, 0:-7]
        # self.x_data = np.array([np.expand_dims(data, 0) for data in self.x_data])
        # self.x_data = torch.from_numpy(self.x_data)
        # self.y_data = torch.from_numpy(xy[:, -1])
        # self.y_data = self.y_data.long()
        # self.len = xy.shape[0]
        self.x_data = np.loadtxt(features_file, delimiter=',', dtype=np.float32)[1:, 45:75]
        self.x_data = np.array([np.expand_dims(data, 0) for data in self.x_data])
        self.y_data = np.loadtxt(labels_file, delimiter=',', dtype=np.float32)[1:]  # The first column is index
        # To tensor
        self.x_data = torch.from_numpy(self.x_data)
        self.y_data = torch.from_numpy(self.y_data)
        self.y_data = self.y_data.long()
        self.len = self.x_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index] - 1  # 不减1就outofboundary

    def __len__(self):
        return self.len


def show_train_hist(hist, show=False, save=False, path='Train_hist.png'):
    x = range(1, 101)
    y1 = hist['loss']
    y2 = hist['train_acc']
    y3 = hist['validation_acc']

    plt.subplot(3, 1, 1)
    plt.plot(x, y1, 'o-')
    plt.title('Loss and Accuracy')
    plt.ylabel('train loss')
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(10))
    plt.subplot(3, 1, 2)
    plt.plot(x, y2, '.-')
    plt.xlabel('epochs')
    plt.ylabel('train accuracy')
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(10))
    plt.subplot(3, 1, 3)
    plt.plot(x, y3, '.-')
    plt.xlabel('epochs')
    plt.ylabel('validation accuracy')
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(10))

    if save:
        plt.savefig(path)
    if show:
        plt.show()
    plt.close()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')

    network = Classifier(30, 100, 8).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)
    # optimizer = torch.optim.Adam(network.parameters(), lr=0.00001) # weight_decay： L2 penalty
    # 实例化这个类，然后我们就得到了Dataset类型的数据，记下来就将这个类传给DataLoader，就可以了。
    train_dataset = DealDataset('train-features/Bag/train_features.csv', 'train-features/Bag/train_labels.csv')
    validation_dataset = DealDataset('validation-features/Bag/validation_features.csv', 'validation-features/Bag/validation_labels.csv')

    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=128, shuffle=True)

    # Train the model
    train_hist = {}
    train_hist['loss'] = []
    train_hist['train_acc'] = []
    train_hist['validation_acc'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []
    print(time.localtime())
    for epoch in range(100):
        avg_loss = 0
        cnt = 0
        Loss = []
        correct = 0

        # Training step
        network.train()
        epoch_start_time = time.time()
        for (inputs, labels) in tqdm(train_loader):
            # 将数据从 train_loader 中读出来,一次读取的样本数是64个
            # for i, data in enumerate(train_loader):
            # print(data)
            # print(i)
            # inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            # print(labels)
            outputs = network(inputs)
            # print(outputs[0])
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            Loss.append(loss.item())

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        epoch_loss = np.mean(Loss)

        # Record the loss for every epoch
        train_hist['loss'].append(epoch_loss)
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

        # Evaluate model
        network.eval()
        with torch.no_grad():
            # Calculate the number of correct predictions
            # Training set
            train_correct = (sum(network(inputs.to(device)).argmax(1) == labels.to(device).data).item() for
                             inputs, labels
                             in train_loader)
            train_correct_num = sum(train_correct)
            train_epoch_acc = train_correct_num / len(train_dataset)

            # Validation set
            validation_correct = (sum(network(inputs.to(device)).argmax(1) == labels.to(device).data).item() for
                                  inputs, labels
                                  in validation_loader)
            validation_correct_num = sum(validation_correct)
            validation_epoch_acc = validation_correct_num / len(validation_dataset)

        train_hist['train_acc'].append(train_epoch_acc)
        train_hist['validation_acc'].append(validation_epoch_acc)
        print("[Epoch: %d] loss: %f, train accuracy: %f, validation accuracy: %f.\n" % (
            epoch, epoch_loss, train_epoch_acc, validation_epoch_acc))

    print("Avg per epoch ptime: %.2f." % (np.mean(train_hist['per_epoch_ptimes'])))
    show_train_hist(train_hist, show=True, path='train_hist_gyr_lacc_mag_Bag_100epoches.png')

    torch.save(network, 'models/train_hist_gyr_lacc_mag_Bag_100epoches.pkl')
