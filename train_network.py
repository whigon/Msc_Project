import os
import time

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib import ticker
from torch.utils.data import DataLoader, Dataset
from network import Classifier
from tqdm import tqdm


class SHLDataset(Dataset):
    def __init__(self, dataset_type='train', positions=['Bag', 'Hand', 'Hips', 'Torso']):
        path = '{}-features'.format(dataset_type)
        if dataset_type is 'train' or dataset_type is 'validation':
            # positions = ['Bag', 'Hand', 'Hips', 'Torso']

            self.x_data = None
            self.y_data = None
            for position in positions:
                feature_path = os.path.join(path, position, '{}_features.csv'.format(dataset_type))
                label_path = os.path.join(path, position, '{}_labels.csv'.format(dataset_type))

                features = np.loadtxt(feature_path, delimiter=',', dtype=np.float32)[1:, 30:60]
                labels = np.loadtxt(label_path, delimiter=',', dtype=np.float32)[1:]  # The first column is index

                # features = np.array([np.expand_dims(data, 0) for data in features])
                if self.x_data is None:
                    self.x_data = features
                    self.y_data = labels
                else:
                    self.x_data = np.append(self.x_data, features, axis=0)
                    self.y_data = np.append(self.y_data, labels, axis=0)
        elif dataset_type is 'test':
            feature_path = os.path.join(path, 'test_features.csv')
            label_path = os.path.join(path, 'test_labels.csv')

            self.x_data = np.loadtxt(feature_path, delimiter=',', dtype=np.float32)[1:, 30:60]
            # self.x_data = np.array([np.expand_dims(data, 0) for data in self.x_data])
            self.y_data = np.loadtxt(label_path, delimiter=',', dtype=np.float32)[1:]  # The first column is index
        else:
            print("Dataset type should be train, validation or test.")

        self.x_data = np.array([np.expand_dims(data, 0) for data in self.x_data])
        self.x_data = torch.from_numpy(self.x_data)
        self.y_data = torch.from_numpy(self.y_data)
        self.y_data = self.y_data.long()

        self.len = self.x_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index] - 1  # 不减1就outofboundary

    def __len__(self):
        return self.len


def show_train_hist(hist, show=False, save=False, path='Train_hist.png'):
    x = range(1, 51)
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
    train_dataset = SHLDataset(dataset_type='train', positions=['Torso'])
    # train_dataset = SHLDataset(dataset_type='train')
    validation_dataset = SHLDataset(dataset_type='validation', positions=['Torso'])
    # validation_dataset = SHLDataset(dataset_type='validation')

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
    for epoch in range(50):
        avg_loss = 0
        cnt = 0
        Loss = []
        correct = 0

        # Training step
        network.train()
        epoch_start_time = time.time()
        for (inputs, labels) in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = network(inputs)
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
    show_train_hist(train_hist, show=True, path='train_hist_gyr_lacc.png')

    torch.save(network, 'models/model_conv_gyr_lacc_50epoches_Torso.pkl')
