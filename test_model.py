import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from data_source import DealDataset
from train_network import SHLDataset


def plot_confusion_matrix(y, y_p):
    matrix_labels = list(set(y))
    matrix = confusion_matrix(y_true=y, y_pred=y_p, labels=matrix_labels)

    print(matrix)
    sns.set()

    f, ax = plt.subplots()
    sns.heatmap(matrix, annot=True, ax=ax, fmt='g')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    labels = ["Still", "Walk", "Run", "Bike", "Car", "Bus", "Train", "Subway"]
    ax.set_xticklabels(labels, ha='center')
    ax.set_yticklabels(labels, va='center')
    plt.show()

def load_model(path):
    if path != None:
        model = torch.load(path)
    return model


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    # test_dataset = DealDataset('test-features/test_features.csv', 'test-features/test_labels.csv')
    test_dataset = SHLDataset(dataset_type='test')
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=True)

    network = load_model('models/model_conv_acc_gyr_lacc_mag_50epoches.pkl')
    # network = load_model('model_conv_gyr_lacc.pkl')

    start_time = time.time()
    with torch.no_grad():
        test_correct = (sum(network(inputs.to(device)).argmax(1) == labels.to(device).data).item() for
                        inputs, labels
                        in test_loader)
        test_correct_num = sum(test_correct)
        test_acc = test_correct_num / len(test_loader)

        end_time = time.time()
        print("test accuracy: %f.\n" % (test_acc))
        print("time: %f s" % (end_time - start_time))

    y = np.array([])
    y_p = np.array([])
    for inputs, labels in test_loader:
        # print(labels.data.cpu().detach().numpy().tolist())
        y = np.append(y, labels.data.cpu().detach().numpy())
        y_p = np.append(y_p, network(inputs.to(device)).argmax(1).cpu().detach().numpy())
    plot_confusion_matrix(y, y_p)
