import torch
from torch.utils.data import DataLoader

from data_source import DealDataset
from train_network import SHLDataset


def load_model(path):
    if path != None:
        model = torch.load(path)
    return model


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    test_dataset = DealDataset('test-features/test_features.csv', 'test-features/test_labels.csv')
    test_dataset = SHLDataset(dataset_type='test')
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=True)

    network = load_model('models/model_conv_gyr_lacc_50epoches_Torso.pkl')  # 64%
    # network = load_model('model_conv_gyr_lacc.pkl') # 60.73

    with torch.no_grad():
        test_correct = (sum(network(inputs.to(device)).argmax(1) == labels.to(device).data).item() for
                        inputs, labels
                        in test_loader)
        test_correct_num = sum(test_correct)
        test_acc = test_correct_num / len(test_loader)

        print("test accuracy: %f.\n" % (test_acc))
