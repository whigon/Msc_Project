import os

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


# path = 'challenge-2019-train_bag/train/Bag'
# path = 'challenge-2020-validation/validation/Bag'


class Feature():
    def __init__(self, path='challenge-2019-train_bag/train/Bag'):
        self.path = path
        # self.path = 'challenge-2020-validation/validation/Bag'

    def getZeroCrossingRate(self, array):
        array = np.array(array)
        return (((array[:-1] * array[1:]) < 0).sum()) / (len(array) - 1)

    def getMeanCrossingRate(self, array):
        array = array - np.nanmean(array)
        return (((array[:-1] * array[1:]) < 0).sum()) / (len(array) - 1)

    def load_labels(self):
        label_path = os.path.join(self.path, 'Label.txt')

        labels = []
        with open(label_path, "r") as f:
            lines = f.readlines()

            for line in lines:
                data = line.split(' ')
                labels.append(data[0])

        return labels

    def get_mean(self, file_name):
        file = os.path.join(self.path, file_name)

        mean = []
        with open(file, "r") as f:
            lines = f.readlines()

            for line in lines:
                data = line.split(' ')
                data = [float(d) for d in data]
                mean.append(np.nanmean(data))

        return mean

    def get_std(self, file_name):
        file = os.path.join(self.path, file_name)

        std = []
        with open(file, "r") as f:
            lines = f.readlines()

            for line in lines:
                data = line.split(' ')
                data = [float(d) for d in data]
                std.append(np.nanstd(data))

        return std

    def get_zcr(self, file_name):
        file = os.path.join(self.path, file_name)

        zcr = []
        with open(file, "r") as f:
            lines = f.readlines()

            for line in lines:
                data = line.split(' ')
                data = [float(d) for d in data]
                zcr.append(self.getZeroCrossingRate(data))

        return zcr

    def get_mcr(self, file_name):
        file = os.path.join(self.path, file_name)

        mcr = []
        with open(file, "r") as f:
            lines = f.readlines()

            for line in lines:
                data = line.split(' ')
                data = [float(d) for d in data]
                mcr.append(self.getMeanCrossingRate(data))

        return mcr

    def get_kurtosis(self, file_name):
        file = os.path.join(self.path, file_name)

        kurt = []
        with open(file, "r") as f:
            lines = f.readlines()

            for line in lines:
                data = line.split(' ')
                data = [float(d) for d in data]
                kurt.append(kurtosis(data, nan_policy='omit'))

        return kurt

    def get_skew(self, file_name):
        file = os.path.join(self.path, file_name)

        skew_features = []
        with open(file, "r") as f:
            lines = f.readlines()

            for line in lines:
                data = line.split(' ')
                data = [float(d) for d in data]
                skew_features.append(skew(data, nan_policy='omit'))

        return skew_features

    def __extract_features__(self, file_x, file_y, file_z, file_w=None):
        mean_x = self.get_mean(file_x)
        std_x = self.get_std(file_x)
        mcr_x = self.get_mcr(file_x)
        kurtosis_x = self.get_kurtosis(file_x)
        skew_x = self.get_skew(file_x)

        mean_y = self.get_mean(file_y)
        std_y = self.get_std(file_y)
        mcr_y = self.get_mcr(file_y)
        kurtosis_y = self.get_kurtosis(file_y)
        skew_y = self.get_skew(file_y)

        mean_z = self.get_mean(file_z)
        std_z = self.get_std(file_z)
        mcr_z = self.get_mcr(file_z)
        kurtosis_z = self.get_kurtosis(file_z)
        skew_z = self.get_skew(file_z)

        if file_w is None:
            features = [list(item) for item in
                        zip(mean_x, std_x, mcr_x, kurtosis_x, skew_x,
                            mean_y, std_y, mcr_y, kurtosis_y, skew_y,
                            mean_z, std_z, mcr_z, kurtosis_z, skew_z)]
        else:
            mean_w = self.get_mean(file_w)
            std_w = self.get_std(file_w)
            mcr_w = self.get_mcr(file_w)
            kurtosis_w = self.get_kurtosis(file_w)
            skew_w = self.get_skew(file_w)

            features = [list(item) for item in
                        zip(mean_x, std_x, mcr_x, kurtosis_x, skew_x,
                            mean_y, std_y, mcr_y, kurtosis_y, skew_y,
                            mean_z, std_z, mcr_z, kurtosis_z, skew_z,
                            mean_w, std_w, mcr_w, kurtosis_w, skew_w)]

        return features

    def create_feature_files(self, prefix):
        # 3 features
        acc_features = self.__extract_features__('Acc_x.txt', 'Acc_y.txt', 'Acc_z.txt')
        acc_features_df = pd.DataFrame(data=acc_features)
        acc_features_df.to_csv('{}_features_acc.csv'.format(prefix), encoding='utf-8', index=False)

        # 3 features
        gra_features = self.__extract_features__('Gra_x.txt', 'Gra_y.txt', 'Gra_z.txt')
        gra_features_df = pd.DataFrame(data=gra_features)
        gra_features_df.to_csv('{}_features_gra.csv'.format(prefix), encoding='utf-8', index=False)

        # 3 features
        gyr_features = self.__extract_features__('Gyr_x.txt', 'Gyr_y.txt', 'Gyr_z.txt')
        gyr_features_df = pd.DataFrame(data=gyr_features)
        gyr_features_df.to_csv('{}_features_gyr.csv'.format(prefix), encoding='utf-8', index=False)

        # 3 features
        lacc_features = self.__extract_features__('LAcc_x.txt', 'LAcc_y.txt', 'LAcc_z.txt')
        lacc_features_df = pd.DataFrame(data=lacc_features)
        lacc_features_df.to_csv('{}_features_lacc.csv'.format(prefix), encoding='utf-8', index=False)

        # 3 features
        mag_features = self.__extract_features__('Mag_x.txt', 'Mag_y.txt', 'Mag_z.txt')
        mag_features_df = pd.DataFrame(data=mag_features)
        mag_features_df.to_csv('{}_features_mag.csv'.format(prefix), encoding='utf-8', index=False)

        # 4 features
        ori_features = self.__extract_features__('Ori_x.txt', 'Ori_y.txt', 'Ori_z.txt', 'Ori_w.txt')
        ori_features_df = pd.DataFrame(data=ori_features)
        ori_features_df.to_csv('{}_features_ori.csv'.format(prefix), encoding='utf-8', index=False)

        total_features = [item[0] + item[1] + item[2] + item[3] + item[4] + item[5] for item in
                          zip(acc_features, gra_features, gyr_features, lacc_features, mag_features, ori_features)]
        print(total_features[0])
        total_features_df = pd.DataFrame(data=total_features)

        total_features_df.to_csv('{}_features.csv'.format(prefix), encoding='utf-8', index=False)

    # def load_features(self):
    #     mean_acc_x = self.get_mean('Acc_x.txt')
    #     std_acc_x = self.get_std('Acc_x.txt')
    #     mcr_acc_x = self.get_mcr('Acc_x.txt')
    #     kurtosis_acc_x = self.get_kurtosis('Acc_x.txt')
    #     skew_acc_x = self.get_skew('Acc_x.txt')
    #
    #     mean_acc_y = self.get_mean('Acc_y.txt')
    #     std_acc_y = self.get_std('Acc_y.txt')
    #     mcr_acc_y = self.get_mcr('Acc_y.txt')
    #     kurtosis_acc_y = self.get_kurtosis('Acc_y.txt')
    #     skew_acc_y = self.get_skew('Acc_y.txt')
    #
    #     mean_acc_z = self.get_mean('Acc_z.txt')
    #     std_acc_z = self.get_std('Acc_z.txt')
    #     mcr_acc_z = self.get_mcr('Acc_z.txt')
    #     kurtosis_acc_z = self.get_kurtosis('Acc_z.txt')
    #     skew_acc_z = self.get_skew('Acc_z.txt')
    #
    #     acc_features = [list(item) for item in
    #                     zip(mean_acc_x, std_acc_x, mcr_acc_x, kurtosis_acc_x, skew_acc_x,
    #                         mean_acc_y, std_acc_y, mcr_acc_y, kurtosis_acc_y, skew_acc_y,
    #                         mean_acc_z, std_acc_z, mcr_acc_z, kurtosis_acc_z, skew_acc_z)]
    #
    #     mean_gra_x = self.get_mean('Gra_x.txt')
    #     std_acc_x = self.get_std('Acc_x.txt')
    #     mcr_acc_x = self.get_mcr('Acc_x.txt')
    #     kurtosis_acc_x = self.get_kurtosis('Acc_x.txt')
    #     skew_acc_x = self.get_skew('Acc_x.txt')
    #
    #     mean_acc_y = self.get_mean('Acc_y.txt')
    #     std_acc_y = self.get_std('Acc_y.txt')
    #     mcr_acc_y = self.get_mcr('Acc_y.txt')
    #     kurtosis_acc_y = self.get_kurtosis('Acc_y.txt')
    #     skew_acc_y = self.get_skew('Acc_y.txt')
    #
    #     mean_acc_z = self.get_mean('Acc_z.txt')
    #     std_acc_z = self.get_std('Acc_z.txt')
    #     mcr_acc_z = self.get_mcr('Acc_z.txt')
    #     kurtosis_acc_z = self.get_kurtosis('Acc_z.txt')
    #     skew_acc_z = self.get_skew('Acc_z.txt')
    #
    #     mean_gyr_x = self.get_mean('Gyr_x.txt')
    #     std_gyr_x = self.get_std('Gyr_x.txt')
    #     mean_gyr_y = self.get_mean('Gyr_y.txt')
    #     std_gyr_y = self.get_std('Gyr_y.txt')
    #     mean_gyr_z = self.get_mean('Gyr_z.txt')
    #     std_gyr_z = self.get_std('Gyr_z.txt')
    #
    #     labels = self.load_labels()
    #     features = [list(item) for item in
    #                 zip(mean_x, std_x, mcr_x, kurtosis_x, skew_x, mean_y, std_y, mcr_y, kurtosis_y, skew_y, mean_z,
    #                     std_z,
    #                     mcr_z, kurtosis_z, skew_z, mean_gyr_x, std_gyr_x, mean_gyr_y, std_gyr_y, mean_gyr_z, std_gyr_z,
    #                     labels)]
    #
    #     return features


if __name__ == '__main__':
    # Train feature extraction
    train_path = 'train-features/'
    if not os.path.exists(train_path):
        os.mkdir(train_path)

    # # Train Bag
    # train_bag_path = os.path.join(train_path, 'Bag/')
    # if not os.path.exists(train_bag_path):
    #     os.mkdir(train_bag_path)
    # f1 = Feature('challenge-2019-train_bag/train/Bag')
    # f1.create_feature_files('{}train'.format(train_bag_path))
    #
    # train_labels = np.array(f1.load_labels())
    # train_labels_df = pd.DataFrame(data=train_labels)
    # train_labels_df.to_csv('{}train_labels.csv'.format(train_bag_path), encoding='utf-8', index=False)
    #
    # # Train Hand
    # train_hand_path = os.path.join(train_path, 'Hand/')
    # if not os.path.exists(train_hand_path):
    #     os.mkdir(train_hand_path)
    # f1 = Feature('challenge-2020-train_hand/train/Hand')
    # f1.create_feature_files('{}train'.format(train_hand_path))
    #
    # train_labels = np.array(f1.load_labels())
    # train_labels_df = pd.DataFrame(data=train_labels)
    # train_labels_df.to_csv('{}train_labels.csv'.format(train_hand_path), encoding='utf-8', index=False)

    # Train Hips
    # The 121217th data in gra has nan
    train_hips_path = os.path.join(train_path, 'Hips/')
    if not os.path.exists(train_hips_path):
        os.mkdir(train_hips_path)
    f1 = Feature('challenge-2019-train_hips/train/Hips')
    f1.create_feature_files('{}train'.format(train_hips_path))

    train_labels = np.array(f1.load_labels())
    train_labels_df = pd.DataFrame(data=train_labels)
    train_labels_df.to_csv('{}train_labels.csv'.format(train_hips_path), encoding='utf-8', index=False)
    #
    # # Train Torso
    # train_torso_path = os.path.join(train_path, 'Torso/')
    # if not os.path.exists(train_torso_path):
    #     os.mkdir(train_torso_path)
    # f1 = Feature('challenge-2019-train_torso/train/Torso')
    # f1.create_feature_files('{}train'.format(train_torso_path))
    #
    # train_labels = np.array(f1.load_labels())
    # train_labels_df = pd.DataFrame(data=train_labels)
    # train_labels_df.to_csv('{}train_labels.csv'.format(train_torso_path), encoding='utf-8', index=False)
    #
    # # Validation feature extraction
    # validation_path = 'validation-features/'
    # if not os.path.exists(validation_path):
    #     os.mkdir(validation_path)
    #
    # # Validation Bag
    # validation_bag_path = os.path.join(validation_path, 'Bag/')
    # if not os.path.exists(validation_bag_path):
    #     os.mkdir(validation_bag_path)
    # f2 = Feature('challenge-2020-validation/validation/Bag')
    # f2.create_feature_files('{}validation'.format(validation_bag_path))
    #
    # validation_labels = np.array(f2.load_labels())
    # validation_labels_df = pd.DataFrame(data=validation_labels)
    # validation_labels_df.to_csv('{}validation_labels.csv'.format(validation_bag_path), encoding='utf-8', index=False)
    #
    # # Validation Hand
    # validation_hand_path = os.path.join(validation_path, 'Hand/')
    # if not os.path.exists(validation_hand_path):
    #     os.mkdir(validation_hand_path)
    # f2 = Feature('challenge-2020-validation/validation/Hand')
    # f2.create_feature_files('{}validation'.format(validation_hand_path))
    #
    # validation_labels = np.array(f2.load_labels())
    # validation_labels_df = pd.DataFrame(data=validation_labels)
    # validation_labels_df.to_csv('{}validation_labels.csv'.format(validation_hand_path), encoding='utf-8', index=False)
    #
    # # Validation Hips
    # validation_hips_path = os.path.join(validation_path, 'Hips/')
    # if not os.path.exists(validation_hips_path):
    #     os.mkdir(validation_hips_path)
    # f2 = Feature('challenge-2020-validation/validation/Hips')
    # f2.create_feature_files('{}validation'.format(validation_hips_path))
    #
    # validation_labels = np.array(f2.load_labels())
    # validation_labels_df = pd.DataFrame(data=validation_labels)
    # validation_labels_df.to_csv('{}validation_labels.csv'.format(validation_hips_path), encoding='utf-8', index=False)
    #
    # # Validation Torso
    # validation_torso_path = os.path.join(validation_path, 'Torso/')
    # if not os.path.exists(validation_torso_path):
    #     os.mkdir(validation_torso_path)
    # f2 = Feature('challenge-2020-validation/validation/Torso')
    # f2.create_feature_files('{}validation'.format(validation_torso_path))
    #
    # validation_labels = np.array(f2.load_labels())
    # validation_labels_df = pd.DataFrame(data=validation_labels)
    # validation_labels_df.to_csv('{}validation_labels.csv'.format(validation_torso_path), encoding='utf-8', index=False)
    #
    # # Test
    # test_path = 'test-features/'
    # if not os.path.exists(test_path):
    #     os.mkdir(test_path)
    #
    # f3 = Feature('challenge-2020-test-15062020')
    # f3.create_feature_files('{}test'.format(test_path))
    #
    # # Test labels
    # test_labels = []
    # test_label_path = os.path.join('challenge-2020-test_label', 'Label.txt')
    # with open(test_label_path, "r") as f:
    #     lines = f.readlines()
    #
    #     for line in lines:
    #         data = line.split(' ')
    #         test_labels.append(data[0])
    # test_labels = np.array(test_labels)
    # test_labels_df = pd.DataFrame(data=test_labels)
    # test_labels_df.to_csv('test-features/test_labels.csv', encoding='utf-8', index=False)
