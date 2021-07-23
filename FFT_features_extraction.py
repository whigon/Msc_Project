import math
import os
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.stats import kurtosis, skew, entropy

SAMPLE_RATE = 100
DURATION = 5
N = SAMPLE_RATE * DURATION


class FFT_Feature():
    def __init__(self, path='challenge-2019-train_bag/train/Bag'):
        self.path = path

    def get_fft(self, file_name):
        file = os.path.join(self.path, file_name)

        features = []
        with open(file, "r") as f:
            lines = f.readlines()

            for line in lines:
                data = line.split(' ')
                data = [float(d) for d in data]
                data = [d for d in data if np.isnan(d) != True]
                yf = fft(data)
                xf = fftfreq(N, 1 / SAMPLE_RATE)
                energy = self.cal_energy(yf)
                mean = self.cal_mean(xf, yf, energy)
                kurtosis = self.cal_kurtosis(yf)
                skew = self.cal_skew(yf)

                features.append([energy, mean, kurtosis, skew])

        return features

    def cal_energy(self, yf):
        return sum([abs(y) ** 2 for y in yf])

    def cal_mean(self, xf, yf, energy):
        yf = [abs(y) ** 2 for y in yf]
        product = [x * y for x, y in zip(xf, yf)]

        return sum(product) / energy

    def cal_kurtosis(self, yf):
        return kurtosis(abs(yf), nan_policy='omit')

    def cal_skew(self, yf):
        return skew(abs(yf), nan_policy='omit')

    def cal_entropy(self, yf):
        return entropy(yf)

    def __extract_features__(self, file_x, file_y, file_z):
        feature_x = self.get_fft(file_x)
        feature_y = self.get_fft(file_y)
        feature_z = self.get_fft(file_z)
        features = [item[0] + item[1] + item[2] for item in zip(feature_x, feature_y, feature_z)]

        return features
        # x_data = []
        # with open(os.path.join(self.path, file_x), "r") as f:
        #     lines = f.readlines()
        #
        #     for line in lines:
        #         data = line.split(' ')
        #         x_data.append([float(d) for d in data])
        #
        # y_data = []
        # with open(os.path.join(self.path, file_y), "r") as f:
        #     lines = f.readlines()
        #
        #     for line in lines:
        #         data = line.split(' ')
        #         y_data.append([float(d) for d in data])
        #
        # z_data = []
        # with open(os.path.join(self.path, file_z), "r") as f:
        #     lines = f.readlines()
        #
        #     for line in lines:
        #         data = line.split(' ')
        #         z_data.append([float(d) for d in data])
        #
        # norm_data = []
        # for i in range(len(x_data)):
        #     data = []
        #     for j in range(len(x_data[i])):
        #         data.append(math.sqrt(x_data[i][j] ** 2 + z_data[i][j] ** 2 + z_data[i][j] ** 2))
        #
        #     norm_data.append(data)

    def create_feature_files(self, prefix):
        # 3 features
        gyr_features = self.__extract_features__('Gyr_x.txt', 'Gyr_y.txt', 'Gyr_z.txt')
        gyr_features_df = pd.DataFrame(data=gyr_features)
        gyr_features_df.to_csv('{}_fft_gyr.csv'.format(prefix), encoding='utf-8', index=False)

        # 3 features
        lacc_features = self.__extract_features__('LAcc_x.txt', 'LAcc_y.txt', 'LAcc_z.txt')
        lacc_features_df = pd.DataFrame(data=lacc_features)
        lacc_features_df.to_csv('{}_fft_lacc.csv'.format(prefix), encoding='utf-8', index=False)

        # 3 features
        mag_features = self.__extract_features__('Mag_x.txt', 'Mag_y.txt', 'Mag_z.txt')
        mag_features_df = pd.DataFrame(data=mag_features)
        mag_features_df.to_csv('{}_fft_mag.csv'.format(prefix), encoding='utf-8', index=False)

        total_features = [item[0] + item[1] + item[2] for item in zip(gyr_features, lacc_features, mag_features)]
        print(total_features[0])
        total_features_df = pd.DataFrame(data=total_features)

        total_features_df.to_csv('{}_fft_features.csv'.format(prefix), encoding='utf-8', index=False)


if __name__ == '__main__':
    train_path = 'train-features/'
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    # Train Hips
    # The 121217th data in gra has nan
    train_hips_path = os.path.join(train_path, 'Hips/')
    if not os.path.exists(train_hips_path):
        os.mkdir(train_hips_path)
    f1 = FFT_Feature('challenge-2019-train_hips/train/Hips')
    f1.create_feature_files('{}train'.format(train_hips_path))

    # Validation feature extraction
    validation_path = 'validation-features/'
    if not os.path.exists(validation_path):
        os.mkdir(validation_path)

    # Validation Hips
    validation_hips_path = os.path.join(validation_path, 'Hips/')
    if not os.path.exists(validation_hips_path):
        os.mkdir(validation_hips_path)
    f2 = FFT_Feature('challenge-2020-validation/validation/Hips')
    f2.create_feature_files('{}validation'.format(validation_hips_path))

    # Test
    test_path = 'test-features/'
    if not os.path.exists(test_path):
        os.mkdir(test_path)

    f3 = FFT_Feature('challenge-2020-test-15062020')
    f3.create_feature_files('{}test'.format(test_path))
