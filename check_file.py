import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


# Check Nan data in hips gra
def getMeanCrossingRate(array):
    array = array - np.nanmean(array)
    return (((array[:-1] * array[1:]) < 0).sum()) / (len(array) - 1)


if __name__ == '__main__':
    file = 'challenge-2019-train_hips/train/Hips/Gra_y.txt'

    skew_features = []
    with open(file, "r") as f:
        lines = f.readlines()

        for line in lines:
            data = line.split(' ')
            data = [float(d) for d in data]
            skew_features.append(data)

    # for data in skew_features[121210:121220]:
    #     print(data)
    data = skew_features[121217]
    data = [float(d) for d in data]
    print(data)
    print(np.nanmean(data))
    print(np.nanstd(data))
    print(getMeanCrossingRate(data))
    print(kurtosis(data, nan_policy='omit'))
    print(skew(data, nan_policy='omit'))
