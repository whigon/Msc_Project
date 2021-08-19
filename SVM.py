import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
import seaborn as sns
from feature_extraction import Feature
from sklearn.metrics import f1_score


def plot_confusion_matrix(y_t, yt_p, y_val, yval_p):
    matrix_labels = list(set(y_t))
    train_confusion_matrix = confusion_matrix(y_true=y_t, y_pred=yt_p, labels=matrix_labels)
    val_confusion_matrix = confusion_matrix(y_true=y_val, y_pred=yval_p, labels=matrix_labels)

    print(train_confusion_matrix)
    sns.set()

    f, ax = plt.subplots()
    sns.heatmap(train_confusion_matrix, annot=True, ax=ax, fmt='g')
    ax.set_title('Training confusion matrix')
    ax.set_xlabel('predict')
    ax.set_ylabel('true')

    f1, ax1 = plt.subplots()
    sns.heatmap(val_confusion_matrix, annot=True, ax=ax1, fmt='g')
    ax1.set_title('Validation confusion matrix')
    ax1.set_xlabel('predict')
    ax1.set_ylabel('true')

    plt.show()


def train_on_all(data):
    X_train = np.array([x[:-1] for x in data])
    y_train = np.array([str(x[-1]) for x in data])

    model = svm.SVC()
    model.fit(X_train, y_train)

    prediction = model.predict(X_train)
    print('Training Accuracy', np.mean(prediction == y_train))
    print(f1_score(y_train, prediction, average='macro'))

    # SVM on Validation dataset
    f = Feature('challenge-2020-validation/validation/Bag')
    validation_labels = np.array(f.load_labels())
    validation_data = np.loadtxt('validation-features/Bag/validation_features_press.csv', delimiter=',', dtype=np.float32)[1:]
    # validation_data = np.loadtxt('validation_features_gyr.csv', delimiter=',', dtype=np.float32)[1:, :10]
    test_p = model.predict(validation_data)
    print('Validation Accuracy', np.mean(test_p == validation_labels))
    print(f1_score(validation_labels, test_p, average='macro'))
    plot_confusion_matrix(y_train, prediction, validation_labels, test_p)


if __name__ == '__main__':
    f = Feature('challenge-2019-train_bag/train/Bag')
    labels = np.array(f.load_labels())
    x = np.loadtxt('train-features/Bag/train_features_press.csv', delimiter=',', dtype=np.float32)[1:]
    # x = np.loadtxt('train_features_gyr.csv', delimiter=',', dtype=np.float32)[1:, :10]

    data = list(np.c_[x, labels])
    train_on_all(data)
    # samples = random.sample(data, k=20000)
    # predict_x = np.array([x[:-1] for x in samples])
    # predict_y = np.array([str(x[-1]) for x in samples])
    #
    # X_train, X_val, y_train, y_val = train_test_split(predict_x, predict_y, test_size=0.2)
    # X_train.shape, X_val.shape, y_train.shape, y_val.shape
    #
    # model = svm.SVC()
    # model.fit(X_train, y_train)
    #
    # yt_p = model.predict(X_train)
    # yv_p = model.predict(X_val)
    #
    # print('Training Accuracy', np.mean(yt_p == y_train))
    # print('Validation Accuracy', np.mean(yv_p == y_val))
    #
    # data_x = np.array([x[:-1] for x in data])
    # data_y = np.array([str(x[-1]) for x in data])
    # prediction = model.predict(data_x)
    #
    # print('Overall Accuracy', np.mean(prediction == data_y))
    #
    # # SVM on Validation dataset
    # f = Feature('challenge-2020-validation/validation/Bag')
    # validation_labels = np.array(f.load_labels())
    # validation_data = np.loadtxt('validation_features_mag.csv', delimiter=',', dtype=np.float32)[1:]
    # # validation_data = np.loadtxt('validation_features_gyr.csv', delimiter=',', dtype=np.float32)[1:, :10]
    # test_p = model.predict(validation_data)
    # print('Validation Accuracy', np.mean(test_p == validation_labels))
    # print(f1_score(validation_labels, test_p, average='macro'))
    # plot_confusion_matrix(data_y, prediction, validation_labels, test_p)
