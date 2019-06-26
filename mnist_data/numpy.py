import numpy as np
from os import path

file_path = path.abspath(path.join(path.dirname(__file__), 'mnist.npz'))


def load_data():
    row, col = 28, 28
    f = np.load(file_path)

    train_xs = f['x_train'].astype('float32')
    train_xs = train_xs / 255
    train_xs = train_xs.reshape(train_xs.shape[0], row, col, 1)
    print('shape of train data set: ' + str(train_xs.shape))

    train_ys = f['y_train']
    # train_ys = train_ys.reshape(train_ys.shape[0], 1)
    print('shape of train label set: ' + str(train_ys.shape))

    test_xs = f['x_test'].astype('float32')
    test_xs = test_xs / 255
    test_xs = test_xs.reshape(test_xs.shape[0], row, col, 1)
    print('shape of test data set: ' + str(test_xs.shape))

    test_ys = f['y_test']
    # test_ys = test_ys.reshape(test_ys.shape[0], 1)
    print('shape of test label set: ' + str(test_ys.shape))

    return train_xs, train_ys, test_xs, test_ys


train_Xs, train_Ys, test_Xs, test_Ys = load_data()
num_train = train_Xs.shape[0]
num_test = test_Xs.shape[0]


def convert_to_one_hot(y, num_category=10):
    return np.eye(num_category)[y.reshape(-1)]


def train_get_batch(num):
    _Xs = []
    _Ys = []
    for _ in range(num):
        _id = np.random.randint(0, num_train)
        _Xs.append(train_Xs[_id])
        _Ys.append(train_Ys[_id])
    return np.array(_Xs), convert_to_one_hot(np.array(_Ys))


def test_get_batch(num):
    _Xs = []
    _Ys = []
    for _ in range(num):
        _id = np.random.randint(0, num_test)
        _Xs.append(test_Xs[_id])
        _Ys.append(test_Ys[_id])
    return np.array(_Xs), convert_to_one_hot(np.array(_Ys))
    pass


if __name__ == '__main__':
    # data = load_data()
    # print(data)

    # batch = train_get_batch(10)
    # batch = test_get_batch(10)
    # print(batch)
    pass


