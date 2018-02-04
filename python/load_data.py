import scipy.io as sio
import numpy as np


def load_data(batch_size=100):
    data = sio.loadmat('data.mat')
    # data
    test_data = data['data'][0][0][0]
    train_data = data['data'][0][0][1]
    valid_data = data['data'][0][0][2]
    vocab = data['data'][0][0][3]
    print(test_data.shape)
    print(train_data.shape)
    print(valid_data.shape)
    print(vocab.shape)

    numdims = train_data.shape[0]
    D = numdims - 1
    M = train_data.shape[1] / batch_size

    train_input = np.reshape(train_data[0:D, 0:batch_size * M], (D, batch_size, M))
    train_target = np.reshape(train_data[D, 0:batch_size * M], (1, batch_size, M))
    valid_input = valid_data[0:D, :]
    valid_target = valid_data[D, :]
    test_input = test_data[0:D, :]
    test_target = test_data[D, :]
    vocab = vocab

    print(train_input.shape)
    print(train_target.shape)
    print(valid_input.shape)
    print(valid_target.shape)
    print(test_input.shape)
    print(test_target.shape)
    print(vocab.shape)

    return train_input, train_target, valid_input, valid_target, test_input, test_target, vocab