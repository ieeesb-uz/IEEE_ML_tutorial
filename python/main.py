import os
import time  # This is required to include time module.
import numpy as np
import scipy.io as sio
from load_data import load_data
from train import train





def main():

    os.system('wget http://www.ucode.es/data.mat')

    epochs = 10

    train_input, train_target, valid_input, valid_target, test_input, test_target, vocab = load_data(batchsize)



if __name__ == '__main__':
    main()