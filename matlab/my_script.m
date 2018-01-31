clc
clear all
close all
%
load data.mat
disp(fieldnames(data))

batch_size = 100;
n_epochs = 1;

[train_x, train_t, valid_x, valid_t, test_x, test_t, vocab] =...
    load_data(batch_size);
%%
model = train(5);
%%