import argparse
import pandas as pd
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--his_length', type=int, default=7)
parser.add_argument("--pred_length", type=int, default=7)
parser.add_argument('--datapath', type=str, default='data/delay.csv')
parser.add_argument('--train_ratio', type=float, default=0.7)
parser.add_argument('--val_ratio', type=float, default=0.1)

args = parser.parse_args()


def seq2instance(data, his_length, pred_length):
    sample_num = data.shape[0] - his_length - pred_length + 1
    x = torch.zeros(sample_num, his_length, data.shape[1])
    y = torch.zeros(sample_num, pred_length, data.shape[1])
    for i in range(sample_num):
        x[i] = data[i:i + his_length]
        y[i] = data[i + his_length:i + his_length + pred_length]
    return x, y


def axisup(array):
    d = array[:, np.newaxis, :, :]
    d = np.tile(d, (2, 1, 1))
    return d


df = pd.read_csv(args.datapath, index_col=0)
data = torch.from_numpy(df.values)
train_num = round(df.shape[0] * args.train_ratio)
val_num = round(df.shape[0] * args.val_ratio)
test_num = df.shape[0] - train_num - val_num
train = data[:train_num]
val = data[train_num:train_num + val_num]
test = data[-test_num:]
trainx, trainy = seq2instance(train, args.his_length, args.pred_length)
valx, valy = seq2instance(val, args.his_length, args.pred_length)
testx, testy = seq2instance(test, args.his_length, args.pred_length)
trainx = axisup(trainx)
trainy = axisup(trainy)
valx = axisup(valx)
valy = axisup(valy)
testx = axisup(testx)
testy = axisup(testy)
np.savez('data/train.npz', x=trainx, y=trainy,allow_pickle=True)
np.savez('data/val.npz', x=valx, y=valy,allow_pickle=True)
np.savez('data/test.npz', x=testx, y=testy,allow_pickle=True)