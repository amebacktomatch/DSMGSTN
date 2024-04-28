import numpy as np
import os

import pandas as pd
import scipy.sparse as sp
import torch
import math
from PyEMD import EMD
global iternum

iternum=0
def getiternum(iter):
    global iternum
    iternum = iter

class DataLoader(object):

    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1
        return _wrapper()


class StandardScaler():

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        mask = (data == 0)
        data[mask] = self.mean
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())

    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    torch.Tensor(data['x_train'])


    return data, data['x_train'].shape[0]
def sym_adj(adj):
    #calculate normalized laplacian
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # d -1/2
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # https://github.com/tkipf/gcn/issues/142
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def load_adj(path):
    adjmartix=np.load(path)
    adj= [sym_adj(adjmartix),sym_adj(np.transpose(adjmartix))]
    return adj
def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse
def calculate_imf(iter,xdata,batch_size):

    imflist=[]

    for i in range(50):

            imfs=EMD().emd(xdata[iter*batch_size:iter*batch_size+batch_size,i],np.arange(xdata[iter*batch_size:iter*batch_size+batch_size,i].shape[0]))
            imflist.append(imfs)

    return imflist

def get_imf(batch_size,instancexnum):

    iter = math.ceil(instancexnum / batch_size)

    origindata = pd.read_csv("./data/delay.csv",index_col=0)
    origindata = origindata.values
    originxdata = origindata[:round(origindata.shape[0]*0.7)]
    imflist=[0 for i in range(iter)]
    for j in range(iter):
      imflist[j]=calculate_imf(j,originxdata,batch_size)

    return imflist,iter

def getimfmartix(batch_size,instancexnum):
    imflist,iter=get_imf(batch_size,instancexnum)
    dyanmic_martix = np.zeros((iter,4,50,50))
    for i in range(iter):
        imfvec=imflist[i]

        imfmartix1=np.zeros((50,50))
        imfmartix2=np.zeros((50,50))
        imfmartix3=np.zeros((50,50))
        imfmartix4=np.zeros((50,50))



        for k in range(50):
                for l in range(k,50):
                    try:
                        array1 = imfvec[k]
                        array2 = imfvec[l]

                        imfmartix1[k,l]=np.abs(np.corrcoef(array1[0,:],array2[0,:]))[0][1]
                        imfmartix2[k,l]=np.abs(np.corrcoef(array1[1,:],array2[1,:]))[0][1]
                        imfmartix3[k, l] = np.abs(np.corrcoef(array1[2, :], array2[2,: ]))[0][1]
                        imfmartix4[k, l] = np.abs(np.corrcoef(array1[3, :], array2[3,: ]))[0][1]
                    except:
                        array1 = imfvec[k]
                        array2 = imfvec[l]
                        imfmartix1[k, l] = np.abs(np.corrcoef(array1[0, :], array2[0, :]))[0][1]
                        imfmartix2[k, l] = np.abs(np.corrcoef(array1[1, :], array2[1,:]))[0][1]
                        imfmartix3[k, l] = np.abs(np.corrcoef(array1[2, :], array2[2,: ]))[0][1]
                        imfmartix4[k, l] = ( imfmartix1[k, l] +  imfmartix2[k, l] +  imfmartix3[k, l]) /3
        for o in range(50):
                for p in range(o):
                    imfmartix1[o,p] = imfmartix1 [p,o]
                    imfmartix2[o, p] = imfmartix2[p, o]
                    imfmartix3[o, p] = imfmartix3[p, o]
                    imfmartix4[o, p] = imfmartix4[p, o]

        imfmartix1=np.reshape(imfmartix1,(1,50,50))
        imfmartix2 = np.reshape(imfmartix2, (1, 50, 50))
        imfmartix3 = np.reshape(imfmartix3, (1, 50, 50))
        imfmartix4 = np.reshape(imfmartix4, (1, 50, 50))
        imfmiddle = np.concatenate((imfmartix1,imfmartix2,imfmartix3,imfmartix4),axis=0)
        dyanmic_martix[i,:,:,:] = imfmiddle
    return dyanmic_martix


def getdgcnmartix(martix):

    adj=np.zeros((martix.shape[0],50,50))
    adj_mxlist=[]
    for i in range(martix.shape[0]):
        adj[i,:,:]=(martix[i,0,:,:]+martix[i,1,:,:]+martix[i,2,:,:]+martix[i,3,:,:])/4
    for i in range(adj.shape[0]):
        adj1 = [sym_adj(adj[i,:,:]),sym_adj(np.transpose(adj[i,:,:]))]
        adj_mxlist.append(adj1)
    return adj_mxlist





class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean