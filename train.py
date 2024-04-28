import utils
import torch
import numpy as np
import argparse
import time
import os
from engine import *



parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='0',help='graphics card')
parser.add_argument('--data',type=str,default='./data',help="data path")
parser.add_argument('--seq_length',type=int,default=7,help='prediction length')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--nhid',type=int,default=40,help='')
parser.add_argument('--num_nodes',type=int,default=50,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--clip', type=int, default=3, help='Gradient Clipping')
parser.add_argument('--lr_decay_rate', type=float, default=0.97, help='learning rate')
parser.add_argument('--epochs',type=int,default=1,help='')
parser.add_argument('--static_graph1',type=str,default='./data/distancemartix.npy',help="static graph")
parser.add_argument('--static_graph2',type=str,default='./data/flowmartix.npy')
parser.add_argument('--static_graph3',type=str,default='./data/dtwmartix.npy')
parser.add_argument('--static_graph4',type=str,default='./data/poimartix.npy')
parser.add_argument('--imfnum',type=int,default=4)
parser.add_argument('--seed',type=int,default=114514,help='random seed')
parser.add_argument('--save',type=str,default='./garage/',help='save path')
parser.add_argument('--print_every',type=int,default=5,help='')
args = parser.parse_args()

def setup_seed(seed):
    np.random.seed(seed) # Numpy module
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU


def main():
    setup_seed(args.seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    adj_mx1 = utils.load_adj(args.static_graph1)
    supports1 = [torch.tensor(i).cuda() for i in adj_mx1]
    adj_mx2 = utils.load_adj(args.static_graph2)
    supports2 = [torch.tensor(i).cuda() for i in adj_mx2]
    adj_mx3 = utils.load_adj(args.static_graph3)
    supports3 = [torch.tensor(i).cuda() for i in adj_mx3]
    adj_mx4 = utils.load_adj(args.static_graph4)
    supports4 = [torch.tensor(i).cuda() for i in adj_mx4]

    dataloader, insxnum = utils.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size) #insnum is the number of the instance
    scaler = dataloader['scaler']
    dynamic_martix = utils.getimfmartix(args.batch_size, insxnum)


    dgcn_adj=utils.getdgcnmartix(dynamic_martix)
    dynamic_supports=[]
    for j in range(len(dgcn_adj)):
        middle = [torch.tensor(i).cuda() for i in dgcn_adj[j]]
        dynamic_supports.append(middle)





    engine = trainer(args.batch_size, scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                     args.learning_rate, args.weight_decay, dynamic_supports,  args.clip, args.lr_decay_rate,supports1,supports2,supports3,supports4,dynamic_martix)

    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    for i in range(1, args.epochs + 1):
        print('***** Epoch: %03d START *****' % i)
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            utils.getiternum(iter)
            trainx = torch.Tensor(x).cuda()
            trainx = trainx.transpose(2, 3)
            trainy = torch.Tensor(y).cuda()
            trainy = trainy.transpose(2, 3)
            # trainx 2789 2 12 50
            metrics = engine.train(trainx, trainy[:, 0, :, :])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)
        t2 = time.time()
        train_time.append(t2 - t1)

        engine.scheduler.step()

        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).cuda()
            testx = testx.transpose(2, 3)
            testy = torch.Tensor(y).cuda()
            testy = testy.transpose(2, 3)
            metrics = engine.eval(testx, testy[:, 0, :, :])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, \nValid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),
              flush=True)
        torch.save(engine.model.state_dict(),
                   args.save + "_epoch_" + str(i) + "_" + str(round(mvalid_loss, 2)) + ".pth")

        print('***** Epoch: %03d END *****' % i)
        print('\n')

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(
        torch.load(args.save + "_epoch_" + str(bestid + 1) + "_" + str(round(his_loss[bestid], 2)) + ".pth"))

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).cuda()
    realy = realy.transpose(2, 3)[:, 0, :, :]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).cuda()
        testx = testx.transpose(2, 3)
        with torch.no_grad():
            preds = engine.model(testx).transpose(2, 3)
           
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]
    print(yhat.shape)
    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid], 4)))
    print("Best model epoch:", str(bestid + 1))

    amae = []
    amape = []
    armse = []

    yhat = yhat.transpose(1,2)


    for i in range(7):

        pred = scaler.inverse_transform(yhat[:,:,i])

        real = realy[:,:,i]



        metrics = utils.metric(pred, real)

        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 7 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))
    torch.save(engine.model.state_dict(), args.save + "_best_" + str(round(his_loss[bestid], 2)) + ".pth")


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))