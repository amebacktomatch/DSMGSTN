import torch.optim as optim
import utils
from model import *
class trainer():
    def __init__(self,batch_size,scaler,in_dim,seq_length,num_nodes,n_hid,dropout,learing_rate,weight_decay,dyanmic_supoorts,clip,lr_de_rate,
                 static_supports1,static_supports2,static_supports3,static_supports4,dgcn_adj):
        self.model=mynet(dyanmic_supoorts,num_nodes,in_dim,static_supports1,static_supports2,static_supports3,static_supports4,dgcn_adj)
        self.model.cuda()
        self.optimizer=optim.Adam(self.model.parameters(),lr=learing_rate,weight_decay=weight_decay)
        self.loss=utils.masked_mae
        self.scaler=scaler
        self.clip=clip
        lr_decay_rate = lr_de_rate
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: lr_decay_rate ** epoch)

    def train(self,input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        input= nn.functional.pad(input,(1,0,0,0))
        output= self.model(input)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = utils.masked_mape(predict, real, 0.0).item()
        rmse = utils.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mape, rmse

    def eval(self,input, real_val):
        self.model.eval()
        input = nn.functional.pad(input, (1, 0, 0, 0))
        output = self.model(input)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = utils.masked_mape(predict, real, 0.0).item()
        rmse = utils.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mape, rmse