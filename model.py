import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils
def getstaticsupports(adjmartix):
    adj = [utils.sym_adj(adjmartix), utils.sym_adj(np.transpose(adjmartix))]
    return adj


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,wv->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 2), dilation=2,padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,len_support=3,order=2):
        super(gcn,self).__init__()
        self.nconv=nconv()
        c_in=(order*len_support+1)*c_in
        self.mlp=linear(c_in,c_out)
        self.dropout=dropout
        self.order=order

    def forward(self,x,support):
        out=[x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h
class mynet(nn.Module):
    def __init__(self,dynamic_supports,num_nodes,in_dim, static_supports1,static_supports2,static_supports3,static_supports4,dgcn_adj,
                 kernel_size=2,layers=1,dilation_channels=40,skip_channels=320,dropout=0.3,residual_channels=40,
                 end_channels=640,out_dim=7,blocks=3):
        super(mynet, self).__init__()
        receptive_field = 1
        self.dropout=dropout
        self.blocks=blocks
        self.layers=layers
        self.supports=static_supports1
        self.residual_channels=residual_channels
        self.supports_len = 0
        self.supports_len += len(static_supports1)
        self.dynamic_supports = dynamic_supports
        self.static_supports1 = static_supports1
        self.static_supports2 = static_supports2
        self.static_supports3 = static_supports3
        self.static_supports4 = static_supports4
        #自适应图
        self.nodevec1=nn.Parameter(torch.randn(num_nodes,10).cuda(),requires_grad=True).cuda()
        self.nodevec2=nn.Parameter(torch.randn(10,num_nodes).cuda(),requires_grad=True).cuda()
        #sgcn
        self.staticnodevec11 = nn.Parameter(torch.randn(num_nodes,10).cuda(),requires_grad=True).cuda()
        self.staticnodevec12 = nn.Parameter(torch.randn(10,num_nodes).cuda(),requires_grad=True).cuda()
        self.staticnodevec21 = nn.Parameter(torch.randn(num_nodes, 10).cuda(), requires_grad=True).cuda()
        self.staticnodevec22 = nn.Parameter(torch.randn(10, num_nodes).cuda(), requires_grad=True).cuda()
        self.staticnodevec31 = nn.Parameter(torch.randn(num_nodes, 10).cuda(), requires_grad=True).cuda()
        self.staticnodevec32 = nn.Parameter(torch.randn(10, num_nodes).cuda(), requires_grad=True).cuda()
        self.staticnodevec41 = nn.Parameter(torch.randn(num_nodes, 10).cuda(), requires_grad=True).cuda()
        self.staticnodevec42 = nn.Parameter(torch.randn(10, num_nodes).cuda(), requires_grad=True).cuda()
        self.supports_len +=1
        self.tcn1=nn.ModuleList()
        self.tcn2=nn.ModuleList()
        self.dgcn=nn.ModuleList()
        self.sgcn=nn.ModuleList()
        self.skipconnection=nn.ModuleList()
        self.batchnorm=nn.ModuleList()
        self.batchnorm1=nn.BatchNorm2d(in_dim,affine=False)
        self.beginconv=nn.Conv2d(in_channels=in_dim,out_channels=residual_channels,kernel_size=(1,1))
        self.endconv1 = nn.Conv2d(in_channels=skip_channels,out_channels=end_channels,kernel_size=(1,1),bias=True)
        self.endconv2 = nn.Conv2d(in_channels=end_channels,out_channels=out_dim,kernel_size=(1,1),bias=True)
        self.dgcn_adj = dgcn_adj
        for b in range(blocks):
           for i in range(layers):
               self.tcn1.append((nn.Conv2d(in_channels=residual_channels,out_channels=dilation_channels,
                                           kernel_size=(1,kernel_size),dilation=2)))
               self.tcn2.append((nn.Conv2d(in_channels=residual_channels,out_channels=dilation_channels,
                                           kernel_size=(1,kernel_size),dilation=2)))
               self.skipconnection.append(nn.Conv2d(in_channels=dilation_channels,out_channels=skip_channels,
                                                    kernel_size=(1,1)))
               self.dgcn.append(gcn(dilation_channels,int(residual_channels),dropout))
               self.sgcn.append(gcn(dilation_channels,int(residual_channels),dropout,len_support=5,order=2))
               receptive_field += (kernel_size
                                    * 2)
               self.batchnorm.append(nn.BatchNorm2d(int(residual_channels)))
        self.receptive_field=receptive_field
        self.startbatchnorm = nn.BatchNorm2d(in_dim,affine=False)
    def forward(self,input):
        gcnused_static_supports1 = [0,0]
        gcnused_static_supports2 = [0,0]
        gcnused_static_supports3 = [0,0]
        gcnused_static_supports4 = [0,0]
        gunused_static_supports12 = [0,0]
        gunused_static_supports22 = [0, 0]
        gunused_static_supports32 = [0, 0]
        gunused_static_supports42 = [0, 0]
        supports_in_gcn1 = [0,0,0,0,0]
        supports_in_gcn2 = [0, 0, 0, 0, 0]
        supports_in_gcn3 = [0, 0, 0, 0, 0]
        supports_in_gcn4 = [0, 0, 0, 0, 0]


        iter = utils.iternum
        in_len=input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        adp=F.softmax(F.relu(torch.mm(self.nodevec1,self.nodevec2)),dim=1)
        sadp1=F.softmax(F.relu(torch.mm(self.staticnodevec11,self.staticnodevec12)),dim=1)
        sadp2 = F.softmax(F.relu(torch.mm(self.staticnodevec21,self.staticnodevec22)),dim=1)
        sadp3 = F.softmax(F.relu(torch.mm(self.staticnodevec31, self.staticnodevec32)), dim=1)
        sadp4 = F.softmax(F.relu(torch.mm(self.staticnodevec41, self.staticnodevec42)), dim=1)

        #x=self.batchnorm1(x)
        x=self.startbatchnorm(x)
        x=self.beginconv(x)
        skip = 0
        x1 = x
        x2 = x
        x3 = x
        x4 = x
        x5 = x
        dynamic_supports=self.dynamic_supports[iter]+[adp]

        static_imf_adjmx1 = self.dgcn_adj[iter,0,:,:]
        static_imf_adjmx2 = self.dgcn_adj[iter,1, :, :]
        static_imf_adjmx3 = self.dgcn_adj[iter,2, :, :]
        static_imf_adjmx4 = self.dgcn_adj[iter,3,:,:]
        imf1 = getstaticsupports(static_imf_adjmx1)
        imf2 = getstaticsupports(static_imf_adjmx2)
        imf3 = getstaticsupports(static_imf_adjmx3)
        imf4 = getstaticsupports(static_imf_adjmx4)
        imf_supports1 = [torch.tensor(i).cuda() for i in imf1]
        imf_supports2 = [torch.tensor(i).cuda() for i in imf2]
        imf_supports3 = [torch.tensor(i).cuda() for i in imf3]
        imf_supports4 = [torch.tensor(i).cuda() for i in imf4]

        gcnused_static_supports1[0] = self.static_supports1[0]*imf_supports1[0]
        gcnused_static_supports1[1] = self.static_supports1[1]*imf_supports1[1]
        gunused_static_supports12[0] = self.static_supports1[0]*dynamic_supports[0]
        gunused_static_supports12[1] = self.static_supports1[1]*dynamic_supports[1]
        supports_in_gcn1 = gcnused_static_supports1 + gunused_static_supports12 + [sadp1]
        gcnused_static_supports2[0] = self.static_supports2[0] * imf_supports2[0]*dynamic_supports[0]
        gcnused_static_supports2[1] = self.static_supports2[1] * imf_supports2[1]*dynamic_supports[1]
        gunused_static_supports22[0] = self.static_supports2[0] * dynamic_supports[0]
        gunused_static_supports22[1] = self.static_supports2[1] * dynamic_supports[1]
        supports_in_gcn2 = gcnused_static_supports2 + gunused_static_supports22 + [sadp2]
        gcnused_static_supports3[0] = self.static_supports3[0] * imf_supports3[0]*dynamic_supports[0]
        gcnused_static_supports3[1] = self.static_supports3[1] * imf_supports3[1]*dynamic_supports[1]
        gunused_static_supports32[0] = self.static_supports3[0] * dynamic_supports[0]
        gunused_static_supports32[1] = self.static_supports3[1] * dynamic_supports[1]
        supports_in_gcn3 = gcnused_static_supports3 + gunused_static_supports32 + [sadp3]
        gcnused_static_supports4[0] = self.static_supports4[0] * imf_supports4[0]*dynamic_supports[0]
        gcnused_static_supports4[1] = self.static_supports4[1] * imf_supports4[1]*dynamic_supports[1]
        gunused_static_supports42[0] = self.static_supports4[0] * dynamic_supports[0]
        gunused_static_supports42[1] = self.static_supports4[1] * dynamic_supports[1]
        supports_in_gcn4 = gcnused_static_supports4 + gunused_static_supports42 + [sadp4]
        
        #Gate TCN
        for i in range(self.blocks*self.layers):
            # dynamic graph conv
            res1 = x1  #64 40 50 13
            dtcn1 = self.tcn1[i](res1)
            dtcn1 = torch.tanh(dtcn1)
            dtcn2 = self.tcn2[i](res1)
            dtcn2 = torch.sigmoid(dtcn2)
            x1 = dtcn1*dtcn2
            x1 = self.dgcn[i](x1, dynamic_supports)
            x1 = self.batchnorm[i](x1) #x1.shape= 64 20 50 11
           
            x1 = x1 + res1[:,:,:,-x1.size(3):]
        
            # static graph conv
            #graph 1
            res2 = x2
            s1tcn1 = self.tcn1[i](res2)
            s1tcn1 = torch.tanh(s1tcn1)
            s1tcn2 = self.tcn2[i](res2)
            s1tcn2 = torch.sigmoid(s1tcn2)
            x2 = s1tcn1*s1tcn2
            x2 = self.sgcn[i](x2, supports_in_gcn1)
            x2 = self.batchnorm[i](x2)
            x2 = x2 + res2[:, :, :, -x2.size(3):]
            # graph 2
            res3 = x3
            s2tcn1 = self.tcn1[i](res3)
            s2tcn1 = torch.tanh(s2tcn1)
            s2tcn2 = self.tcn2[i](res3)
            s2tcn2 = torch.sigmoid(s2tcn2)
            x3 = s2tcn1 * s2tcn2
            x3 = self.sgcn[i](x3, supports_in_gcn2)
            x3 = self.batchnorm[i](x3)
            x3= x3 + res3[:, :, :, -x3.size(3):]
            # graph 3
            res4 = x4
            s3tcn1 = self.tcn1[i](res4)
            s3tcn1 = torch.tanh(s3tcn1)
            s3tcn2 = self.tcn2[i](res4)
            s3tcn2 = torch.sigmoid(s3tcn2)
            x4 = s3tcn1 * s3tcn2
            x4 = self.sgcn[i](x4, supports_in_gcn3)
            x4 = self.batchnorm[i](x4)
            x4 = x4 + res4[:, :, :, -x4.size(3):]
            # graph 4
            res5 = x5
            s4tcn1 = self.tcn1[i](res5)
            s4tcn1 = torch.tanh(s4tcn1)
            s4tcn2 = self.tcn2[i](res5)
            s4tcn2 = torch.sigmoid(s4tcn2)
            x5 = s4tcn1 * s4tcn2
            x5 = self.sgcn[i](x5, supports_in_gcn4)
            x5 = self.batchnorm[i](x5)
            x5 = x5 + res5[:, :, :, -x5.size(3):]


            xall = (x1 + x2 + x3 + x4 + x5)/5
            xall = self.batchnorm[i](xall)
            


            s = xall
            s = self.skipconnection[i](s)

            
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip



        x = F.leaky_relu(skip)

        x = F.leaky_relu(self.endconv1(x))

        x = self.endconv2(x)
        
        return x