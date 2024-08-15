import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch.conv import ChebConv
import dgl
#from dgl.nn import EGATConv
from torch.nn.utils import weight_norm
import torch as th
from layers import BottleneckLayer,EGATLayer,MLPPredictor,MergeLayer


#from models import EGAT
class LSN_module(nn.Module):
    def __init__(self, sta_fes, sta_h1,droup):
        super(LSN_module, self).__init__()
        
        self.fc1 = nn.Linear(sta_fes, sta_h1)
        self.convS = dgl.nn.GraphConv(sta_h1, sta_h1)
        #self.convS = EGATConv(sta_h1, sta_h1,3)
        self.dropout1 = nn.Dropout(droup)
        self.relu1 = nn.ReLU()

    def reset_param(self):
        for layer in [self.conv1]:
            nn.itnit.kaiming_uniform_(layer.weight, a=0, mode = 'fan_in',nonlinearity = 'relu')
    
    def forward(self, g, static_x):
        x1 =  self.fc1(static_x)   
        x1 =  nn.functional.elu(x1)
        
        x = self.convS(g, x1)
        x = self.relu1(x)
        #e_x = self.relu11(e_x)
        
        return self.dropout1(x) #,self.dropout11(e_x)


class Chomp2d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp2d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        裁剪的模块，裁剪多出来的padding
        """
        return x[:, :, :-self.chomp_size[0], :-self.chomp_size[1]].contiguous()

class TN_module(nn.Module):
    def __init__(self, c_in, c_out, dropout, dia=1):
        super(TN_module, self).__init__()
        
        self.convT = weight_norm(nn.Conv2d(c_in, c_out, (3,1),
                                           stride=1, padding=(0,0), dilation=dia))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.dropout2 = nn.Dropout(dropout)
        self.chomp1 = Chomp2d((2,0))  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        
        self.conv = nn.Conv2d(
            c_in, c_out, (3, 1), 1, dilation=dia, padding=(0, 0)
        )  #（2，1）卷积核的尺寸
        self.net = nn.Sequential(self.convT, self.relu2, self.dropout2)
        self.init_weights()
    
    def init_weights(self):
        """
        参数初始化
        """
        self.convT.weight.data.normal_(0, 0.01)
    
    def forward(self, x):
        return self.net(x)



class TTST_module(nn.Module):
    def __init__(self, c_in, c_out,sta_fes, sta_h1,droput):
        super(TTST_module, self).__init__()
        
        self.T1 = TN_module(c_in,c_out,droput,dia=1)
        self.S1 = LSN_module(sta_fes, sta_h1,droput)
        
        self.T2 = TN_module(c_out+sta_h1,64,droput,dia = 2)
        #node_feats, edge_feats, f_h,f_e,lamda, num_heads, dropout,pred_hid,l_num
        self.S2 = EGAT(64, 4, 48, 16,0.85,3,droput,48,2)
        self.T3 = TN_module(48,64,droput,dia = 4)
        
        self.out = OutputLayer(64, 16, 23140)
        
    def forward(self, g, dy_x, stat_x,edge_feats):
        
        x1 = self.T1(dy_x)         
        x2 = self.S1(g, stat_x)    
        
        new_LS = torch.zeros((x1.shape[0], 28, 23140,64))
        x2 = x2.repeat(new_LS.shape[0], new_LS.shape[1], 1, 1)
        x2 = x2.transpose(1, 3)
        x2 = x2.transpose(2, 3)
        
        x= torch.cat((x1,x2),dim = 1)  
        del x1,x2,new_LS,dy_x,stat_x
        
        
        x = self.T2(x)
        x = x.transpose(0, 2)
        x = x.transpose(1, 3)
        x = x.transpose(1, 2)
        
        egat_outputs = []
        for i in range(x.shape[0]):
            x_ = self.S2(g, x[i,:,:,:], edge_feats)  
            egat_outputs.append(x_)
        x = torch.stack(egat_outputs, dim=0)  #[looks_back,23140，64]
        x = x.view(1,x.shape[0],x.shape[1],x.shape[2])  #[batch_size,looks_back,23140，64]
        #del g,edge_feats
        #print(x.shape)
        x = x.transpose(1, 2)                #[batch_size,23140,looks_back,64]
        x = x.transpose(1, 3)            
        #print(x.shape)
        x = self.T3(x)            
        #print(x.shape)
        x = self.out(x).view(-1,23140)
        
        return x

class FullyConvLayer(nn.Module):
    def __init__(self, c):
        super(FullyConvLayer, self).__init__()
        self.conv = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        return self.conv(x)


class OutputLayer(nn.Module):
    def __init__(self, c, T, n):
        super(OutputLayer, self).__init__()
        self.tconv1 = nn.Conv2d(c, c, (T, 1), 1, dilation=1, padding=(0, 0))
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)
        #self.ln = nn.LayerNorm([n, c])
        self.tconv2 = nn.Conv2d(c, c, (1, 1), 1, dilation=1, padding=(0, 0))
        self.relu4 = nn.ReLU()
        self.drop4 = nn.Dropout(0.5)
        self.fc = FullyConvLayer(c)
        
        

    def forward(self, x):
        x_t1 = self.relu3(self.tconv1(x))
        x_t1 = self.drop3(x_t1)
        #print(x_t1.shape)
        #x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        #print(x_ln.shape)
        x_t2 = self.relu4(self.tconv2(x_t1))
        x_t2 = self.drop4(x_t2)
        return self.fc(x_t2)

    

    
class EGAT(nn.Module):
    def __init__(self, node_feats, edge_feats, f_h,f_e,lamda, num_heads, dropout,pred_hid,l_num):
        #node_feats=node_features.shape[1],edge_feats=edge_features.shape[1],
        #f_h = 128,f_e = 128,lamda = args.lamda,num_heads = args.num_heads,dropout=args.dropout,pred_hid = 128, l_num = args.l_nums, 
        super(EGAT, self).__init__()
        device = torch.device('cuda:1')
        self.l = l_num
        self.bottleneck = BottleneckLayer(node_feats, edge_feats,f_h,f_e)  #first 
        self.egat = EGATLayer(f_h, f_e, lamda, num_heads)
        self.pred = MLPPredictor(f_h,64,pred_hid)
        self.merge = MergeLayer(f_h,f_e,l_num)  #last 
        self.dropout = dropout

    def forward(self, graph, h_in,e_in):
        
        h_out,e_out = self.bottleneck(h_in,e_in)  #first layer
        h_final = h_out.view((-1,h_out.shape[-1]))
        e_final = e_out.view((-1,e_out.shape[-1]))
        for i in range(self.l):
          h_out,e_out = self.egat(graph,h_out,e_out)  #second layer
          #print('h_out,e_out',h_out.shape,e_out.shape)
          if i != 0:#Merge Layer
            h_final = th.cat([h_final,h_out], dim=-1)
            e_final = th.cat([e_final,e_out], dim=-1)

        
        h_final, e_final = self.merge(h_final, e_final)   # last layer
        #print(h_final.shape)
        return self.pred(h_final)     