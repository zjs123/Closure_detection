import math

from requests import head
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch_geometric_temporal.nn.recurrent import GConvGRU
from torch_geometric_temporal.nn.recurrent import A3TGCN, TGCN
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops,degree, softmax, to_dense_batch, sort_edge_index
from torch_geometric.nn.dense.linear import Linear
import torch_scatter
from typing import Optional, Tuple, Union
from torch import Tensor
from torch.nn import LSTM
from torch_sparse import SparseTensor

import torch
import math
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops,remove_self_loops,softmax
from torch_geometric.datasets import Planetoid
import ssl
import torch.nn.functional as F
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptPairTensor,
    OptTensor,
    Size,
)

class H_GAT(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(H_GAT, self).__init__()

        self.real_GNN = GraphConv(in_channels = in_channels, out_channels = out_channels).cuda()
        self.plan_GNN = GraphConv(in_channels = in_channels, out_channels = out_channels).cuda()
        self.other_GNN = GraphConv(in_channels = in_channels, out_channels = out_channels).cuda()

        self.Agg_Encoder = Seq_Encoder_ATT(110, 110, 3)
    
    def forward(self, x, edge_index, edge_weight, cat_list):
        return self.real_GNN(x, edge_index, edge_weight)
        dst_list = edge_index[1]
        real_edge_index = []
        plan_edge_index = []
        other_edge_index = []

        real_edge_weight = []
        plan_edge_weight = []
        other_edge_weight = []
        for index in range(len(dst_list)):
            if cat_list[dst_list[index]] == 0 or cat_list[dst_list[index]] == 1:
                real_edge_index.append(edge_index[:,index].unsqueeze(-1))
                real_edge_weight.append(edge_weight[index])
            elif cat_list[dst_list[index]] == 1 or cat_list[dst_list[index]] == 2:
                plan_edge_index.append(edge_index[:,index].unsqueeze(-1))
                plan_edge_weight.append(edge_weight[index])
            else:
                other_edge_index.append(edge_index[:,index].unsqueeze(-1))
                other_edge_weight.append(edge_weight[index])
        
        # real sub GNN
        if len(real_edge_weight) == 0:
            real_gnn_output = x.unsqueeze(1)
        else:
            #print('real')
            edge_index = torch.cat(real_edge_index, -1).cuda()
            edge_weight = torch.LongTensor(real_edge_weight).cuda()
            row, col = edge_index
            adj = SparseTensor(row=row, col=col, sparse_sizes=(x.shape[0], x.shape[0]))
            real_gnn_output = self.real_GNN(x, edge_index, edge_weight).unsqueeze(1)
        
        if len(plan_edge_weight) == 0:
            plan_gnn_output = x.unsqueeze(1)
        else:
            #print('plan')
            edge_index = torch.cat(plan_edge_index, -1).cuda()
            edge_weight = torch.LongTensor(plan_edge_weight).cuda()
            row, col = edge_index
            adj = SparseTensor(row=row, col=col, sparse_sizes=(x.shape[0], x.shape[0]))
            plan_gnn_output = self.plan_GNN(x, edge_index, edge_weight).unsqueeze(1)

        if len(other_edge_weight) == 0:
            other_gnn_output = x.unsqueeze(1)
        else:
            #print('other')
            edge_index = torch.cat(other_edge_index, -1).cuda()
            edge_weight = torch.LongTensor(other_edge_weight).cuda()
            other_gnn_output = self.other_GNN(x, edge_index, edge_weight).unsqueeze(1)
        #print(plan_gnn_output.size())
        #print(other_gnn_output.size())
        #print(real_gnn_output.size())

        all_feature = torch.cat([real_gnn_output, plan_gnn_output, other_gnn_output], 1).permute(1,0,2)
        
        out = self.Agg_Encoder(all_feature, None)
        return  out

class GraphConv_LSTM(MessagePassing):
    """自定义GraphConv层的实现"""
    def __init__(self, in_channels, out_channels, normalize=True, bias=False, **kwargs):
        super(GraphConv_LSTM, self).__init__(**kwargs)

        self.in_channels = in_channels    # 输入张量尺寸
        self.out_channels = out_channels  # 输出张量尺寸
        self.normalize = normalize        # 是否正则化

        self.lin_l = nn.Linear(in_channels, out_channels)  # 应用于嵌入中心节点的线性变换，即权重矩阵Wl
        self.lin_r = nn.Linear(in_channels, out_channels)  # 应用于来自邻居的聚合消息的线性变换，即权重矩阵Wr

        self.lin_c = nn.Linear(2*in_channels, out_channels) # 聚合中心节点与邻域节点信息

        self.edge_EMB = nn.Embedding(10, out_channels)
        torch.nn.init.uniform_(self.edge_EMB.weight.data) # 边类型权重

        self.lstm = LSTM(out_channels, out_channels, batch_first=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        self.lin_c.reset_parameters()

    def forward(self, x, edge_index, edge_weight, size=None):
        """
        消息传递时的逻辑都由forward实现
        :param x: 中心节点及其邻居节点特征相等，都为x
        :param edge_index: 图的邻接表
        :param size:
        :return:
        """
        '''
        if isinstance(edge_index, Tensor):
            num_nodes = int(edge_index.max()) + 1
            edge_index = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes))
        '''
        #print(x.size())
        #print(edge_index.size())
        # 与PyG库的propagate()不同，这里只输出聚合后的邻居消息
        prop = self.propagate(edge_index, x=(x, x), edge_weight = edge_weight, size=size)

        # 下一层输出为中心节点线性变换+聚合邻居消息的线性变换
        out = self.lin_l(x) + self.lin_r(prop)
        
        #if self.normalize:
        #    out = F.normalize(out, p=2)  # L2正则化
        

        return out

    def message(self, x_i, x_j, edge_weight):
        """
        重写了父类MessagePassing的message()方法,在propagate()中被调用
        将邻居节点的信息转换为消息，以备聚合
        在GraphSAGE中邻居节点的消息等于其特征本身
        :param x_j: 邻居节点的特征
        :return: 待聚合的邻居消息
        """
        edge_weight_emb = self.edge_EMB(edge_weight) # edge_num * out_dim
        neighbor_messge = edge_weight_emb*x_j #edge_weight_emb*(x_j+x_i) #torch.cat([edge_weight_emb*x_j, x_i], -1)

        final_messge = neighbor_messge#self.lin_c(neighbor_messge)

        return final_messge

    def aggregate(self, x: Tensor, index: Tensor, ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:

        #sort_edge_index(index, sort_by_row=False)
        # LSTM aggregation:
        '''
        if ptr is None and not torch.all(index[:-1] <= index[1:]):
            raise ValueError(f"Can not utilize LSTM-style aggregation inside "
                             f"'{self.__class__.__name__}' in case the "
                             f"'edge_index' tensor is not sorted by columns. "
                             f"Run 'sort_edge_index(..., sort_by_row=False)' "
                             f"in a pre-processing step.")
        '''
        #print(index.size())
        x, mask = to_dense_batch(x, batch=index, batch_size=dim_size)
        #print(x.size())
        out, _ = self.lstm(x)
        #print(out.size())
        return out[:, -1]


class GraphConv(MessagePassing):
    """自定义GraphConv层的实现"""
    def __init__(self, in_channels, out_channels, normalize=True, bias=False, **kwargs):
        super(GraphConv, self).__init__(**kwargs)

        self.in_channels = in_channels    # 输入张量尺寸
        self.out_channels = out_channels  # 输出张量尺寸
        self.normalize = normalize        # 是否正则化

        self.lin_l = nn.Linear(in_channels, out_channels)  # 应用于嵌入中心节点的线性变换，即权重矩阵Wl
        self.lin_r = nn.Linear(in_channels, out_channels)  # 应用于来自邻居的聚合消息的线性变换，即权重矩阵Wr

        self.lin_c = nn.Linear(2*in_channels, out_channels) # 聚合中心节点与邻域节点信息

        self.edge_EMB = nn.Embedding(10, out_channels)
        torch.nn.init.uniform_(self.edge_EMB.weight.data) # 边类型权重

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        self.lin_c.reset_parameters()

    def forward(self, x, edge_index, edge_weight, size=None):
        """
        消息传递时的逻辑都由forward实现
        :param x: 中心节点及其邻居节点特征相等，都为x
        :param edge_index: 图的邻接表
        :param size:
        :return:
        """
        # 与PyG库的propagate()不同，这里只输出聚合后的邻居消息
        prop = self.propagate(edge_index, x=(x, x), edge_weight = edge_weight, size=size)

        # 下一层输出为中心节点线性变换+聚合邻居消息的线性变换
        out = self.lin_l(x) + self.lin_r(prop)
        
        #if self.normalize:
        #    out = F.normalize(out, p=2)  # L2正则化
        

        return out

    def message(self, x_i, x_j, edge_weight):
        """
        重写了父类MessagePassing的message()方法,在propagate()中被调用
        将邻居节点的信息转换为消息，以备聚合
        在GraphSAGE中邻居节点的消息等于其特征本身
        :param x_j: 邻居节点的特征
        :return: 待聚合的邻居消息
        """
        edge_weight_emb = self.edge_EMB(edge_weight) # edge_num * out_dim
        neighbor_messge = edge_weight_emb*x_j #edge_weight_emb*(x_j+x_i) #torch.cat([edge_weight_emb*x_j, x_i], -1)

        final_messge = neighbor_messge#self.lin_c(neighbor_messge)

        return final_messge

    
    def aggregate(self, inputs, index, dim_size=None):
        """
        重写了父类MessagePassing的aggregate()方法,在propagate()中被调用
        将邻居节点的消息进行聚合
        :param inputs: 由message()转换的消息
        :param index: 待聚合的邻居节点索引
        :param dim_size:
        :return:
        """
        # The axis along which to index number of nodes.
        node_dim = self.node_dim

        # GraphSAGE采用平均聚合
        out = torch_scatter.scatter(inputs, index, node_dim, dim_size=dim_size)

        return out
        


class XGB_Encoder(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(XGB_Encoder, self).__init__()
        self.linear_1 = torch.nn.Linear(in_dim, in_dim, bias = False)
        self.linear_2 = torch.nn.Linear(in_dim, out_dim, bias = False)
    
    def forward(self, x):
        x = self.linear_1(x)
        x = torch.tanh(x)
        x = self.linear_2(x)
        x = torch.tanh(x)

        x = F.dropout(x, p = 0.3, training=self.training)
        return x

class Seq_Encoder_GRU(torch.nn.Module):
    def __init__(self, in_dim, out_dim, layer):
        super(Seq_Encoder_GRU, self).__init__()
        self.layer = layer
        self.out_dim = out_dim

        # GRU encoder
        self.LSTM = nn.GRU(in_dim, out_dim, layer, bias = False)
        self.padding_weight = torch.nn.Parameter(torch.FloatTensor(out_dim), requires_grad=True)

        # aux task linear
        self.predictor = nn.Linear(out_dim, 1)

    def forward(self, f, padding):
        f = self.LSTM(f)[0]
        if padding == None:
            return f[-1]
        else:
            f_reshaped = f.permute(1,0,2) # n*T*f_dim
            final_index = (len(padding[0]) - torch.sum(padding, 1) - 1).long()
            
            # traj f collect
            traj_f_list = []
            for index in range(f_reshaped.size()[0]):
                if final_index[index] != -1:
                    traj_f_list.append(f_reshaped[index][final_index[index]])
                    #traj_f_list.append(torch.mean(f_reshaped[index][:(final_index[index]+1).long()], 0, keepdim = False))
                else:
                    #traj_f_list.append(torch.zeros(self.out_dim).cuda())
                    traj_f_list.append(self.padding_weight)
            f_collected = torch.stack(traj_f_list,0)
            #f_collected = [torch.mean(f_reshaped[index][:(final_index[index]+1).long()], 0, keepdim = True) for index in range(f_reshaped.size()[0])]
            #f_collected = torch.cat(f_collected, 0)

            return f_collected


class Seq_Encoder_ATT(torch.nn.Module):
    def __init__(self, dim, output_dim, head_num):
        super(Seq_Encoder_ATT, self).__init__()
        self.head_num = head_num
        self.output_dim = output_dim

        # project linear
        self.project_linear = nn.Linear(dim, output_dim, bias = True)

        # self-attention encoder
        self.self_attention_1 = nn.MultiheadAttention(output_dim*head_num, head_num, dropout=0.3, bias=True)
        self.traj_linear_1 = nn.Linear(output_dim*head_num, output_dim*head_num, bias = True)

        self.self_attention_2 = nn.MultiheadAttention(output_dim*head_num, head_num, dropout=0.3, bias=True)
        self.traj_linear_2 = nn.Linear(output_dim*head_num, output_dim*head_num, bias = True)

        # positional encoder
        
        pe = torch.zeros(100, output_dim)
        position = torch.arange(0, 100, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, output_dim, 2).float() * (-math.log(10000.0) / output_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

        # learnable positional encoder
        #self.pe = torch.nn.Parameter(torch.FloatTensor(100, 1, output_dim), requires_grad=True)
    
    def add_position(self, traj_f):
        position_embedding = self.pe[:traj_f.size(0), :]
        return traj_f + position_embedding
    
    def forward(self, traj_f, traj_padding):
        # perpare padding tensor
        if traj_padding != None :
            traj_padding_reshaped = traj_padding.repeat(self.head_num, 1).unsqueeze(1) # (N*head_num)*1*point_sample
            traj_padding_reshaped = traj_padding_reshaped.repeat(1, traj_padding.size()[1], 1) # (N*head_num)*point_sample*point_sample
        else:
            traj_padding_reshaped = None

        traj_f = self.project_linear(traj_f)
        traj_f = self.add_position(traj_f)
        traj_f_reshaped = traj_f.repeat(1, 1, self.head_num)
        
        traj_f_new = self.self_attention_1(query = traj_f_reshaped, key = traj_f_reshaped, value = traj_f_reshaped, attn_mask = traj_padding_reshaped)[0] + traj_f_reshaped
        #traj_f_new = self.traj_linear_1(traj_f_new)#+traj_f_reshaped
        
        #traj_f_new = self.self_attention_2(query = traj_f_new, key = traj_f_new, value = traj_f_new, attn_mask = traj_padding_reshaped)[0] + traj_f_new
        #traj_f_new = self.traj_linear_2(traj_f_new)#+traj_f_new

        #traj_f_mean = torch.mean(traj_f_new, 0)
        traj_f_mean = torch.mean(torch.mean(traj_f_new, 0).view(-1, self.head_num, self.output_dim), 1)

        return traj_f_mean 


class Basic_GNN_LSTM(torch.nn.Module):
    def __init__(self, node_dim, out_channel, his_len, layer_num):
        super(Basic_GNN_LSTM, self).__init__()
        self.layer_num = layer_num
        self.his_len = his_len
        # n layer GNN_LSTM for spatial and temporal modeling
        self.GNN_LSTM = [[H_GAT(in_channels = node_dim, out_channels = node_dim).cuda(), nn.GRU(node_dim,node_dim,1).cuda(), nn.Linear(node_dim, node_dim).cuda()] for i in range(layer_num)]

        # attention encoder for traj feature
        self.Traj_Encoder = Seq_Encoder_ATT(3, 50, 3)
        #self.uv_Encoder = Seq_Encoder_ATT(3, 50, 3)

        # GRU encoder for traj feature
        #self.Traj_Encoder = Seq_Encoder_GRU(5, 50, 1)
        self.uv_Encoder = Seq_Encoder_GRU(3,50,1)
        
        # Linear for regress and classify
        self.linear_main_task = torch.nn.Linear(node_dim, out_channel, bias = False)
        self.traj_project = torch.nn.Linear(5, 10, bias = True)
        self.linear_order_task = torch.nn.Linear(node_dim, out_channel, bias = False)
        
        # Init Embeddings
        self.cat_EMB = nn.Embedding(10, 10)
        torch.nn.init.uniform_(self.cat_EMB.weight.data)

        self.lane_EMB = nn.Embedding(10, 10)
        torch.nn.init.uniform_(self.lane_EMB.weight.data)

        self.direction_EMB = nn.Embedding(10, 10)
        torch.nn.init.uniform_(self.direction_EMB.weight.data)

        self.fc_EMB = nn.Embedding(10, 10)
        torch.nn.init.uniform_(self.fc_EMB.weight.data)

        self.speed_class_EMB = nn.Embedding(10, 10)
        torch.nn.init.uniform_(self.speed_class_EMB.weight.data)

        self.park_EMB = nn.Embedding(10, 10)
        torch.nn.init.uniform_(self.park_EMB.weight.data)

        self.status_EMB = nn.Embedding(10, 10)
        torch.nn.init.uniform_(self.status_EMB.weight.data)

        # aux task loss
        self.criterion = torch.nn.BCELoss()

    def graph_pooling(self, batch_index, h):
        batch_len = list(batch_index.bincount())
        
        h_central = torch.stack([i[0] for i in h.split(batch_len, 0)]) # batch_size*F
        h_mean = torch.stack([torch.mean(i, dim = 0) for i in h.split(batch_len, 0)]) # batch_size*F

        return h_central
    
    def GNN_LSTM_block(self, x, edge_index, edge_weight, batch_list, order_list, cat_list):
        GNN_outputs = [] # T*N*32
        order_list = order_list.permute(1,0)
        for step in range(len(x)):
            node_f = x[step]
            edge = edge_index[step]
            weight = edge_weight[step]
            cat = cat_list[step]
            for i in range(1):
                node_f = self.GNN_LSTM[i][0](node_f, edge, weight, cat.cuda())
            graph_emb = self.graph_pooling(batch_list[step], node_f)
            GNN_outputs.append((graph_emb).unsqueeze(0))
            #GNN_outputs.append(graph_emb.unsqueeze(0)+order_emb)

        spatial_embeddings = torch.cat(GNN_outputs, 0)
        h, _ = self.GNN_LSTM[0][1](spatial_embeddings)# T*batch_size*32
        #h = self.GNN_LSTM[0][2](h)

        #print(h.size())
        return h, spatial_embeddings
    
    def traj_aux_task(self, batch_index, traj_sample, traj_emb):
        
        # prepare the embedding
        collected_traj_sample = torch.LongTensor(traj_sample).cuda()
        if collected_traj_sample.size()[0] == 0:
            return 0
        
        traget_traj_emb = torch.index_select(traj_emb, 0, collected_traj_sample[:,0])
        pos_traj_emb = torch.index_select(traj_emb, 0, collected_traj_sample[:,1])
        neg_traj_emb = pos_traj_emb[torch.randperm(pos_traj_emb.size(0))]
        #neg_traj_emb = torch.index_select(traj_emb, 0, collected_traj_sample[:,2])

        # caculate the loss hinge_loss
        '''
        pos_exp = torch.exp(torch.sum(traget_traj_emb*pos_traj_emb, 1))
        neg_exp = torch.exp(torch.sum(traget_traj_emb*neg_traj_emb, 1))
        aux_loss = -torch.log(pos_exp/(pos_exp+neg_exp))
        '''
        
        pos_score = torch.norm(traget_traj_emb-pos_traj_emb, 2, 1)
        neg_score = torch.norm(traget_traj_emb-neg_traj_emb, 2, 1)
        aux_loss = pos_score-neg_score+1
        aux_loss[aux_loss <= 0] = 0
        

        return torch.mean(aux_loss)


    def forward(self, batch):
        """
        features = Node features for T time steps T*N(batched by snapshot)*(1+uv_f_dim)
        edge_indices = Graph edge indices T*2*E(batched by snapshot)
        edge_weight = Graph edge weight T*E(batched by snapshot)
        edge_weights = Batch split for T time steps T*N(batched by snapshot)
        targets = label for each node in T time steps T*N(batched by snapshot)*1
        traj_f = raw traj features in T time steps T*N(batched by snapshot)*6*point_sample
        traj_padding = padding index for raw traj features, 1:pad, 0:unpad T*N(batched by snapshot)*point_sample
        seq_padding = final index of each order squence (batch_size)
        """
        
        seq_padding = batch[1].cuda().long()
        seq_span = batch[3].permute(1,0).cuda()
        order_type_list = batch[4].cuda()
        time_steps = len(batch[0].edge_indices)
        traj_aux_sample_batch = batch[2]
        
        edge_index_list = []
        edge_weight_list = []
        target_list = []
        batch_list =  []
        f_list = [] # T*N*F
        cat_list = []

        traj_aux_loss = 0

        # perpare the input features 
        for step in range(time_steps):
            edge_index = torch.Tensor(batch[0].edge_indices[step]).long().cuda()
            edge_weight = torch.Tensor(batch[0].edge_weights[step]).long().cuda()
            features = torch.Tensor(batch[0].features[step]).cuda()
            batch_index = torch.Tensor(batch[0].batches[step]).long().cuda()
            target = torch.Tensor(batch[0].targets[step]).cuda()

            # encode the raw traj feature
            traj_raw_feature = torch.Tensor(batch[0].traj_f[step]).cuda() # N(batched by snapshot)*6*point_sample
            traj_raw_feature = traj_raw_feature.permute(2,0,1) # point_sample*N(batched by snapshot)*6
            
            traj_raw_feature_num = traj_raw_feature[:,:,2:-1] # point_sample*N(batched by snapshot)*5
            status_emb = self.status_EMB(traj_raw_feature[:,:,-1].long()) # point_sample*N(batched by snapshot)*10
            traj_raw_feature = traj_raw_feature_num #torch.cat([traj_raw_feature_num, status_emb], -1)
            
            traj_padding = torch.Tensor(batch[0].traj_padding[step]).float().cuda() # N(batched by snapshot)*point_sample

            traj_feature = self.Traj_Encoder(traj_raw_feature, traj_padding) # N(batched by snapshot)*(traj_emb)

            # caculate the traj aux loss
            #traj_aux_loss += self.traj_aux_task(batch_index, traj_aux_sample_batch[step], traj_feature)

            # encode the raw uv feature
            uv_feature = features[:,1:31] # N*uv_dim
            uv_feature = uv_feature.reshape(-1, 3, 10) # N*3*point_sample
            uv_feature = uv_feature.permute(2,0,1) # point_sample*N*3
            uv_feature = self.uv_Encoder(uv_feature, None) # N(batched by snapshot)*(uv_emb)
            #traj_aux_loss += aux_loss

            # look up the category embedding
            cat_embedding = self.cat_EMB(features[:,0].long()) # N*node_dim
            static_feature = features[:,31:] # N*7
            cat_list.append(features[:,0].long())

            lane_emb = self.lane_EMB(static_feature[:,0].long()) # N*node_dim
            direction_emb = self.direction_EMB(static_feature[:,1].long())
            fc_emb = self.fc_EMB(static_feature[:,4].long())
            speed_class_emb = self.speed_class_EMB(static_feature[:,5].long())
            park_emb = self.park_EMB(static_feature[:,6].long())

            # concate the features
            h = torch.cat([cat_embedding, lane_emb, direction_emb, fc_emb, speed_class_emb, park_emb, traj_feature], -1) # N*(node_dim*4+uv_dim)

            # caculate the encoder aux loss
            traj_uv_emb = h #torch.cat([uv_feature, traj_feature], -1)
            traj_aux_loss += self.traj_aux_task(batch_index, traj_aux_sample_batch[step], traj_uv_emb)

            edge_index_list.append(edge_index)
            edge_weight_list.append(edge_weight)
            target_list.append(target)
            batch_list.append(batch_index)
            f_list.append(h)

        # encode the data via n layer GNN and n layer LSTM
        LSTM_outputs, GNN_outputs = self.GNN_LSTM_block(f_list, edge_index_list, edge_weight_list, batch_list, order_type_list, cat_list) # T*batch_size*dim

        # collect the output features
        GNN_outputs_reshaped = GNN_outputs.permute(1,0,2)
        LSTM_outputs_reshaped = LSTM_outputs.permute(1,0,2) # batch_size*T*dim
        collected_gnn_outputs = []
        collected_lstm_outputs = []
        predict_emb = []
        order_task_y = []
        order_task_label = []

        for batch in range(GNN_outputs_reshaped.size()[0]):
            collected_gnn_outputs.append(GNN_outputs_reshaped[batch][:seq_padding[batch]])
            collected_lstm_outputs.append(LSTM_outputs_reshaped[batch][:seq_padding[batch]])
            predict_emb.append(LSTM_outputs_reshaped[batch][seq_padding[batch]-1].unsqueeze(0))
            order_task_y.append(torch.sigmoid(self.linear_order_task(GNN_outputs_reshaped[batch][:seq_padding[batch]])))
            order_task_label.append(order_type_list[batch][:seq_padding[batch]])

        predict_emb = torch.cat(predict_emb, 0)
        order_task_y = torch.cat(order_task_y)
        order_task_label = torch.cat(order_task_label).unsqueeze(1)
        #traj_aux_loss += self.criterion(order_task_y, order_task_label)

        predict_emb = F.dropout(predict_emb, p = 0.3, training=self.training) # batch_size*F

        # main task loss
        batch_len = list(batch_list[-1].bincount())
        target_main = torch.stack([i[0] for i in target_list[-1].split(batch_len, 0)])

        y_main = torch.sigmoid(self.linear_main_task(predict_emb)) # batch_size*1

        return y_main, target_main, traj_aux_loss


'''
class LSTM(nn.Module):
    
    def __init__(self, input_num, output_num, layer):
        super(LSTM, self).__init__()
        self.hidden_size = input_num
        self.grucell = span_GRUCell(input_num, output_num)
        self.out_linear = nn.Linear(output_num, output_num)

    def forward(self, x, span_list):
        y_list = []
        if span_list is None:
            span_list = torch.zero(x.shape[0])
        for step in range(x.shape[0]):
            if step == 0:
                hid = torch.randn(x[step].shape[0], self.hidden_size)
            hid = self.grucell(x[step].cuda(), hid.cuda(), span_list[step].cuda())  # 需要传入隐藏层状态
            y = self.out_linear(hid)
            y_list.append(y.unsqueeze(0))
        y_result = torch.cat(y_list, 0)
        return y_result, hid.detach()  # detach()和detach_()都可以使用


class span_GRUCell(nn.Module):

    def __init__(self, input_size, output_size, bias=True):
        super(span_GRUCell, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * output_size, bias=bias).cuda()
        self.h2h = nn.Linear(input_size, 3 * output_size, bias=bias).cuda()

        self.decay_weight = nn.Linear(1, output_size, bias=bias).cuda()
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.output_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, hidden, span):
        
        x = x.view(-1, x.size(1))
        
        gate_x = self.x2h(x) 
        gate_h = self.h2h(hidden)
        
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()
        
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        
        # 公式1
        resetgate = F.sigmoid(i_r + h_r)
        # 公式2
        inputgate = F.sigmoid(i_i + h_i)#*torch.exp((-span*0.1)).unsqueeze(1)
        # 公式3
        newgate = F.tanh(i_n + (resetgate * h_n))
        # 公式4，不过稍微调整了一下公式形式
        hy = (1-inputgate)*newgate + inputgate*hidden
        
        return hy
'''