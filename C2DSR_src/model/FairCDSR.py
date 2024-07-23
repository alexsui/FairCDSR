import numpy as np
import torch
import torch.nn as nn
import math
from sklearn.preprocessing import StandardScaler
class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs



class scaled_dot_product_attention(torch.nn.Module):
    def forward(self, q, k, v, mask = None, dropout = None): #q,k,v皆為(B, num_heads, seq_len, h)
        score = torch.matmul(q,k.transpose(-1,-2))/math.sqrt(q.size(-1))
        if mask is not None:
            score = score.masked_fill(mask==0, -1e9) # if true, then fill -1e9
        prob_score = torch.softmax(score, dim = -1)
        if dropout is not None:
            prob_score = dropout(prob_score)
        attention_val = torch.matmul(prob_score, v)
        return attention_val, prob_score            
class MultiHeadAttention(torch.nn.Module):
    def __init__(self,  num_heads, qk_d_model, v_d_model, dropout_rate = 0.2):
        super(MultiHeadAttention, self).__init__()
        assert qk_d_model%num_heads==0
        assert v_d_model%num_heads==0
        self.hidden_units = v_d_model
        self.num_heads = num_heads
        self.qk_head_dim = qk_d_model // num_heads
        self.v_head_dim = v_d_model // num_heads
        self.W_Q = torch.nn.Linear(qk_d_model, qk_d_model)
        self.W_K = torch.nn.Linear(qk_d_model, qk_d_model)
        self.W_V = torch.nn.Linear(v_d_model, v_d_model)

        self.fc = torch.nn.Linear(v_d_model, v_d_model)
        self.attention = scaled_dot_product_attention()
        self.dropout = torch.nn.Dropout(p=dropout_rate)
       
    def forward(self, queries, keys, values, mask=None):
        # queries, keys, values: (N, T, C)
        batch_size = queries.size(0)
        q = self.W_Q(queries).view(batch_size,-1, self.num_heads, self.qk_head_dim)
        k = self.W_K(keys).view(batch_size,-1, self.num_heads, self.qk_head_dim)
        v = self.W_V(values).view(batch_size,-1, self.num_heads, self.v_head_dim)
        q,k,v = [x.transpose(1,2) for x in [q,k,v]]
        attention_output, attention_weights = self.attention(q, k, v, mask = mask, dropout = self.dropout)
        attention_output = attention_output.transpose(1,2).contiguous().view(batch_size, -1, self.hidden_units)
        outputs = self.fc(attention_output)
        return outputs, attention_weights

class ATTENTION(torch.nn.Module):
    def __init__(self, opt):
        super(ATTENTION, self).__init__()
        self.opt = opt
        self.emb_dropout = torch.nn.Dropout(p=self.opt["dropout"])
        self.pos_emb = torch.nn.Embedding(self.opt["maxlen"]+1, self.opt["hidden_units"], padding_idx=0)  # TO IMPROVE
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(self.opt["hidden_units"], eps=1e-8)
            
            
        for _ in range(self.opt["num_blocks"]):
            new_attn_layernorm = torch.nn.LayerNorm(self.opt["hidden_units"], eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layer = MultiHeadAttention(self.opt["num_heads"], self.opt["hidden_units"], self.opt["hidden_units"], self.opt["dropout"])
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(self.opt["hidden_units"], eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(self.opt["hidden_units"], self.opt["dropout"])
            self.forward_layers.append(new_fwd_layer)
    def learn_time_embedding(self, tt):
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
    def forward(self, seqs_data, seqs, position, ts=None, causality_mask = True):
        
        #build attention mask
        timeline_mask = torch.BoolTensor(seqs_data.cpu() != self.opt["itemnum"] - 1) #若為padding則為False
        if self.opt["cuda"]:
            timeline_mask = timeline_mask.cuda()
        if causality_mask:
            tl = seqs.shape[1] # time dim len for enforce causality
            causal_mask = torch.tril(torch.ones((tl, tl), dtype=torch.bool)) #若為False則不會被attention到
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(1).expand(seqs.size(0), 1, tl, tl) # (B, 1, seq_len, seq_len)
            if self.opt["cuda"]:
                causal_mask = causal_mask.cuda()
            extended_timeline_mask = timeline_mask.unsqueeze(1).unsqueeze(2) #[B,1,1,seq_len]
            attention_mask = extended_timeline_mask * causal_mask
            # print("extended_attention_mask:", extended_attention_mask.shape) #[B,1,seq_len,seq_len]
        else:
            attention_mask = timeline_mask.unsqueeze(1).unsqueeze(2)#[B,1,1,seq_len]
        seqs += self.pos_emb(position)
        seqs = self.emb_dropout(seqs)
        seqs *= timeline_mask.unsqueeze(-1) # broadcast in last dim ,過濾掉padding的部分=>padding的部分embedding為0
        
        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, mask = attention_mask)
            seqs = Q + mha_outputs # residual connection
            seqs = self.forward_layernorms[i](seqs)
            outs = self.forward_layers[i](seqs)
            seqs = seqs + outs     # residual connection
            seqs *=  timeline_mask.unsqueeze(-1)
        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)
        
        return log_feats
class CL_Projector(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.dense = nn.Linear(opt["hidden_units"], opt["hidden_units"])
        self.activation = nn.ReLU()
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, first_token_tensor):
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
class FairCDSR(torch.nn.Module):
    def __init__(self, opt):
        super(FairCDSR, self).__init__()
        self.opt = opt
        self.item_emb_X = torch.nn.Embedding(self.opt["itemnum"] + 1, self.opt["hidden_units"],
                                           padding_idx= self.opt["itemnum"] - 1)
        self.item_emb_Y = torch.nn.Embedding(self.opt["itemnum"] + 1, self.opt["hidden_units"],
                                           padding_idx= self.opt["itemnum"] - 1)
        self.item_emb = torch.nn.Embedding(self.opt["itemnum"] + 1, self.opt["hidden_units"],
                                           padding_idx = self.opt["itemnum"] - 1)   #opt["itemnum"] = opt["source_item_num"] + opt["target_item_num"] + 1

        self.lin_X = nn.Linear(self.opt["hidden_units"], self.opt["source_item_num"])
        self.lin_Y = nn.Linear(self.opt["hidden_units"], self.opt["target_item_num"])                      
        self.lin_PAD = nn.Linear(self.opt["hidden_units"], 1)
        self.encoder = ATTENTION(opt)
        self.encoder_X = ATTENTION(opt)
        self.encoder_Y = ATTENTION(opt)
        
        #for in-domain proto_CL
        self.CL_projector = CL_Projector(opt)
        self.CL_projector_X = CL_Projector(opt)
        self.CL_projector_Y = CL_Projector(opt)
        
        # for I2C contrastive learning
        self.interest_projector = CL_Projector(opt)
        self.interest_projector_X = CL_Projector(opt)
        self.interest_projector_Y = CL_Projector(opt)
        
        # for cross domain proto_CL
        self.projector = nn.Sequential(
            nn.Linear(self.opt["hidden_units"], self.opt["hidden_units"]*2),
            nn.ReLU(),
            nn.Linear(self.opt["hidden_units"]*2, self.opt["hidden_units"])
        )
        self.projector_X = nn.Sequential(
            nn.Linear(self.opt["hidden_units"], self.opt["hidden_units"]*2),
            nn.ReLU(),
            nn.Linear(self.opt["hidden_units"]*2, self.opt["hidden_units"])
        )
        self.projector_Y = nn.Sequential(
            nn.Linear(self.opt["hidden_units"], self.opt["hidden_units"]*2),
            nn.ReLU(),
            nn.Linear(self.opt["hidden_units"]*2, self.opt["hidden_units"])
        )
        self.initialize_weights(self.projector)
        self.initialize_weights(self.projector_X)
        self.initialize_weights(self.projector_Y)
        self.source_item_index = torch.arange(0, self.opt["source_item_num"], 1)
        self.target_item_index = torch.arange(self.opt["source_item_num"], self.opt["source_item_num"] + self.opt["target_item_num"], 1)
        self.item_index = torch.arange(0, self.opt["itemnum"], 1)
        if self.opt["cuda"]:
            self.source_item_index = self.source_item_index.cuda()
            self.target_item_index = self.target_item_index.cuda()
            self.item_index = self.item_index.cuda()
    def initialize_weights(self,projector):
        for m in projector:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def my_index_select_embedding(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = memory(index)
        ans = ans.view(tmp)
        return ans

    def my_index_select(self, memory, index):#從memory中選擇index的item
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = torch.index_select(memory, 0, index)
        ans = ans.view(tmp)
        return ans

    def forward(self, o_seqs, x_seqs, y_seqs, position, x_position, y_position, ts_d=None, ts_xd=None, ts_yd=None): #_o_seqs為mixed domain seqs, _x_seqs為source domain seqs, _y_seqs為target domain seqs
        
        if o_seqs is None:
            seqs_fea = None
        else:
            seqs = self.item_emb(o_seqs)
            seqs *= self.item_emb.embedding_dim ** 0.5
            seqs_fea = self.encoder(o_seqs, seqs, position, ts=ts_d)
        if x_seqs is None:
            x_seqs_fea = None
        else:            
            seqs = self.item_emb_X(x_seqs)
            seqs *= self.item_emb.embedding_dim ** 0.5
            x_seqs_fea = self.encoder_X(x_seqs, seqs, x_position, ts=ts_xd)
        if y_seqs is None:
            y_seqs_fea = None
        else:
            seqs = self.item_emb_Y(y_seqs)
            seqs *= self.item_emb.embedding_dim ** 0.5
            y_seqs_fea = self.encoder_Y(y_seqs, seqs, y_position, ts=ts_yd)
        return seqs_fea, x_seqs_fea, y_seqs_fea

    def false_forward(self, false_seqs, position):
        seqs = self.my_index_select(self.cross_emb, false_seqs) + self.item_emb(false_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        false_seqs_fea = self.encoder(false_seqs, seqs, position)
        return false_seqs_fea

class GenderDiscriminator(nn.Module):
    def __init__(self,opt):
        super(GenderDiscriminator, self).__init__()
        self.opt = opt
        self.attention=nn.Linear(opt['hidden_units'],1)
        self.layer = nn.Sequential(
            nn.Linear(opt['hidden_units'], 2*opt['hidden_units']),
            nn.ReLU(),
            nn.Linear(2*opt['hidden_units'], opt['hidden_units']),
            nn.ReLU(),
            nn.Linear(opt['hidden_units'],1),
            nn.Sigmoid()
        )
    def forward(self, x):
        attention_weights = torch.nn.functional.softmax(self.attention(x), dim=1)
        weighted_average = torch.sum(x * attention_weights, dim=1)
        
        return self.layer(weighted_average), attention_weights
        
class ClusterRepresentation(nn.Module):
    def __init__(self, opt, feature_dim, num_clusters, topk):#topk most similar cluster
        super(ClusterRepresentation, self).__init__()
        self.opt = opt
        self.num_clusters = num_clusters
        self.feature_dim = feature_dim
        self.topk = topk
        self.cluster_prototypes = nn.Parameter(torch.randn(num_clusters, feature_dim))
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim*topk, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        
    def forward(self, features):
        sim = features@self.cluster_prototypes.T #[X, num_clusters]
        sim /= sim.max(-1,keepdim = True)[0]
        weight = torch.softmax(sim, dim=-1)
        new_cluster = weight.T@features
        new_sim = features@new_cluster.T
        _, top_k_indice  = torch.topk(new_sim, self.topk, dim=-1)#[X, topk]
        multi_interest = new_cluster[top_k_indice.squeeze()]#[B,topk,feature_dim]
        multi_interest = self.feature_extractor(multi_interest.reshape(-1, self.feature_dim*self.topk))#[B, feature_dim]
        return new_cluster, multi_interest
