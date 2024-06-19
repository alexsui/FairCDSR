import torch
import torch.nn as nn
import copy
import ipdb
from random import sample
from utils.time_transformation import TimeTransformation
from utils.augmentation import TimeSpeedUp, TimeSlowDown, TimeReverse,TimeShift,TimeReorder

class MoCo_Interest(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, opt, model, T=0.1):
        super(MoCo_Interest, self).__init__()
        self.T = T
        self.opt = opt
        self.pooling = opt["pooling"]
        self.model = model
        # if domain == "X":
        #     self.encoder_q = model.encoder_X
        #     self.encoder_k = copy.deepcopy(model.encoder_X)
        #     self.item_embed = model.item_emb_X
        #     self.CL_projector_q = model.CL_projector_X                   # for instance discrimination
        #     self.CL_projector_k = copy.deepcopy(model.CL_projector_X)
        #     self.projector = model.projector_X                           #for cross-domain clustering : project x,y to joint space
        # elif domain == "Y":
        #     self.encoder_q = model.encoder_Y
        #     self.encoder_k = copy.deepcopy(model.encoder_Y)
        #     self.item_embed = model.item_emb_Y
        #     self.CL_projector_q = model.CL_projector_Y
        #     self.CL_projector_k = copy.deepcopy(model.CL_projector_Y)
        #     self.projector = model.projector_Y                          
        # elif domain == "mixed":
        #     self.encoder_q = model.encoder
        #     self.encoder_k = copy.deepcopy(model.encoder)
        #     self.item_embed = model.item_emb
        #     self.CL_projector_q = model.CL_projector
        #     self.CL_projector_k = copy.deepcopy(model.CL_projector)
        #     self.projector = model.projector 
        # for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
        #     param_k.data.copy_(param_q.data)  # initialize
        #     param_k.requires_grad = False  #not update by gradient
        # for param_q, param_k in zip(self.CL_projector_q.parameters(), self.CL_projector_k.parameters()):
        #     param_k.data.copy_(param_q.data)
        #     param_k.requires_grad = False
        
    def encode(self, encoder,item_embed, seq, ts):
        non_zero_mask = (seq != (self.opt["source_item_num"] + self.opt["target_item_num"])).long()
        position_id = non_zero_mask.cumsum(dim=1) * non_zero_mask
        seq_feat = item_embed(seq)
        if ts is not None:
            ts = torch.sort(ts,dim=-1)[0]
        feat = encoder(seq, seq_feat, position_id, ts, causality_mask = False)  # keys: NxC

        if self.pooling == "bert":
            feat = feat[:,0,:]
        elif self.pooling == "ave":
            feat= torch.sum(feat, dim=1)/torch.sum(non_zero_mask, dim=1).unsqueeze(-1)
        feat = nn.functional.normalize(feat, dim=1)
        return feat
    def forward(self, mixed_seq, target_seq, cluster_result=None, index=None, ts = None):
        mixed_feature = self.encode(self.model.encoder, self.model.item_emb, mixed_seq,ts = ts)
        target_feature = self.encode(self.model.encoder_Y, self.model.item_emb_Y, target_seq, ts = ts)
        target_feature = self.model.interest_projector_Y(target_feature) # project to the joint space of the prototypes
        if cluster_result is not None:  
            proto_logits = []
            for n, (im2cluster,prototypes,density) in enumerate(zip(cluster_result['im2cluster'],cluster_result['centroids'],cluster_result['density'])):
                
                # MoCo本來的作法
                # pos_proto_id = im2cluster[index]
                # pos_prototypes = prototypes[pos_proto_id]   
                # all_proto_id = [i for i in range(im2cluster.max()+1)]       
                # neg_proto_id = set(all_proto_id)-set(pos_proto_id.tolist())
                # neg_proto_id = sample(neg_proto_id,self.opt['r']) #sample r negative prototypes 
                # neg_prototypes = prototypes[neg_proto_id]    
                # proto_selected = torch.cat([pos_prototypes,neg_prototypes],dim=0)
                # logits_proto = torch.mm(q,proto_selected.t())
                # labels_proto = torch.linspace(0, q.size(0)-1, steps=q.size(0)).long().cuda()
                # temp_proto = density[torch.cat([pos_proto_id,torch.LongTensor(neg_proto_id).cuda()],dim=0)]
                
                # find multi-interest for each user
                sim = mixed_feature@prototypes.T
                top_k_values, top_k_indice  = torch.topk(sim, self.opt['topk_cluster'], dim=-1)
                multi_interest = prototypes[top_k_indice].squeeze()
                l_pos = torch.einsum('nc,nkc->nk', [target_feature, multi_interest])#[B,number_of_interest]
                all_proto_id = torch.arange(0, len(prototypes),dtype = torch.long).repeat(mixed_feature.size(0),1) 
                all_proto_id = all_proto_id.cuda() if self.opt['cuda'] else all_proto_id
                top_k_indice_expanded = top_k_indice.unsqueeze(1)
                all_proto_id_expanded = all_proto_id.unsqueeze(2)
                mask = (all_proto_id_expanded != top_k_indice_expanded)
                mask = mask.all(dim=-1)
                neg_proto_id = all_proto_id[mask].view(mixed_feature.size(0),-1)
                neg_prototypes = prototypes[neg_proto_id] #[batch_size, num_neg, dim]
                l_neg = torch.einsum('nc,nkc->nk', [target_feature, neg_prototypes])
                logits_proto = torch.cat([l_pos, l_neg], dim=1)
                temp_proto = density[torch.cat([top_k_indice,neg_proto_id],dim=1)]  
                logits_proto /= temp_proto
                proto_logits.append(logits_proto)
            return proto_logits
        # if cluster_result is not None:  
        #     proto_logits = []
        #     for n, (im2cluster,prototypes) in enumerate(zip(cluster_result['im2cluster'],cluster_result['centroids'])):
        #         # compute prototypical logits
        #         sim = torch.mm(q,prototypes.t())/self.opt['temp']
        #         sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        #         sim = sim - sim_max
        #         logits_proto = torch.nn.functional.softmax(sim, dim=1)
        #         # logits_proto = torch.exp(sim)
        #         # z = torch.sum(logits_proto, dim=-1) + 1e-5
        #         # logits_proto = logits_proto/z.unsqueeze(1)
        #         proto_logits.append(logits_proto)
        #     return None, None, proto_logits, None, None, None
        # else:
        #     return None, None, None, None, None, None


# # utils
# @torch.no_grad()
# def concat_all_gather(tensor):
#     """
#     Performs all_gather operation on the provided tensors.
#     *** Warning ***: torch.distributed.all_gather has no gradient.
#     """
#     tensors_gather = [torch.ones_like(tensor)
#         for _ in range(torch.distributed.get_world_size())]
#     torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

#     output = torch.cat(tensors_gather, dim=0)
#     return output