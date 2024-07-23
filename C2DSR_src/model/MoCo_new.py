import torch
import torch.nn as nn
class MoCo_Interest(nn.Module):

    def __init__(self, opt, model, T=0.1):
        super(MoCo_Interest, self).__init__()
        self.T = T
        self.opt = opt
        self.pooling = opt["pooling"]
        self.model = model
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
                
                # find multi-interest for each user
                sim = mixed_feature@prototypes.T
                _, top_k_indice  = torch.topk(sim, self.opt['topk_cluster'], dim=-1)
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