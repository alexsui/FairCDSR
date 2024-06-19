import torch
import torch.nn as nn
import copy
import ipdb
from random import sample
from utils.time_transformation import TimeTransformation
from utils.augmentation import TimeSpeedUp, TimeSlowDown, TimeReverse,TimeShift,TimeReorder

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, opt, model, dim=128, r=1280, m=0.999, T=0.1, mlp = False, domain = "X"):
        """
        dim: feature dimension (default: 128)
        r: queue size; number of negative samples/prototypes (default: 16384)
        m: momentum for updating key encoder (default: 0.999)
        T: softmax temperature 
        mlp: whether to use mlp projection
        """
        super(MoCo, self).__init__()

        self.r = r
        self.m = m
        self.T = T
        self.mlp = mlp
        self.opt = opt
        self.pooling = opt["pooling"]
        # create the encoders
        self.speed_classifier = model.speed_classifier
        self.direction_classifier = model.direction_classifier
        if domain == "X":
            self.encoder_q = model.encoder_X
            self.encoder_k = copy.deepcopy(model.encoder_X)
            self.item_embed = model.item_emb_X
            self.CL_projector_q = model.CL_projector_X                   # for instance discrimination
            self.CL_projector_k = copy.deepcopy(model.CL_projector_X)
            self.projector = model.projector_X                           #for cross-domain clustering : project x,y to joint space
            self.Equivariance_Projector = model.Equivariance_Projector_X #for time equivariance discrimination
        elif domain == "Y":
            self.encoder_q = model.encoder_Y
            self.encoder_k = copy.deepcopy(model.encoder_Y)
            self.item_embed = model.item_emb_Y
            self.CL_projector_q = model.CL_projector_Y
            self.CL_projector_k = copy.deepcopy(model.CL_projector_Y)
            self.projector = model.projector_Y                          
            self.Equivariance_Projector = model.Equivariance_Projector_Y
        elif domain == "mixed":
            self.encoder_q = model.encoder
            self.encoder_k = copy.deepcopy(model.encoder)
            self.item_embed = model.item_emb
            self.CL_projector_q = model.CL_projector
            self.CL_projector_k = copy.deepcopy(model.CL_projector)
            self.projector = model.projector 
            self.Equivariance_Projector = model.Equivariance_Projector
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for param_q, param_k in zip(self.CL_projector_q.parameters(), self.CL_projector_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        # create the queue
        self.register_buffer("queue", torch.randn(dim, r))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.time_transformation = TimeTransformation(opt)
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        if self.mlp:
            for param_q, param_k in zip(self.CL_projector_q.parameters(), self.CL_projector_k.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.r % batch_size == 0  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.r  # move pointer
        self.queue_ptr[0] = ptr
        
    def encode(self,encoder, seq, mlp, ts, mode = "q"):
        non_zero_mask = (seq != (self.opt["source_item_num"] + self.opt["target_item_num"])).long()
        position_id = non_zero_mask.cumsum(dim=1) * non_zero_mask
        seq_feat = self.item_embed(seq)
        if ts is not None:
            ts = torch.sort(ts,dim=-1)[0]
        feat = encoder(seq, seq_feat, position_id, ts, causality_mask = False)  # keys: NxC

        if self.pooling == "bert":
            feat = feat[:,0,:]
        elif self.pooling == "ave":
            feat= torch.sum(feat, dim=1)/torch.sum(non_zero_mask, dim=1).unsqueeze(-1)
        if mlp:
            if mode == "q":
                feat = self.CL_projector_q(feat)
            elif mode == "k":
                feat = self.CL_projector_k(feat)
        feat = nn.functional.normalize(feat, dim=1)
        return feat
    def equivariance_encode(self, encoder, seq, ts, aug_method, scale):
        non_zero_mask = (seq != (self.opt["source_item_num"] + self.opt["target_item_num"])).long()
        position_id = non_zero_mask.cumsum(dim=1) * non_zero_mask
        seq_feat = self.item_embed(seq)
        ts = torch.sort(ts,dim=-1)[0]
        if isinstance(aug_method,TimeReorder) or isinstance(aug_method,TimeReverse):
            new_ts, position_id = aug_method(ts, position_id, scale)
        else:
            new_ts = aug_method(ts, scale)
        feat = encoder(seq, seq_feat, position_id, new_ts, causality_mask = True)  # keys: NxC
        if self.pooling == "bert":
            feat = feat[:,0,:]
        elif self.pooling == "ave":
            feat= torch.sum(feat, dim=1)/torch.sum(non_zero_mask, dim=1).unsqueeze(-1)
        feat = nn.functional.normalize(feat, dim=1)
        return feat 
    def forward(self, seq_q, seq_k=None, is_eval=False, cluster_result=None, index=None, task="in-domain", train_mode = "train", ts = None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            is_eval: return momentum embeddings (used for clustering)
            cluster_result: cluster assignments, centroids, and density
            index: indices for training samples
        Output:
            logits, targets, proto_logits, proto_targets
        """
        
        if is_eval:
            k = self.encode(self.encoder_k, seq_q, mlp = self.mlp, ts = ts, mode = "k")  #???           
            return k
        if task == "in-domain":
            # Instance Discrimination
            with torch.no_grad():
                if train_mode == "train":
                    # compute key features
                        self._momentum_update_key_encoder()  # update the key encoder
                        k = self.encode(self.encoder_k, seq_k, mlp = self.mlp, ts = ts, mode = "k")  # keys: NxC
                else:
                        k = self.encode(self.encoder_k, seq_k, mlp = self.mlp, ts = ts, mode = "k")  # keys: NxC
            q = self.encode(self.encoder_q, seq_q, mlp = self.mlp, ts = ts, mode = "q")
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
            logits = torch.cat([l_pos, l_neg], dim=1)
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            
            time_ssl_task = None
            equivariance_loss = None
            if self.opt["time_encode"]:
                # Time equivariance learning - Equivariance via Transformation Discrimination
                pos_aug1, pos_aug2, neg_aug_1, neg_aug_2, scale = self.time_transformation(mode ="discriminate")
                pos_feature_1 = self.equivariance_encode(self.encoder_q, seq_q, ts = ts, aug_method = pos_aug1, scale=scale)
                pos_feature_2 = self.equivariance_encode(self.encoder_q, seq_q, ts = ts, aug_method = pos_aug2, scale=scale)
                neg_feature_1 = self.equivariance_encode(self.encoder_q, seq_q, ts = ts, aug_method = neg_aug_1, scale=scale)
                neg_feature_2 = self.equivariance_encode(self.encoder_q, seq_q, ts = ts, aug_method= neg_aug_2, scale=scale)
                pos_feature = torch.cat([pos_feature_1,pos_feature_2],dim=1) #B,2H
                pos_feature = self.Equivariance_Projector(pos_feature)
                detached_pos_feature = pos_feature.clone().detach()
                neg_feature = torch.cat([neg_feature_1,neg_feature_2],dim=1)
                neg_feature = self.Equivariance_Projector(neg_feature)
                detached_neg_feature = neg_feature.clone().detach()
                batch_size = pos_feature.size(0)
                pos_mask = torch.triu(torch.ones(batch_size,batch_size,dtype=torch.long),diagonal=1).to(pos_feature.device)
                pos_sim = torch.matmul(pos_feature, detached_pos_feature.T)
                logits_max, _ = torch.max(pos_sim, dim=1, keepdim=True)
                pos_sim = pos_sim - logits_max.detach()
                nominator = torch.sum(torch.exp(pos_sim/self.T)*pos_mask,dim=1,keepdim=True)
                neg_mask =torch.ones(batch_size,batch_size,dtype=torch.long).to(pos_feature.device)
                neg_mask = neg_mask.fill_diagonal_(0)
                neg_sim = torch.matmul(pos_feature, detached_neg_feature.T)
                logits_max, _ = torch.max(neg_sim, dim=1, keepdim=True)
                neg_sim = neg_sim - logits_max.detach()
                
                denominator = torch.sum(torch.exp(neg_sim/self.T)*neg_mask,dim=1,keepdim=True)
                equivariance_loss = -torch.sum(torch.log((nominator+1e-6)/(denominator+1e-6)))
                # Time equivariance learning - auxiliary ssl task
                # 1. speed classification
                aug_method, scale = self.time_transformation(mode ="prediction", task = "speed_classification")
                for s in scale:
                    speedup_feature = self.equivariance_encode(self.encoder_q, seq_q, ts = ts, aug_method = aug_method, scale=s)
                    if s == scale[0]:
                        speedup_feature_all = speedup_feature
                    else:
                        speedup_feature_all = torch.cat([speedup_feature_all,speedup_feature],dim=0)
                speed_pred = self.speed_classifier(speedup_feature_all)
                speed_labels = torch.arange(0, len(scale),dtype = torch.long).repeat_interleave(batch_size).cuda()
                perm_indices = torch.randperm(speed_pred.size(0))
                shuffled_speed_feature, shuffled_speed_labels = speed_pred[perm_indices],speed_labels[perm_indices]
                shuffled_speed_pred = torch.argmax(shuffled_speed_feature,dim=1)
                speed_acc = (shuffled_speed_pred == shuffled_speed_labels).float().mean()
                # 2. direction classification
                aug_method, _ = self.time_transformation(mode ="prediction", task = "direction_classification")
                reversed_feature = self.equivariance_encode(self.encoder_q, seq_q, ts = ts, aug_method = aug_method, scale=None)
                feature = self.encode(self.encoder_q, seq_q, mlp = False, ts =ts)
                all_feature = torch.cat([feature,reversed_feature],dim=0)
                direction_pred = self.direction_classifier(all_feature).squeeze()
                direction_labels = torch.arange(0, 2,dtype = torch.float).repeat_interleave(batch_size).cuda()
                perm_indices = torch.randperm(all_feature.size(0))
                shuffled_direction_feature, shuffled_direction_labels = direction_pred[perm_indices],direction_labels[perm_indices]
                shuffled_direction_pred = torch.round(shuffled_direction_feature)
                direction_acc = (shuffled_direction_pred == shuffled_direction_labels).float().mean()
                time_ssl_task = {
                    "speed_classification" : (shuffled_speed_feature, shuffled_speed_labels,speed_acc.item()),
                    "direction_classification" : (shuffled_direction_feature, shuffled_direction_labels,direction_acc.item())
                }
            if train_mode == "train":
                self._dequeue_and_enqueue(k)
        elif task == "cross-domain":
            q = self.encode(self.encoder_q, seq_q, mlp = False, ts = ts, mode = "q")
            q = self.projector(q) # project to the joint space of the prototypes
        # prototypical contrast
        if task=="in-domain":
            if cluster_result is not None:  
                proto_labels = []
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
                    
                    # 新作法 : 把其他所有的prototype都視為negative=>只適用於proto number小的狀況
                    pos_proto_id = im2cluster[index]
                    pos_prototypes = prototypes[pos_proto_id]
                    l_pos = torch.einsum('nc,nc->n', [q, pos_prototypes]).unsqueeze(-1)
                    all_proto_id = torch.arange(0, im2cluster.max()+1,dtype = torch.long).repeat(pos_proto_id.size(0),1).cuda() 
                    # indices = torch.arange(all_proto_id.size(1)).expand(all_proto_id.size(0), -1)
                    mask = (all_proto_id != pos_proto_id[:, None]) 
                    neg_proto_id = all_proto_id[mask].view(pos_proto_id.size(0),-1)
                    neg_prototypes = prototypes[neg_proto_id] #[batch_size, num_neg, dim]
                    l_neg = torch.einsum('nc,nkc->nk', [q, neg_prototypes])
                    logits_proto = torch.cat([l_pos, l_neg], dim=1)
                    labels_proto = torch.zeros(logits_proto.shape[0], dtype=torch.long).cuda() 
                    temp_proto = density[torch.cat([pos_proto_id[:, None],neg_proto_id],dim=1)]  
                    logits_proto /= temp_proto
                    proto_labels.append(labels_proto)
                    proto_logits.append(logits_proto)
                return logits, labels, proto_logits, proto_labels, equivariance_loss,time_ssl_task
            else:
                return logits, labels, None, None, equivariance_loss, time_ssl_task
        elif task=="cross-domain":
            if cluster_result is not None:  
                proto_logits = []
                for n, (im2cluster,prototypes) in enumerate(zip(cluster_result['im2cluster'],cluster_result['centroids'])):
                    # compute prototypical logits
                    sim = torch.mm(q,prototypes.t())/self.opt['temp']
                    sim_max, _ = torch.max(sim, dim=1, keepdim=True)
                    sim = sim - sim_max
                    logits_proto = torch.nn.functional.softmax(sim, dim=1)
                    # logits_proto = torch.exp(sim)
                    # z = torch.sum(logits_proto, dim=-1) + 1e-5
                    # logits_proto = logits_proto/z.unsqueeze(1)
                    proto_logits.append(logits_proto)
                return None, None, proto_logits, None, None, None
            else:
                return None, None, None, None, None, None


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