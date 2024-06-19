import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from datetime import datetime
import numpy as np
import ipdb
import random
import copy
import time
from pathlib import Path
from utils import torch_utils
from model.FairCDSR import FairCDSR
from model.MoCo import MoCo
from model.MoCo_new import MoCo_Interest
from utils.MoCo_utils import compute_features, compute_embedding_for_target_user,compute_features_for_I2C
from utils.cluster import run_kmeans
from scipy.spatial import distance
from model.FairCDSR import *
from utils.loader import CustomDataLoader,NonoverlapDataLoader
from utils.torch_utils import *
from utils.collator import CLDataCollator
from model.evaluator import Evaluator
from model.item_generator import Generator
class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)
    def forward(self, x, y):
        return self.cos(x, y) / self.temp
class Trainer(object):
    def __init__(self, opt):
        self.opt =opt
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):  # here should change
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        # self.model.load_state_dict(checkpoint['model'])
        # self.opt = checkpoint['config']
        # self.model.load_state_dict(checkpoint)
        state_dict = checkpoint['model']
        # print(state_dict['lin_X.weight'])
        # if self.opt['main_task'] == "X":
        #     state_dict.pop('lin_X.weight', None)
        #     state_dict.pop('lin_X.bias', None)
        # elif self.opt['main_task'] == "Y":
        #     state_dict.pop('lin_y.weight', None)
        #     state_dict.pop('lin_y.bias', None)
        # state_dict.pop('lin_PAD.weight', None)
        # state_dict.pop('lin_PAD.bias', None)
        self.model.load_state_dict(state_dict, strict=False)
    def save(self, filename):
        params = {
            'model': self.model.state_dict(),
            'config': self.opt,
        }
        torch.save(params, filename)
        print("model saved to {}".format(filename))
    def unpack_batch(self, batch):
        if self.opt["cuda"]:
            try:
                inputs = [Variable(b.cuda()) for b in batch]
            except:
                print("error")
                ipdb.set_trace()
        else:
            inputs = [Variable(b) for b in batch]
        index = inputs[0]
        seq = inputs[1]
        x_seq = inputs[2]
        y_seq = inputs[3]
        position = inputs[4]
        x_position = inputs[5]
        y_position = inputs[6]
        ts_d = inputs[7].to(torch.float32)
        ts_xd = inputs[8].to(torch.float32)
        ts_yd = inputs[9].to(torch.float32)
        ground = inputs[10]
        share_x_ground = inputs[11]
        share_y_ground = inputs[12]
        x_ground = inputs[13]
        y_ground = inputs[14]
        ground_mask = inputs[15]
        share_x_ground_mask = inputs[16]
        share_y_ground_mask = inputs[17]
        x_ground_mask = inputs[18]
        y_ground_mask = inputs[19]
        corru_x = inputs[20]
        corru_y = inputs[21]
        augmented_d = inputs[22]
        augmented_xd = inputs[23]
        augmented_yd = inputs[24]
        gender = inputs[25]
        return index, seq, x_seq, y_seq, position, x_position, y_position, ts_d, ts_xd, ts_yd, ground, share_x_ground, share_y_ground, x_ground, y_ground, ground_mask, share_x_ground_mask, share_y_ground_mask, x_ground_mask, y_ground_mask, corru_x, corru_y, augmented_d, augmented_xd, augmented_yd,gender
    def unpack_batch_valid(self, batch):
        if self.opt["cuda"]:
            inputs = [Variable(b.cuda()) for b in batch]
        else:
            inputs = [Variable(b) for b in batch]
        seq = inputs[0]
        x_seq = inputs[1]
        y_seq = inputs[2]
        position = inputs[3]
        x_position = inputs[4]
        y_position = inputs[5]
        ts_d = inputs[6].to(torch.float32)
        ts_xd = inputs[7].to(torch.float32)
        ts_yd = inputs[8].to(torch.float32)
        X_last = inputs[9]
        Y_last = inputs[10]
        XorY = inputs[11]
        ground_truth = inputs[12]
        neg_list = inputs[13]
        index = inputs[14]
        gender = inputs[15]
        return index, seq, x_seq, y_seq, position, x_position, y_position, ts_d, ts_xd, ts_yd, X_last, Y_last, XorY, ground_truth, neg_list, gender
    def unpack_batch_for_gen(self, batch):
        if self.opt["cuda"]:
            inputs = [Variable(b.cuda()) for b in batch]
        else:
            inputs = [Variable(b) for b in batch]
        index = inputs[0]
        seq = inputs[1]
        position = inputs[2]
        ts_d = inputs[3]
        ground = inputs[4]
        ground_mask = inputs[5]
        masked_d = inputs[6]
        neg_d = inputs[7]
        target_sentence = inputs[8]
        gender = inputs[9]
        return index,seq, position, ts_d, ground,ground_mask ,masked_d, neg_d,target_sentence, gender
class GTrainer(Trainer):
    def __init__(self, opt):
        self.opt = opt
        self.model = Generator(opt,type = self.opt["generate_type"])
        self.CE_criterion = nn.CrossEntropyLoss(ignore_index =self.opt["source_item_num"] + self.opt["target_item_num"] )
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.model.parameters(), opt['lr'])
        if opt['cuda']:
            self.model.cuda()
    def train_batch(self, batch):
        index,seq, position, ts_d, ground, ground_mask ,masked_d, neg_d, target_sentence, gender = self.unpack_batch_for_gen(batch)
        if self.opt['time_encode']:
            mip_pred = self.model(masked_d, position, ts_d)
        else:
            mip_pred = self.model(masked_d, position)
        batch_size, seq_len, item_num = mip_pred.shape
        mlm_predictions = mip_pred.argmax(dim=2)
        flatten_mip_pred = mip_pred.view(-1,item_num)
        flatten_target_sentence = target_sentence.view(batch_size * seq_len)
        if self.opt["generate_type"] == "Y":
            mask = flatten_target_sentence != self.opt["source_item_num"] + self.opt["target_item_num"]
            flatten_target_sentence[mask] = flatten_target_sentence[mask] - self.opt["source_item_num"]
        mip_loss = self.CE_criterion(flatten_mip_pred, flatten_target_sentence)
        self.optimizer.zero_grad()
        mip_loss.backward()
        self.optimizer.step()
        
        return mip_loss.item()
    def train(self ,train_dataloader, val_dataloader):
        global_step = 0
        current_lr = self.opt["lr"]
        format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/epoch), lr: {:.6f}'
        num_batch = len(train_dataloader)
        max_steps =  self.opt['pretrain_epoch'] * num_batch
        print("Start training:")        
        patience =  self.opt["finetune_patience"]
        train_pred_loss = []
        val_pred_loss = []
        for epoch in range(1, self.opt['pretrain_epoch'] + 1):
            epoch_start_time = time.time()
            self.prediction_loss = 0
            self.model.train()
            for _,batch in enumerate(train_dataloader):
                global_step += 1
                loss = self.train_batch(batch)
                self.prediction_loss += loss
                
            duration = time.time() - epoch_start_time
            print(format_str.format(datetime.now(), global_step, max_steps, epoch, \
                                            self.opt['pretrain_epoch'],  self.prediction_loss/num_batch, duration, current_lr))
            print("train mip loss:", self.prediction_loss/num_batch)
            train_pred_loss.append(self.prediction_loss/num_batch)
            if epoch%20==0:
                folder = self.opt['model_save_dir'] +f'{epoch}'
                Path(folder).mkdir(parents=True, exist_ok=True)
                self.save(folder + '/model.pt')
            #validation => 最後
            if epoch%500==0:
                self.model.eval()
                with torch.no_grad():
                    self.val_prediction_loss = 0
                    for i,batch in enumerate(val_dataloader):
                        index, seq, position, ts_d, ground, ground_mask ,masked_d, neg_d, target_sentence, gender = self.unpack_batch_for_gen(batch)
                        mip_pred = self.model(seq,position)
                        batch_size, seq_len, item_num = mip_pred.shape
                        flatten_mip_res = mip_pred.view(-1,item_num)
                        flatten_target_sentence = target_sentence.view(batch_size * seq_len)
                        mip_loss = self.CE_criterion(flatten_mip_res,flatten_target_sentence)
                        self.val_prediction_loss += mip_loss.item()
                    print("-"*50)
                    print(f"Start validation at epoch {epoch}:")
                    print("validation mip loss:", self.val_prediction_loss/len(val_dataloader))
                    print("-"*50)
                    val_pred_loss.append(self.val_prediction_loss/len(val_dataloader))
                    if self.val_prediction_loss/len(val_dataloader) < min(val_pred_loss):
                        patience =  self.opt["finetune_patience"]
                    else:
                        patience -= 1
                        print("Early stop counter:", 5-patience)
                        if patience == 0:
                            print("Early stop at epoch", epoch)
                            self.save(self.opt['model_save_dir'] + '/best_model.pt')
                            break
class CDSRTrainer(Trainer):
    def __init__(self, opt):
        self.opt = opt
        self.pooling = self.opt['pooling']
        self.sim = Similarity(self.opt['temp'])
        if opt["model"] == "FairCDSR":
            self.model = FairCDSR(opt)
        else:
            print("please select a valid model")
            exit(0)
        self.initialize_losses()
        self.init_GCL_setting()
        self.init_GMiT_setting()
        self.initialize_cuda_settings()
        self.configure_optimizers()
        self.evaluator = Evaluator(self.model, self.opt)
    def initialize_cuda_settings(self):
        """ Move models and criteria to CUDA if enabled. """
        if self.opt['cuda']:
            self.cudaify(self.model, self.BCE_criterion, self.CS_criterion, self.CL_criterion,\
                getattr(self, 'group_classifier', None), getattr(self, 'MoCo_Interest', None), getattr(self, 'male_cluster', None), getattr(self, 'female_cluster', None), getattr(self, 'cluster', None))

    def cudaify(self, *args):
        """ Utility to move tensors or modules to CUDA. """
        for arg in args:
            if arg is not None:
                arg.cuda()
    def initialize_losses(self):
        self.prediction_loss = 0
        self.BCE_criterion = nn.BCELoss() # for adversarial loss
        self.CS_criterion = nn.CrossEntropyLoss(reduction='none') # for item prediction loss
        self.CL_criterion = nn.CrossEntropyLoss()
    def init_GMiT_setting(self):
        self.I2C_loss = 0
        self.warmup_for_GMiT = self.opt['warmup_epoch']
        if self.opt['ssl'] in ['GMiT','both']:
            self.MoCo_Interest =MoCo_Interest(self.opt, self.model, T = self.opt['temp'])        
        if self.opt['ssl'] in ['GMiT','both']:
            if self.opt['cluster_mode'] =="separate":
                self.male_cluster = ClusterRepresentation(self.opt,self.opt['hidden_units'], int(self.opt['num_cluster'][2]*(1-self.opt['cluster_ratio'])), topk = self.opt['topk_cluster'])
                self.female_cluster = ClusterRepresentation(self.opt,self.opt['hidden_units'], int(self.opt['num_cluster'][2]*self.opt['cluster_ratio']), topk = self.opt['topk_cluster'])                                                                                             
            elif self.opt['cluster_mode'] == "joint":
                self.cluster = ClusterRepresentation(self.opt,self.opt['hidden_units'], self.opt['num_cluster'][2], topk = self.opt['topk_cluster'])
    def configure_optimizers(self):
        if self.opt['param_group'] : 
            param_name = []
            for name, param in self.model.named_parameters():
                param_name.append(name)
            target_param_name = [s for s in param_name if "encoder_X" in s]
            print("target_param_name:",target_param_name)
            group1 =[p for n, p in self.model.named_parameters() if n not in target_param_name and p.requires_grad]
            group2 =[p for n, p in self.model.named_parameters() if n in target_param_name and p.requires_grad]
            self.optimizer = torch_utils.get_optimizer(self.opt['optim'],
                                                    [{'params': group1, 'lr': self.opt['lr']},
                                                        {'params': group2, 'lr': self.opt['lr']*0.01}],
                                                    self.opt['lr'])
        else:
            if self.opt['training_mode'] == 'joint_learn' and self.opt['ssl'] in ['GMiT','both']: 
                if self.opt['cluster_mode'] =="separate":
                    self.optimizer = torch_utils.get_optimizer(self.opt['optim'], list(self.model.parameters())+list(self.male_cluster.parameters())+list(self.female_cluster.parameters()), self.opt['lr'])
                elif self.opt['cluster_mode'] == "joint":
                    self.optimizer = torch_utils.get_optimizer(self.opt['optim'], list(self.model.parameters())+list(self.cluster.parameters()), self.opt['lr'])
            else:
                self.optimizer = torch_utils.get_optimizer(self.opt['optim'],self.model.parameters(), self.opt['lr'])
    def init_GCL_setting(self):
        self.group_classifier = GenderDiscriminator(self.opt)
        self.g_optimizer = torch_utils.get_optimizer(self.opt['optim'], self.group_classifier.parameters(), self.opt['lr'])
        self.is_group_classifier_pretrained = False
        self.warmup_for_GCL = 5
        self.group_classifier_pretrain_epoch = 10
        self.group_CL_loss = 0
    def unpack_batch_predict(self, batch):
        if self.opt["cuda"]:
            inputs = [Variable(b.cuda()) for b in batch]
        else:
            inputs = [Variable(b) for b in batch]
        seq = inputs[0]
        x_seq = inputs[1]
        y_seq = inputs[2]
        position = inputs[3]
        x_position = inputs[4]
        y_position = inputs[5]
        ts_d = inputs[6].to(torch.float32)
        ts_xd = inputs[7].to(torch.float32)
        ts_yd = inputs[8].to(torch.float32)
        X_last = inputs[9]
        Y_last = inputs[10]
        XorY = inputs[11]
        ground_truth = inputs[12]
        neg_list = inputs[13]
        index = inputs[14]
        gender = inputs[15]
        x_last_3 = inputs[16]
        y_last_3 = inputs[17]
        return index, seq, x_seq, y_seq, position, x_position, y_position, ts_d, ts_xd, ts_yd, X_last, Y_last, XorY, ground_truth, neg_list,gender, x_last_3,y_last_3

    def isnan(self,x):
        return torch.any(torch.isnan(x))
    def my_index_select(self, memory, index):#從memory中選擇index的item
        tmp = list(index.size()) + [-1]
        index = index.reshape(-1)
        ans = torch.index_select(memory, 0, index)
        ans = ans.reshape(tmp)
        return ans
    def get_embedding_for_ssl(self, data, encoder, item_embed, CL_projector, 
                          encoder_causality_mask = False, add_graph_node=False, graph_embedding = None):
        batch_size = data.shape[0]
        device = data.device
        non_zero_mask = (data != (self.opt["source_item_num"] +self.opt["target_item_num"])).long()
        position_id = non_zero_mask.cumsum(dim=1) * non_zero_mask
        if add_graph_node:
            if self.pooling == "bert":
                node_feature = torch.cat((torch.zeros(data.size(0),1,self.opt['hidden_units'],device = device),self.my_index_select(graph_embedding, data[:,1:])),1)
                seqs = item_embed(data) + node_feature #[B*2, seq_len, hidden_units]
            elif self.pooling == "ave":
                seqs = item_embed(data) + self.my_index_select(graph_embedding, data)
            if self.opt['augment_type'] == "dropout":
                seq_feature1 = encoder(data, seqs, position_id, causality_mask = encoder_causality_mask).unsqueeze(1)
                seq_feature2 = encoder(data, seqs, position_id, causality_mask = encoder_causality_mask).unsqueeze(1)
                seq_feature = torch.cat([seq_feature1, seq_feature2], dim=1).view(batch_size*2 ,seq_feature1.size(2),-1)
            else:
                seq_feature = encoder(data, seqs, position_id,  causality_mask = encoder_causality_mask)
        else:
            seqs = item_embed(data)
            seq_feature = encoder(data, seqs, position_id,  causality_mask = encoder_causality_mask)
        if self.pooling == "bert":
            out = seq_feature[:,0,:]
        elif self.pooling == "ave":
            if self.opt['augment_type']=="dropout":
            #     non_zero_mask = non_zero_mask.unsqueeze(1).expand(-1,2,-1).reshape(batch_size*2,-1)
            #     out = torch.sum(seq_feature, dim=1)/torch.sum(non_zero_mask, dim=1).unsqueeze(-1)
            # else:
                out = torch.sum(seq_feature, dim=1)/torch.sum(non_zero_mask, dim=1).unsqueeze(-1)
            
        out = CL_projector(out)
        return out
    def nonoverlap_user_generation(self):
        
        # get non-overlap data
        nonoverlap_dataLoader = NonoverlapDataLoader(self.opt['data_dir'], self.opt['batch_size'],  self.opt)
        nonoverlap_seq_feat, nonoverlap_feat = compute_embedding_for_target_user(self.opt, nonoverlap_dataLoader, self.model, name = 'non-overlap')
        target_gt = [data[3] for data in nonoverlap_dataLoader.all_data]
        target_gt_mask = [data[4] for data in nonoverlap_dataLoader.all_data]        
        # get overlap data
        overlap_dataLoader = CustomDataLoader(self.opt['data_dir'], self.opt['batch_size'], self.opt, -1)
        overlap_seq_feat, _ = compute_embedding_for_target_user(self.opt, overlap_dataLoader, self.model, name = 'overlap')
        # get topk similar user for nonoverlap female data
        sim = nonoverlap_seq_feat@overlap_seq_feat.T
        top_k_values, top_k_indice = torch.topk(sim, 20, dim=1)
        try:
            lookup_dict = {item[0]: item[1:] for item in overlap_dataLoader.all_data}
            lookup_keys = set(lookup_dict.keys())
            top_k_data = [[lookup_dict[idx] for idx in sorted(random.sample(indices, 10))] for indices in top_k_indice.tolist() if all([x in lookup_keys for x in indices])] #從20個中隨機選10個
        except:
            ipdb.set_trace()
        
        #select [required_num] augmented user to be added in each batch 
        required_num = 1000 #number of aumented user to be added in each batch
        if len(top_k_data)<required_num:
            required_num = len(top_k_data)
        selected_idx = random.sample(list(range(len(top_k_data))), required_num)
        selected_top_k_data = [top_k_data[idx] for idx in selected_idx]
        
        augmented_seq_fea = []
        augmented_share_y_ground = []
        augmented_share_y_ground_mask = []
        for query, topk in zip(nonoverlap_seq_feat, selected_top_k_data):
            mixed_seq = [seq[0] for seq in topk] # 1,4,12
            target_seq = [seq[2] for seq in topk]
            position = [seq[3] for seq in topk] 
            share_y_grounds = [seq[11] for seq in topk] 
            # share_y_ground_masks = [seq[16] for seq in topk] 
            value = torch.tensor(mixed_seq)
            key = torch.tensor(target_seq)
            query = query.unsqueeze(0)
            
            if self.opt['cuda']:
                query, key, value = query.cuda(), key.cuda(), value.cuda()
            ts_key = [seq[9] for seq in topk] if self.opt['time_encode'] else None
            ts_value = [seq[7] for seq in topk] if self.opt['time_encode'] else None

            key, _ = get_sequence_embedding(self.opt,key,self.model.encoder_Y,self.model.item_emb_Y, encoder_causality_mask = False, ts = ts_key) #[10, 128] (sequence_embedding, item embedding)
            _, value = get_sequence_embedding(self.opt,value,self.model.encoder,self.model.item_emb, encoder_causality_mask = True, ts = ts_value) #[10, 50, 128]
            weight = torch.softmax(torch.matmul(query, key.T)/math.sqrt(query.size(-1)), dim=-1) #[1, 10]
            aug_mixed_seq = torch.einsum('ij,jkl->ikl', weight, value).squeeze()#[1, 50, 128]
            max_position_id = max([max(s) for s in position])
            aug_position = [0]*(50-max_position_id) + list(range(0,max_position_id+1))[1:]
            
            share_y_grounds = torch.tensor(share_y_grounds)
            indices = torch.randint(share_y_grounds.size(0), (share_y_grounds.size(-1),)) 
            aug_share_y_ground = share_y_grounds[indices, torch.arange(share_y_grounds.size(-1))]    
            aug_share_y_ground_mask =  (aug_share_y_ground != self.opt['target_item_num']).to(torch.int)                          
            augmented_share_y_ground.append(aug_share_y_ground)
            augmented_share_y_ground_mask.append(copy.deepcopy(aug_share_y_ground_mask))
            augmented_seq_fea.append(aug_mixed_seq.clone())
        # nonoverlap_feat = nonoverlap_feat.repeat_interleave(target_num,dim=0)
        # target_gt = torch.tensor(target_gt).repeat_interleave(target_num,dim=0)
        # target_gt_mask = torch.tensor(target_gt_mask).repeat_interleave(target_num,dim=0)
        
        def add_noise(representation): # do perturbation
            noise = torch.randn_like(representation)
            scaled_noise = torch.nn.functional.normalize(noise,dim=2)
            return representation + scaled_noise
        
        nonoverlap_feat = nonoverlap_feat[selected_idx]
        nonoverlap_feat = add_noise(nonoverlap_feat)
        target_gt = torch.tensor(target_gt)[selected_idx]
        target_gt_mask = torch.tensor(target_gt_mask)[selected_idx]
        augmented_seq_fea = add_noise(torch.stack(augmented_seq_fea))        
        augmented_share_y_ground = torch.stack(augmented_share_y_ground)
        augmented_share_y_ground_mask = torch.stack(augmented_share_y_ground_mask)
        if self.opt['cuda']:
            augmented_seq_fea = augmented_seq_fea.cuda()
            augmented_share_y_ground = augmented_share_y_ground.cuda()
            augmented_share_y_ground_mask = augmented_share_y_ground_mask.cuda()
            nonoverlap_feat = nonoverlap_feat.cuda()
            target_gt = target_gt.cuda()
            target_gt_mask = target_gt_mask.cuda()
        
        return augmented_seq_fea, augmented_share_y_ground, augmented_share_y_ground_mask, nonoverlap_feat, target_gt - self.opt['source_item_num'], target_gt_mask
    def mask_correlated_samples(self, batch_size):
        N =batch_size*2
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for  i in range(batch_size):
            mask[i,i+batch_size] = 0
            mask[i+batch_size,i] = 0
        return mask
    def group_CL(self, seq, seq_xd, seq_yd):
        seq = seq.reshape(-1,seq.size(-1))
        seq_xd = seq_xd.reshape(-1,seq_xd.size(-1))
        seq_yd = seq_yd.reshape(-1,seq_yd.size(-1))
        out = get_embedding_for_ssl(self.opt, seq, self.model.encoder, self.model.item_emb, projector = None)
        out_yd = get_embedding_for_ssl(self.opt, seq_yd, self.model.encoder_Y, self.model.item_emb_Y, projector = None)
        out = self.model.CL_projector(out+out_yd)
        # out = out+out_yd
        out = out.reshape(-1,2,out.size(-1))
        z1,z2 = out[:,0,:], out[:,1,:]
        batch_size = z1.size(0)
        N = batch_size*2
        z = torch.cat([z1,z2],dim=0)
        sim = torch.mm(z, z.T)/self.opt['temp']
        pos = torch.cat([torch.diag(sim, z1.size(0)), torch.diag(sim, -z1.size(0))])
        
        if batch_size != self.opt['batch_size']:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_correlated_samples(self.opt['batch_size'])
        neg = sim[mask].reshape(batch_size*2,-1)
        labels = torch.zeros(N).to(pos.device).long()
        logits = torch.cat([pos.unsqueeze(1), neg], dim=1)
        loss = self.CL_criterion(logits, labels)
        return loss
    def GMiT_Kmeans(self, index, mixed_seq, target_seq, ts_d, ts_yd, gender,cluster_result):
        ts_d = ts_d if self.opt['time_encode'] else None
        ts_yd = ts_yd if self.opt['time_encode'] else None
        cluster_result_mixed, cluster_result_male_mixed, cluster_result_female_mixed = cluster_result[0], cluster_result[1], cluster_result[2]
        if self.opt['cluster_mode'] == "joint":
            logits= self.MoCo_Interest(mixed_seq, target_seq, cluster_result = cluster_result_mixed, index = index,ts =ts_d)
            log_probabilities = F.log_softmax(logits[0] / self.opt['temp'], dim=1)
            positive_log_probabilities = log_probabilities[:, :self.opt['topk_cluster']]
            loss = -positive_log_probabilities.mean(dim=1).mean()
            return loss
        elif self.opt['cluster_mode'] == "separate":
            gender = gender[:,0]
            target_male_seq = target_seq[gender==1]
            target_female_seq = target_seq[gender==0]
            mixed_male_seq = mixed_seq[gender==1]
            mixed_female_seq = mixed_seq[gender==0]
            
            male_logits = self.MoCo_Interest(mixed_male_seq, target_male_seq, cluster_result = cluster_result_male_mixed, index = index,ts =ts_d)
            female_logits = self.MoCo_Interest(mixed_female_seq, target_female_seq, cluster_result = cluster_result_female_mixed, index = index,ts =ts_d)
            # if the number of clusters for male and female are equal
            if int(self.opt['num_cluster'][2]*(1-self.opt['cluster_ratio'])) == int(self.opt['num_cluster'][2]*self.opt['cluster_ratio']):
                logits = torch.cat([male_logits[0],female_logits[0]],dim=0)
                log_probabilities = F.log_softmax(logits / self.opt['temp'], dim=1)
                positive_log_probabilities = log_probabilities[:, :self.opt['topk_cluster']]
                loss = -positive_log_probabilities.mean(dim=1).mean()
            else:
                male_log_probabilities = F.log_softmax(male_logits[0] / self.opt['temp'], dim=1)
                male_positive_log_probabilities = male_log_probabilities[:, :self.opt['topk_cluster']]
                male_loss = -male_positive_log_probabilities.mean(dim=1).mean()
                female_log_probabilities = F.log_softmax(female_logits[0] / self.opt['temp'], dim=1)
                female_positive_log_probabilities = female_log_probabilities[:, :self.opt['topk_cluster']]
                female_loss = -female_positive_log_probabilities.mean(dim=1).mean()
                loss = male_loss + female_loss
            return loss      
        else:
            raise ValueError("cluster_mode should be joint or separate")
    def GMiT(self, index, mixed_seq, target_seq, ts_d, ts_yd, gender):
        ts_d = ts_d if self.opt['time_encode'] else None
        ts_yd = ts_yd if self.opt['time_encode'] else None
        if self.opt['cluster_mode'] == "joint":
            mixed_feature = get_embedding_for_ssl(self.opt , mixed_seq, self.model.encoder, self.model.item_emb,ts=ts_d, projector = self.model.interest_projector)
            target_feature = get_embedding_for_ssl(self.opt , target_seq, self.model.encoder_Y, self.model.item_emb_Y,ts=ts_yd,projector=self.model.interest_projector_Y)
            new_cluster, multi_interest = self.cluster(mixed_feature)
            pos = torch.sum(multi_interest*target_feature,dim=1,keepdim=True)
            # pos = torch.einsum('nc,nkc->nk', [target_feature, multi_interest])
            neg = target_feature@new_cluster.T
            logits = torch.cat([pos,neg],dim=1)
            logits = logits/self.opt['temp']
            labels = torch.zeros(logits.shape[0], dtype=torch.long)
            if self.opt['cuda']:
                labels = labels.cuda() 
            loss = self.CL_criterion(logits, labels)
            # log_probabilities = F.log_softmax(logits / self.opt['temp'], dim=1)
            # positive_log_probabilities = log_probabilities[:, :self.opt['topk_cluster']]
            # loss = -positive_log_probabilities.mean(dim=1).mean()
            return loss
        elif self.opt['cluster_mode'] == "separate":
            gender = gender[:,0]
            target_male_seq = target_seq[gender==1]
            target_female_seq = target_seq[gender==0]
            mixed_male_seq = mixed_seq[gender==1]
            mixed_female_seq = mixed_seq[gender==0]
            
            mixed_male_feature = get_embedding_for_ssl(self.opt , mixed_male_seq, self.model.encoder, self.model.item_emb,ts=ts_d, projector = self.model.interest_projector)
            mixed_female_feature = get_embedding_for_ssl(self.opt , mixed_female_seq, self.model.encoder, self.model.item_emb,ts=ts_d, projector = self.model.interest_projector)
            target_male_feature = get_embedding_for_ssl(self.opt , target_male_seq, self.model.encoder_Y, self.model.item_emb_Y,ts=ts_yd,projector=self.model.interest_projector_Y)
            target_female_feature = get_embedding_for_ssl(self.opt , target_female_seq, self.model.encoder_Y, self.model.item_emb_Y,ts=ts_yd,projector=self.model.interest_projector_Y)
            
            new_male_cluster, male_multi_interest = self.male_cluster(mixed_male_feature)
            new_female_cluster, female_multi_interest = self.female_cluster(mixed_female_feature)

            new_male_cluster =torch.nn.functional.normalize(new_male_cluster, dim=-1)
            male_multi_interest = torch.nn.functional.normalize(male_multi_interest, dim=-1)
            new_female_cluster =torch.nn.functional.normalize(new_female_cluster, dim=-1)
            female_multi_interest = torch.nn.functional.normalize(female_multi_interest, dim=-1)
            # male_pos = torch.einsum('nc,nkc->nk', [target_male_feature, male_multi_interest])
            # female_pos = torch.einsum('nc,nkc->nk', [target_female_feature, female_multi_interest])
            male_pos = torch.sum(male_multi_interest*target_male_feature,dim=1,keepdim=True)
            female_pos = torch.sum(female_multi_interest*target_female_feature, dim=1,keepdim=True)
            male_neg = target_male_feature@new_male_cluster.T
            female_neg = target_female_feature@new_female_cluster.T
            male_logits = torch.cat([male_pos, male_neg],dim=1)
            female_logits = torch.cat([female_pos, female_neg],dim=1)
            
            # if the number of clusters for male and female are equal
            if int(self.opt['num_cluster'][2]*(1-self.opt['cluster_ratio'])) == int(self.opt['num_cluster'][2]*self.opt['cluster_ratio']):
                logits = torch.cat([male_logits,female_logits],dim=0)
                logits = logits/self.opt['temp']
                labels = torch.zeros(logits.shape[0], dtype=torch.long)
                if self.opt['cuda']:
                    labels = labels.cuda() 
                loss = self.CL_criterion(logits, labels)
                # log_probabilities = F.log_softmax(logits / self.opt['temp'], dim=1)
                # positive_log_probabilities = log_probabilities[:, :self.opt['topk_cluster']]
                # loss = -positive_log_probabilities.mean(dim=1).mean()
            else:
                male_logits = male_logits/self.opt['temp']
                female_logits = female_logits/self.opt['temp'] 
                male_labels = torch.zeros(male_logits.shape[0], dtype=torch.long)
                female_labels = torch.zeros(female_logits.shape[0], dtype=torch.long)
                if self.opt['cuda']:
                    male_labels = male_labels.cuda() 
                    female_labels = female_labels.cuda()
                male_loss = self.CL_criterion(male_logits, male_labels)
                female_loss = self.CL_criterion(female_logits, female_labels)
                # male_log_probabilities = F.log_softmax(male_logits / self.opt['temp'], dim=1)
                # male_positive_log_probabilities = male_log_probabilities[:, :self.opt['topk_cluster']]
                # male_loss = -male_positive_log_probabilities.mean(dim=1).mean()
                # female_log_probabilities = F.log_softmax(female_logits / self.opt['temp'], dim=1)
                # female_positive_log_probabilities = female_log_probabilities[:, :self.opt['topk_cluster']]
                # female_loss = -female_positive_log_probabilities.mean(dim=1).mean()
                loss = male_loss + female_loss
            return loss      
        else:
            raise ValueError("cluster_mode should be joint or separate")
    
    ### adversarial training for bias free embedding (not used in final experiment)###
    def adversarial_training(self, seq, x_seq, y_seq, gender):
        dis_loss, adv_loss = None, None
        ### update generator ###
        target = gender[:,0].float()
        # fake_target = (target.clone()==0).to(int).float()
        seqs_fea,_ = get_sequence_embedding(self.opt,seq,self.model.encoder,self.model.item_emb, encoder_causality_mask = True)
        if self.opt['main_task'] == "X":
            specific_seq_fea,_ = get_sequence_embedding(self.opt,x_seq,self.model.encoder_X,self.model.item_emb_X, encoder_causality_mask = True)
        elif self.opt['main_task'] == "Y":
            specific_seq_fea,_ = get_sequence_embedding(self.opt,y_seq,self.model.encoder_Y,self.model.item_emb_Y, encoder_causality_mask = True)
        pred = self.group_classifier(seqs_fea + specific_seq_fea).squeeze()
        adv_loss = self.BCE_criterion(pred, target)
        loss = loss - adv_loss    
        ###update discriminator###
        pred = self.group_classifier(seqs_fea.detach() + specific_seq_fea.detach()).squeeze()
        dis_loss = self.BCE_criterion(pred, target)
        dis_loss.backward()
        self.g_optimizer.step()
    def train_batch(self, epoch, batch, i, cluster_result):
        def compute_loss(result, ground, mask, target):
            if target == 'X':
                item_num = self.opt['source_item_num']
            elif target == 'Y':
                item_num = self.opt['target_item_num'] 
            loss = self.CS_criterion(result.reshape(-1, item_num + 1), ground.reshape(-1))
            loss = (loss * mask.reshape(-1)).mean()
            return loss

        def process_task(seqs_fea, x_seqs_fea=None, y_seqs_fea=None, x_only=False, y_only=False):
            if seqs_fea is not None:
                share_result = self.model.lin_X(seqs_fea[:, -used:]) if x_only else self.model.lin_Y(seqs_fea[:, -used:])
                share_pad_result = self.model.lin_PAD(seqs_fea[:, -used:])
                share_trans_result = torch.cat((share_result, share_pad_result), dim=-1)
                specific_fea = x_seqs_fea if x_only else y_seqs_fea
                specific_result = self.model.lin_X(seqs_fea[:,-used:]+specific_fea[:, -used:]) if x_only else self.model.lin_Y(seqs_fea[:,-used:]+specific_fea[:, -used:])
                specific_pad_result = self.model.lin_PAD(specific_fea[:, -used:])
                specific_trans_result = torch.cat((specific_result, specific_pad_result), dim=-1)

                ground = x_ground if x_only else y_ground
                ground_mask = x_ground_mask if x_only else y_ground_mask
                share_ground = share_x_ground if x_only else share_y_ground
                share_ground_mask = share_x_ground_mask if x_only else share_y_ground_mask
                target = 'X' if x_only else 'Y'
                share_loss = compute_loss(share_trans_result, share_ground, share_ground_mask, target)
                specific_loss = compute_loss(specific_trans_result, ground, ground_mask, target)
                loss = share_loss + specific_loss
                self.prediction_loss += (share_loss.item() + specific_loss.item())
            else:
                specific_fea = x_seqs_fea if x_only else y_seqs_fea
                specific_result = self.model.lin_X(specific_fea[:, -used:])
                specific_pad_result = self.model.lin_PAD(specific_fea[:, -used:])
                specific_trans_result = self.concat_results(specific_result, specific_pad_result)
                loss = self.compute_loss(specific_trans_result, x_ground if x_only else y_ground, x_ground_mask if x_only else y_ground_mask)
                self.prediction_loss += loss.item()
            if self.opt['training_mode'] == "joint_learn":
                additional_loss = compute_additional_loss()
                loss =  loss + additional_loss
            return loss
        def compute_additional_loss():
            if self.opt['ssl'] in ['GMiT']:
                loss = I2C_loss
                self.I2C_loss += I2C_loss.item()
            elif self.opt['ssl'] in ['group_CL']:
                loss = group_CL_loss
                self.group_CL_loss += group_CL_loss.item()
            elif self.opt['ssl'] in ['both']:
                loss = I2C_loss * self.opt['lambda_'][1] + group_CL_loss * self.opt['lambda_'][0]
                self.group_CL_loss += group_CL_loss.item()
                self.I2C_loss += I2C_loss.item()
            return loss
        self.model.train()
        self.optimizer.zero_grad()
        index, seq, x_seq, y_seq, position, x_position, y_position, ts_d, ts_xd, ts_yd, ground, share_x_ground, share_y_ground, x_ground, y_ground, ground_mask, share_x_ground_mask, share_y_ground_mask, x_ground_mask, y_ground_mask, corru_x, corru_y, augmented_d, augmented_xd,augmented_yd,gender = self.unpack_batch(batch)
        if self.opt['training_mode'] =="joint_learn":
            if self.opt['ssl'] in ['GMiT','both']:
                if epoch>=self.warmup_for_GMiT:
                    if self.opt['cluster_algo'] == "prototype":
                        I2C_loss = self.GMiT(index, seq, y_seq, ts_d, ts_yd, gender)
                    elif self.opt['cluster_algo'] == "kmeans":
                        I2C_loss = self.GMiT_Kmeans(index, seq, y_seq, ts_d, ts_yd, gender, cluster_result)
                else:
                    I2C_loss = torch.Tensor([0]).cuda() if self.opt['cuda'] else torch.Tensor([0])
            if self.opt['ssl'] in ["both","group_CL"]:
                if self.opt['substitute_mode'] in ['attention_weight','hybrid']:
                    if epoch>=self.warmup_for_GCL:
                        group_CL_loss = self.group_CL(augmented_d,augmented_xd,augmented_yd)
                    else:
                        group_CL_loss = torch.Tensor([0]).cuda() if self.opt['cuda'] else torch.Tensor([0])
                else:
                    group_CL_loss = self.group_CL(augmented_d,augmented_xd,augmented_yd)
        # control if dual or single domain recommendation task and if the main task is X or Y
        if self.opt['domain'] =="single":
            seq = None 
            if self.opt['main_task'] == "X":
                y_seq = None
            elif self.opt['main_task'] == "Y":
                x_seq = None  
        
        model_input = [seq, x_seq, y_seq, position, x_position, y_position]
        if self.opt['time_encode']:
            model_input += [ts_d, ts_xd, ts_yd]
        seqs_fea, x_seqs_fea, y_seqs_fea = self.model(*model_input)
        used = 50
        ground = ground[:,-used:]
        ground_mask = ground_mask[:, -used:]
        share_x_ground = share_x_ground[:, -used:]
        share_x_ground_mask = share_x_ground_mask[:, -used:]
        share_y_ground = share_y_ground[:, -used:]
        share_y_ground_mask = share_y_ground_mask[:, -used:]
        x_ground = x_ground[:, -used:]
        x_ground_mask = x_ground_mask[:, -used:]
        y_ground = y_ground[:, -used:]
        y_ground_mask = y_ground_mask[:, -used:]
        
        '''overlap user generation'''
        aug_data = None
        if epoch > 10 and i==0 and self.opt['data_augmentation']=="user_generation":
            aug_data = self.nonoverlap_user_generation()
        if aug_data is not None:
            augmented_seqs_fea, augmented_share_y_ground, augmented_share_y_ground_mask, target_feat, target_gt, target_gt_mask = aug_data
            print(f"\033[34m{len(augmented_seqs_fea)} Non-overlap user generation\033[0m")
            seqs_fea = torch.cat([seqs_fea,augmented_seqs_fea])
            share_y_ground = torch.cat([share_y_ground,augmented_share_y_ground[:, -used:]],dim=0)
            share_y_ground_mask = torch.cat([share_y_ground_mask,augmented_share_y_ground_mask[:, -used:]],dim=0)
            y_seqs_fea = torch.cat([y_seqs_fea, target_feat])
            y_ground = torch.cat([y_ground, target_gt[:, -used:]],dim=0)
            y_ground_mask = torch.cat([y_ground_mask, target_gt_mask[:, -used:]],dim=0)
        # if self.opt['main_task'] == "dual":
        #     share_x_result =  self.model.lin_X(seqs_fea[:,-used:]) # b * seq * X_num
        #     share_y_result = self.model.lin_Y(seqs_fea[:, -used:])  # b * seq * Y_num
        #     share_pad_result = self.model.lin_PAD(seqs_fea[:, -used:])  # b * seq * 1 #最後一維，即padding，score要是零
        #     share_trans_x_result = torch.cat((share_x_result, share_pad_result), dim=-1)
        #     share_trans_y_result = torch.cat((share_y_result, share_pad_result), dim=-1)

        #     specific_x_result = self.model.lin_X(seqs_fea[:,-used:] + x_seqs_fea[:, -used:])  # b * seq * X_num
        #     specific_x_pad_result = self.model.lin_PAD(x_seqs_fea[:, -used:])  # b * seq * 1
        #     specific_x_result = torch.cat((specific_x_result, specific_x_pad_result), dim=-1)
            
        #     specific_y_result = self.model.lin_Y(seqs_fea[:,-used:] + y_seqs_fea[:, -used:])  # b * seq * Y_num
        #     specific_y_pad_result = self.model.lin_PAD(y_seqs_fea[:, -used:])  # b * seq * 1
        #     specific_y_result = torch.cat((specific_y_result, specific_y_pad_result), dim=-1)

        #     x_share_loss = self.CS_criterion(
        #         share_trans_x_result.reshape(-1, self.opt["source_item_num"] + 1),
        #         share_x_ground.reshape(-1))  # b * seq
        #     y_share_loss = self.CS_criterion(
        #         share_trans_y_result.reshape(-1, self.opt["target_item_num"] + 1),
        #         share_y_ground.reshape(-1))  # b * seq
        #     x_loss = self.CS_criterion(
        #         specific_x_result.reshape(-1, self.opt["source_item_num"] + 1),
        #         x_ground.reshape(-1))  # b * seq
        #     y_loss = self.CS_criterion(
        #         specific_y_result.reshape(-1, self.opt["target_item_num"] + 1),
        #         y_ground.reshape(-1))  # b * seq

        #     x_share_loss = (x_share_loss * (share_x_ground_mask.reshape(-1))).mean() #只取預測x的部分
        #     y_share_loss = (y_share_loss * (share_y_ground_mask.reshape(-1))).mean() #只取預測y的部分
        #     x_loss = (x_loss * (x_ground_mask.reshape(-1))).mean()
        #     y_loss = (y_loss * (y_ground_mask.reshape(-1))).mean()
            
        #     if self.opt['training_mode'] =="joint_learn":
        #         if self.opt['ssl'] in ['GMiT']:
        #             loss = x_share_loss + y_share_loss + x_loss + y_loss + I2C_loss
        #             self.I2C_loss += I2C_loss.item()
        #         elif self.opt['ssl'] in ['group_CL']:
        #             loss = x_share_loss + y_share_loss + x_loss + y_loss + group_CL_loss
        #             self.group_CL_loss += group_CL_loss.item()
        #         elif self.opt['ssl'] in ['both']:
        #             loss = x_share_loss + y_share_loss + x_loss + y_loss + group_CL_loss*self.opt['lambda_'][0] + I2C_loss*self.opt['lambda_'][1] 
        #             self.I2C_loss += I2C_loss.item()
        #             self.group_CL_loss += group_CL_loss.item()
        #     else:
        #         loss = x_share_loss + y_share_loss + x_loss + y_loss
        
            
        #     self.prediction_loss += (x_share_loss.item() + y_share_loss.item() + x_loss.item() + y_loss.item())
        # elif self.opt['main_task'] == "X":
        #     if seqs_fea is not None:
        #         share_x_result =  self.model.lin_X(seqs_fea[:,-used:]) # b * seq * X_num
        #         share_pad_result = self.model.lin_PAD(seqs_fea[:, -used:])  # b * seq * 1 #最後一維，即padding，score要是零
        #         share_trans_x_result = torch.cat((share_x_result, share_pad_result), dim=-1)
            
        #         specific_x_result = self.model.lin_X(seqs_fea[:,-used:] + x_seqs_fea[:, -used:])  # b * seq * X_num
        #         specific_x_pad_result = self.model.lin_PAD(x_seqs_fea[:, -used:])  # b * seq * 1
        #         specific_x_result = torch.cat((specific_x_result, specific_x_pad_result), dim=-1)
        #         x_share_loss = self.CS_criterion(
        #         share_trans_x_result.reshape(-1, self.opt["source_item_num"] + 1),
        #         share_x_ground.reshape(-1))  # b * seq
        #         x_share_loss = (x_share_loss * (share_x_ground_mask.reshape(-1))).mean() #只取預測x的部分
        #     else:
        #         specific_x_result = self.model.lin_X(x_seqs_fea[:, -used:])
        #         specific_x_pad_result = self.model.lin_PAD(x_seqs_fea[:, -used:])  # b * seq * 1
        #         specific_x_result = torch.cat((specific_x_result, specific_x_pad_result), dim=-1)
            
        #     x_loss = self.CS_criterion(
        #         specific_x_result.reshape(-1, self.opt["source_item_num"] + 1),
        #         x_ground.reshape(-1))  # b * seq
        #     x_loss = (x_loss * (x_ground_mask.reshape(-1))).mean()
            
        #     if self.opt['training_mode'] =="joint_learn":
        #         if self.opt['ssl'] in ['GMiT']:
        #             loss = x_share_loss + x_loss + I2C_loss
        #             self.I2C_loss += I2C_loss.item()
        #         elif self.opt['ssl'] in ['group_CL']:
        #             loss = x_share_loss + x_loss + group_CL_loss
        #             self.group_CL_loss += group_CL_loss.item()
        #         elif self.opt['ssl'] in ['both']:
        #             loss = x_share_loss + x_loss + I2C_loss*self.opt['lambda_'][1] + group_CL_loss*self.opt['lambda_'][0]
        #             self.I2C_loss += I2C_loss.item()
        #             self.group_CL_loss += group_CL_loss.item()
        #     else:
        #         if seqs_fea is not None:
        #             loss = x_share_loss + x_loss
        #             self.prediction_loss += (x_share_loss.item() + x_loss.item())
        #         else:
        #             loss = x_loss
        #             self.prediction_loss += x_loss.item()
        # elif self.opt['main_task'] == "Y":
        #     if seqs_fea is not None:                                  
        #         share_y_result =  self.model.lin_Y(seqs_fea[:,-used:]) # b * seq * X_num
        #         share_pad_result = self.model.lin_PAD(seqs_fea[:, -used:])  # b * seq * 1 #最後一維，即padding，score要是零
        #         share_trans_y_result = torch.cat((share_y_result, share_pad_result), dim=-1)
        #         specific_y_result = self.model.lin_Y(seqs_fea[:,-used:] + y_seqs_fea[:, -used:])  # b * seq * X_num
        #         specific_y_pad_result = self.model.lin_PAD(y_seqs_fea[:, -used:])  # b * seq * 1
        #         specific_y_result = torch.cat((specific_y_result, specific_y_pad_result), dim=-1)
        #         y_share_loss = self.CS_criterion(
        #         share_trans_y_result.reshape(-1, self.opt["target_item_num"] + 1),share_y_ground.reshape(-1))
        #         y_share_loss = (y_share_loss * (share_y_ground_mask.reshape(-1))).mean()  # 只取預測y的部分
        #     else:
        #         specific_y_result = self.model.lin_Y(y_seqs_fea[:, -used:])
        #         specific_y_pad_result = self.model.lin_PAD(y_seqs_fea[:, -used:])
        #         specific_y_result = torch.cat((specific_y_result, specific_y_pad_result), dim=-1)
        #     y_loss = self.CS_criterion(
        #         specific_y_result.reshape(-1, self.opt["target_item_num"] + 1),
        #         y_ground.reshape(-1))  # b * seq
        #     y_loss = (y_loss * (y_ground_mask.reshape(-1))).mean()
            
        #     if self.opt['training_mode'] =="joint_learn":
        #         if self.opt['ssl'] =='GMiT':
        #             loss = y_share_loss + y_loss + I2C_loss
        #             self.I2C_loss += I2C_loss.item()
        #         elif self.opt['ssl'] =='group_CL':
        #             loss = y_share_loss + y_loss + group_CL_loss
        #             self.group_CL_loss += group_CL_loss.item()
        #         elif self.opt['ssl'] =='both':
        #             loss = y_share_loss + y_loss + group_CL_loss*self.opt['lambda_'][0] + I2C_loss*self.opt['lambda_'][1] 
        #             self.I2C_loss += I2C_loss.item()
        #             self.group_CL_loss += group_CL_loss.item()
        #     else:
        #         if seqs_fea is not None:
        #             loss = y_share_loss + y_loss
        #             self.prediction_loss += (y_share_loss.item() + y_loss.item())
        #         else:
        #             loss = y_loss
        #             self.prediction_loss += y_loss.item()
        
        # self.adversarial_training(seq, x_seq, y_seq, gender)
        if self.opt['main_task'] == "X":
            loss = process_task(seqs_fea, x_seqs_fea = x_seqs_fea, x_only=True)
        elif self.opt['main_task'] == "Y":
            loss = process_task(seqs_fea, y_seqs_fea = y_seqs_fea, y_only=True)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    def pretrain_group_classifier(self):
        self.group_classifier.train()
        for epoch in range(self.group_classifier_pretrain_epoch):
            dataloader = CustomDataLoader(self.opt['data_dir'], self.opt['batch_size'], self.opt, evaluation = -1, collate_fn  = None, generator = self.generator, model = self.model, balanced =True)            
            batch_dis_loss = 0
            batch_dis_acc = 0
            batch_predict_male_num = 0
            batch_gt_male_num = 0
            initial_params_A = [param.clone() for param in self.model.parameters()]
            for _,batch in enumerate(dataloader):  
                self.g_optimizer.zero_grad()
                batch = self.unpack_batch(batch)
                seq, gender = batch[1], batch[-1]
                seqs_fea = get_item_embedding_for_sequence(self.opt, seq, self.model.encoder, self.model.item_emb, self.model.CL_projector,encoder_causality_mask = False,cl =False)
                pred, _ = self.group_classifier(seqs_fea.detach())
                pred = pred.squeeze()
                dis_loss = self.BCE_criterion(pred,gender[:,0].float())
                dis_loss.backward()
                self.g_optimizer.step()
                acc = (torch.round(pred)== gender[:,0]).sum()/len(pred)
                batch_dis_loss += dis_loss.item()
                batch_dis_acc += acc.item()
                batch_predict_male_num += (torch.round(pred)==1).sum().item()
                batch_gt_male_num += gender[:,0].sum().item()
                for param, initial_param in zip(self.model.parameters(), initial_params_A):
                    if not torch.equal(param, initial_param):
                        print("Warning: Parameters of model A are changing!")
                        break
            print("-"*20)
            print(f"Epoch {epoch+1} discriminator pretrain loss:",batch_dis_loss/len(dataloader))
            print(f"Epoch {epoch+1} discriminator accuracy:",batch_dis_acc/len(dataloader))
            print(f"Number of real male:",batch_gt_male_num/len(dataloader))
            print(f"Number of predicted male:",batch_predict_male_num/len(dataloader))
            print("-"*20)
    def train(self, epoch, train_dataloader, valid_dataloader, test_dataloader, file_logger):
        global_step = 0    
        current_lr = self.opt["lr"]
        format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/epoch), lr: {:.6f}'
        num_batch = len(train_dataloader)
        max_steps =  self.opt['num_epoch'] * num_batch
        print("Start training:")
        
        begin_time = time.time()
        dev_score_history=[0]
        patience =  self.opt["finetune_patience"]
        train_pred_loss = []
        val_pred_loss = []
        
        for epoch in range(1, self.opt['num_epoch'] + 1):
            # if self.opt['ssl'] in ['group_CL','both'] and self.opt['substitute_mode'] in ["attention_weight","hybrid"] and epoch>=1 \
            #     and not self.is_group_classifier_pretrained: #10epoch之後才pretrain discriminator
            #     self.pretrain_group_classifier()
            #     self.is_group_classifier_pretrained = True
                    #     collator = CLDataCollator(self.opt,-1, self.generator[2],self.group_classifier,self.model)
                    #     train_dataloader = CustomDataLoader(self.opt['data_dir'], self.opt['batch_size'], self.opt, evaluation = -1, collate_fn  = collator, generator = None)
                    #     self.is_group_classifier_pretrained = True
                    # else:
                    #     collator = CLDataCollator(self.opt,-1, self.generator[2],self.group_classifier,self.model)
                    #     train_dataloader = CustomDataLoader(self.opt['data_dir'], self.opt['batch_size'], self.opt, evaluation = -1, collate_fn  = collator, generator = None)
            train_loss = 0
            epoch_start_time = time.time()
            self.prediction_loss = 0
            self.adv_loss = 0
            self.dis_loss = 0
            self.group_CL_loss = 0
            self.I2C_loss = 0
            cluster_result_male_mixed = None
            cluster_result_female_mixed = None
            cluster_result_mixed = None
            cluster_result = None
            
            if self.opt['ssl'] in ["GMiT","both"] and self.opt['cluster_algo'] == "kmeans":
                if epoch >= self.warmup_for_GMiT:
                    if self.opt['cluster_mode']=="separate":
                        # compute cluster results
                        male_mixed_feature = compute_features(self.opt, train_dataloader, self.model, gender = "male")
                        male_mixed_feature[torch.norm(male_mixed_feature,dim=1)>1.5] /= 2 #account for the few samples that are computed twice  
                        male_mixed_feature = male_mixed_feature.numpy()
                        cluster_result_male_mixed = run_kmeans(male_mixed_feature, self.opt, gender = "male") 
                        female_mixed_feature = compute_features(self.opt, train_dataloader, self.model, gender = "female")
                        female_mixed_feature[torch.norm(female_mixed_feature,dim=1)>1.5] /= 2 #account for the few samples that are computed twice  
                        female_mixed_feature = female_mixed_feature.numpy()
                        cluster_result_female_mixed = run_kmeans(female_mixed_feature, self.opt, gender = "female")
                    else:
                        mixed_feature = compute_features(self.opt, train_dataloader, self.model, gender = None)
                        mixed_feature[torch.norm(mixed_feature,dim=1)>1.5] /= 2     
                        mixed_feature = mixed_feature.numpy()
                        cluster_result_mixed = run_kmeans(mixed_feature, self.opt, gender = None) 
                    cluster_result = (cluster_result_mixed, cluster_result_male_mixed, cluster_result_female_mixed)
                
                
            if self.opt['ssl'] in ['group_CL','both']:
                if self.opt['substitute_mode'] in ["attention_weight","hybrid"]:
                    if epoch>=self.warmup_for_GCL and not self.is_group_classifier_pretrained:
                        self.pretrain_group_classifier()
                        self.is_group_classifier_pretrained = True
                    if self.is_group_classifier_pretrained:
                        collator = CLDataCollator(self.opt, eval=-1, mixed_generator = self.generator[2], model = self.model, attribute_predictor=self.group_classifier)
                    else:
                        collator = None
                else: #subtitute_mode = "DGIR" or "AGIR"
                    collator = CLDataCollator(self.opt, eval=-1, mixed_generator = self.generator[2])
            else:
                collator = None
                
            # 若使用item augmentation，則每個epoch都要重新生成DataLoader
            generator = self.generator if self.opt['data_augmentation']=="item_augmentation"  else None
            train_dataloader = CustomDataLoader(self.opt['data_dir'], self.opt['batch_size'], self.opt, evaluation = -1,collate_fn=collator, generator=generator)
            for i,batch in enumerate(train_dataloader):
                global_step += 1
                loss = self.train_batch(epoch, batch, i, cluster_result = cluster_result)
                train_loss+=loss
            duration = time.time() - epoch_start_time
            print(format_str.format(datetime.now(), global_step, max_steps, epoch, \
                                            self.opt['num_epoch'], train_loss/num_batch, duration, current_lr))
            print("I2C loss:", self.I2C_loss/num_batch)
            print("Group_CL loss:", self.group_CL_loss/num_batch)
            train_pred_loss.append(self.prediction_loss/num_batch)
            # eval model
            if epoch%5 == 0:
                if self.evaluator.run_val(epoch,valid_dataloader):
                    break
        loss_save_path = f"./loss/{self.opt['data_dir']}/{self.opt['id']}/{self.opt['seed']}"
        print(f"write loss into path {loss_save_path}")
        Path(loss_save_path).mkdir(parents=True, exist_ok=True)
        np.save(f"{loss_save_path}/val_pred_loss_{self.opt['main_task']}.npy", np.array(self.evaluator.val_pred_loss))
        np.save(f"{loss_save_path}/train_pred_loss.npy", np.array(train_pred_loss))
        
        # save X, Y domain Encoder weights for embedding plotting
        model_save_dir = f"./models/{self.opt['data_dir']}/{self.opt['id']}/{self.opt['seed']}"
        print(f"write models into path {model_save_dir} for embedding plotting")
        Path(model_save_dir).mkdir(parents=True, exist_ok=True)
        self.save(f"{model_save_dir}/model.pt")
        
        print("Start testing:")
        best_test,best_test_male,best_test_female = self.evaluator.run_test(test_dataloader, file_logger)
        return best_test,best_test_male,best_test_female