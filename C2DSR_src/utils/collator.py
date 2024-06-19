from argparse import Namespace
import torch
from utils.augmentation import *
import ipdb
from sklearn.preprocessing import StandardScaler
import pandas as pd
import json
from utils.torch_utils import get_item_embedding_for_sequence
class CLDataCollator:
    def __init__(self, opt, eval, mixed_generator,attribute_predictor=None,model=None) -> None:
        self.opt = opt
        self.eval = eval
        self.mixed_generator = mixed_generator #for all substitute_mode
        self.attribute_predictor = attribute_predictor  # for substitute_mode = "attention_weight" or "hybrid"
        self.model = model # for substitute_mode = "attention_weight" or "hybrid"
        with open(f"./fairness_dataset/{self.opt['dataset']}/{self.opt['data_dir']}/male_IR.json","r") as f:
            self.male_IR = json.load(f)
        with open(f"./fairness_dataset/{self.opt['dataset']}/{self.opt['data_dir']}/female_IR.json","r") as f:
            self.female_IR = json.load(f)
    def __call__(self, batch):
        if self.eval == -1: #for training
            augmented_d = self.augment_new(batch[1],all_ts = batch[7], gender=batch[-1]) #[B,2,max_len]
            # augmented_xd = self.augment(batch[26])
            # augmented_yd = self.augment(batch[27])   
            augmented_xd, augmented_yd,augmented_d = self.decompose(augmented_d)  
            return (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]), torch.LongTensor(batch[2]), torch.LongTensor(batch[3]),torch.LongTensor(batch[4]), torch.LongTensor(batch[5]), torch.LongTensor(batch[6]), torch.LongTensor(batch[7]),torch.LongTensor(batch[8]), torch.LongTensor(batch[9]), torch.LongTensor(batch[10]), torch.LongTensor(batch[11]), torch.LongTensor(batch[12]), torch.LongTensor(batch[13]), torch.LongTensor(batch[14]), torch.LongTensor(batch[15]), torch.LongTensor(batch[16]), torch.LongTensor(batch[17]),\
                        torch.LongTensor(batch[18]),torch.LongTensor(batch[19]),torch.LongTensor(batch[20]),torch.LongTensor(batch[21]), torch.LongTensor(augmented_d),torch.LongTensor(augmented_xd),torch.LongTensor(augmented_yd),torch.LongTensor(batch[25]))
        elif self.eval == 2: #for  validation
            augmented_d = self.augment(batch[0])
            augmented_xd = self.augment(batch[1])
            augmented_yd = self.augment(batch[2])
            return (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]), torch.LongTensor(batch[2]), torch.LongTensor(batch[3]),torch.LongTensor(batch[4]), torch.LongTensor(batch[5]), torch.tensor(batch[6]), torch.tensor(batch[7]),torch.tensor(batch[8]), torch.LongTensor(batch[9]),\
                    torch.LongTensor(batch[10]),torch.LongTensor(batch[11]),torch.LongTensor(batch[12]),torch.LongTensor(batch[13]), torch.LongTensor(batch[14]), torch.LongTensor(batch[15]), torch.LongTensor(batch[16]), torch.LongTensor(batch[17]), torch.LongTensor(batch[18]), torch.LongTensor(augmented_d), torch.LongTensor(augmented_xd), torch.LongTensor(augmented_yd))
    def decompose(self,seq):
        # for i in range(2):
        augmented_xd = []
        augmented_yd = []
        augmented_d = []
        for s1,s2 in zip(seq[0],seq[1]):
            xd_1 = [j for j in s1 if j < self.opt['source_item_num']]
            yd_1 = [j for j in s1 if j >= self.opt['source_item_num'] and j != self.opt['source_item_num'] + self.opt['target_item_num']]
            xd_2 = [j for j in s2 if j < self.opt['source_item_num']]
            yd_2 = [j for j in s2 if j >= self.opt['source_item_num'] and j != self.opt['source_item_num'] + self.opt['target_item_num']]
            if len(xd_1) == 0 or len(yd_1) == 0 or len(xd_2) == 0 or len(yd_2) == 0:
                continue
            xd_1 = [self.opt['source_item_num'] + self.opt['target_item_num']]*(self.opt['maxlen']-len(xd_1))+xd_1
            yd_1 = [self.opt['source_item_num'] + self.opt['target_item_num']]*(self.opt['maxlen']-len(yd_1))+yd_1
            xd_2 = [self.opt['source_item_num'] + self.opt['target_item_num']]*(self.opt['maxlen']-len(xd_2))+xd_2
            yd_2 = [self.opt['source_item_num'] + self.opt['target_item_num']]*(self.opt['maxlen']-len(yd_2))+yd_2
            augmented_xd.append([xd_1,xd_2])
            augmented_yd.append([yd_1,yd_2])
            augmented_d.append([s1,s2])
        return augmented_xd, augmented_yd,augmented_d
    def generate(self,seq, positions, timestamp, female_IR):
        
        torch_seq = torch.LongTensor(np.array(seq)) #[X,max_len]
        mask = torch_seq == self.opt['itemnum']
        torch_position = torch.LongTensor(positions) #[X,max_len]
        torch_ts = torch.LongTensor(timestamp) #[X,max_len]
        self.mixed_generator.eval()
        if self.opt['cuda']:
            self.mixed_generator =  self.mixed_generator.cuda()
            torch_seq = torch_seq.cuda()
            torch_position = torch_position.cuda()
            torch_ts = torch_ts.cuda()
        with torch.no_grad():
            if self.opt['time_encode']:
                seq_fea =  self.mixed_generator(torch_seq,torch_position,torch_ts) #[X,max_len,item_num]
            else:
                seq_fea =  self.mixed_generator(torch_seq,torch_position)
            target_fea = seq_fea[mask]
            target_fea /= torch.max(target_fea, dim=1, keepdim=True)[0]
            probabilities = torch.nn.functional.softmax(target_fea, dim=1)
            if self.opt['substitute_mode'] in ["attention_weight","DGIR","AGIR","hybrid"]:
                num_sample = 10
                sampled_indices = torch.multinomial(probabilities, num_sample, replacement=False).squeeze() #[X,10]
                ### insert highest female interaction ratio item from 10 item### 
                mapping = torch.zeros(self.opt['source_item_num']+self.opt['target_item_num'],dtype=torch.float32)
                if self.opt['cuda']:
                    mapping = mapping.cuda()
                for k in self.female_IR.keys():
                    mapping[int(k)] = self.female_IR[k]
                selected_idx_index = mapping[sampled_indices].argmax(dim=-1)
                sampled_indices = sampled_indices[torch.arange(len(sampled_indices)),selected_idx_index]
            elif self.opt['substitute_mode'] == "random":
                # sampled_indices = torch.multinomial(probabilities, 1, replacement=False).squeeze()
                num_sample = 10
                sampled_indices = torch.multinomial(probabilities, num_sample, replacement=False).squeeze()
                selected_idx_index = torch.randint(0, num_sample, (len(sampled_indices),))
                sampled_indices = sampled_indices[torch.arange(len(sampled_indices)),selected_idx_index]
            # if type =="Y":
            #     sampled_indices = sampled_indices + self.opt['source_item_num']
            torch_seq[mask] = sampled_indices
            new_seq = torch_seq.tolist()
        
        # new_seq = [(g,[[x,ts] for x,ts in zip(sublist,timestamp) if x != self.opt['source_item_num'] + self.opt['target_item_num']]) for g, sublist, timestamp in zip(gender, new_seq, torch_ts.tolist())]
        return new_seq
    def augment_new(self,item_seqs, all_ts, gender):
        
        ratio = self.opt['substitute_ratio']
        augmented_d = []
        for _ in range(2):
            if self.attribute_predictor is not None and self.model is not None and self.opt['substitute_mode'] in ["attention_weight","hybrid"]:
                # gender_mask = torch.tensor(gender)[:,0]==1
                item_seqs = torch.tensor(item_seqs).clone().detach()
                item_seqs = item_seqs.cuda() if self.opt['cuda'] else item_seqs
                initial_params_A = [param.clone() for param in self.attribute_predictor.parameters()]
                self.attribute_predictor.eval()
                self.model.eval()
                with torch.no_grad():
                    seqs_fea = get_item_embedding_for_sequence(self.opt,item_seqs,self.model.encoder,self.model.item_emb, self.model.CL_projector,encoder_causality_mask = False,cl =False)
                    pred, weight = self.attribute_predictor(seqs_fea)
                    pred, weight = pred.detach(), weight.detach()
                    # gender_mask = torch.round(pred).squeeze()==1
                    gender_mask = torch.tensor(gender)[:,0]==1
                    weight = weight.squeeze()
                    item_seqs= item_seqs.squeeze()
                    mask = (item_seqs!=(self.opt['source_item_num'] + self.opt['target_item_num']))
                    item_seq_len = mask.int().sum(dim=-1)
                    #將padding item的attention weight設為-inf
                    filtered_weight = torch.where(mask, weight, torch.tensor(float('-inf')).to(weight.device))

                    #取最高的一半 attention weight 的indices
                    sorted_indices = torch.argsort(filtered_weight, dim=1,descending=True)
                    male_sorted_indices = sorted_indices[gender_mask]
                    female_sorted_indices = sorted_indices[~gender_mask]
                    male_sorted_idx = [indices[:item_seq_len[i]//2] for i,indices in enumerate(male_sorted_indices.tolist())]
                    female_sorted_idx = [indices[item_seq_len[i]//2:item_seq_len[i]] for i,indices in enumerate(female_sorted_indices.tolist())]
                    male_item_seqs = item_seqs[gender_mask]
                    female_item_seqs = item_seqs[~gender_mask]
                    new_positions = []
                    new_seqs = []
                    def get_new_seq(sorted_idx,item_seqs,gender_mask,gender):
                        sub_weight = weight[gender_mask]
                        # ipdb.set_trace()
                        max_len = max([len(i) for i in sorted_idx]) 
                        #取最高的一半 attention weight，長度不足的部分補-inf(確保過softmax後機率為0不會被選到)            
                        values = torch.tensor([w[i].tolist() + [float('-inf')]*(max_len-len(i))for w,i in zip(sub_weight,sorted_idx)]).to(sub_weight.device)
                        # ipdb.set_trace()
                        if self.opt['substitute_mode'] == "hybrid":
                            IR_val = []
                            for i,item_seq in enumerate(item_seqs):
                                item_seq = item_seq[sorted_idx[i]]
                                if gender==1:
                                    item_IR = {item.item():self.male_IR[str(item.item())] if str(item.item()) in self.male_IR.keys() else 0 for item in item_seq}
                                else:
                                    item_IR = {item.item():self.female_IR[str(item.item())] if str(item.item()) in self.female_IR.keys() else 0 for item in item_seq}
                                value = list(item_IR.values())
                                IR_val.append(value+[float('-inf')]*(max_len-len(value)))
                            IR_val = torch.tensor(IR_val).to(sub_weight.device)
                            mask = IR_val==float('-inf')
                            if gender==1:  
                                # ranking
                                IR_val_rank = IR_val.argsort().argsort()
                                values_rank = values.argsort().argsort()
                                values = (values_rank+1)*(IR_val_rank+1)
                                
                                #harmonic mean
                                # values = 2 * (IR_val * values) / (IR_val + values + 1e-6)
                                
                            else:
                                # ranking
                                IR_val[mask] = float('nan')
                                values[mask] = float('nan')
                                IR_val_rank = (IR_val*-1).argsort().argsort()
                                values_rank = (values*-1).argsort().argsort()
                                values = (values_rank+1)*(IR_val_rank+1)
                                
                                #harmonic mean
                                # epsilon = 1e-6  # Small number to avoid division by zero
                                # Transform A and B by taking the reciprocal
                                # A_prime = 1 / (IR_val + epsilon)
                                # B_prime = 1 / (values + epsilon)
                                # values = 2 * (A_prime * B_prime) / (A_prime + B_prime + 1e-6)
                            values = values.float()
                            values[mask] = float('inf')
                            min_ = values.min(dim=1, keepdim=True).values
                            values[mask] = float('-inf')
                            max_ = values.max(dim=1, keepdim=True).values
                            values = (values-min_)/(max_-min_+1e-9)
                        #sample item from the highest 50% attention weight items
                        probabilities = torch.nn.functional.softmax(values, dim=1)
                        # num_substitute = [int(item_seq_len[i] * ratio).item() if int(item_seq_len[i] * ratio) <= item_seq_len[i]//2 else item_seq_len[i]//2  for i in range(probabilities.shape[0])]
                        # selected_idx = []
                        # try:
                        # # Iterate over each probability distribution and its corresponding index
                        #     for i, p in enumerate(probabilities):

                        #         # Get the sorted indices for the i-th element and move them to the same device as 'weight'
                        #         sorted_indices = torch.tensor(sorted_idx[i]).to(weight.device)
                                
                        #         # Calculate the number of elements to sample, ensuring at least 1 element is selected
                        #         num_samples = max(round(((item_seq_len[i] // 2) * ratio).item()), 1)
                                
                        #         # Sample 'num_samples' indices from the probability distribution 'p' without replacement
                        #         sampled_indices = torch.multinomial(p, num_samples, replacement=False)
                        #         sampled_indices = sampled_indices[sampled_indices<len(sorted_indices)]
                        #         # Convert the sampled indices to a list and store them in 'selected_idx'
                        #         selected_idx.append(sorted_indices[sampled_indices].tolist())
                        # except:
                        #     ipdb.set_trace()
                        try:
                            selected_idx = [torch.tensor(sorted_idx[i]).to(sub_weight.device)[torch.multinomial(p, max(round((len(sorted_idx[i])*ratio)),1) if (p!=0).sum()>round((len(sorted_idx[i])*ratio), 1) else (p!=0).sum(), replacement=False)].tolist() for i,p in enumerate(probabilities)]
                        except:
                            ipdb.set_trace()
                        selected_item = [seq[idx].cpu().detach().numpy() for seq,idx in zip(item_seqs,selected_idx)]
                        item_seqs = item_seqs.clone().cpu().detach().numpy()
                        for i,seq in enumerate(item_seqs):
                            try:
                                substitute_idxs = np.where(seq ==selected_item[i][:, None])[1] #more than one item
                            except:
                                substitute_idxs = np.where(item_seqs ==selected_item[i])[0] #only one item
                            seq[substitute_idxs] = self.opt['itemnum']
                            position = [0]*(self.opt['maxlen']-item_seq_len[i]) + list(range(item_seq_len[i]+1)[1:])[-self.opt['maxlen']:]
                            new_seq = seq[-self.opt['maxlen']:]
                            new_seqs.append(new_seq)
                            new_positions.append(position)
                    if male_sorted_idx:
                        get_new_seq(male_sorted_idx,male_item_seqs, gender_mask, 1)
                    if female_sorted_idx:
                        get_new_seq(female_sorted_idx,female_item_seqs, ~gender_mask, 0)
                for param, initial_param in zip(self.attribute_predictor.parameters(), initial_params_A):
                    if not torch.equal(param, initial_param):
                        print("Warning: Parameters of model A are changing!")
                        break
            else:
                new_seqs = []
                new_positions = []
                for i in range(len(item_seqs)):
                    item_seq = np.array(item_seqs[i].copy())
                    item_seq_len = len(item_seq[item_seq!=(self.opt['source_item_num'] + self.opt['target_item_num'])])
                    if self.opt['substitute_mode'] in ["AGIR","DGIR"]:
                        # choose the highest 50% male IR items or lowest 50% female IR items
                        if self.opt['substitute_mode']=="AGIR":
                            item_IR = {item:self.male_IR[str(item)] if str(item) in self.male_IR.keys() else 0 for item in item_seq}
                            pair = dict(sorted(item_IR.items(), key=lambda item: item[1], reverse=True)[:item_seq_len//2])
                        elif self.opt['substitute_mode']=="DGIR":
                            item_IR = {item:self.female_IR[str(item)] if str(item) in self.female_IR.keys() else 0 for item in item_seq}
                            pair = dict(sorted(item_IR.items(), key=lambda item: item[1], reverse=True)[item_seq_len//2:])    
                        
                        values = np.array(list(pair.values()))
                        # sample item from the highest 50% male IR items or lowest 50% female IR items
                        probabilities = np.exp(values - np.max(values)) / np.sum(np.exp(values - np.max(values)))
                        size = max(round(item_seq_len//2 * ratio), 1)
                        if size > len(values):
                            size = len(values)
                        try:
                            sampled_pos = np.random.choice(len(values), size=size, replace=False, p=probabilities)
                        except:
                            ipdb.set_trace()
                        selected_item = [k for i,(k,v) in enumerate(pair.items()) if i in sampled_pos]
                        substitute_idxs = np.where(item_seq == np.array(selected_item)[:, None])[1]
                    elif self.opt['substitute_mode'] =="random":
                        substitute_idxs = np.random.choice(item_seq_len, max(int(item_seq_len * ratio), 1), replace=False)
                    else:
                        raise ValueError("substitute_mode should be attention_weight or AGIR, DGIR, hybrid")
                    for index in substitute_idxs:
                        item_seq[index] = self.opt['itemnum']
                    position = [0]*(self.opt['maxlen']-item_seq_len) + list(range(item_seq_len+1)[1:])[-self.opt['maxlen']:]
                    new_seq = item_seq.tolist()[-self.opt['maxlen']:]
                    xd = [j for j in new_seq if j < self.opt['source_item_num']]
                    yd = [j for j in new_seq if j >= self.opt['source_item_num'] and j != self.opt['source_item_num'] + self.opt['target_item_num']]
                    # if len(xd) == 0 or len(yd) == 0:
                    #     skip_id.append(i)
                    #     continue
                    new_seqs.append(new_seq)
                    new_positions.append(position)
            augmented_seq= self.generate(new_seqs, new_positions, all_ts, self.female_IR)
            augmented_d.append(augmented_seq)
            # augmented_xd.append([[i for i in seq if i < self.opt['source_item_num']] for seq in new_seq])
            # augmented_yd.append([[i for i in seq if i >= self.opt['source_item_num']] for seq in new_seq])
        return augmented_d
    def augment(self,item_seqs, all_ts, gender):
        ratio = self.opt['substitute_ratio']
        augmented_d = []
        for _ in range(2):
            new_seqs = []
            new_positions = []
            for i in range(len(item_seqs)):
                item_seq = np.array(item_seqs[i].copy())
                ts= all_ts[i]
                # item_IR = {item:male_IR[str(item)] if str(item) in male_IR.keys() else 0 for item in item_seq}
                item_IR = {item:self.female_IR[str(item)] if str(item) in self.female_IR.keys() else 0 for item in item_seq}
                item_seq_len = len(item_seq[item_seq!=(self.opt['source_item_num'] + self.opt['target_item_num'])])
                ###find male item to substitute with attention weight###
                if self.attribute_predictor is not None and self.model is not None and self.opt['substitute_mode'] == "attention_weight":
                    item_seq = torch.tensor(item_seq).unsqueeze(0)
                    item_seq = item_seq.cuda() if self.opt['cuda'] else item_seq
                    seqs_fea = get_item_embedding_for_sequence(self.opt,item_seq,self.model.encoder,self.model.item_emb, self.model.CL_projector,encoder_causality_mask = False,cl =False)
                    _, weight = self.attribute_predictor(seqs_fea)
                    weight = weight.squeeze()
                    item_seq= item_seq.squeeze()
                    mask = item_seq!=(self.opt['source_item_num'] + self.opt['target_item_num'])
                    idx = torch.nonzero(mask)[0]
                    sorted_idx = torch.argsort(weight[idx:])[-item_seq_len//2:]#取最高的一半 attention weight
                    values = weight[idx:][sorted_idx] 
                    #sample item from the highest 50% attention weight items
                    probabilities = torch.nn.functional.softmax(values, dim=0)
                    sampled_pos = torch.multinomial(probabilities, max(round(((item_seq_len[i]//2)*ratio).item()), 1), replacement=False).squeeze()
                    selected_idx = sorted_idx[sampled_pos]
                    selected_item = item_seq[idx:][selected_idx].cpu().detach().numpy()
                    item_seq = item_seq.cpu().detach().numpy()
                    try:
                        substitute_idxs = np.where(item_seq ==selected_item[:, None])[1] #more than one item
                    except:
                        substitute_idxs = np.where(item_seq ==selected_item)[0] #only one item
                elif self.opt['substitute_mode'] =="IR":
                ###find male item to substitute with IR###
                    # choose the highest 50% male IR items
                    # pair = dict(sorted(item_IR.items(), key=lambda item: item[1], reverse=True)[:item_seq_len//2]) #取male IR最高的一半
                    pair = dict(sorted(item_IR.items(), key=lambda item: item[1], reverse=True)[item_seq_len//2:]) #取female IR最低的一半
                    values = np.array(list(pair.values()))
                    # sample item from the highest 50% male IR items
                    probabilities = np.exp(values - np.max(values)) / np.sum(np.exp(values - np.max(values)))
                    size = max(int(item_seq_len * ratio), 1)
                    if size > len(values):
                        size = len(values)
                    try:
                        sampled_pos = np.random.choice(len(values), size=size, replace=False, p=probabilities)
                    except:
                        ipdb.set_trace()
                    selected_item = [k for i,(k,v) in enumerate(pair.items()) if i in sampled_pos]
                    substitute_idxs = np.where(item_seq == np.array(selected_item)[:, None])[1]
                elif self.opt['substitute_mode'] =="random":
                    substitute_idxs = np.random.choice(item_seq_len, max(int(item_seq_len * ratio), 1), replace=False)
                else:
                    raise ValueError("substitute_mode should be attention_weight or IR")
                for index in substitute_idxs:
                    item_seq[index] = self.opt['itemnum']
                position = [0]*(self.opt['maxlen']-item_seq_len) + list(range(item_seq_len+1)[1:])[-self.opt['maxlen']:]
                new_seq = item_seq.tolist()[-self.opt['maxlen']:]
                xd = [j for j in new_seq if j < self.opt['source_item_num']]
                yd = [j for j in new_seq if j >= self.opt['source_item_num'] and j != self.opt['source_item_num'] + self.opt['target_item_num']]
                # if len(xd) == 0 or len(yd) == 0:
                #     skip_id.append(i)
                #     continue
                new_seqs.append(new_seq)
                new_positions.append(position)
            augmented_seq= self.generate(new_seqs, new_positions, ts, self.female_IR)
            augmented_d.append(augmented_seq)
            # augmented_xd.append([[i for i in seq if i < self.opt['source_item_num']] for seq in new_seq])
            # augmented_yd.append([[i for i in seq if i >= self.opt['source_item_num']] for seq in new_seq])
        return augmented_d
class GDataCollator:
    def __init__(self, opt) -> None:
        self.opt = opt
        
    def __call__(self, batch):
        seq,target_sentences, masked_seqs, neg_seqs = self.augment(batch[1])    
        return (torch.LongTensor(batch[0]), torch.LongTensor(seq), torch.LongTensor(batch[2]), torch.LongTensor(batch[3]),\
                torch.LongTensor(batch[4]),torch.LongTensor(batch[5]),torch.LongTensor(masked_seqs),torch.LongTensor(neg_seqs),torch.LongTensor(target_sentences),torch.LongTensor(batch[6]))
                        
    def augment(self,item_seqs):
        max_len = self.opt["maxlen"]
        target_sentences = []
        masked_seqs = []
        neg_seqs = []
        d = []
        for seq in item_seqs:
            target_sentence = []
            masked_seq = []
            neg_seq = []
            #確保至少mask一個X或Y item
            last_x = [s for s in seq if s < self.opt["source_item_num"]][-1]
            x_idx = seq.index(last_x)
            last_y = [s for s in seq if s >= self.opt["source_item_num"]][-1]
            y_idx = seq.index(last_y)
            for i in seq:
                if self.opt["generate_type"] == "X":
                    condition = i < self.opt["source_item_num"]
                    
                elif self.opt["generate_type"] == "Y":
                    condition = (i >= self.opt["source_item_num"]) and (i!= self.opt["source_item_num"] + self.opt["target_item_num"])
                else:
                    condition = i!= self.opt["source_item_num"] + self.opt["target_item_num"]
                if condition and (random.random() < self.opt['mask_prob']):
                    masked_seq.append(self.opt["source_item_num"] + self.opt["target_item_num"]+1)
                    neg = random.sample(set(range(self.opt['source_item_num']+self.opt['target_item_num'])) - set(seq),1)[0]
                    neg_seq.append(neg)
                    target_sentence.append(i)
                else:
                    masked_seq.append(i)
                    neg_seq.append(i)
                    target_sentence.append(self.opt["source_item_num"] + self.opt["target_item_num"])
            if self.opt["generate_type"] == "mixed":
                masked_seq[-1] = self.opt["source_item_num"] + self.opt["target_item_num"]+1
                neg_seq[-1] = random.sample(set(range(self.opt['source_item_num']+self.opt['target_item_num'])) - set(seq),1)[0]
                target_sentence[-1] = seq[-1]
            elif self.opt["generate_type"] == "X":
                masked_seq[x_idx] = self.opt["source_item_num"] + self.opt["target_item_num"]+1
                neg_seq[x_idx] = random.sample(set(range(self.opt['source_item_num']+self.opt['target_item_num'])) - set(seq),1)[0]
                target_sentence[x_idx] = seq[x_idx]
            elif self.opt["generate_type"] == "Y":
                masked_seq[y_idx] = self.opt["source_item_num"] + self.opt["target_item_num"]+1
                neg_seq[y_idx] = random.sample(set(range(self.opt['source_item_num']+self.opt['target_item_num'])) - set(seq),1)[0]
                target_sentence[y_idx] = seq[y_idx]
            
            masked_seq = [self.opt["source_item_num"] + self.opt["target_item_num"]]*(max_len - len(masked_seq)) + masked_seq
            neg_seq = [self.opt["source_item_num"] + self.opt["target_item_num"]]*(max_len - len(neg_seq)) + neg_seq
            target_sentence = [self.opt["source_item_num"] + self.opt["target_item_num"]]*(max_len - len(target_sentence)) + target_sentence
            seq = [self.opt["source_item_num"] + self.opt["target_item_num"]]*(max_len - len(seq)) + seq
            d.append(seq)
            target_sentences.append(target_sentence)
            masked_seqs.append(masked_seq)
            neg_seqs.append(neg_seq)
        return d,target_sentences, masked_seqs, neg_seqs
