
import json
import random
import torch
import numpy as np
import codecs
import copy
import ipdb
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from utils.augmentation import *
from utils.MoCo_utils import compute_embedding_for_female_mixed_user
import pandas as pd 
class VanillaDataLoader(DataLoader): # Base class for all data loaders
    def __init__(self, filename, batch_size, opt):
        self.batch_size = batch_size
        self.opt = opt
        self.filename  = filename
    def read_item(self, fname):
        with codecs.open(fname, "r", encoding="utf-8") as fr:
            item_num = [int(d.strip()) for d in fr.readlines()[:2]]
        return item_num
    def read_train_data(self, train_file):
        with codecs.open(train_file, "r", encoding="utf-8") as infile:
            train_data = []
            for id, line in enumerate(infile.readlines()):
                if id<2:
                    continue
                res = []
                line = line.strip()
                gender = list(map(int, line.split()[1]))[0]
                i_t = [list(map(int,tmp.split("|"))) for tmp in line.split()[2:]]
                data = (gender,i_t)
                train_data.append(data)
        return train_data
    def __len__(self):
        return len(self.data)
    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
class CustomDataLoader(VanillaDataLoader):
    """
    Args:
        filename (str): The path to the file.
        batch_size (int): The size of each data batch.
        opt: Configuration options or parameters.
        evaluation (bool): -1 : train, 2 : valid, 1 : test
        collate_fn (callable, optional): apply data augmentation to batches for GCL
        generator (tuple of generators, optional): A tuple containing three generators for source, target, and mixed data.
        model: for user representation generation
        balanced (bool, optional): Whether to balance the number of users between males and females
    """
    def __init__(self, filename, batch_size, opt, evaluation, collate_fn = None, generator = None, model = None,balanced =False):
        super().__init__(filename, batch_size, opt)
        self.eval = evaluation
        self.collate_fn = collate_fn
        self.model = model
        self.balanced = balanced
        if generator is not None:
            self.source_generator = generator[0]
            self.target_generator = generator[1]
            self.mixed_generator = generator[2]
        
        self.opt["maxlen"] = 50
        # ************* item_id *****************
        opt["source_item_num"], opt["target_item_num"] = self.read_item(f"./fairness_dataset/{self.opt['dataset']}/" + filename + "/train.txt")
        opt['itemnum'] = opt['source_item_num'] + opt['target_item_num'] + 1
        # ************* sequential data *****************
        source_train_data = f"./fairness_dataset/{self.opt['dataset']}/" + filename + f"/train.txt"
        source_valid_data = f"./fairness_dataset/{self.opt['dataset']}/" + filename + f"/valid.txt"
        source_test_data = f"./fairness_dataset/{self.opt['dataset']}/" + filename + f"/test.txt"
        
        if evaluation < 0:
            self.train_data = self.read_train_data(source_train_data)
            if (self.opt['data_augmentation']=="user_generation" and self.model is not None) or self.balanced: # for group classifier pretraining
                self.overlap_user_generation()
            if (self.opt['data_augmentation']=="item_augmentation" and generator is not None) and not self.balanced:
                self.item_generate_female()
            data = self.preprocess()
            self.num_examples = len(data)
            self.female_data = [d for d in data if d[-1][0]==0]
            self.male_data = [d for d in data if d[-1][0]==1]
            self.all_data = data
            print("Number of train_data:",len(data))
        elif evaluation == 2:
            self.test_data = self.read_test_data(source_valid_data)
            data = self.preprocess_for_predict()
        else :
            self.test_data = self.read_test_data(source_test_data)
            data = self.preprocess_for_predict()
            print("Number of test_data:",len(data))
        
        # shuffle for training
        if evaluation == -1:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
            if batch_size > len(data):
                batch_size = len(data)
                self.batch_size = batch_size
            if len(data)%batch_size != 0:
                data += data[:batch_size]
            data = data[: (len(data)//batch_size) * batch_size]
        else :
            batch_size = 256
            
        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        
    def overlap_user_generation(self):
        print("*"*50)
        print("\033[33mOverlap user generation\033[0m")
        female_seq = [d for d in self.train_data if d[0]==0]
        overlap_seq_feat, _ = compute_embedding_for_female_mixed_user(self.opt, female_seq, self.model, name = 'overlap')
        sim = overlap_seq_feat@overlap_seq_feat.T #dot product
        # sim = torch.nn.CosineSimilarity(dim=1)(overlap_seq_feat,overlap_seq_feat) #cosine similarity                         
        _, top_k_indice = torch.topk(sim, 20, dim=1)
        n_repeat = 1
        augmented_seq= []
        for _ in range(n_repeat):
            lookup_dict = { i: seq[1] for i, seq in enumerate(female_seq)}
            top_k_data = [[lookup_dict[idx] for idx in sorted(random.sample(indices, 10))] for indices in top_k_indice.tolist()] #從20個中隨機選10個
            # try:
            for topk in top_k_data:
                item_seq = [[self.opt["source_item_num"] + self.opt["target_item_num"]]*(self.opt["maxlen"] - len(seq)) + [i_t[0] for i_t in seq] for seq in topk]
                item_seq = torch.tensor(item_seq)
                if self.opt['cuda']:
                    item_seq = item_seq.cuda()
                    
                ### pick similar sequential item ###
                # embedding = self.model.item_emb(item_seq)# [10, 50, 128]
                # vectors_i = embedding[:, :-1, :].permute(1,0,2)    # [49, 10, 128]
                # vectors_i_plus_1 = embedding[:, 1:, :].permute(1,2,0)  # [49, 128, 10]
                # sim = vectors_i@vectors_i_plus_1 #[49, 10, 10] 
                # topk_idx = torch.topk(sim,3, dim=2)[1] #[49, 10, 5] 
                # # all_idx = torch.argmax(sim, dim=2) #[49, 10] 
                # cur = random.randint(0, item_seq.size(0)-1)
                # cur = 0
                # selected_idx = [cur]
                # for idx in topk_idx:
                #     i = random.randint(0, 2)
                #     selected_idx.append(idx[cur,i].item())
                #     cur = idx[cur,i].item()
                # item_seq = item_seq[selected_idx, torch.arange(item_seq.size(-1))] 
                
                ### random pick item per position ###
                indices = torch.randint(item_seq.size(0), (item_seq.size(-1),)) 
                item_seq = item_seq[indices, torch.arange(item_seq.size(-1))]  
                
                new_item_seq = item_seq[item_seq!=self.opt["source_item_num"] + self.opt["target_item_num"]]
                if len([s for s in new_item_seq if s >= self.opt["source_item_num"]]) <3:
                    continue
                ts = [i_t[1] for seq in topk for i_t in seq]
                sorted_ts = sorted(ts)
                new_ts = random.sample(sorted_ts, len(new_item_seq))
                new_ts = sorted(new_ts)
                new_data = [0,[[i,t] for i, t in zip(new_item_seq.tolist(), new_ts)]]
                augmented_seq.append(new_data)
                if len(augmented_seq)+len(female_seq)==len([s for s in self.train_data if s[0]==1]):
                    break
            if len(augmented_seq)+len(female_seq)==len([s for s in self.train_data if s[0]==1]):
                break

        self.train_data = self.train_data + augmented_seq
        print("Number of female augmented sequence:",len(augmented_seq))
        print("Number of female sequence:",len([s for s in self.train_data if s[0]==0]))
        print("Number of male sequence:",len([s for s in self.train_data if s[0]==1]))
        print("*"*50)
    def generate(self, seq, positions, timestamp, gender, type):
        """
        Generate new items for the input sequences according to assigned type.
        """
        with open(f"./fairness_dataset/{self.opt['dataset']}/{self.opt['data_dir']}/item_IF.json", "r") as f:
            item_if = json.load(f)
        # Choose the correct generator based on the type
        if type == "X":
            generator = self.source_generator
        elif type == "Y":
            generator = self.target_generator
        elif type == "mixed":
            generator = self.mixed_generator

        torch_seq = torch.LongTensor(seq)  # [X, max_len]
        all_mask = torch_seq == self.opt['itemnum']
        torch_position = torch.LongTensor(positions)  # [X, max_len]
        torch_ts = torch.LongTensor(timestamp)  # [X, max_len]

        generator.eval()
        eval_batch_size = 8192

        if self.opt['cuda']:
            generator = generator.cuda()
            torch_seq = torch_seq.cuda()
            torch_position = torch_position.cuda()
            torch_ts = torch_ts.cuda()
            all_mask = all_mask.cuda()

        new_seqs = []
        with torch.no_grad():
            for i in range(0, len(torch_seq), eval_batch_size):
                batch_seq = torch_seq[i:i+eval_batch_size]
                batch_position = torch_position[i:i+eval_batch_size]
                batch_ts = torch_ts[i:i+eval_batch_size]
                batch_mask = all_mask[i:i+eval_batch_size]

                # Generate sequence features
                if self.opt['time_encode']:
                    seq_fea = generator(batch_seq, batch_position, batch_ts)
                else:
                    seq_fea = generator(batch_seq, batch_position)

                target_fea = seq_fea[batch_mask]
                target_fea /= torch.max(target_fea, dim=1, keepdim=True)[0]
                probabilities = torch.nn.functional.softmax(target_fea, dim=1)
                probabilities = torch.clamp(probabilities, min=0)
                probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)

                sampled_indices = torch.multinomial(probabilities, 5, replacement=False).squeeze()  # [X, 10]

                # Insert high interaction fairness score item
                mapping = torch.zeros(self.opt['source_item_num'] + self.opt['target_item_num'], dtype=torch.float32)
                if self.opt['cuda']:
                    mapping = mapping.cuda()

                for k in item_if.keys():
                    mapping[int(k)] = item_if[k]

                random_idx = torch.randint(0, 3, (len(sampled_indices),))
                selected_idx_index = mapping[sampled_indices].topk(k=3, dim=-1)[1][torch.arange(len(sampled_indices)), random_idx]

                sampled_indices = sampled_indices[torch.arange(len(sampled_indices)), selected_idx_index]
                if type == "Y":
                    sampled_indices = sampled_indices + self.opt['source_item_num']

                batch_seq[batch_mask] = sampled_indices
                new_seq = batch_seq.tolist()
                new_seqs.extend(new_seq)

        new_seqs = [
            (g, [[x, ts] for x, ts in zip(sublist, timestamp) if x != self.opt['source_item_num'] + self.opt['target_item_num']])
            for g, sublist, timestamp in zip(gender, new_seqs, torch_ts.tolist())
        ]

        return new_seqs
    def item_generate_female(self):
        print("\033[33mitem generation for female\033[0m")
        female_seq = [d for d in self.train_data if d[0]==0]
        # ipdb.set_trace()
        male_seq = [d for d in self.train_data if d[0]==1]
        new_seqs = []
        timestamp = []
        positions = []
        alpha = self.opt['alpha']
        for gender,seq in female_seq:
            #確保mixed sequence不全是source item
            min_,max_ = 3, int(alpha*len(seq))
            if max_<=min_:
                min_ = max_
            inserted_item_num = torch.randint(min_, max_+1, (1,))
            if all([x[0] < self.opt['source_item_num'] for x in seq]):
                continue
            # idxs = random.choices(list(range(0, len(seq))), k = self.opt['generate_num']) #random insertion
            insert_idxs = np.argsort(np.abs(np.diff(np.array([s[1] for s in seq]))))[-inserted_item_num :] + 1 # interval insertion
            new_seq = copy.deepcopy(seq)
            for index in insert_idxs:
                if index == len(new_seq)-1:
                    new_seq.append([self.opt['itemnum'],new_seq[index][1]])
                    continue
                t1 = new_seq[index-1][1]
                t2 = new_seq[index][1]
                new_ts = int((t1+t2)/2)
                new_seq.insert(index, [self.opt['itemnum'],new_ts])
            position = list(range(len(new_seq)+1))[1:]
            ts = [0] * (self.opt["maxlen"] - len(new_seq)) + [t for i,t in new_seq]
            new_seq = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (self.opt["maxlen"] - len(new_seq)) + [ i for i,t in new_seq ]
            position = [0] * (self.opt["maxlen"] - len(position)) + position
            # ipdb.set_trace()
            new_seqs.append(new_seq[-self.opt['maxlen']:])
            positions.append(position[:self.opt['maxlen']])
            timestamp.append(ts[-self.opt['maxlen']:])
        gender = [0]*len(new_seqs)
        new_female_seq = self.generate(new_seqs, positions, timestamp, gender, type = "mixed")
        
        self.train_data = new_female_seq + male_seq
    def item_generate_all(self):
        with open(f"./fairness_dataset/{self.opt['dataset']}/{self.filename}/average_sequence_length.json","r")  as f:
            avg_length = json.load(f)
        source_flag = 1 if avg_length['source_male'] < avg_length['source_female'] else 0
        target_flag = 1 if avg_length['target_male'] < avg_length['target_female'] else 0
        source_item_inserted_num = abs(avg_length['source_male'] -avg_length['source_female']) if abs(avg_length['source_male'] -avg_length['source_female'])>0 else 1
        target_item_inserted_num = abs(avg_length['target_male'] -avg_length['target_female']) if abs(avg_length['target_male'] -avg_length['target_female'])>0 else 1
        if source_flag and  target_flag:
            male_insert_type = "both"
            female_insert_type = None
        elif source_flag:
            male_insert_type = "X"
            female_insert_type = "Y"
        elif target_flag:
            male_insert_type = "Y"
            female_insert_type ="X"
        else:
            male_insert_type = None
            female_insert_type = "both"
        print("\033[33mitem generation\033[0m")
        female_seq = [d for d in self.train_data if d[0]==0]
        male_seq = [d for d in self.train_data if d[0]==1]
        source_l, source_positions, source_timestamp = [], [], []
        target_l, target_positions, target_timestamp = [], [], []
        both_l, both_positions, both_timestamp = [], [], []
        no_augment_data = []
        genders = []
        max_item_num =7
        min_item_num = 1
        for gender, seq in self.train_data:
            #確保mixed sequence不全是source item
            if all([x[0] < self.opt['source_item_num'] for x in seq]):
                continue
            insert_type = male_insert_type if gender == 1 else female_insert_type
            if insert_type == "X" :
                # source_item_seq_len = len([i for i,t in seq if i<self.opt['source_item_num']])
                # if gender == 1:
                #     max_item_num = int(source_item_seq_len/2)-1 if source_item_seq_len>3 else 2
                # else:    
                #     max_item_num = int(source_item_seq_len/2) if source_item_seq_len>3 else 2
                item_inserted_num = source_item_inserted_num-1 if gender == 1 else source_item_inserted_num
                sampled_item_inserted_num = torch.clamp(torch.poisson(torch.tensor([item_inserted_num], dtype = torch.float32)),min=min_item_num, max = max_item_num).int()
            elif insert_type == "Y":
                # target_item_seq_len = len([i for i,t in seq if i>=self.opt['source_item_num']])
                # if gender == 1:
                #     max_item_num = int(target_item_seq_len/2)-1 if target_item_seq_len>3 else 2
                # else:
                #     max_item_num = int(target_item_seq_len/2) if target_item_seq_len>3 else 2
                item_inserted_num = target_item_inserted_num-1 if gender == 1 else target_item_inserted_num
                sampled_item_inserted_num = torch.clamp(torch.poisson(torch.tensor([item_inserted_num], dtype = torch.float32)),min=min_item_num, max = max_item_num).int()
            elif insert_type == "both":
                # max_item_num = int(len(seq)/2) if len(seq)>3 else 2
                if source_item_inserted_num + target_item_inserted_num >max_item_num:
                    sampled_item_inserted_num = max_item_num-1 if gender == 1 else max_item_num
                else:
                    sampled_item_inserted_num = source_item_inserted_num + target_item_inserted_num-1 if gender == 1 else source_item_inserted_num + target_item_inserted_num
            else:
                no_augment_data.append([gender,seq])
                continue
            insert_idxs = np.argsort(np.abs(np.diff(np.array([s[1] for s in seq]))))[-sampled_item_inserted_num :] + 1
            # insert_idxs = random.choices(list(range(0, len(seq))), k = sampled_item_inserted_num)
            new_seq = copy.deepcopy(seq)
            for index in insert_idxs:
                if index == len(new_seq)-1:
                    new_seq.append([self.opt['itemnum'],new_seq[index][1]])
                    continue
                t1 = new_seq[index-1][1]
                t2 = new_seq[index][1]
                new_ts = int((t1+t2)/2)
                new_seq.insert(index, [self.opt['itemnum'],new_ts])
            if len(new_seq) < self.opt['maxlen']:
                position = list(range(len(new_seq)+1))[1:]
                ts = [0] * (self.opt["maxlen"] - len(new_seq)) + [t for i,t in new_seq]
                new_seq = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (self.opt["maxlen"] - len(new_seq)) + [i for i,t in new_seq]
                position = [0] * (self.opt["maxlen"] - len(position)) + position
            else:
                position = list(range(len(new_seq)+1))[1:]
                ts = [t for i,t in new_seq]
                new_seq = [i for i,t in new_seq][-self.opt['maxlen']:]
                position = position[:self.opt['maxlen']]
                ts = ts[-self.opt['maxlen']:]
            genders.append(gender)
            if (gender==0 and female_insert_type =="X") or (gender==1 and male_insert_type =="X"): 
                source_l.append([new_seq])
                source_positions.append(position)
                source_timestamp.append(ts)
            elif (gender==0 and female_insert_type =="Y") or (gender==1 and male_insert_type =="Y"):
                target_l.append([new_seq])
                target_positions.append(position)
                target_timestamp.append(ts)
            else:
                both_l.append([new_seq])
                both_positions.append(position)
                both_timestamp.append(ts)
        
        # if female_insert_type is not None:
        #     for gender,seq in female_seq:
        #         #確保mixed sequence不全是source item
        #         if all([x[0] < self.opt['source_item_num'] for x in seq]):
        #             continue
        #         idxs = random.choices(list(range(0, len(seq))), k = self.opt['generate_num'])
        #         new_seq = copy.deepcopy(seq)
        #         for index in idxs:
        #             if index == len(new_seq)-1:
        #                 new_seq.append([self.opt['itemnum'],new_seq[index][1]])
        #                 continue
        #             t1 = new_seq[index-1][1]
        #             t2 = new_seq[index][1]
        #             new_ts = int((t1+t2)/2)
        #             new_seq.insert(index, [self.opt['itemnum'],new_ts])
        #         position = list(range(len(new_seq)+1))[1:]
        #         ts = [0] * (self.opt["maxlen"] - len(new_seq)) + [t for i,t in new_seq]
        #         new_seq = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (self.opt["maxlen"] - len(new_seq)) + [i for i,t in new_seq]
        #         position = [0] * (self.opt["maxlen"] - len(position)) + position
        #         l.append(new_seq)
        #         positions.append(position)
        #         timestamp.append(ts)
        new_seq1, new_seq2, new_seq3 = [], [], []
        if source_l:
            new_seq1 = self.generate(source_l, source_positions, source_timestamp,genders, type = "X")
        if target_l:
            new_seq2 = self.generate(target_l, target_positions, target_timestamp,genders, type = "Y")
        if both_l:
            new_seq3 = self.generate(both_l, both_positions, both_timestamp,genders, type ="mixed")
        self.train_data = new_seq1 + new_seq2 + new_seq3 + no_augment_data
        # print("Number of source item sequnece detected:",count)
    def read_test_data(self, test_file):
        with codecs.open(test_file, "r", encoding="utf-8") as infile:
            test_data = []
            deleted_test_data_num = 0
            for id, line in enumerate(infile.readlines()):
                if id<2:
                    continue
                res = []
                line = line.strip()
                gender = list(map(int, line.split()[1]))[0]
                i_t = [list(map(int,tmp.split("|"))) for tmp in line.split()[2:]]
                res = (gender, i_t)
                res_2 = []
                for r in res[1]:
                    res_2.append(r)
                    
                target_idx = -1
                if all([x[0] < self.opt['source_item_num'] for x in res[1]]) or all([x[0] >= self.opt['source_item_num'] for x in res[1]]): # test data has no X/Y domain item
                    deleted_test_data_num+=1
                    continue
                for idx in range(len(res[1])-1,0,-1):
                    if self.opt['main_task']=="X":
                        if res[1][idx][0]<self.opt["source_item_num"]:
                            target_idx = idx
                            break
                    elif self.opt['main_task']=="Y":
                        if res[1][idx][0]>=self.opt["source_item_num"]:
                            target_idx = idx
                            break
                # ipdb.set_trace()
                if target_idx < 3: # test data is too short to predict
                    deleted_test_data_num += 1
                    continue
                if self.opt['main_task']=="X":
                    test_data.append([res_2[:target_idx], 0, res[1][target_idx][0], res[0]])
                elif self.opt['main_task']=="Y":
                    test_data.append([res_2[:target_idx], 1, res[1][target_idx][0], res[0]])
                # if res[1][-1][0] >= self.opt["source_item_num"]: # denoted the corresponding validation/test entry , 若為y domain的item_id
                #     test_data.append([res_2, 1, res[1][-1][0], res[0]]) #[整個test sequence, 1, 最後一個item_id, gender]
                # else :
                #     test_data.append([res_2, 0, res[1][-1][0], res[0]])
        print("Number of deleted test data:",deleted_test_data_num)
        return test_data
    def encode_time_features(self, timestamps):
        datetimes = pd.to_datetime(timestamps, unit='s')
        times_of_day = datetimes.hour + datetimes.minute / 60
        days_of_week = datetimes.weekday
        days_of_year = datetimes.dayofyear
        times_of_day_sin = np.sin(2 * np.pi * times_of_day / 24)
        times_of_day_cos = np.cos(2 * np.pi * times_of_day / 24)
        days_of_week_sin = np.sin(2 * np.pi * days_of_week / 7)
        days_of_week_cos = np.cos(2 * np.pi * days_of_week / 7)
        days_of_year_sin = np.sin(2 * np.pi * days_of_year / 365)
        days_of_year_cos = np.cos(2 * np.pi * days_of_year / 365)

        time_features = np.vstack((times_of_day_sin, times_of_day_cos,
                                days_of_week_sin, days_of_week_cos,
                                days_of_year_sin, days_of_year_cos)).T

        scaler = StandardScaler()
        time_features = scaler.fit_transform(time_features)
        return time_features
    def preprocess_for_predict(self):
        processed=[]
        for index, d in enumerate(self.test_data): # the pad is needed! but to be careful. [res[0], res_2, 1, res[1][-1]]
            gender = d[-1]
            position = list(range(len(d[0])+1))[1:]
            xd = []
            xcnt = 1
            x_position = []
            ts_xd = []

            yd = []
            ycnt = 1
            y_position = []
            ts_yd = []
            for item,ts in d[0]:
                
                if item < self.opt["source_item_num"]:
                    xd.append(item)
                    x_position.append(xcnt)
                    xcnt += 1
                    yd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
                    y_position.append(0)
                    ts_xd.append(ts)
                    ts_yd.append(0)
                else:
                    xd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
                    x_position.append(0)
                    yd.append(item)
                    y_position.append(ycnt)
                    ycnt += 1
                    ts_xd.append(0)
                    ts_yd.append(ts)
                
            ts_d = [t for i,t in d[0]]
            seq = [item for item,ts in d[0]]
            if len(d[0]) < self.opt["maxlen"]:
                position = [0] * (self.opt["maxlen"] - len(d[0])) + position
                x_position = [0] * (self.opt["maxlen"] - len(d[0])) + x_position
                y_position = [0] * (self.opt["maxlen"] - len(d[0])) + y_position

                xd = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (self.opt["maxlen"] - len(d[0])) + xd
                yd = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (self.opt["maxlen"] - len(d[0])) + yd
                seq = [self.opt["source_item_num"] + self.opt["target_item_num"]]*(self.opt["maxlen"] - len(d[0])) + seq
                
                ts_xd = [0] * (self.opt["maxlen"] - len(ts_xd)) + ts_xd
                ts_yd = [0] * (self.opt["maxlen"] - len(ts_yd)) + ts_yd
                ts_d = [0]*(self.opt["maxlen"] - len(ts_d)) + ts_d
                gender = [gender]*self.opt["maxlen"]
            x_last = -1
            for id in range(len(x_position)):
                id += 1
                if x_position[-id]:
                    x_last = -id
                    break
            x_last_3 = []
            c = 0
            for id in range(len(x_position)):
                id += 1
                if x_position[-id]:
                    c+=1
                    x_last_3.insert(0,-id)
                    if c==3:
                        break
            y_last = -1
            for id in range(len(y_position)):
                id += 1
                if y_position[-id]:
                    y_last = -id
                    break
            y_last_3 = []
            c = 0
            for id in range(len(y_position)):
                id += 1
                if y_position[-id]:
                    c+=1
                    y_last_3.insert(0,-id)
                    if c==3:
                        break
            count = 0
            if len(x_last_3)<3 or len(y_last_3)<3 :
                count+=1
                continue
            negative_sample = []
            for i in range(999):
                while True:
                    if d[1] : # in Y domain, the validation/test negative samples
                        sample = random.randint(0, self.opt["target_item_num"] - 1)
                        if sample != d[2] - self.opt["source_item_num"]: #若沒有sample到最後一個item_id
                            negative_sample.append(sample)
                            break
                    else : # in X domain, the validation/test negative samples
                        sample = random.randint(0, self.opt["source_item_num"] - 1)
                        if sample != d[2]:
                            negative_sample.append(sample)
                            break
            if d[1]:
                processed.append([seq, xd, yd, position, x_position, y_position, ts_d, ts_xd, ts_yd, x_last, y_last, d[1],
                                  d[2]-self.opt["source_item_num"], negative_sample, index,gender,x_last_3, y_last_3])
            else:
                processed.append([seq, xd, yd, position, x_position, y_position, ts_d, ts_xd, ts_yd, x_last, y_last, d[1],
                                  d[2], negative_sample, index, gender, x_last_3, y_last_3])
        if self.eval==2: #if is validation data
            print("Number of valid sequence deleted after preprocess:",count)
            print("Number of valid male:",len([tmp for tmp in processed if tmp[-3][0]==1]))
            print("Number of valid female:",len([tmp for tmp in processed if tmp[-3][0]==0]))
        else:
            print("Number of test sequence deleted after preprocess:",count)
            print("Number of test male:",len([tmp for tmp in processed if tmp[-3][0]==1]))
            print("Number of test female:",len([tmp for tmp in processed if tmp[-3][0]==0]))
        return processed
    def preprocess(self):
        processed = []
        female_delete_num = 0
        male_delete_num = 0
        for index, d in enumerate(self.train_data): # the pad is needed! but to be careful.
            gender = d[0]
            d = d[1]
            # d = [[tmp,0] for tmp in d]
            i_t = copy.deepcopy(d)
            d = [i[0] for i in d]
            ground = copy.deepcopy(d)[1:]

            share_x_ground = []
            share_x_ground_mask = []
            share_y_ground = []
            share_y_ground_mask = []
            for w in ground:
                if w < self.opt["source_item_num"]:
                    share_x_ground.append(w)
                    share_x_ground_mask.append(1)
                    share_y_ground.append(self.opt["target_item_num"])
                    share_y_ground_mask.append(0)
                else:
                    share_x_ground.append(self.opt["source_item_num"])
                    share_x_ground_mask.append(0)
                    share_y_ground.append(w - self.opt["source_item_num"])
                    share_y_ground_mask.append(1)
            
            d = d[:-1]  # delete the ground truth
            
            position = list(range(len(d)+1))[1:]
            ground_mask = [1] * len(d)



            xd = [] # the input of X domain
            xcnt = 1 
            x_position = [] # the position of X domain
            ts_xd = [] # the timestamp of X domain

            yd = [] 
            ycnt = 1
            y_position = []
            ts_yd = [] # the timestamp of Y domain
            
            # a list of (item,ts) pairs
            i_t = i_t[:-1]
            i_t_yd = []
            i_t_xd = []
            
            #augmentation
            augment_xd = []
            augment_yd = []
            #corru_x和corru_y是為了後續的contrastive learning而構建的關聯序列。corru_x包含原始的X域項目加上隨機的Y域項目;corru_y包含原始的Y域項目加上隨機的X域項目。
            corru_x = [] # the corrupted input of X domain
            corru_y = [] # the corrupted input of Y domain
            
            
            masked_xd = []
            neg_xd = []
            masked_yd = []
            neg_yd = []
            for i, k in enumerate(i_t):
                item = k[0]
                ts = k[1]
                if item < self.opt["source_item_num"]:
                    corru_x.append(item)
                    xd.append(item)
                    augment_xd.append(item)
                    x_position.append(xcnt)
                    xcnt += 1
                    corru_y.append(random.randint(0, self.opt["source_item_num"] - 1))
                    yd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
                    y_position.append(0)
                    i_t_xd.append(k)
                    ts_xd.append(ts)
                    ts_yd.append(0)
                    if self.opt['ssl'] =="mask_prediction":
                        if random.random()<self.opt['mask_prob']:
                            masked_xd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
                            neg_xd.append(random.sample(set(range(self.opt['source_item_num'])) - set(xd),1)[0])
                        else:
                            masked_xd.append(item)
                            neg_xd.append(item)
                        masked_yd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
                        neg_yd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
                    
                else: #if item is y-domain
                    corru_x.append(random.randint(self.opt["source_item_num"], self.opt["source_item_num"] + self.opt["target_item_num"] - 1))
                    xd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
                    x_position.append(0)
                    corru_y.append(item)
                    yd.append(item)
                    augment_yd.append(item)
                    y_position.append(ycnt)
                    ycnt += 1
                    i_t_yd.append(k)
                    ts_yd.append(ts)
                    ts_xd.append(0)
            ts_d = [t for i,t in i_t]
            #產生單域序列的ground truth
            now = -1
            x_ground = [self.opt["source_item_num"]] * len(xd) # caution!
            x_ground_mask = [0] * len(xd)
            for id in range(len(xd)):
                id+=1
                if x_position[-id]:
                    if now == -1: #若為第一個ground truth
                        now = xd[-id]
                        if ground[-1] < self.opt["source_item_num"]: #若為x domain
                            x_ground[-id] = ground[-1]
                            x_ground_mask[-id] = 1
                        else:
                            xd[-id] = self.opt["source_item_num"] + self.opt["target_item_num"]
                            ts_xd[-id] = 0
                            augment_xd = augment_xd[:-1]
                            x_position[-id] = 0
                    else:
                        x_ground[-id] = now
                        x_ground_mask[-id] = 1
                        now = xd[-id]
            if sum(x_ground_mask) == 0:
                # print("pass sequence x")
                if gender ==1:
                    male_delete_num+=1
                else:
                    female_delete_num+=1
                continue

            now = -1
            y_ground = [self.opt["target_item_num"]] * len(yd) # caution!
            y_ground_mask = [0] * len(yd)
            for id in range(len(yd)):
                id+=1
                if y_position[-id]:
                    if now == -1:
                        now = yd[-id] - self.opt["source_item_num"]
                        if ground[-1] > self.opt["source_item_num"]:
                            y_ground[-id] = ground[-1] - self.opt["source_item_num"]
                            y_ground_mask[-id] = 1
                        else:
                            yd[-id] = self.opt["source_item_num"] + self.opt["target_item_num"]
                            y_position[-id] = 0
                            ts_yd[-id] = 0
                            augment_yd = augment_yd[:-1]
                    else:
                        y_ground[-id] = now
                        y_ground_mask[-id] = 1
                        now = yd[-id] - self.opt["source_item_num"]
                        
            if sum(y_ground_mask) == 0:
                # print("pass sequence y")
                if gender ==1:
                    male_delete_num+=1
                else:
                    female_delete_num+=1
                continue
            
            if len(d) < self.opt['maxlen']:
                position = [0] * (self.opt['maxlen'] - len(d)) + position
                x_position = [0] * (self.opt['maxlen'] - len(d)) + x_position
                y_position = [0] * (self.opt['maxlen'] - len(d)) + y_position

                ground = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (self.opt['maxlen'] - len(d)) + ground
                share_x_ground = [self.opt["source_item_num"]] * (self.opt['maxlen'] - len(d)) + share_x_ground
                share_y_ground = [self.opt["target_item_num"]] * (self.opt['maxlen'] - len(d)) + share_y_ground
                x_ground = [self.opt["source_item_num"]] * (self.opt['maxlen'] - len(d)) + x_ground
                y_ground = [self.opt["target_item_num"]] * (self.opt['maxlen'] - len(d)) + y_ground

                ground_mask = [0] * (self.opt['maxlen'] - len(d)) + ground_mask
                share_x_ground_mask = [0] * (self.opt['maxlen'] - len(d)) + share_x_ground_mask
                share_y_ground_mask = [0] * (self.opt['maxlen'] - len(d)) + share_y_ground_mask
                x_ground_mask = [0] * (self.opt['maxlen'] - len(d)) + x_ground_mask
                y_ground_mask = [0] * (self.opt['maxlen'] - len(d)) + y_ground_mask

                corru_x = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (self.opt['maxlen'] - len(d)) + corru_x
                corru_y = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (self.opt['maxlen'] - len(d)) + corru_y  
                xd = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (self.opt['maxlen'] - len(d)) + xd             #x domain sequence
                yd = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (self.opt['maxlen'] - len(d)) + yd             #y domain sequence
                d = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (self.opt['maxlen'] - len(d)) + d               #mixed domain sequence
                # timestamp padding
                ts_xd = [0] * (self.opt['maxlen'] - len(ts_xd)) + ts_xd
                ts_yd = [0] * (self.opt['maxlen'] - len(ts_yd)) + ts_yd
                ts_d = [0]*(self.opt['maxlen'] - len(ts_d)) + ts_d
                gender = [gender]*self.opt['maxlen']
                augment_xd = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (self.opt['maxlen'] - len(augment_xd)) + augment_xd
                augment_yd = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (self.opt['maxlen'] - len(augment_yd)) + augment_yd
                if self.opt['ssl']=='mask_prediction':
                    masked_xd = [self.opt["source_item_num"] + self.opt["target_item_num"]]*(self.opt['maxlen'] - len(masked_xd)) + masked_xd
                    neg_xd = [self.opt["source_item_num"] + self.opt["target_item_num"]]*(self.opt['maxlen'] - len(neg_xd)) + neg_xd
                    masked_yd = [self.opt["source_item_num"] + self.opt["target_item_num"]]*(self.opt['maxlen'] - len(masked_yd)) + masked_yd
                    neg_yd = [self.opt["source_item_num"] + self.opt["target_item_num"]]*(self.opt['maxlen'] - len(neg_yd)) + neg_yd
            else:
                print("pass")
            processed.append([index, d, xd, yd, position, x_position, y_position, ts_d, ts_xd, ts_yd, ground, share_x_ground, share_y_ground, x_ground, y_ground, ground_mask, 
                                  share_x_ground_mask, share_y_ground_mask, x_ground_mask, y_ground_mask, corru_x, corru_y, 0, augment_xd, augment_yd, gender])
        print(f"number of training male deleted after preprocess: {male_delete_num}")
        print(f"number of training female deleted after preprocess: {female_delete_num}")
        return processed
    
    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch = list(zip(*batch))
        
        if self.eval!=-1: # valid or test
            return [torch.LongTensor(item) if not isinstance(item, torch.Tensor) else item for item in batch]
        else : # train
            if self.collate_fn:
                batch = self.collate_fn(batch)
                return batch
            else:
                return [torch.LongTensor(item) if not isinstance(item, torch.Tensor) else item for item in batch]
class GDataLoader(VanillaDataLoader):
    
    def __init__(self, filename, batch_size, opt, eval, collate_fn=None):
        super().__init__(filename, batch_size, opt)
        self.collate_fn = collate_fn
        self.eval = eval
        # ************* item_id *****************
        opt["source_item_num"], opt["target_item_num"] = self.read_item(f"./fairness_dataset/{self.opt['dataset']}/" + filename + "/train.txt")
        opt['itemnum'] = opt['source_item_num'] + opt['target_item_num'] + 1
        self.opt["maxlen"] = 50
        self.mask_id = opt["source_item_num"]+ opt["target_item_num"] +1
        # ************* sequential data *****************
        # if self.opt['domain'] =="cross":
        source_train_data = f"./fairness_dataset/{self.opt['dataset']}/" + filename + f"/train.txt"
        source_valid_data = f"./fairness_dataset/{self.opt['dataset']}/" + filename + f"/valid.txt"
        if self.eval == -1: 
            self.train_data = self.read_train_data(source_train_data)
        else:
            self.train_data = self.read_train_data(source_valid_data)
        self.part_sequence = []
        self.split_sequence()
        data = self.preprocess()
        print("pretrain_data length:",len(data))
        
        # shuffle for training
        indices = list(range(len(data)))
        random.shuffle(indices)
        data = [data[i] for i in indices]
        if batch_size > len(data):
            batch_size = len(data)
            self.batch_size = batch_size
        if len(data)%batch_size != 0:
            data += data[:batch_size]
        data = data[: (len(data)//batch_size) * batch_size]
        self.num_examples = len(data)
        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
    def split_sequence(self):
        for seq in self.train_data:
            input_ids = seq[1][-(self.opt["maxlen"]+1):-1] # keeping same as train set
            for i in range(4,len(input_ids)): # 5 is the minimum length of sequence
                self.part_sequence.append((seq[0],input_ids[:i+1]))          
    def preprocess(self):
        processed = []
        max_len =self.opt["maxlen"]
        for index, d in enumerate(self.part_sequence): # the pad is needed! but to be careful.
            gender = d[0]
            d = d[1]
            # d = [[tmp,0] for tmp in d] # add redundant timestamp
            i_t = copy.deepcopy(d)
            d = [i[0] for i in d]
            ground = copy.deepcopy(d)[1:]
            position = list(range(len(d)+1))[1:]
            ground_mask = [1] * len(d)
            i_t = i_t[:-1]
            ts_d = [t for i,t in i_t]
            
            if not [i for i in d if i < self.opt["source_item_num"]]:
                continue
            if not [i for i in d if i >= self.opt["source_item_num"]]:
                continue
            #ensure the length of sequence is max_len
            if len(d) < max_len:
                position = [0] * (max_len - len(d)) + position
                ground = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(d)) + ground
                ground_mask = [0] * (max_len - len(d)) + ground_mask
                ts_d = [0]*(max_len - len(ts_d)) + ts_d
                gender = [gender]*max_len
            else:
                print("pass")
            processed.append([index, d, position , ts_d, ground, ground_mask, gender]) #後面由collate_fn做masking
        return processed
    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch = list(zip(*batch))
        if self.collate_fn:
            batch = self.collate_fn(batch)
            return batch
        return [torch.tensor(item) if not isinstance(item, torch.Tensor) else item for item in batch]

class NonoverlapDataLoader(VanillaDataLoader):
    def __init__(self, filename, batch_size, opt, collate_fn=None):
        self.batch_size = batch_size
        self.opt = opt
        self.filename  = filename
        self.collate_fn = collate_fn
        # ************* item_id *****************
        data1,data2 = filename.split("_")   
        if 'sci-fi' == data1:
            data1 = 'Sci-Fi' 
        if 'sci-fi' == data2:
            data2 = 'Sci-Fi'
        if 'film-noir' == data1:
            data1 = 'Film-Noir'
        if 'film-noir' == data2:
            data2 = 'Film-Noir'
        # folder_name = f"{data1.lower()}_{data2.lower()}"
        A_data_name = data1.capitalize() if data1!="Sci-Fi" and data1!="Film-Noir" else data1
        B_data_name = data2.capitalize() if data2!="Sci-Fi" and data2!="Film-Noir" else data2
        opt["source_item_num"], opt["target_item_num"] = self.read_item(f"./fairness_dataset/{self.opt['dataset']}/" + filename + f"/nonoverlap_Y_female_{B_data_name}.txt")
        opt['itemnum'] = opt['source_item_num'] + opt['target_item_num'] + 1       
        self.opt["maxlen"] = 50
        self.mask_id = opt["source_item_num"]+ opt["target_item_num"] +1
        source_train_data = f"./fairness_dataset/{self.opt['dataset']}/" + filename + f"/nonoverlap_Y_female_{B_data_name}.txt"
        self.train_data = self.read_train_data(source_train_data)
        data = self.preprocess()
        self.all_data = data
        print("number of Nonoverlap user:",len(data))
        indices = list(range(len(data)))
        random.shuffle(indices)
        data = [data[i] for i in indices]
        if batch_size > len(data):
            batch_size = len(data)
            self.batch_size = batch_size
        if len(data)%batch_size != 0:
            data += data[:batch_size]
        data = data[: (len(data)//batch_size) * batch_size]
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
    def preprocess(self):
        processed = []
        max_len =self.opt["maxlen"]
        for index, d in enumerate(self.train_data): # the pad is needed! but to be careful.
            gender = d[0]
            d = d[1]
            i_t = copy.deepcopy(d)
            d = [i[0] for i in d]
            ground = copy.deepcopy(d)[1:]
            d = d[:-1]  # delete the ground truth
            ts_d = [t for i,t in i_t]
            position = list(range(len(d)+1))[1:]
            ground_mask = [1] * len(d)
            if len(d) < max_len:
                position = [0] * (max_len - len(d)) + position
                ground = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(d)) + ground
                ground_mask = [0] * (max_len - len(d)) + ground_mask
                d = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(d)) + d
                ts_d = [0]*(max_len - len(ts_d)) + ts_d
                gender = [gender]*max_len
            else:
                print("pass")
            processed.append([index, d, position , ground, ground_mask, gender, ts_d])
        return processed
    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch = list(zip(*batch))
        return [torch.LongTensor(item) if not isinstance(item, torch.Tensor) else item for item in batch]

