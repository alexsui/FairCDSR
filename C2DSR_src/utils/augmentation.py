import numpy as np 
import torch 
import math
import random
import torch.nn as nn
import ipdb
# from utils import neg_sample

class Mask(object):
    def __init__(self,args,mask_p,mask_id,similarity_dict):
        self.mask_p = mask_p
        self.mask_id = mask_id
        self.args = args
        self.similarity_dict = similarity_dict
    def softmax(self,x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    def __call__(self,input_ids):
        if len(input_ids)<2:
            return input_ids,len(input_ids)
        else:
            n_mask = math.ceil( self.mask_p *len(input_ids))
            masked_item_seq = input_ids.copy()
            if self.args.mask_mode == 'mask_id':
                mask_index = random.sample(range(len(input_ids)),k = n_mask)
                mask = [self.mask_id for _ in range(n_mask)]
            elif self.args.mask_mode == 'random_negative':
                item_set = set(input_ids)
                mask_index = random.sample(range(len(input_ids)),k = n_mask)
                mask = [neg_sample(item_set, self.args.item_size) for _ in range(n_mask)]
            elif self.args.mask_mode == 'similar_negative':
                #random mask with similar item
                mask_index = random.sample(range(len(input_ids)),k = n_mask)
                mask = []
                for index in mask_index:
                    # choose item with weight
                    related_item = self.similarity_dict[input_ids[index]]
                    item, sim = zip(*related_item)
                    # res = np.random.normal(500,125,1)
                    l = 250
                    w = [0.1/l]*l+[0.2/l]*l+[0.3/l]*l+[0.4/l]*l
                    res = np.random.choice(item,p=w)
                    mask.append(res)
                    #random
                    # r = np.random.randint(0,1000)
                    # item, sim = self.similarity_dict[input_ids[index]][r]
                    # mask.append(item)
                #topn similar mask 
                # input_index = list(range(len(input_ids)))
                # mask = []
                # get_sim = np.zeros((len(input_index),3))
                # for i, index in enumerate(input_index):
                #     sim_item = self.similarity_dict[input_ids[index]][self.args.n-1] #取topn(item, similarity)
                #     get_sim[i]= [index ,sim_item[0] ,sim_item[1]]
                # sorted_array = get_sim[get_sim[:, 2].argsort()[::-1]]
                # mask_index = [l[0].astype('int')for l in sorted_array[:n_mask]]
                # dic = {sim[0]:sim[1] for sim in get_sim}
                # for index in mask_index:
                #     mask.append(int(dic[index]))
                
            for idx, mask_val in zip(mask_index,mask):
                masked_item_seq[idx] = mask_val 
            # print("input_ids:",input_ids)
            # print("masked_item_seq:",masked_item_seq)
            return masked_item_seq, len(masked_item_seq)
class SegmentMask(object):
    def __init__(self,mask_id):
        self.mask_id = mask_id
    def __call__(self,input_ids):
        if len(input_ids)<2:
            return input_ids,len(input_ids)
        else:
            sample_length = random.randint(1, len(input_ids) // 2)
            start = random.randint(0,len(input_ids)-sample_length)
            segment = input_ids.copy()
            segment = segment[:start]+sample_length*[self.mask_id]+segment[start+sample_length:]
            return segment, len(segment)
class Crop(object):
    def __init__(self,tao=0.6):
        self.tao = tao
    def __call__(self,input_ids):
        if len(input_ids)<2:
            return input_ids,len(input_ids)
        else:
            crop_length = math.ceil(len(input_ids)*self.tao)
            start = random.randint(0,len(input_ids)-crop_length)
            segment = input_ids.copy()
            segment = segment[start:start+crop_length]
            return segment, len(segment)
# class Reorder(object):
#     def __init__(self,beta = 0.2):
#         self.beta = beta
#     def __call__(self,input_ids):
#         if len(input_ids)<2:
#             return input_ids,len(input_ids)
#         elif len(input_ids)==2:
#             reorder_seq = input_ids.copy()
#             reorder_seq[0] ,reorder_seq[1] = reorder_seq[1] ,reorder_seq[0]
#             return reorder_seq, len(reorder_seq) 
#         else:
#             reorder_len = math.ceil(len(input_ids)*self.beta) #2
#             if reorder_len == 1:
#                 reorder_len = reorder_len+1
#             start = random.randint(0,len(input_ids)-reorder_len) #1
#             reorder_seq = input_ids.copy()
#             shuffle_seq = reorder_seq[start:start + reorder_len]
#             # if len(shuffle_seq)==2:
#             #     shuffle_seq[0], shuffle_seq[1] = shuffle_seq[1], shuffle_seq[0]
#             # else:
#             random.shuffle(shuffle_seq)
#             reorder_seq = reorder_seq[:start] + shuffle_seq + reorder_seq[start+reorder_len:]
#             return reorder_seq, len(reorder_seq)
# # for time transformation
class TimeSpeedUp(object):
    def __init__(self) -> None:
        pass
        # self.scale_range = scale_range
    def __call__(self, ts_sequence,scale):
        def scale_non_zero_elements(row):
            non_zero_indices = np.nonzero(row)[0]
            non_zero_elements = row[non_zero_indices]
            intervals = np.abs(np.diff(non_zero_elements))
            scaled_intervals = np.round(intervals * scale).astype(int)
            new_non_zero_sequence = np.cumsum(np.insert(scaled_intervals, 0, non_zero_elements[0]))
            new_sequence = np.zeros_like(row)
            new_sequence[non_zero_indices] = new_non_zero_sequence
            return new_sequence
        sequence = ts_sequence.cpu().numpy()
        new_sequence = np.apply_along_axis(scale_non_zero_elements, axis=1, arr = sequence)
        return torch.LongTensor(new_sequence).cuda()
class TimeSlowDown(object):
    def __init__(self) -> None:
        # self.scale_range = scale_range
        pass
    def __call__(self, ts_sequence,scale):
        # scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        def scale_non_zero_elements(row):
            non_zero_indices = np.nonzero(row)[0]
            non_zero_elements = row[non_zero_indices]
            intervals = np.abs(np.diff(non_zero_elements))
            scaled_intervals = np.round(intervals * (1/scale)).astype(int)
            new_non_zero_sequence = np.cumsum(np.insert(scaled_intervals, 0, non_zero_elements[0]))
            new_sequence = np.zeros_like(row)
            new_sequence[non_zero_indices] = new_non_zero_sequence
            return new_sequence
        sequence = ts_sequence.cpu().numpy()
        new_sequence = np.apply_along_axis(scale_non_zero_elements, axis=1, arr=sequence)
        return torch.LongTensor(new_sequence).cuda()
class TimeReverse(object):
    def __init__(self) -> None:
        pass
    def __call__(self, ts_sequence, position, scale= None):
        def reverse_non_zero_elements(row):
            non_zero_indices = np.nonzero(row)[0]
            non_zero_elements = row[non_zero_indices]
            reversed_non_zero_elements = non_zero_elements[::-1]
            reversed_row = np.zeros_like(row)
            reversed_row[non_zero_indices] = reversed_non_zero_elements
            return reversed_row
        ts_sequence = ts_sequence.cpu().numpy()
        pos_sequence = position.cpu().numpy()
        reversed_ts= np.apply_along_axis(reverse_non_zero_elements, axis=1, arr=ts_sequence)
        # reversed_position = np.apply_along_axis(reverse_non_zero_elements, axis=1, arr=pos_sequence)
        return torch.LongTensor(reversed_ts).cuda(), torch.LongTensor(pos_sequence).cuda()
    
class TimeShift(object):
    def __init__(self) -> None:
        # self.scale_range = scale_range
        pass
    def __call__(self, ts_sequence, ratio):
        # scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        def scale_non_zero_elements(row):
            non_zero_indices = np.nonzero(row)[0]
            non_zero_elements = row[non_zero_indices]
            shift_len = math.ceil((1/ratio)*len(non_zero_elements))
            non_zero_elements[-shift_len:] = non_zero_elements[-shift_len:] + np.random.randint(100000,200000)
            new_sequence = np.zeros_like(row)
            new_sequence[non_zero_indices] = non_zero_elements
            return new_sequence
        sequence = ts_sequence.cpu().numpy()
        new_sequence = np.apply_along_axis(scale_non_zero_elements, axis=1, arr=np.array(sequence))
        return torch.LongTensor(new_sequence).cuda()
class TimeReorder(object):
    def __call__(self, ts_sequence, position, ratio):
        def reorder_non_zero_elements(row):
            mid = len(row)//2
            ts, pos = row[:mid], row[mid:]
            if len(ts)<1:
                return ts,pos
            elif len(ts)==2:
                return ts[::-1],pos[::-1]
            try:
                random.seed(random.randint(0,1000))
                non_zero_indices = np.nonzero(ts)[0]
                non_zero_elements = ts[non_zero_indices]
                reorder_len = math.ceil(len(non_zero_elements)*1/ratio)
                if reorder_len == 1:
                    reorder_len = reorder_len+1
                start = random.randint(0,len(non_zero_elements)-reorder_len)
                sub_array_indices = non_zero_indices[start:start + reorder_len]
                perm = np.random.permutation(len(sub_array_indices))
                ts_non_zero_shuffled = ts[sub_array_indices][perm]
                pos_non_zero_shuffled = pos[sub_array_indices][perm]
                ts_new_sequence = ts.copy()
                ts_new_sequence[sub_array_indices] = ts_non_zero_shuffled
                pos_new_sequence = pos.copy()
                pos_new_sequence[sub_array_indices] = pos_non_zero_shuffled
                new_combined = np.concatenate([ts_new_sequence, pos_new_sequence], axis=0)
            except Exception as e:
                print(e)
                ipdb.set_trace()
            return new_combined    
        ts_sequence = ts_sequence.cpu().numpy()
        pos_sequence = position.cpu().numpy()
        combined = np.concatenate([ts_sequence, pos_sequence], axis=1)
        new_combined = np.apply_along_axis(reorder_non_zero_elements, axis=1, arr=combined)
        ts_new_sequence, pos_new_sequence =new_combined[:,:len(ts_sequence[0])], new_combined[:,len(ts_sequence[0]):]
        
        return torch.LongTensor(ts_new_sequence).cuda(), torch.LongTensor(pos_new_sequence).cuda()