import torch
import ipdb
import numpy as np
import json
from scipy.spatial import distance
from pathlib import Path
from torch.autograd import Variable
class Evaluator:
    def __init__(self, model, opt):
        self.model = model
        self.opt = opt
        self.model_save_dir = f"./models/{self.opt['data_dir']}/{self.opt['id']}/{self.opt['seed']}"
        self.dev_score_history = [0]
        self.val_pred_loss = []
        self.patience = self.opt["finetune_patience"] #5
        self.if_early_stop = False
        self.CS_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    def save(self, filename):
        params = {
            'model': self.model.state_dict(),
            'config': self.opt,
        }
        torch.save(params, filename)
        print("model saved to {}".format(filename))
    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        state_dict = checkpoint['model']
        self.model.load_state_dict(state_dict, strict=False)
    def run_val(self, epoch, valid_dataloader)-> bool:
        print("Evaluating on dev set...")
        self.model.eval()
        val_pred, val_loss = self.get_evaluation_result(valid_dataloader, mode = "valid", target  = self.opt['main_task'])
        self.val_pred_loss.append(val_loss)
        val_MRR, val_NDCG_5, val_NDCG_10, val_HR_1, val_HR_5, val_HR_10 = self.cal_test_score(val_pred)
        print("")
        print('val epoch:%d, MRR: %.4f, NDCG@10: %.4f, HR@10: %.4f'
            % (epoch, val_MRR, val_NDCG_10, val_HR_10))

        if (val_MRR > max(self.dev_score_history)):
            self.patience = self.opt["finetune_patience"]
            print("")
            print('Best validation result!')
            print(f"write models into path {self.model_save_dir}")
            Path(self.model_save_dir).mkdir(parents=True, exist_ok=True)
            self.save(f"{self.model_save_dir}/{self.opt['main_task']}_model.pt")
        else:
            self.patience -=1
            print("early stop counter:", self.opt["finetune_patience"]-self.patience)
            if self.patience == 0:
                print("Early stop triggered!")
                self.if_early_stop = True
        print("")
        self.dev_score_history.append(val_MRR)
        return self.if_early_stop
    def run_test(self, test_dataloader, file_logger)-> list: 
        self.load(f"{self.model_save_dir}/{self.opt['main_task']}_model.pt")
        self.model.eval()
        with torch.no_grad():
            test_pred, test_male_pred, test_female_pred, test_loss = self.get_evaluation_result(test_dataloader, mode = "test",target  = self.opt['main_task'])
            test_MRR, test_NDCG_5, test_NDCG_10, test_HR_1, test_HR_5, test_HR_10 = self.cal_test_score(test_pred)
            test_MRR_male, test_NDCG_5_male, test_NDCG_10_male, test_HR_1_male, test_HR_5_male, test_HR_10_male = self.cal_test_score(test_male_pred)
            test_MRR_female, test_NDCG_5_female, test_NDCG_10_female, test_HR_1_female, test_HR_5_female, test_HR_10_female = self.cal_test_score(test_female_pred)
            # DIF, X_DP, Y_DP, X_EO, Y_EO = self.get_fairness_metric_for_test(test_dataloader) 
            result_str =""
            print("")
            best_test = [test_MRR, test_NDCG_5, test_NDCG_10, test_HR_5, test_HR_10]
            best_test_male = [test_MRR_male, test_NDCG_5_male, test_NDCG_10_male, test_HR_5_male, test_HR_10_male]
            best_test_female = [test_MRR_female, test_NDCG_5_female, test_NDCG_10_female, test_HR_5_female, test_HR_10_female]
            result_str += str({"Best overall": best_test})+"\n"
            result_str += str({"Best male":best_test_male})+"\n"
            result_str += str({"Best female":best_test_female})+"\n"
            print(f"test overall: MRR-{best_test[0]}, NDCG_5-{best_test[1]}, NDCG_10-{best_test[2]}, HR_5-{best_test[3]}, HR_10-{best_test[4]}")
            print(f"test male: MRR-{best_test_male[0]}, NDCG_5-{best_test_male[1]}, NDCG_10-{best_test_male[2]}, HR_5-{best_test_male[3]}, HR_10-{best_test_male[4]}")
            print(f"test female: MRR-{best_test_female[0]}, NDCG_5-{best_test_female[1]}, NDCG_10-{best_test_female[2]}, HR_5-{best_test_female[3]}, HR_10-{best_test_female[4]}")
            file_logger.log(result_str)
        return best_test, best_test_male,best_test_female
    def cal_test_score(self, predictions):
        MRR=0.0
        HR_1 = 0.0
        HR_5 = 0.0
        HR_10 = 0.0
        NDCG_5 = 0.0
        NDCG_10 = 0.0
        valid_entity = 0.0
        for pred in predictions:
            valid_entity += 1
            MRR += 1 / pred
            if pred <= 1:
                HR_1 += 1
            if pred <= 5:
                NDCG_5 += 1 / np.log2(pred + 1)
                HR_5 += 1
            if pred <= 10:
                NDCG_10 += 1 / np.log2(pred + 1)
                HR_10 += 1
            if valid_entity % 100 == 0:
                print('.', end='')
        if valid_entity == 0:
            valid_entity = 1
        return MRR/valid_entity, NDCG_5 / valid_entity, NDCG_10 / valid_entity, HR_1 / valid_entity, HR_5 / (valid_entity*2/3), HR_10 / (valid_entity*2/3)
    
    def get_evaluation_result(self, evaluation_batch, mode = "valid", target = "Y"):
        total_pred = []
        total_pred_female = []
        total_pred_male = []
        total_val_loss = 0
        for i, batch in enumerate(evaluation_batch):
            pred, pred_male, pred_female, batch_loss= self.test_batch(batch, mode = mode, target = target)
            total_pred = pred + total_pred
            total_pred_male = total_pred_male + pred_male
            total_pred_female = total_pred_female + pred_female 
            total_val_loss += batch_loss
        if mode == "valid":
            return total_pred, total_val_loss / len(evaluation_batch)
        else:
            return total_pred, total_pred_male, total_pred_female, total_val_loss / len(evaluation_batch)
    def get_fairness_metric_for_test(self, evaluation_batch):
        with open(f"./fairness_dataset/{self.opt['dataset']}/{self.opt['data_dir']}/item_IF.json","r") as f:
            item_if = json.load(f)
        all_DIF = 0
        c =0
        # DP
        X_pred_female = []
        X_pred_male = []
        Y_pred_female = []
        Y_pred_male = []
        # EO
        EO_X_pred_female = []
        EO_Y_pred_female = []
        EO_X_pred_male = []
        EO_Y_pred_male = []
        
        X_pred_female_num =0
        X_pred_male_num =0
        Y_pred_female_num =0
        Y_pred_male_num =0
        
        for i, batch in enumerate(evaluation_batch):
            
            index, seq, x_seq, y_seq, position, x_position, y_position, ts_d, ts_xd, ts_yd, X_last, Y_last, XorY, ground_truth, neg_list,gender,x_last_3,y_last_3 = self.unpack_batch_predict(batch)
            if self.opt['domain'] =="single":
                seq = None
            seqs_fea, x_seqs_fea, y_seqs_fea = self.model(seq, x_seq, y_seq, position, x_position, y_position)
            if self.opt['domain'] =="single":
                seq = None
                if self.opt['main_task'] == "X":
                    tmp = x_seq
                elif self.opt['main_task'] == "Y":
                    tmp = y_seq
            else:
                tmp =seq
            for id, (fea, sex) in enumerate(zip(tmp,gender)): # b * s * f
                if XorY[id] == 0: #if x domain
                    # print("share_fea:", share_fea.shape)#[256]
                    specific_fea = x_seqs_fea[id, X_last[id]]
                    if seqs_fea is not None:
                        # print("seqs_fea:", seqs_fea.shape)
                        share_fea = seqs_fea[id, -1]
                        X_score = self.model.lin_X(share_fea + specific_fea).squeeze(0) #256-> self.opt["source_item_num"]
                    else:
                        X_score = self.model.lin_X(specific_fea).squeeze(0)
                    topk_item = torch.topk(X_score, self.opt['topk'])[1].detach().cpu().numpy()
                    try:
                        predicted_IF = [item_if[str(d)] for d in topk_item if str(d) in list(item_if.keys())]
                        gt_item_id = [x_seq[id][i].item() for i in x_last_3[id]]
                        gt_IF = [item_if[str(d)] for d in gt_item_id if str(d) in list(item_if.keys())]
                        DIF = np.sum(predicted_IF) - np.sum(gt_IF)
                        all_DIF += DIF
                        c+=1
                    except:
                        print("Something wrong with IF!")
                        ipdb.set_trace()
                    if sex[0]==0:
                        X_pred_female_num+=1
                        X_pred_female+=topk_item.tolist()
                        EO_X_pred_female+=[item for item in topk_item.tolist()if item in x_seq[id]] 
                    elif sex[0]==1:
                        X_pred_male_num +=1
                        X_pred_male+=topk_item.tolist()
                        EO_X_pred_male+=[item for item in topk_item.tolist()if item in x_seq[id]] 
                else :
                    specific_fea = y_seqs_fea[id, Y_last[id]]
                    if seqs_fea is not None:
                        share_fea = seqs_fea[id, -1]
                        Y_score = self.model.lin_Y(share_fea + specific_fea).squeeze(0)
                    else:
                        Y_score = self.model.lin_Y(specific_fea).squeeze(0)
                    topk_item = torch.topk(Y_score, self.opt['topk'])[1].detach().cpu().numpy()
                    try:
                        predicted_IF = [item_if[str(d)] for d in topk_item + self.opt['source_item_num'] if str(d) in list(item_if.keys())]
                        gt_item_id = [y_seq[id][i].item() for i in y_last_3[id]]
                        gt_IF = [item_if[str(d)] for d in gt_item_id if str(d) in list(item_if.keys())]
                        DIF = np.sum(predicted_IF) - np.sum(gt_IF)
                        all_DIF+=DIF
                        c+=1
                    except:
                        print("Something wrong with IF!")
                        ipdb.set_trace()
                    if sex[0]==0:
                        Y_pred_female_num+=1
                        Y_pred_female+=topk_item.tolist()
                        EO_Y_pred_female+=[item for item in topk_item.tolist() if item in y_seq[id]] 
                    elif sex[0]==1:
                        Y_pred_male_num+=1
                        Y_pred_male+=topk_item.tolist()
                        EO_Y_pred_male+=[item for item in topk_item.tolist() if item in y_seq[id]] 
        avg_DIF = all_DIF / c
        if self.opt['main_task'] == "X" or self.opt['main_task'] == "dual":
            X_pred_female = torch.tensor(X_pred_female)
            X_pred_male = torch.tensor(X_pred_male)
            EO_X_pred_female = torch.tensor(EO_X_pred_female)
            EO_X_pred_male = torch.tensor(EO_X_pred_male)
        if self.opt['main_task'] == "Y" or self.opt['main_task'] == "dual":
            Y_pred_female = torch.tensor(Y_pred_female)
            Y_pred_male = torch.tensor(Y_pred_male)
            EO_Y_pred_female = torch.tensor(EO_Y_pred_female)
            EO_Y_pred_male = torch.tensor(EO_Y_pred_male)
        X_DP, X_EO = None, None
        Y_DP, Y_EO = None, None
        
        if self.opt['main_task'] == "X" or self.opt['main_task'] == "dual":
            #DP calculation    
            X_pred_female_dist= torch.zeros(self.opt['source_item_num'],dtype=torch.int64)
            num, count = X_pred_female.unique(return_counts=True)
            X_pred_female_dist[num] = count
            X_pred_female_dist = X_pred_female_dist.to(torch.float32)/X_pred_female_num
            # max_count = X_pred_female_dist.max()
            # if max_count > 0:
            #     X_pred_female_dist /= max_count
            
            X_pred_male_dist= torch.zeros(self.opt['source_item_num'],dtype=torch.int64)
            num, count = X_pred_male.unique(return_counts=True)
            X_pred_male_dist[num] = count
            X_pred_male_dist = X_pred_male_dist.to(torch.float32)/X_pred_male_num
            X_DP = distance.jensenshannon(X_pred_female_dist+1e-12,X_pred_male_dist+1e-12).item() if X_pred_female_dist is not None else None
            
            #EO calculation
            EO_X_pred_female_dist= torch.zeros(self.opt['source_item_num'],dtype=torch.int64)
            num, count = EO_X_pred_female.unique(return_counts=True)
            EO_X_pred_female_dist[num] = count
            EO_X_pred_female_dist = EO_X_pred_female_dist.to(torch.float32)/X_pred_female_num
            EO_X_pred_male_dist= torch.zeros(self.opt['source_item_num'],dtype=torch.int64)
            num, count = EO_X_pred_male.unique(return_counts=True)
            EO_X_pred_male_dist[num] = count
            EO_X_pred_male_dist = EO_X_pred_male_dist.to(torch.float32)/X_pred_male_num
            X_EO = distance.jensenshannon(EO_X_pred_female_dist+1e-12,EO_X_pred_male_dist+1e-12).item() 
        elif self.opt['main_task'] == "Y" or self.opt['main_task'] == "dual":
            #DP calculation
            Y_pred_female_dist= torch.zeros(self.opt['target_item_num'],dtype=torch.int64)
            tmp_num, tmp_count = Y_pred_female.unique(return_counts=True)
            Y_pred_female_dist[tmp_num] = tmp_count
            Y_pred_female_dist = Y_pred_female_dist.to(torch.float32)/Y_pred_female_num
            Y_pred_male_dist= torch.zeros(self.opt['target_item_num'],dtype=torch.int64)
            num, count = Y_pred_male.unique(return_counts=True)
            Y_pred_male_dist[num] = count
            Y_pred_male_dist = Y_pred_male_dist.to(torch.float32)/Y_pred_male_num
            Y_DP = distance.jensenshannon(Y_pred_female_dist+1e-12, Y_pred_male_dist+1e-12).item() 
            #EO calculation
            # try:
            EO_Y_pred_female_dist= torch.zeros(self.opt['target_item_num'],dtype=torch.int64)
            num, count = EO_Y_pred_female.unique(return_counts=True)
            if num.tolist():
                EO_Y_pred_female_dist[num] = count
                EO_Y_pred_female_dist = EO_Y_pred_female_dist.to(torch.float32)/Y_pred_female_num
            EO_Y_pred_male_dist= torch.zeros(self.opt['target_item_num'],dtype=torch.int64)
            num, count = EO_Y_pred_male.unique(return_counts=True)
            if num.tolist():
                EO_Y_pred_male_dist[num] = count
                EO_Y_pred_male_dist = EO_Y_pred_male_dist.to(torch.float32)/Y_pred_male_num
                Y_EO = distance.jensenshannon(EO_Y_pred_female_dist+1e-12, EO_Y_pred_male_dist+1e-12).item() if Y_pred_female_dist is not None else None
                    
            # except:
            #     ipdb.set_trace()
            #     pass

        
        print("number of X tested male:",X_pred_male_num)
        print("number of X tested female:",X_pred_female_num)
        print("number of Y tested male:",Y_pred_male_num)
        print("number of Y tested female:",Y_pred_female_num)
        print("number of total tested user:",Y_pred_male_num+Y_pred_female_num)
        print("number of test data:",Y_pred_male_num+Y_pred_female_num)
        # return distance.jensenshannon(X_pred_female_dist+1e-12,X_pred_male_dist+1e-12).item() ,distance.jensenshannon(Y_pred_female_dist+1e-12,Y_pred_male_dist+1e-12).item()
        return avg_DIF, X_DP, Y_DP, X_EO, Y_EO
    def test_batch(self, batch, mode, target):
        if mode == "valid":
            index, seq, x_seq, y_seq, position, x_position, y_position, ts_d, ts_xd, ts_yd, X_last, Y_last, XorY, ground_truth, neg_list, gender = self.unpack_batch_valid(batch)
        elif mode == "test":
            index, seq, x_seq, y_seq, position, x_position, y_position, ts_d, ts_xd, ts_yd, X_last, Y_last, XorY, ground_truth, neg_list, gender,x_last_3,y_last_3 = self.unpack_batch_test(batch)
        
        if self.opt['domain'] =="single":
            seq = None
        if self.opt['time_encode']:
            seqs_fea, x_seqs_fea, y_seqs_fea = self.model(seq, x_seq, y_seq, position, x_position, y_position, ts_d, ts_xd, ts_yd)
        else:
            seqs_fea, x_seqs_fea, y_seqs_fea = self.model(seq, x_seq, y_seq, position, x_position, y_position)
        pred = []
        pred_female = []
        pred_male = []
        batch_loss = 0
        if self.opt['domain'] =="single":
            seq = None
            if self.opt['main_task'] == "X":
                tmp = x_seq
            elif self.opt['main_task'] == "Y":
                tmp = y_seq
        else:
            tmp = seq
        if target == 'X':
            for id, (fea, g) in enumerate(zip(tmp,gender)): # b * s * f
                if XorY[id] == 0: #if x domain
                    # print("share_fea:", share_fea.shape)#[256]
                    specific_fea = x_seqs_fea[id, X_last[id]]
                    if seqs_fea is not None:
                        # print("seqs_fea:", seqs_fea.shape)
                        share_fea = seqs_fea[id, -1]
                        X_score = self.model.lin_X(share_fea + specific_fea).squeeze(0) #256-> self.opt["source_item_num"]
                    else:
                        X_score = self.model.lin_X(specific_fea).squeeze(0)
                    cur = X_score[ground_truth[id]] 
                    score_larger = (X_score[neg_list[id]] > (cur + 0.00001)).data.cpu().numpy() #1000
                    true_item_rank = np.sum(score_larger) + 1
                    if g[0]==0:
                        pred_female.append(true_item_rank)
                    elif g[0]==1:
                        pred_male.append(true_item_rank)
                    pred.append(true_item_rank)
                    batch_loss+=self.CS_criterion(X_score, ground_truth[id]).item()
        elif target == 'Y':
            for id, (fea, g) in enumerate(zip(tmp,gender)): # b * s * f
                if XorY[id] == 1: #if Y domain
                    specific_fea = y_seqs_fea[id, Y_last[id]]
                    if seqs_fea is not None:
                        share_fea = seqs_fea[id, -1]
                        Y_score = self.model.lin_Y(share_fea + specific_fea).squeeze(0)
                    else:
                        Y_score = self.model.lin_Y(specific_fea).squeeze(0)
                    cur = Y_score[ground_truth[id]]
                    score_larger = (Y_score[neg_list[id]] > (cur + 0.00001)).data.cpu().numpy()
                    true_item_rank = np.sum(score_larger) + 1
                    if g[0]==0:
                        pred_female.append(true_item_rank)
                    elif g[0]==1:
                        pred_male.append(true_item_rank)
                    pred.append(true_item_rank)
                    batch_loss+=self.CS_criterion(Y_score, ground_truth[id]).item()
        else:
            raise ValueError("Invalid target")
        return pred, pred_male, pred_female,  batch_loss#[B,1]
    def evaluate(self,test_dataloader,file_logger):
        self.model.eval()
        with torch.no_grad():
            test_pred, test_male_pred, test_female_pred, test_loss = self.get_evaluation_result(test_dataloader, mode = "test",target  = self.opt['main_task'])
            test_MRR, test_NDCG_5, test_NDCG_10, test_HR_1, test_HR_5, test_HR_10 = self.cal_test_score(test_pred)
            
            test_MRR_male, test_NDCG_5_male, test_NDCG_10_male, test_HR_1_male, test_HR_5_male, test_HR_10_male = self.cal_test_score(test_male_pred)
            test_MRR_female, test_NDCG_5_female, test_NDCG_10_female, test_HR_1_female, test_HR_5_female, test_HR_10_female = self.cal_test_score(test_female_pred)
            # DIF, X_DP, Y_DP, X_EO, Y_EO = self.get_fairness_metric_for_test(test_dataloader) 
            result_str =""
            print("")
            best_test = [test_MRR, test_NDCG_5, test_NDCG_10, test_HR_5, test_HR_10]
            best_test_male = [test_MRR_male, test_NDCG_5_male, test_NDCG_10_male, test_HR_5_male, test_HR_10_male]
            best_test_female = [test_MRR_female, test_NDCG_5_female, test_NDCG_10_female, test_HR_5_female, test_HR_10_female]
            result_str += str({"Best overall": best_test})+"\n"
            result_str += str({"Best male":best_test_male})+"\n"
            result_str += str({"Best female":best_test_female})+"\n"
            print("test overall",best_test)
            print("test male:",best_test_male)
            print("test female:",best_test_female)
            file_logger.log(result_str)
        return best_test, best_test_male,best_test_female
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
    def unpack_batch_test(self, batch):
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