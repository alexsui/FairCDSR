import torch
import torch.nn as nn
from model.FairCDSR import ATTENTION
class Generator(nn.Module):
    def __init__(self, opt, type):
        super(Generator, self).__init__()
        self.opt = opt
        self.item_embed = torch.nn.Embedding(self.opt["itemnum"] + 1, self.opt["hidden_units"],
                                            padding_idx= self.opt["itemnum"] - 1)   
        self.encoder = ATTENTION(opt)
        if type == "X":
            self.pred_head = nn.Linear(opt['hidden_units'], opt['source_item_num'])
        elif type == "Y":
            self.pred_head = nn.Linear(opt['hidden_units'], opt['target_item_num'])
        else:
            self.pred_head = nn.Linear(opt['hidden_units'], opt['itemnum'] - 1)
    def forward(self, o_seqs, position, ts = None):
        seqs = self.item_embed(o_seqs)
        seqs_fea = self.encoder(o_seqs, seqs, position,causality_mask = False,ts = ts)
        return self.pred_head(seqs_fea)