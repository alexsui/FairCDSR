from .augmentation import TimeSpeedUp, TimeSlowDown, TimeReverse,TimeShift,TimeReorder
import numpy as np
import random
class TimeTransformation:
    def __init__(self, opt) -> None:
        self.opt = opt
        self.scale = [2, 3, 4, 5, 6]
        self.scale2 = [2, 4, 8, 16]
        self.method = ["speedup","slowdown","reverse","reorder"]
        self.method_combination = [(x,s) for s in self.scale for x in self.method]
        self.time_augmentation = {"speedup": TimeSpeedUp(),
                                  "slowdown" : TimeSlowDown(),
                                  "reverse" : TimeReverse(),
                                #   "timeshift" : TimeShift(),
                                  "reorder" : TimeReorder()
                                }        
    def __call__(self, mode = "discriminate",task = None):
        if mode == "discriminate":
            pos_aug_method = random.sample(self.method,2)
            neg_aug_method = [x for x in self.method if x not in pos_aug_method]
            pos_aug1 = self.time_augmentation[pos_aug_method[0]]
            pos_aug2 = self.time_augmentation[pos_aug_method[1]]
            neg_aug1 = self.time_augmentation[neg_aug_method[0]]
            neg_aug2 = self.time_augmentation[neg_aug_method[1]]
            scale = random.sample(self.scale,1)[0]
            return pos_aug1, pos_aug2, neg_aug1, neg_aug2,scale
        elif mode == "prediction":
            if task == "speed_classification":
                aug_method = "speedup"
                aug = self.time_augmentation[aug_method]
                return aug, self.scale2
            elif task == "direction_classification":
                return self.time_augmentation["reverse"],None
                