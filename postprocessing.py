# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 15:07:38 2018

@author: eduardo.andrade
"""

def clip_pred(pred, X, return_counts=True):
    '''Clip pred to values between min and max of features'''
    ret = pred.copy()
    min_ = X.min(axis=1)
    max_ = X.max(axis=1)
    
    count = 0
    for i in range(len(ret)):
        p = ret[i]
        ret[i] = np.clip(p, min_[i], max_[i])
        if return_counts:
            if ret[i] != p:
                count +=1
    
    if return_counts:
        return ret, count
    else:
        return ret
    
def round_to_closer_feature(pred, X):
    '''Round pred to the closer feature value'''
    ret = pred.copy()
    for i in range(len(ret)):
        features = X[i, :]
        dists = abs(ret[i] - features)
        ret[i] = features[np.argmin(dists)]
    return ret