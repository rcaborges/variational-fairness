
import numpy as np
import bottleneck as bn
from utils import *

def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    return DCG / IDCG

def Recall_at_k_batch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall

def dcg_k_rounds(scores_rounds):
    dcgs_rounds = []
    for iround in range(scores_rounds.shape[0]):
        dcg_rounds.append(dcg_k_users(scores_rounds[iround,:,:]))
    return np.array(dcgs_rounds)

def dcg_k_users(scores):
    dcg_round = []
    for user in range(scores.shape[0]):
        dcg_round.append(dcg_single_ranking(scores[user,:]))
    return dcg_round 

def dcg_single_ranking(scores):
    dcg = 0.0
    for idx in range(len(scores)):
        curr = scores[idx]/np.log2(idx + 2)   
        dcg += curr
    return dcg

def Fairness_at_k_rounds(X_pred, dcg_gt):
    debug = False
    print(X_pred.shape)
    ufair_all = []
    ndcg_all = []
    for user in range(X_pred.shape[1]):
        ufair, ndcg = 0.0, 0.0
        for item in range(X_pred.shape[2]):
            sum_a, sum_r = 0,0
            item_ufair, item_ndcg = [], []
            for iround in range(X_pred.shape[0]):
                att, rel = 0,0
                if X_pred[iround,user,item] != -np.inf:
                    if debug: print(user,item,iround)
                    # NORMALIZE SCORES BETWEEN THE MINIMUN AND MAXIMUN VALUES
                    #print(min(X_pred[iround,user,:]),max(X_pred[iround,user,:]))
                    norm_scores = ((X_pred[iround,user,:] - min(X_pred[iround,user,:]))/(max(X_pred[iround,user,:])- min(X_pred[iround,user,:])))
                    #norm_scores = (X_pred[iround,user,:]/max(X_pred[iround,user,:]))
                    if debug: print(norm_scores)
                    # INDEX OF ITEMS SORTED IN DESCENDING ORDER
                    if debug: print(np.argsort(norm_scores)[::-1]) 
                    # POSITION OF THE CURRENT ITEM
                    if debug: print(max(np.argsort(norm_scores)) - np.where(np.argsort(norm_scores) == item)[0][0]) 
                    att = np.where(np.argsort(norm_scores) == item)[0][0]/max(np.argsort(norm_scores))
                    if debug: print('Attention: '+ str(att)) 
                    sum_a = sum_a + att
                    rel = norm_scores[item]
                    if debug: print('Relevance: '+str(rel))
                    sum_r = sum_r + rel
                    if debug: print('Unfairness: '+str(abs(att - rel)))
                    if debug: print('DCG: '+str(dcg_single_ranking(X_pred[iround,user,:])))
                    if debug: print('NDCG: '+str(dcg_single_ranking(X_pred[iround,user,:])/dcg_gt[user]))
                    if debug: input()
                    item_ufair.append(abs(att - rel)) 
                    item_ndcg.append(dcg_single_ranking(X_pred[iround,user,:])/dcg_gt[user])
                else: 
                    print("ALERT")    
                    print(user,item,iround) 
            #plot_curve(item_ufair,item_ndcg)
            # SUMS THE UNFAIRNESS OF CURRENT ITEM
            ufair = ufair + (abs(sum_a - sum_r)/X_pred.shape[0])
            ndcg = ndcg + (np.sum(item_ndcg)/X_pred.shape[0])
            if debug: print('Unfairness of Item: '+str((abs(sum_a - sum_r)/X_pred.shape[0])))
            if debug: print('Total Current Unfairness: '+str(ufair/(item+1)))
            if debug: print('-----------------------------------')
        if debug: print('Normalized Total Unfairness: '+str(ufair/X_pred.shape[2]))
        if debug: print('Normalized Total NDCG: '+str(ndcg/X_pred.shape[2]))
        # normalize by the number of items
        ufair_all.append(ufair/X_pred.shape[2])
        ndcg_all.append(ndcg/X_pred.shape[2])
    return np.array(ufair_all), np.array(ndcg_all)
