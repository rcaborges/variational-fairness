import os
import shutil
import sys

import numpy as np
from scipy import sparse

import matplotlib.pyplot as plt

import seaborn as sn
sn.set()

import pandas as pd

import tensorflow as tf
from tensorflow.contrib.layers import apply_regularization, l2_regularizer

import bottleneck as bn
from multi_dae import MultiDAE, MultiVAE
from metrics import *
from utils import *

DATA_DIR = '../../../data/ml-20m/'
pro_dir = os.path.join(DATA_DIR, 'pro_sg')

unique_sid = list()
with open(os.path.join(pro_dir, 'unique_sid.txt'), 'r') as f:
    for line in f:
        unique_sid.append(line.strip())

n_items = len(unique_sid)
print("NUMERO ITENS: "+str(n_items))

p_dims = [200, 600, n_items]
# the total number of gradient updates for annealing
total_anneal_steps = 200000
# largest annealing parameter
anneal_cap = 0.2

# Test data
test_data_tr, test_data_te = load_tr_te_data(
    os.path.join(pro_dir, 'test_tr.csv'),
    os.path.join(pro_dir, 'test_te.csv'),
    n_items)
N_test = test_data_tr.shape[0]
idxlist_test = range(N_test)

batch_size_test = 2000

tf.reset_default_graph()
vae = MultiVAE(p_dims, lam=0.0)
saver, logits_var, _, _, _ = vae.build_graph(0)

arch_str = "I-%s-I" % ('-'.join([str(d) for d in vae.dims[1:-1]]))

chkpt_dir = './chkpt/ml-20m/VAE_anneal{}K_cap{:1.1E}/{}'.format(
    total_anneal_steps/1000, anneal_cap, arch_str)
print("chkpt directory: %s" % chkpt_dir)

n100_list, r20_list, r50_list = [], [], []

preds = []

with tf.Session() as sess:
    saver.restore(sess, '{}/model'.format(chkpt_dir))
    for bnum, st_idx in enumerate(range(0, N_test, batch_size_test)):
        end_idx = min(st_idx + batch_size_test, N_test)
        X = test_data_tr[idxlist_test[st_idx:end_idx]]

        if sparse.isspmatrix(X):
            X = X.toarray()
        X = X.astype('float32')

        pred_val = sess.run(logits_var, feed_dict={vae.input_ph: X})
        # exclude examples from training and validation (if any)
        pred_val[X.nonzero()] = -np.inf
        preds.extend(pred_val)
        n100_list.append(NDCG_binary_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=100))
        r20_list.append(Recall_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=20))
        r50_list.append(Recall_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=50))


num_users = 1000
num_items = 100
num_rounds = 100

print("USUARIOS: "+str(test_data_tr.shape[0]))
preds = np.array(preds)
rnd_users = np.random.choice(range(test_data_tr.shape[0]), num_users, replace=False)
preds = preds[rnd_users,:]
#preds = preds[1000:1100,:]
#plot_sorted_preds(preds)
#print(sorted(preds))
topkItens = np.argsort(preds)[:,-num_items:]
print(topkItens.shape)

predsn = []
for user in range(topkItens.shape[0]):
    predsn.append(preds[user,sorted(topkItens[user,:])])
predsn = np.array(predsn)
print(predsn.shape)
dcg_gt = dcg_k_users(predsn)
print(dcg_gt)
#subSet = test_data_tr.T[topkItens,0]
test_data_tr = test_data_tr[rnd_users,:]
print(test_data_tr.shape)


def test_model_k_rounds(p_dims, noise, test_data_tr, num_rounds, num_users, num_items, topkItens, dcg_gt):

    tf.reset_default_graph()
    vae = MultiVAE(p_dims, lam=0.0)
    saver, logits_var, _, _, _ = vae.build_graph(noise)
    arch_str = "I-%s-I" % ('-'.join([str(d) for d in vae.dims[1:-1]]))
    chkpt_dir = './chkpt/ml-20m/VAE_anneal{}K_cap{:1.1E}/{}'.format(
        total_anneal_steps/1000, anneal_cap, arch_str)
    print("chkpt directory: %s" % chkpt_dir)
    
    preds = []
    preds_k_rounds = []
    
    for k in range(num_rounds):
        preds = []
        with tf.Session() as sess:
            saver.restore(sess, '{}/model'.format(chkpt_dir))
            X = test_data_tr
        
            if sparse.isspmatrix(X):
                X = X.toarray()
            X = X.astype('float32')
        
            pred_val = sess.run(logits_var, feed_dict={vae.input_ph: X})
            # exclude examples from training and validation (if any)
            pred_val[X.nonzero()] = -np.inf
            preds.extend(pred_val)
        preds_k_rounds.append(preds)    
    
    preds_k_rounds = np.array(preds_k_rounds)
    print(preds_k_rounds.shape)
    
    preds_k_rounds_filter = np.zeros((num_rounds,num_users,num_items))
    for user in range(num_users):
        preds_k_rounds_filter[:,user,:] = preds_k_rounds[:,user,sorted(topkItens[user,:])]
    
    #preds_k_rounds = preds_k_rounds[:,:,sorted(topkItens)]
    preds_k_rounds_filter = np.array(preds_k_rounds_filter)
    print(preds_k_rounds_filter.shape)
    
    ufairs, ndcgs = Fairness_at_k_rounds(preds_k_rounds_filter, dcg_gt)
    return ufairs,ndcgs

# NO NOISE
ufairs, ndcgs = test_model_k_rounds(p_dims, 0, test_data_tr, num_rounds, num_users, num_items, topkItens, dcg_gt)
# NORMAL NOISE STD 0.5
ufairs_n05, ndcgs_n05 = test_model_k_rounds(p_dims, 1, test_data_tr, num_rounds, num_users, num_items, topkItens, dcg_gt)
# NORMAL NOISE STD 1.0
ufairs_n10, ndcgs_n10 = test_model_k_rounds(p_dims, 2, test_data_tr, num_rounds, num_users, num_items, topkItens, dcg_gt)
# NORMAL NOISE STD 2.0
ufairs_n20, ndcgs_n20 = test_model_k_rounds(p_dims, 3, test_data_tr, num_rounds, num_users, num_items, topkItens, dcg_gt)
# UNIFORM NOISE 
ufairs_unif, ndcgs_unif = test_model_k_rounds(p_dims, 4, test_data_tr, num_rounds, num_users, num_items, topkItens, dcg_gt)

plot_comparison([ufairs,ufairs_n05, ufairs_n10,ufairs_n20, ufairs_unif],[1-ndcgs,1-ndcgs_n05, 1-ndcgs_n10,1-ndcgs_n20,1-ndcgs_unif],['original','N(std=0.5)','N(std=1.0)','N(std=2.0)','uniform'])

#n100_list = np.concatenate(n100_list)
#r20_list = np.concatenate(r20_list)
#r50_list = np.concatenate(r50_list)

#print("Test NDCG@100=%.5f (%.5f)" % (np.mean(n100_list), np.std(n100_list) / np.sqrt(len(n100_list))))
#print("Test Recall@20=%.5f (%.5f)" % (np.mean(r20_list), np.std(r20_list) / np.sqrt(len(r20_list))))
#print("Test Recall@50=%.5f (%.5f)" % (np.mean(r50_list), np.std(r50_list) / np.sqrt(len(r50_list))))
