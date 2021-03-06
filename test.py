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
from model import MultiDAE, MultiVAE
from metrics import *
from utils import *


def test_model_k_rounds(p_dims, noise, test_data, num_rounds, num_users, num_items, topkItens, dcg_gt, dataset):
    N_test = test_data.shape[0]
    idxlist_test = range(N_test)
    
    batch_size_test = 2000
    tf.reset_default_graph()
    vae = MultiVAE(p_dims, lam=0.0)
    saver, logits_var, _, _, _ = vae.build_graph(noise)
    arch_str = "I-%s-I" % ('-'.join([str(d) for d in vae.dims[1:-1]]))
    # the total number of gradient updates for annealing
    total_anneal_steps = 200000
    # largest annealing parameter
    anneal_cap = 0.2
    chkpt_dir = './chkpt/'+dataset+'/VAE_anneal{}K_cap{:1.1E}/{}'.format(
        total_anneal_steps/1000, anneal_cap, arch_str)
    print("chkpt directory: %s" % chkpt_dir)
    
    preds_k_rounds = []
    
    for k in range(num_rounds):
        preds = []
        with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
        #with tf.Session() as sess:
            saver.restore(sess, '{}/model'.format(chkpt_dir))
            for bnum, st_idx in enumerate(range(0, N_test, batch_size_test)):
                end_idx = min(st_idx + batch_size_test, N_test)
                X = test_data[idxlist_test[st_idx:end_idx]]
            #    X = test_data
                
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

def main(dataset, test_file):
    #dataset = "netflix"
    #dataset = "ml-20m"

    DATA_DIR = '../data/'+dataset+'/'
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
    #test_data_tr, test_data_te = load_tr_te_data(
    #    os.path.join(pro_dir, 'test_tr.csv'),
    #    os.path.join(pro_dir, 'test_te.csv'),
    #    n_items)
    
    test_data = load_test_data(
        os.path.join(pro_dir, test_file),
        n_items)
    
    N_test = test_data.shape[0]
    idxlist_test = range(N_test)
    
    batch_size_test = 2000
    
    tf.reset_default_graph()
    vae = MultiVAE(p_dims, lam=0.0)
    saver, logits_var, _, _, _ = vae.build_graph(0)
    
    arch_str = "I-%s-I" % ('-'.join([str(d) for d in vae.dims[1:-1]]))
    
    chkpt_dir = './chkpt/'+dataset+'/VAE_anneal{}K_cap{:1.1E}/{}'.format(
        total_anneal_steps/1000, anneal_cap, arch_str)
    print("chkpt directory: %s" % chkpt_dir)
    
    n100_list, r20_list, r50_list = [], [], []
    preds = []
    
    with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
    #with tf.Session() as sess:
        saver.restore(sess, '{}/model'.format(chkpt_dir))
        for bnum, st_idx in enumerate(range(0, N_test, batch_size_test)):
            end_idx = min(st_idx + batch_size_test, N_test)
            X = test_data[idxlist_test[st_idx:end_idx]]
            #X = test_data
                
            if sparse.isspmatrix(X):
                X = X.toarray()
            X = X.astype('float32')
            
            pred_val = sess.run(logits_var, feed_dict={vae.input_ph: X})
            # exclude examples from training and validation (if any)
            pred_val[X.nonzero()] = -np.inf
            preds.extend(pred_val)
        
    num_users = 1000
    num_items = 100
    num_rounds = 100
    
    print("USUARIOS: "+str(test_data.shape[0]))
    preds = np.array(preds)
    np.random.seed(98765)
    rnd_users = np.random.choice(range(test_data.shape[0]), num_users, replace=False)
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
    #subSet = test_data_tr.T[topkItens,0]
    test_data = test_data[rnd_users,:]

    print("NO NOISE")
    ufairs, ndcgs = test_model_k_rounds(p_dims, 0, test_data, num_rounds, num_users, num_items, topkItens, dcg_gt, dataset)
    print(np.mean(ufairs),np.std(ufairs),np.mean(ndcgs),np.std(ndcgs))
    print("NORMAL NOISE STD 0.5")
    ufairs_n05, ndcgs_n05 = test_model_k_rounds(p_dims, 1, test_data, num_rounds, num_users, num_items, topkItens, dcg_gt, dataset)
    print(np.mean(ufairs_n05),np.std(ufairs_n05),np.mean(ndcgs_n05),np.std(ndcgs_n05))
    print("NORMAL NOISE STD 1.0")
    ufairs_n10, ndcgs_n10 = test_model_k_rounds(p_dims, 2, test_data, num_rounds, num_users, num_items, topkItens, dcg_gt, dataset)
    print(np.mean(ufairs_n10),np.std(ufairs_n10),np.mean(ndcgs_n10),np.std(ndcgs_n10))
    print("NORMAL NOISE STD 2.0")
    ufairs_n20, ndcgs_n20 = test_model_k_rounds(p_dims, 3, test_data, num_rounds, num_users, num_items, topkItens, dcg_gt, dataset)
    print(np.mean(ufairs_n20),np.std(ufairs_n20),np.mean(ndcgs_n20),np.std(ndcgs_n20))
    print("UNIFORM NOISE") 
    ufairs_unif, ndcgs_unif = test_model_k_rounds(p_dims, 4, test_data, num_rounds, num_users, num_items, topkItens, dcg_gt, dataset)
    print(np.mean(ufairs_unif),np.std(ufairs_unif),np.mean(ndcgs_unif),np.std(ndcgs_unif))
    
    plot_comparison([ufairs,ufairs_n05, ufairs_n10,ufairs_n20, ufairs_unif],[1-ndcgs,1-ndcgs_n05, 1-ndcgs_n10,1-ndcgs_n20,1-ndcgs_unif],['original','N(std=0.5)','N(std=1.0)','N(std=2.0)','uniform'], dataset, test_file)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

