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

def main(dataset):
    pro_dir = load_data(dataset)
    #pro_dir = load_netflix_data()
    
    # Model Definition and Training
    
    unique_sid = list()
    with open(os.path.join(pro_dir, 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())
    
    n_items = len(unique_sid)
    print("NUMERO ITENS: "+str(n_items))
    
    train_data = load_train_data(os.path.join(pro_dir, 'train.csv'),n_items)
    
    vad_data_tr, vad_data_te = load_tr_te_data(os.path.join(pro_dir, 'validation_tr.csv'),
                                               os.path.join(pro_dir, 'validation_te.csv'),
                                               n_items)
    
    N = train_data.shape[0]
    idxlist = list(range(N))
    
    # training batch size
    batch_size = 500
    batches_per_epoch = int(np.ceil(float(N) / batch_size))
    
    N_vad = vad_data_tr.shape[0]
    idxlist_vad = list(range(N_vad))
    
    # validation batch size (since the entire validation set might not fit into GPU memory)
    batch_size_vad = 2000
    
    # the total number of gradient updates for annealing
    total_anneal_steps = 200000
    # largest annealing parameter
    anneal_cap = 0.2
    
    p_dims = [200, 600, n_items]
    #p_dims = [200, 600, batch_size]
    
    tf.reset_default_graph()
    vae = MultiVAE(p_dims, lam=0.0, random_seed=98765)
    
    saver, logits_var, loss_var, train_op_var, merged_var = vae.build_graph(2)
    
    ndcg_var = tf.Variable(0.0)
    ndcg_dist_var = tf.placeholder(dtype=tf.float64, shape=None)
    ndcg_summary = tf.summary.scalar('ndcg_at_k_validation', ndcg_var)
    ndcg_dist_summary = tf.summary.histogram('ndcg_at_k_hist_validation', ndcg_dist_var)
    merged_valid = tf.summary.merge([ndcg_summary, ndcg_dist_summary])
    
    arch_str = "I-%s-I" % ('-'.join([str(d) for d in vae.dims[1:-1]]))
    
    #log_dir = '/volmount/log/ml-20m/VAE_anneal{}K_cap{:1.1E}/{}'.format(
    log_dir = './log/'+dataset+'/VAE_anneal{}K_cap{:1.1E}/{}'.format(
        total_anneal_steps/1000, anneal_cap, arch_str)
    
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    
    print("log directory: %s" % log_dir)
    summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())
    
    #chkpt_dir = '/volmount/chkpt/ml-20m/VAE_anneal{}K_cap{:1.1E}/{}'.format(
    chkpt_dir = './chkpt/'+dataset+'/VAE_anneal{}K_cap{:1.1E}/{}'.format(
        total_anneal_steps/1000, anneal_cap, arch_str)
    
    if not os.path.isdir(chkpt_dir):
        os.makedirs(chkpt_dir)
    
    print("chkpt directory: %s" % chkpt_dir)
    
    n_epochs = 200
    ndcgs_vad = []
    
    with tf.Session() as sess:
    
        init = tf.global_variables_initializer()
        sess.run(init)
        best_ndcg = -np.inf
        update_count = 0.0
        
        for epoch in range(n_epochs):
            print("epoch: "+str(epoch))
            np.random.shuffle(idxlist)
            # train for one epoch
            for bnum, st_idx in enumerate(range(0, N, batch_size)):
                end_idx = min(st_idx + batch_size, N)
    
                X = train_data[idxlist[st_idx:end_idx]]
    #            X = train_data[train_data.index.isin(idxlist[st_idx:end_idx])]
                if sparse.isspmatrix(X):
                    X = X.toarray()
                X = X.astype('float32')           
    
                if total_anneal_steps > 0:
                    anneal = min(anneal_cap, 1. * update_count / total_anneal_steps)
                else:
                    anneal = anneal_cap
                
                feed_dict = {vae.input_ph: X, 
                             vae.keep_prob_ph: 0.5, 
                             vae.anneal_ph: anneal,
                             vae.is_training_ph: 1}     
                sess.run(train_op_var, feed_dict=feed_dict)
    
                if bnum % 100 == 0:
                    summary_train = sess.run(merged_var, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_train, 
                                               global_step=epoch * batches_per_epoch + bnum) 
                
                update_count += 1
    
            # compute validation NDCG
            ndcg_dist = []
            for bnum, st_idx in enumerate(range(0, N_vad, batch_size_vad)):
                end_idx = min(st_idx + batch_size_vad, N_vad)
                X = vad_data_tr[idxlist_vad[st_idx:end_idx]]
    
                if sparse.isspmatrix(X):
                    X = X.toarray()
                X = X.astype('float32')
            
                pred_val = sess.run(logits_var, feed_dict={vae.input_ph: X} )
                # exclude examples from training and validation (if any)
                pred_val[X.nonzero()] = -np.inf
                ndcg_dist.append(NDCG_binary_at_k_batch(pred_val, vad_data_te[idxlist_vad[st_idx:end_idx]]))
            
            ndcg_dist = np.concatenate(ndcg_dist)
            ndcg_ = ndcg_dist.mean()
            ndcgs_vad.append(ndcg_)
            merged_valid_val = sess.run(merged_valid, feed_dict={ndcg_var: ndcg_, ndcg_dist_var: ndcg_dist})
            summary_writer.add_summary(merged_valid_val, epoch)
    
            # update the best model (if necessary)
            if ndcg_ > best_ndcg:
                saver.save(sess, '{}/model'.format(chkpt_dir))
                best_ndcg = ndcg_       
    
    print(ndcgs_vad)
    
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
    
    chkpt_dir = './chkpt/'+dataset+'/VAE_anneal{}K_cap{:1.1E}/{}'.format(
        total_anneal_steps/1000, anneal_cap, arch_str)
    print("chkpt directory: %s" % chkpt_dir)
    
    n100_list, r20_list, r50_list = [], [], []
    
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
            n100_list.append(NDCG_binary_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=100))
            r20_list.append(Recall_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=20))
            r50_list.append(Recall_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=50))
    
    n100_list = np.concatenate(n100_list)
    r20_list = np.concatenate(r20_list)
    r50_list = np.concatenate(r50_list)
    
    
    print("Test NDCG@100=%.5f (%.5f)" % (np.mean(n100_list), np.std(n100_list) / np.sqrt(len(n100_list))))
    print("Test Recall@20=%.5f (%.5f)" % (np.mean(r20_list), np.std(r20_list) / np.sqrt(len(r20_list))))
    print("Test Recall@50=%.5f (%.5f)" % (np.mean(r50_list), np.std(r50_list) / np.sqrt(len(r50_list))))

if __name__ == "__main__":
    main(sys.argv[1])

