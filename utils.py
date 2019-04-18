import numpy as np
import sys
import pandas as pd
from scipy import sparse
import os

def load_data(dataset):
    if dataset == 'netflix': pro_dir = load_netflix_data()
    if dataset == 'ml-20m': pro_dir = load_movielens_data()
    if dataset == 'msd': pro_dir = load_msd_data()
    return pro_dir

def load_netflix_data():

    DATA_DIR = '../data/netflix/'
    raw_data_train = pd.read_csv(os.path.join(DATA_DIR, 'NF_TRAIN/nf.train.txt'), sep='\t', header=None, names=['userId','movieId','rating'])
    raw_data_valid = pd.read_csv(os.path.join(DATA_DIR, 'NF_VALID/nf.valid.txt'), sep='\t', header=None, names=['userId','movieId','rating'])
    raw_data_test = pd.read_csv(os.path.join(DATA_DIR, 'NF_TEST/nf.test.txt'), sep='\t', header=None, names=['userId','movieId','rating'])
    raw_data = pd.concat([raw_data_train, raw_data_valid, raw_data_test])
    pro_dir = os.path.join(DATA_DIR, 'pro_sg')
    raw_data = raw_data[raw_data['rating'] > 3.5]
    
    # Only keep items that are clicked on by at least 5 users
    raw_data, user_activity, item_popularity = filter_triplets(raw_data)
    raw_data = raw_data.sort_values(by=['userId'])
    raw_data = raw_data.sort_values(by=['userId','movieId'])
    raw_data = raw_data.reset_index(drop=True)
    _, _, _ = get_user_by_mean(raw_data)
    
    sparsity = 1. * raw_data.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])
    
    print("After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)" %
          (raw_data.shape[0], user_activity.shape[0], item_popularity.shape[0], sparsity * 100))
    
    unique_uid = user_activity.index
    
    np.random.seed(98765)
    idx_perm = np.random.permutation(unique_uid.size)
    unique_uid = unique_uid[idx_perm]

    # create train/validation/test users
    n_users = unique_uid.size
    n_heldout_users = 40000
    
    tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
    vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
    te_users = unique_uid[(n_users - n_heldout_users):]
    
    train_plays = raw_data.loc[raw_data['userId'].isin(tr_users)]
    unique_sid = pd.unique(train_plays['movieId'])
    
    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))
    
    if not os.path.exists(pro_dir):
        os.makedirs(pro_dir)
    
    with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
        for sid in unique_sid:
            f.write('%s\n' % sid)
    
    vad_plays = raw_data.loc[raw_data['userId'].isin(vd_users)]
    vad_plays = vad_plays.loc[vad_plays['movieId'].isin(unique_sid)]
    vad_plays_tr, vad_plays_te, vad_plays_raw = split_train_test_proportion(vad_plays)
    
    test_plays = raw_data.loc[raw_data['userId'].isin(te_users)]
    test_plays = test_plays.loc[test_plays['movieId'].isin(unique_sid)]
    test_plays_tr, test_plays_te, test_plays_raw = split_train_test_proportion(test_plays)
    user1, user2, user3 = get_user_by_mean(test_plays_raw)
    
    train_data = numerize(train_plays, profile2id, show2id)
    train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)
    vad_data_tr = numerize(vad_plays_tr, profile2id, show2id)
    vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)
    vad_data_te = numerize(vad_plays_te, profile2id, show2id)
    vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)
    test_data_tr = numerize(test_plays_tr, profile2id, show2id)
    test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)
    test_data_te = numerize(test_plays_te, profile2id, show2id)
    test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)
    test_data = numerize(test_plays_raw, profile2id, show2id)
    test_data.to_csv(os.path.join(pro_dir, 'test.csv'), index=False)
    
    user1 = numerize(user1, profile2id, show2id)
    user1.to_csv(os.path.join(pro_dir, 'test_user1.csv'), index=False)
    user2 = numerize(user2, profile2id, show2id)
    user2.to_csv(os.path.join(pro_dir, 'test_user2.csv'), index=False)
    user3 = numerize(user3, profile2id, show2id)
    user3.to_csv(os.path.join(pro_dir, 'test_user3.csv'), index=False)
 
    return pro_dir

def load_movielens_data():

    DATA_DIR = '../data/ml-20m/'
    raw_data = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'), header=0)
    pro_dir = os.path.join(DATA_DIR, 'pro_sg')
    
    raw_data = raw_data[raw_data['rating'] > 3.5]
    
    # Only keep items that are clicked on by at least 5 users
    raw_data, user_activity, item_popularity = filter_triplets(raw_data)
    raw_data = raw_data.sort_values(by=['userId','movieId'])
    raw_data = raw_data.reset_index(drop=True)
    _, _, _ = get_user_by_mean(raw_data)
    
    sparsity = 1. * raw_data.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])
    
    print("After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)" %
          (raw_data.shape[0], user_activity.shape[0], item_popularity.shape[0], sparsity * 100))
    
    unique_uid = user_activity.index
    
    np.random.seed(98765)
    idx_perm = np.random.permutation(unique_uid.size)
    unique_uid = unique_uid[idx_perm]

    # create train/validation/test users
    n_users = unique_uid.size
    n_heldout_users = 10000
    
    tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
    vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
    te_users = unique_uid[(n_users - n_heldout_users):]
    
    train_plays = raw_data.loc[raw_data['userId'].isin(tr_users)]
    unique_sid = pd.unique(train_plays['movieId'])
    
    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))
    
    if not os.path.exists(pro_dir):
        os.makedirs(pro_dir)
    
    with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
        for sid in unique_sid:
            f.write('%s\n' % sid)
    
    vad_plays = raw_data.loc[raw_data['userId'].isin(vd_users)]
    vad_plays = vad_plays.loc[vad_plays['movieId'].isin(unique_sid)]
    vad_plays_tr, vad_plays_te, vad_plays_raw = split_train_test_proportion(vad_plays)
    
    test_plays = raw_data.loc[raw_data['userId'].isin(te_users)]
    test_plays = test_plays.loc[test_plays['movieId'].isin(unique_sid)]
    test_plays_tr, test_plays_te, test_plays_raw = split_train_test_proportion(test_plays)
    user1, user2, user3 = get_user_by_mean(test_plays_raw)
    
    train_data = numerize(train_plays, profile2id, show2id)
    train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)
    vad_data_tr = numerize(vad_plays_tr, profile2id, show2id)
    vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)
    vad_data_te = numerize(vad_plays_te, profile2id, show2id)
    vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)
    test_data_tr = numerize(test_plays_tr, profile2id, show2id)
    test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)
    test_data_te = numerize(test_plays_te, profile2id, show2id)
    test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)
    test_data = numerize(test_plays_raw, profile2id, show2id)
    test_data.to_csv(os.path.join(pro_dir, 'test.csv'), index=False)
    
    user1 = numerize(user1, profile2id, show2id)
    user1.to_csv(os.path.join(pro_dir, 'test_user1.csv'), index=False)
    user2 = numerize(user2, profile2id, show2id)
    user2.to_csv(os.path.join(pro_dir, 'test_user2.csv'), index=False)
    user3 = numerize(user3, profile2id, show2id)
    user3.to_csv(os.path.join(pro_dir, 'test_user3.csv'), index=False)

    return pro_dir

def load_msd_data():

    DATA_DIR = '../data/msd/'
    raw_data = pd.read_csv(os.path.join(DATA_DIR, 'train_triplets-random.txt'), sep='\t', header=None, names=['userId','movieId','rating'])
    pro_dir = os.path.join(DATA_DIR, 'pro_sg')
    
    #raw_data = raw_data[raw_data['rating'] > 3.5]
    
    # Only keep items that are clicked on by at least 5 users
    raw_data, user_activity, item_popularity = filter_triplets(raw_data, 20, 200)
    raw_data = raw_data.sort_values(by=['userId','movieId'])
    raw_data = raw_data.reset_index(drop=True)
    _, _, _ = get_user_by_mean(raw_data)
    
    sparsity = 1. * raw_data.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])
    
    print("After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)" %
          (raw_data.shape[0], user_activity.shape[0], item_popularity.shape[0], sparsity * 100))
    
    unique_uid = user_activity.index
    
    np.random.seed(98765)
    idx_perm = np.random.permutation(unique_uid.size)
    unique_uid = unique_uid[idx_perm]

    # create train/validation/test users
    n_users = unique_uid.size
    n_heldout_users = 50000
    
    tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
    vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
    te_users = unique_uid[(n_users - n_heldout_users):]
    
    train_plays = raw_data.loc[raw_data['userId'].isin(tr_users)]
    unique_sid = pd.unique(train_plays['movieId'])
    
    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))
    
    if not os.path.exists(pro_dir):
        os.makedirs(pro_dir)
    
    with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
        for sid in unique_sid:
            f.write('%s\n' % sid)
    
    vad_plays = raw_data.loc[raw_data['userId'].isin(vd_users)]
    vad_plays = vad_plays.loc[vad_plays['movieId'].isin(unique_sid)]
    vad_plays_tr, vad_plays_te, vad_plays_raw = split_train_test_proportion(vad_plays)
    
    test_plays = raw_data.loc[raw_data['userId'].isin(te_users)]
    test_plays = test_plays.loc[test_plays['movieId'].isin(unique_sid)]
    test_plays_tr, test_plays_te, test_plays_raw = split_train_test_proportion(test_plays)
    user1, user2, user3 = get_user_by_mean(test_plays_raw)
    
    train_data = numerize(train_plays, profile2id, show2id)
    train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)
    vad_data_tr = numerize(vad_plays_tr, profile2id, show2id)
    vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)
    vad_data_te = numerize(vad_plays_te, profile2id, show2id)
    vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)
    test_data_tr = numerize(test_plays_tr, profile2id, show2id)
    test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)
    test_data_te = numerize(test_plays_te, profile2id, show2id)
    test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)
    test_data = numerize(test_plays_raw, profile2id, show2id)
    test_data.to_csv(os.path.join(pro_dir, 'test.csv'), index=False)
    
    user1 = numerize(user1, profile2id, show2id)
    user1.to_csv(os.path.join(pro_dir, 'test_user1.csv'), index=False)
    user2 = numerize(user2, profile2id, show2id)
    user2.to_csv(os.path.join(pro_dir, 'test_user2.csv'), index=False)
    user3 = numerize(user3, profile2id, show2id)
    user3.to_csv(os.path.join(pro_dir, 'test_user3.csv'), index=False)

    return pro_dir

def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count

def get_ratings_histogram(data, labels):
    user_type = []
    data_grouped_by_rating = data.groupby('rating')
    for i, (_, group) in enumerate(data_grouped_by_rating):
        user_type.append(len(group['rating'])) 
    plot_bar_graph(user_type, labels)
    return 0

def get_user_by_mean(data):
    df1 = data.groupby('userId').size() 
    quant1 = np.quantile(df1.values,1/3)
    quant2 = np.quantile(df1.values,2/3)
    print(quant1,quant2)    

    user1 = data.loc[data['userId'].isin(df1[df1 < quant1].index.values)]
    l1 = list(df1[df1 >= quant1].index.values)
    l2 = list(df1[df1 < quant2].index.values)
    user2 = data.loc[data['userId'].isin(np.intersect1d(l1,l2))]
    user3 = data.loc[data['userId'].isin(df1[df1 >= quant2].index.values)]
    print(len(set(user1['userId'])),len(set(user2['userId'])),len(set(user3['userId'])))

    return user1, user2, user3

def filter_triplets(tp, min_uc=5, min_sc=0):
    # Only keep the triplets for items which were clicked on by at least min_sc users.
    if min_sc > 0:
        itemcount = get_count(tp, 'movieId')
        tp = tp[tp['movieId'].isin(itemcount.index[itemcount >= min_sc])]

    # Only keep the triplets for users who clicked on at least min_uc items
    # After doing this, some of the items will have less than min_uc users, but should only be a small proportion
    if min_uc > 0:
        usercount = get_count(tp, 'userId')
        tp = tp[tp['userId'].isin(usercount.index[usercount >= min_uc])]

    # Update both usercount and itemcount after filtering
    usercount, itemcount = get_count(tp, 'userId'), get_count(tp, 'movieId')
    return tp, usercount, itemcount

def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby('userId')
    tr_list, te_list, raw_list = list(), list(), list()

    np.random.seed(98765)

    for i, (_, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
            raw_list.append(group)
        else:
            tr_list.append(group)
            raw_list.append(group)

        if i % 1000 == 0:
            print("%d users sampled" % i)
            sys.stdout.flush()

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)
    data_raw = pd.concat(raw_list)
    
    return data_tr, data_te, data_raw

def numerize(tp, profile2id, show2id):
    uid = list(map(lambda x: profile2id[x], tp['userId']))
    sid = list(map(lambda x: show2id[x], tp['movieId']))
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])

def numerize_test(tp, profile2id, show2id):
    uid = list(map(lambda x: profile2id[x], tp['uid']))
    sid = list(map(lambda x: show2id[x], tp['sid']))
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])

def load_train_data(csv_file,n_items):
    tp = pd.read_csv(csv_file)
    n_users = tp['uid'].max() + 1

    rows, cols = tp['uid'], tp['sid']
    data = sparse.csr_matrix((np.ones_like(rows),
                             (rows, cols)), dtype='float64',
                             shape=(n_users, n_items))
    return data

def load_test_data(csv_file,n_items):
    tp = pd.read_csv(csv_file)
    tp = tp.sort_values(by=['uid','sid'])
    tp = tp.reset_index(drop=True)

    n_users = set(tp['uid'].values)
    profile2id = dict((pid, i) for (i, pid) in enumerate(n_users))
    show2id = dict((sid, i) for (i, sid) in enumerate(range(n_items)))
    tp = numerize_test(tp, profile2id, show2id)

    start_idx = tp['uid'].min()
    end_idx = tp['uid'].max()
    
    rows, cols = tp['uid'] - start_idx, tp['sid']
    data = sparse.csr_matrix((np.ones_like(rows),
                             (rows, cols)), dtype='float64',
                             shape=(end_idx - start_idx + 1, n_items))
    return data

def load_tr_te_data(csv_file_tr, csv_file_te, n_items):
    tp_tr = pd.read_csv(csv_file_tr)
    tp_te = pd.read_csv(csv_file_te)

    start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
    end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

    rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
    rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

    data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                             (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te),
                             (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    return data_tr, data_te

def plot_curve(ufair,ndcg):
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots()
    plt.plot( range(len(ufair)), ufair)
    plt.plot( range(len(ndcg)), ndcg)
    #plt.ylabel("Validation NDCG@100")
    #plt.xlabel("Epochs")
    #plt.savefig('novelty.pdf', bbox_inches='tight')
    plt.show()

def set_box_color(bp, color):
    import matplotlib.pyplot as plt
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

def plot_comparison(data_a, data_b, ticks, dataset, test_file):
    import matplotlib.pyplot as plt
    plt.figure()
    bpl = plt.boxplot(data_a, positions=np.array(range(len(data_a)))*2.0-0.4, sym='', widths=0.6)
    bpr = plt.boxplot(data_b, positions=np.array(range(len(data_b)))*2.0+0.4, sym='', widths=0.6)
    #bpr = plt.boxplot(data_c, positions=np.array(range(len(data_b)))*2.0+0.4, sym='', widths=0.6)
    set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
    set_box_color(bpr, '#2C7BB6')
    #set_box_color(bpr, '#2C7BB6')
    
    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#D7191C', label='Unfairness@100')
    plt.plot([], c='#2C7BB6', label='1 - NDCG@100')
    #plt.plot([], c='#2C7BB6', label='CNN + STFT')
    plt.legend()
    
    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks)*2)
    #plt.ylim(np.min(np.concatenate((data_a,data_b),axis=1)), np.max(np.concatenate((data_a,data_b),axis=1)))
    plt.tight_layout()
    plt.savefig('plots/boxcompare_'+dataset+'_'+test_file+'.pdf')

def plot_sorted_preds(preds):
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots()
    plt.plot( range(len(preds)), sorted(preds)[::-1])
    plt.ylabel("Scores")
    plt.xlabel("Items")
    plt.savefig('preds_sorted.pdf', bbox_inches='tight')
    #plt.show()

def plot_bar_graph(data, labels):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plt.bar(np.arange(len(data)),data)
    plt.xticks(np.arange(len(data)),labels)
    #plt.show() 
    plt.savefig('ratings_hist.pdf', bbox_inches='tight')

def plot_histogram(data):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    print(max(data))
    plt.hist(data,int(max(data)))
    #plt.show() 
    plt.savefig('user_hist.pdf', bbox_inches='tight')
