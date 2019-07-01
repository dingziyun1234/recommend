# coding:utf-8
# VAE_CF 主函数
# 训练/验证数据，超参数

# 数据的处理
# data：ml-20m,ratings.csv
#
import os
import shutil

import numpy as np
from scipy import sparse

import matplotlib.pyplot as plt

import bottleneck as bn

## change `DATA_DIR` to the location where movielens-20m dataset sits
DATA_DIR = '/Users/dingziyun/Downloads/ml-20m/'

pro_dir = os.path.join(DATA_DIR, 'pro_sg')
#############################  加载先前保存的数据，处理为二值矩阵  ################################
# 加载训练组里 所有的物品id

unique_sid = list()
with open(os.path.join(pro_dir, 'unique_sid.txt'), 'r') as f:
    for line in f:
        unique_sid.append(line.strip())

n_items = len(unique_sid)

# 加载文件数据 {user_index,item_index}，并将数据处理为 二值矩阵
def load_train_data(csv_file):
    tp = pd.read_csv(csv_file)
    n_users = tp['uid'].max() + 1

    rows, cols = tp['uid'], tp['sid']
    data = sparse.csr_matrix((np.ones_like(rows), #返回一个用1填充的跟输入 形状和类型 一致的数组。
                             (rows, cols)), dtype='float64',
                             shape=(n_users, n_items)) # 压缩稀疏行矩阵
    return data

train_data = load_train_data(os.path.join(pro_dir, 'train.csv')) # 加载train.csv 并处理为二值矩阵

# 同时加载 测试组 的tr和te 数据
def load_tr_te_data(csv_file_tr, csv_file_te):
    tp_tr = pd.read_csv(csv_file_tr)
    tp_te = pd.read_csv(csv_file_te)

    start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min()) #
    end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

    rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
    rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

    data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                             (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te),
                             (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    return data_tr, data_te

# 预留组的数据加载
vad_data_tr, vad_data_te = load_tr_te_data(os.path.join(pro_dir, 'validation_tr.csv'),
                                           os.path.join(pro_dir, 'validation_te.csv'))
# Set up training hyperparameters 设置训练超参数
N = train_data.shape[0] # 训练组的 条目 个数
idxlist = list(range(N)) # id的list

# training batch size 训练的批次大小
batch_size = 500
batches_per_epoch = int(np.ceil(float(N) / batch_size))

N_vad = vad_data_tr.shape[0]
idxlist_vad = list(range(N_vad))

# validation batch size (since the entire validation set might not fit into GPU memory)
# 验证批大小(因为整个验证集可能不适合GPU内存)
batch_size_vad = 2000

# the total number of gradient updates for annealing
#退火梯度更新的总数
total_anneal_steps = 200000
# largest annealing parameter
anneal_cap = 0.2

###################################   评估函数    ########################################
#  Evaluate function: Normalized discounted cumulative gain (NDCG@k) and Recall@k
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

#####################  Train a Multi-VAE^{PR} ###########################
from VAE_CF_model import MultiDAE,MultiVAE
import tensorflow as tf

p_dims = [200, 600, n_items] # 模型 MultiVAE 的输入 ，n_items = len(unique_sid)

tf.reset_default_graph() # 用于清除默认图形堆栈并重置全局默认图形。 注意：默认图形是当前线程的一个属性
vae = MultiVAE(p_dims, lam=0.0, random_seed=98765)
# 返回存储的数据，logits_var是预测的xu，loss_var=neg_ELBO是对数似然边际下限，
# train_op_var是最小化neg_ELBO的方法，merged_var 可视化TensorBoard
saver, logits_var, loss_var, train_op_var, merged_var = vae.build_graph()

ndcg_var = tf.Variable(0.0)
ndcg_dist_var = tf.placeholder(dtype=tf.float64, shape=None)
ndcg_summary = tf.summary.scalar('ndcg_at_k_validation', ndcg_var)
ndcg_dist_summary = tf.summary.histogram('ndcg_at_k_hist_validation', ndcg_dist_var)
merged_valid = tf.summary.merge([ndcg_summary, ndcg_dist_summary])

################################  创建检查日志  ########################################
# 日志目录
arch_str = "I-%s-I" % ('-'.join([str(d) for d in vae.dims[1:-1]]))

log_dir = '/Users/dingziyun/Downloads/ml-20m/log/VAE_anneal{}K_cap{:1.1E}/{}'.format(
    total_anneal_steps/1000, anneal_cap, arch_str)

if os.path.exists(log_dir):
    shutil.rmtree(log_dir)

print("log directory: %s" % log_dir)
summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())
# chkpt目录
chkpt_dir = '/Users/dingziyun/Downloads/ml-20m/log/VAE_anneal{}K_cap{:1.1E}/{}'.format(
    total_anneal_steps / 1000, anneal_cap, arch_str)

if not os.path.isdir(chkpt_dir):
    os.makedirs(chkpt_dir)

print("chkpt directory: %s" % chkpt_dir)

#################################     数据模型训练     ########################################
import pandas as pd
n_epochs = 10 # 源码200，代数

ndcgs_vad = []

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    best_ndcg = -np.inf #

    update_count = 0.0 # 更新计数

    for epoch in range(n_epochs):  # 一共训练n_epochs代
        np.random.shuffle(idxlist)
        # train for one epoch 一代的训练
        for bnum, st_idx in enumerate(range(0, N, batch_size)):
            end_idx = min(st_idx + batch_size, N)
            X = train_data[idxlist[st_idx:end_idx]] # 训练数据

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
            sess.run(train_op_var, feed_dict=feed_dict) # train_op_var是最小化neg_ELBO的方法

            if bnum % 10 == 0:
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

            pred_val = sess.run(logits_var, feed_dict={vae.input_ph: X}) # 预测的结果，logits_var是预测的xu
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
            saver.save(sess, '{}/model'.format(chkpt_dir)) # 保存模型
            best_ndcg = ndcg_

# 可视化结果
plt.figure(figsize=(12, 3))
plt.plot(ndcgs_vad)
plt.ylabel("Validation NDCG@100")
plt.xlabel("Epochs")
plt.show()

################################  加载测试数据并计算测试指标  ########################################

test_data_tr, test_data_te = load_tr_te_data(
    os.path.join(pro_dir, 'test_tr.csv'),
    os.path.join(pro_dir, 'test_te.csv'))

N_test = test_data_tr.shape[0]
idxlist_test = range(N_test)

batch_size_test = 2000

tf.reset_default_graph()
vae = MultiVAE(p_dims, lam=0.0)
saver, logits_var, _, _, _ = vae.build_graph()
###################   在验证集上加载性能最佳的模型     ##################################
chkpt_dir = '/Users/dingziyun/Downloads/ml-20m/log/VAE_anneal{}K_cap{:1.1E}/{}'.format(
    total_anneal_steps/1000, anneal_cap, arch_str)
print("chkpt directory: %s" % chkpt_dir)
#####################    测试数据集  ######################################################
n100_list, r20_list, r50_list = [], [], []
# pred_val_all = []
filename = '/Users/dingziyun/Downloads/ml-20m/pred_val_all.txt'

def text_save(filename, data):#filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename,'a')
    for i in range(len(data)):
        tex = list(data[i])
        s = ''
        for x in tex:
            s += str(x) + ' '
        # s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        s = s +'\n'   #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存文件成功")


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
        pred_val[X.nonzero()] = -np.inf # np.array shape=[2000，20108]

        # text_save(filename, pred_val)

        # pred_val_all.append(pred_val)

        n100_list.append(NDCG_binary_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=100))
        r20_list.append(Recall_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=20))
        r50_list.append(Recall_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=50))



n100_list = np.concatenate(n100_list)
r20_list = np.concatenate(r20_list)
r50_list = np.concatenate(r50_list)


print("Test NDCG@100=%.5f (%.5f)" % (np.mean(n100_list), np.std(n100_list) / np.sqrt(len(n100_list))))
print("Test Recall@20=%.5f (%.5f)" % (np.mean(r20_list), np.std(r20_list) / np.sqrt(len(r20_list))))
print("Test Recall@50=%.5f (%.5f)" % (np.mean(r50_list), np.std(r50_list) / np.sqrt(len(r50_list))))
