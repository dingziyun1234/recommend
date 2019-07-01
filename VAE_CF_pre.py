# 数据的处理
# data：ml-20m,ratings.csv
# -> 挑选出评分大于3.5的数据
# -> 依据用户的点击行为次数，和物品被点击的行为次数，进行过滤
# -> 之后分为三个数据组：预留组，测试组，训练组
# -> 提取训练组所有sid，所有uid，对数据整理
# -> 存储格式：（uid，sid）
import os
import sys
import numpy as np
import pandas as pd

## change `DATA_DIR` to the location where movielens-20m dataset sits
DATA_DIR = '/Users/dingziyun/Downloads/ml-20m/'

raw_data = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'), header=0)
raw_data = raw_data[raw_data['rating'] > 3.5]
# 原始数据的格式，{userId,movieId,rating,timestamp}
#  binarize the data (only keep ratings > 3.5)

###################################   数据分裂    ####################################################

# 数据分裂
# 10k用户预留，10k用户验证，剩下的是train set 使用来自训练用户的所有物品作为item set 对于验证和测试用户，
# 80%的子样本作为折叠数据，其余的用于预测

#################   提取点击行为数据   ######################
def get_count(tp, id):
    # 输入：数据tp，和id
    # 以id分组，统计每个id 点击/被点击 的次数
    # 返回：给定id的点击次数
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count
################  筛选条件数据  ############################
def filter_triplets(tp, min_uc=5, min_sc=0):
    # 过滤数据，返回过滤后的数据集，用户点击次数，物品被点击次数
    # 过滤原则：满足最少点击次数的，min_uc 用户最少的点击次数，min_sc 物品最少被点击次数限制
    # 返回：原数据tp，过滤后的用户点击次数usercount，物品被点击次数itemcount
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

# 用户的点击次数作为用户的活跃度，物品的被点击次数作为受欢迎度
raw_data, user_activity, item_popularity = filter_triplets(raw_data)
# 稀疏 = 原始数据的行数/（过滤后用户数*过滤后物品数）
sparsity = 1. * raw_data.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])

# 取唯一的用户id
unique_uid = user_activity.index

np.random.seed(98765)
idx_perm = np.random.permutation(unique_uid.size) # 这个函数的是用来随机排列一个数组的
unique_uid = unique_uid[idx_perm] # 随机打乱用户id的顺序

################### 创建 训练用户||验证用户||测试用户 ################
n_users = unique_uid.size # 全部的用户个数
n_heldout_users = 10000 # 预留10k用户，以及验证用户10k

tr_users = unique_uid[:(n_users - n_heldout_users * 2)] # 训练组用户的id,电脑太渣，跑不了原设定[:(n_users - n_heldout_users * 2)]
vd_users = unique_uid[int((n_users - n_heldout_users * 2)/10): int((n_users - n_heldout_users)/10)] # 验证组用户的id
te_users = unique_uid[(n_users - n_heldout_users):] # 测试组用户的id

train_plays = raw_data.loc[raw_data['userId'].isin(tr_users)] # 取训练组用户的id对应的原始数据
unique_sid = pd.unique(train_plays['movieId']) # 取训练组用户所有产生过行为的物品的id

show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid)) # 字典{movieId,序数}
profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid)) # 字典{userId,序数}

######################  存储数据   ###########################
# 创建文件存储 训练组用户所有产生过行为的物品的id

pro_dir = os.path.join(DATA_DIR, 'pro_sg')

if not os.path.exists(pro_dir):
    os.makedirs(pro_dir)

with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
    for sid in unique_sid:
        f.write('%s\n' % sid)

################## 训练组的数据 为 train 和 test ######################
# 分裂 训练组的数据 为 train 和 test

def split_train_test_proportion(data, test_prop=0.2):
    # 输入：数据data，分裂比例test_prop
    # 输出：训练数据data_tr，测试数据data_te
    data_grouped_by_user = data.groupby('userId')
    tr_list, te_list = list(), list()

    np.random.seed(98765)

    for i, (_, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
        else:
            tr_list.append(group)

        if i % 1000 == 0:
            print("%d users sampled" % i)
            sys.stdout.flush()

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te

###################  验证组和测试组数据的分裂处理   ######################
# 验证组数据
vad_plays = raw_data.loc[raw_data['userId'].isin(vd_users)] # 取 验证组用户id 对应的原数据
vad_plays = vad_plays.loc[vad_plays['movieId'].isin(unique_sid)] # 取 验证组中 同时存在于 训练组用户所有产生过行为的物品的id 的原数据

vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays) # 将 验证组的数据 进行分裂为训练和测试，默认比例test_pro=0.2
# 测试组数据
test_plays = raw_data.loc[raw_data['userId'].isin(te_users)] # 取 测试组用户id 对应的原数据
test_plays = test_plays.loc[test_plays['movieId'].isin(unique_sid)] #取 测试组中 同时存在于 训练组用户所有产生过行为的物品的id 的原数据

test_plays_tr, test_plays_te = split_train_test_proportion(test_plays) # 将 测试组的数据 进行分裂为训练和测试，默认比例test_pro=0.2
##############################  所有处理数据的格式存储  ################################
# 把数据存为(user_index, item_index) 格式

# show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid)) # 字典{movieId,序数}
# profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid)) # 字典{userId,序数}

def numerize(tp):
    uid = list(map(lambda x: profile2id[x], tp['userId'])) # 将原始数据的userId列
    sid = list(map(lambda x: show2id[x], tp['movieId']))
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])

train_data = numerize(train_plays) # 将 训练组用户的id 对应的 原始数据 转换为 {uid,sid}
train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)

vad_data_tr = numerize(vad_plays_tr)
vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)

vad_data_te = numerize(vad_plays_te)
vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)

test_data_tr = numerize(test_plays_tr)
test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)

test_data_te = numerize(test_plays_te)
test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)

