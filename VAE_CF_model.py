# VAE_CF 模型建立
import seaborn as sn
sn.set()

import tensorflow as tf
from tensorflow.contrib.layers import apply_regularization, l2_regularizer

class MultiDAE(object):
    def __init__(self, p_dims, q_dims=None, lam=0.01, lr=1e-3, random_seed=None):
        # p[输入数据维数，神经元个数，n_items）网络的输入维数，自动编码器的输入和输出尺寸必须相等
        # q(n_items,神经元个数,输入数据维数)
        self.p_dims = p_dims
        if q_dims is None:
            self.q_dims = p_dims[::-1]
        else:
            assert q_dims[0] == p_dims[-1], "Input and output dimension must equal each other for autoencoders."
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q-network mismatches." # p-和q-网络不匹配的潜在维数
            self.q_dims = q_dims

        self.dims = self.q_dims + self.p_dims[1:]

        self.lam = lam  #
        self.lr = lr  # learning_rate=0.001 学习速率
        self.random_seed = random_seed

        self.construct_placeholders()

    def construct_placeholders(self):
        # 此函数可以理解为形参，用于定义过程，在执行的时候再赋具体的值
        self.input_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.dims[0]])
        self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=None)

    def build_graph(self):
        # 构建图
    # 1，权重配置
        self.construct_weights()
    # 2，前馈函数
        saver, logits = self.forward_pass() # logits = tf.matmul(h, w) + b，正则化的h
        log_softmax_var = tf.nn.log_softmax(logits) # 生成概率向量，
        # 原文：The output of this transformation （f_theta）is normalized via a softmax function to produce a probability vector pi（zu） over the entire item set
        # tf.nn.log_softmax(logits,axis=None,name=None,dim=None)，logits是一个非空的Tensor，一些参数已被弃用.它们将在未来版本中删除.更新说明：不推荐使用dim,而是使用axis
        # logsoftmax = logits - log(reduce_sum(exp(logits), axis))

        # per-user average negative log-likelihood
        # 每一位用户的 负（平均对数似然）
        neg_ll = -tf.reduce_mean(tf.reduce_sum(log_softmax_var * self.input_ph, axis=1))
        # apply regularization to weights，权重正则化,lam=0.01
        reg = l2_regularizer(self.lam)
        reg_var = apply_regularization(reg, self.weights)
        # tensorflow l2 regularization multiply 0.5 to the l2 norm
        # multiply 2 so that it is back in the same scale
        loss = neg_ll + 2 * reg_var

        train_op = tf.train.AdamOptimizer(self.lr).minimize(loss) #是一个寻找全局最优点的优化算法，引入了二次方梯度校正

        # add summary statistics
        tf.summary.scalar('negative_multi_ll', neg_ll)
        tf.summary.scalar('loss', loss)
        merged = tf.summary.merge_all()
        return saver, logits, loss, train_op, merged

    def forward_pass(self):
        # construct forward graph
        # 前馈函数
        h = tf.nn.l2_normalize(self.input_ph, 1) # L2正则化，对输入的数据
        h = tf.nn.dropout(h, rate=1-self.keep_prob_ph) # 防止过拟合，输入tensor为h，keep prob 每个元素被保留下来的概率

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = tf.matmul(h, w) + b

            if i != len(self.weights) - 1:
                h = tf.nn.tanh(h)
        return tf.train.Saver(), h #saver保存和恢复变量,

    def construct_weights(self):

        self.weights = []
        self.biases = []

        # define weights
        # 定义权重
        for i, (d_in, d_out) in enumerate(zip(self.dims[:-1], self.dims[1:])):
            weight_key = "weight_{}to{}".format(i, i + 1)
            bias_key = "bias_{}".format(i + 1)

            self.weights.append(tf.get_variable(
                name=weight_key, shape=[d_in, d_out],initializer=tf.contrib.layers.xavier_initializer(seed=self.random_seed))) #该函数返回一个用于初始化权重的初始化程序 “Xavier”

            self.biases.append(tf.get_variable(
                name=bias_key, shape=[d_out],initializer=tf.truncated_normal_initializer(stddev=0.001, seed=self.random_seed))) #从截断的正态分布中输出随机值。

            # add summary stats
            tf.summary.histogram(weight_key, self.weights[-1]) # 权重 输出一个直方图的Summary protocol buffer
            tf.summary.histogram(bias_key, self.biases[-1]) # 偏量

class MultiVAE(MultiDAE):

    def construct_placeholders(self):
        super(MultiVAE, self).construct_placeholders()

        # placeholders with default values when scoring
        self.is_training_ph = tf.placeholder_with_default(0., shape=None)
        self.anneal_ph = tf.placeholder_with_default(1., shape=None)

    def build_graph(self):
        self._construct_weights()

        saver, logits, KL = self.forward_pass()  # 返回预测xu，和KL散度
        log_softmax_var = tf.nn.log_softmax(logits) # 向量进行softmax计算，得到一个概率向量

        neg_ll = -tf.reduce_mean(tf.reduce_sum(log_softmax_var * self.input_ph,axis=-1)) # 每一位用户的 负（平均对数似然）

        # apply regularization to weights,lam=0.01,L2正则化的参数
        reg = l2_regularizer(self.lam) # 返回一个 l2(权重) 的函数

        reg_var = apply_regularization(reg, self.weights_q + self.weights_p)
        # tensorflow l2 regularization multiply 0.5 to the l2 norm
        # multiply 2 so that it is back in the same scale
        neg_ELBO = neg_ll + self.anneal_ph * KL + 2 * reg_var # 对数似然边际下限

        train_op = tf.train.AdamOptimizer(self.lr).minimize(neg_ELBO) # 是一个寻找全局最优点的优化算法，引入了二次方梯度校正

        # add summary statistics
        tf.summary.scalar('negative_multi_ll', neg_ll)
        tf.summary.scalar('KL', KL)
        tf.summary.scalar('neg_ELBO_train', neg_ELBO)
        merged = tf.summary.merge_all()

        return saver, logits, neg_ELBO, train_op, merged
        # 返回存储的数据，logits是预测的xu，neg_ELBO是对数似然边际下限，train_op是最小化neg_ELBO的方法，merged可视化版面TensorBoard

    def q_graph(self):
        # 生成每个xu的均值，方差，以及KL
        mu_q, std_q, KL = None, None, None
        h = tf.nn.l2_normalize(self.input_ph, 1)  # L2正则
        h = tf.nn.dropout(h, rate=1-self.keep_prob_ph)  # 防止过拟合

        for i, (w, b) in enumerate(zip(self.weights_q, self.biases_q)):  # 计算q的权重和偏量
            h = tf.matmul(h, w) + b # 计算w和b下的每个预测值

            if i != len(self.weights_q) - 1:
                h = tf.nn.tanh(h) # 激活函数处理h
            else:
                mu_q = h[:, :self.q_dims[-1]]  # 根据计算结果分布的均值和方差，分布的均值和方差
                logvar_q = h[:, self.q_dims[-1]:]

                std_q = tf.exp(0.5 * logvar_q)  # q分布的标准差
                KL = tf.reduce_mean(tf.reduce_sum(
                    0.5 * (-logvar_q + tf.exp(logvar_q) + mu_q ** 2 - 1), axis=1))  # KL散度
        return mu_q, std_q, KL

    def p_graph(self, z):
        # p(z_u) 根据采样的隐变量，生成预测结果x～

        h = z

        for i, (w, b) in enumerate(zip(self.weights_p, self.biases_p)):
            h = tf.matmul(h, w) + b

            if i != len(self.weights_p) - 1:
                h = tf.nn.tanh(h)
        return h

    def forward_pass(self):
        # q-network
        mu_q, std_q, KL = self.q_graph()
        epsilon = tf.random_normal(tf.shape(std_q)) # 随机选取标准高斯分布的epsilon
        # zu的取样，zu = E(q) + epsilon*STD(q)*self.is_training_ph
        sampled_z = mu_q + self.is_training_ph * epsilon * std_q

        # p-network
        logits = self.p_graph(sampled_z) # 根据采样的隐变量，生成预测x～

        return tf.train.Saver(), logits, KL # 返回预测xu，和KL散度

    def _construct_weights(self):
        # 构建 权重，偏量 张量
        self.weights_q, self.biases_q = [], []
        #
        for i, (d_in, d_out) in enumerate(zip(self.q_dims[:-1], self.q_dims[1:])):
            if i == len(self.q_dims[:-1]) - 1:
                # we need two sets of parameters for mean and variance,
                # 需要均值，方差两组参数
                # respectively
                d_out *= 2
            weight_key = "weight_q_{}to{}".format(i, i + 1)
            bias_key = "bias_q_{}".format(i + 1)

            self.weights_q.append(tf.get_variable(name=weight_key, shape=[d_in, d_out],
                initializer=tf.contrib.layers.xavier_initializer(seed=self.random_seed)))

            self.biases_q.append(tf.get_variable(name=bias_key, shape=[d_out],
                initializer=tf.truncated_normal_initializer(stddev=0.001, seed=self.random_seed)))

            # add summary stats
            tf.summary.histogram(weight_key, self.weights_q[-1])
            tf.summary.histogram(bias_key, self.biases_q[-1])
        self.weights_p, self.biases_p = [], []

        for i, (d_in, d_out) in enumerate(zip(self.p_dims[:-1], self.p_dims[1:])):
            weight_key = "weight_p_{}to{}".format(i, i + 1)
            bias_key = "bias_p_{}".format(i + 1)
            self.weights_p.append(tf.get_variable(name=weight_key, shape=[d_in, d_out],
                initializer=tf.contrib.layers.xavier_initializer(seed=self.random_seed)))

            self.biases_p.append(tf.get_variable(name=bias_key, shape=[d_out],
                initializer=tf.truncated_normal_initializer(stddev=0.001, seed=self.random_seed)))

            # add summary stats
            tf.summary.histogram(weight_key, self.weights_p[-1])
            tf.summary.histogram(bias_key, self.biases_p[-1])