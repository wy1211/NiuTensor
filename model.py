# 搭建模型
import os
import time
import sys
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from data import pad_sequences, batch_yield
from utils import get_logger
from eval import cal_main


class BiLSTM_CRF(object):
    def __init__(self, args, embeddings, tag2label, vocab, paths, config):
        self.batch_size = args.batch_size
        # batch_size= 64
        self.epoch_num = args.epoch
        # epoch = 40
        self.hidden_dim = args.hidden_dim
        # hidden_dim = 300
        self.embeddings = embeddings
        self.CRF = args.CRF
        self.update_embedding = args.update_embedding
        self.dropout_keep_prob = args.dropout
        # dropout = 0.5
        self.optimizer = args.optimizer
        # 优化器用的Adam
        self.lr = args.lr
        # 学习率= 0.001
        self.clip_grad = args.clip
        # gradient clipping = 5.0
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.vocab = vocab
        self.shuffle = args.shuffle
        self.model_path = paths['model_path']
        self.summary_path = paths['summary_path']
        self.logger = get_logger(paths['log_path'])
        self.result_path = paths['result_path']
        self.config = config

    def build_graph(self):        # 构筑图
        self.add_placeholders()
        self.lookup_layer_op()
        self.biLSTM_layer_op()
        self.softmax_pred_op()
        self.loss_op()
        self.trainstep_op()
        self.init_op()

    def add_placeholders(self):      # 增加占位符
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")   # 数据集合
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")       # 真实标签
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")   # 输入序列的最大长度
        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")    # 神经元被选中的概率
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")              # 学习率

    def lookup_layer_op(self):
        # 利用随机初始化的embedding矩阵将句子中的每个字从one-hot向量映射为低维稠密的字向量
        with tf.variable_scope("words"):
            # tf.variable_scope可以让变量有相同的命名，包括tf.get_variable得到的变量，还有tf.Variable的变量
            _word_embeddings = tf.Variable(self.embeddings,
                                           dtype=tf.float32,
                                           trainable=self.update_embedding,
                                           name="_word_embeddings")
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,    # 选取张量中对应的元素
                                                     ids=self.word_ids,
                                                     name="word_embeddings")
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_pl)   # 防止过拟合

    def biLSTM_layer_op(self):
        with tf.variable_scope("bi-lstm"):
            # 设hidden_dim为隐藏层神经元数量
            cell_fw = LSTMCell(self.hidden_dim)   # 前向RNN
            cell_bw = LSTMCell(self.hidden_dim)   # 反向RNN
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.word_embeddings,
                sequence_length=self.sequence_lengths,
                # sequence_lengths： 输入序列的长度
                dtype=tf.float32)
            # 对于序列标注问题，通常会在LSTM的输出后接一个CRF层：
            # 将LSTM的输出通过线性变换得到维度为[batch_size, max_seq_len, num_tags]的张量,
            # 这个张量再输入到CRF层。
            # 将两个LSTM结果进行合并
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)   # 拼接起来，相当于axis=2
            output = tf.nn.dropout(output, self.dropout_pl)    # 防止过拟合

        with tf.variable_scope("proj"):
            W = tf.get_variable(name="W",
                                # 变换矩阵，可训练参数
                                shape=[2 * self.hidden_dim, self.num_tags],
                                # 该函数返回一个用于初始化权重的初始化程序 “Xavier”
                                # 这个初始化器是用来保持每一层的梯度大小都差不多相同
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),    # 初始化为0
                                dtype=tf.float32)
            # output的形状为[batch_size,steps,cell_num]
            s = tf.shape(output)    # s为output的形状
            # 线性变换
            # reshape的目的是为了跟w做矩阵乘法
            output = tf.reshape(output, [-1, 2*self.hidden_dim])   # -1表示由程序计算
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])   # 转换为适合CRF输入的形状

    def loss_op(self):         # 损失函数
        if self.CRF:
            # crf_log_likelihood作为损失函数
            # inputs：unary potentials,就是每个标签的预测概率值
            # tag_indices，这个就是真实的标签序列了
            # sequence_lengths,一个样本真实的序列长度，为了对齐长度会做些padding，但是可以把真实的长度放到这个参数里
            # transition_params,转移概率，可以没有，没有的话这个函数也会算出来
            # 输出：log_likelihood:标量;transition_params,转移概率，如果输入没输，它就自己算个给返回
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits, tag_indices=self.labels,
                                                                        sequence_lengths=self.sequence_lengths)
            self.loss = -tf.reduce_mean(log_likelihood)   # 求平均值

        else:    # 交叉熵做损失函数
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=self.labels)
            # 返回一个表示每个单元的前N个位置的mask张量
            mask = tf.sequence_mask(self.sequence_lengths)
            # tf.boolean_mask(a, b)将使a(m维)矩阵仅保留与b中“True”元素同下标的部分，并将结果展开到m - 1维
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)     # 求平均值
        # 添加标量统计结果
        tf.summary.scalar("loss", self.loss)

    def softmax_pred_op(self):     # 如果不加CRF层
        if not self.CRF:
            self.labels_softmax_ = tf.argmax(self.logits, axis=-1)
            self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)   # 转换为int类型

    def trainstep_op(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)     # 记录全局训练步骤
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
                # 此函数是Adam优化算法：是一个寻找全局最优点的优化算法，引入了二次方梯度校正。
                # 相比于基础SGD算法,不容易陷于局部优点,速度更快
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)

            grads_and_vars = optim.compute_gradients(self.loss)
            # 对var_list中的变量计算loss的梯度
            # 该函数为函数minimize()的第一部分，返回一个以元组(gradient, variable)组成的列表
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            # 梯度控制在-5和5之间，小于-5的记为-5，大于5的记为5
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)
            # 更新对应变量的梯度

    def init_op(self):    # 变量初始化
        self.init_op = tf.global_variables_initializer()

    def add_summary(self, sess):
        self.merged = tf.summary.merge_all()   # 将所有summary全部保存到磁盘
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

    def train(self, train, dev):      # 训练
        saver = tf.train.Saver(tf.global_variables())   # 保存程序中的变量
        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)
            self.add_summary(sess)

            for epoch in range(self.epoch_num):
                self.run_one_epoch(sess, train, dev, self.tag2label, epoch, saver)

    def test(self, test):
        saver = tf.train.Saver()
        with tf.Session(config=self.config) as sess:
            self.logger.info('=========== testing ===========')
            saver.restore(sess, self.model_path)
            label_list, seq_len_list = self.dev_one_epoch(sess, test)
            self.evaluate(label_list, seq_len_list, test)

    def demo_one(self, sess, sent):     # demo
        label_list = []
        for seqs, labels in batch_yield(sent, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, _ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label
        tag = [label2tag[label] for label in label_list[0]]
        return tag

    def run_one_epoch(self, sess, train, dev, tag2label, epoch, saver):     # 训练一个epoch
        num_batches = (len(train) + self.batch_size - 1) // self.batch_size    # batch的总数
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        batches = batch_yield(train, self.batch_size, self.vocab, self.tag2label, shuffle=self.shuffle)  # 生成batch
        for step, (seqs, labels) in enumerate(batches):
            sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
            step_num = epoch * num_batches + step + 1
            feed_dict, _ = self.get_feed_dict(seqs, labels, self.lr, self.dropout_keep_prob)
            _, loss_train, summary, step_num_ = sess.run([self.train_op, self.loss, self.merged, self.global_step],
                                                         feed_dict=feed_dict)
            if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:
                self.logger.info(
                    '{} epoch {}, step {}, loss: {:.4}, global_step: {}'.format(start_time, epoch + 1, step + 1,
                                                                                loss_train, step_num))

            self.file_writer.add_summary(summary, step_num)

            if step + 1 == num_batches:
                saver.save(sess, self.model_path, global_step=step_num)

        self.logger.info('===========validation / test===========')
        label_list_dev, seq_len_list_dev = self.dev_one_epoch(sess, dev)
        self.evaluate(label_list_dev, seq_len_list_dev, dev, epoch)

    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None):    # 生成feed_dict
        word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)

        feed_dict = {self.word_ids: word_ids,
                     self.sequence_lengths: seq_len_list}
        if labels is not None:
            # labels经过padding后，喂给feed_dict
            labels_, _ = pad_sequences(labels, pad_mark=0)
            feed_dict[self.labels] = labels_
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout
        # seq_len_list用来统计每个样本的真实长度
        return feed_dict, seq_len_list

    def dev_one_epoch(self, sess, dev):    # 验证
        label_list, seq_len_list = [], []
        for seqs, labels in batch_yield(dev, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, seq_len_list_ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)
        return label_list, seq_len_list

    def predict_one_batch(self, sess, seqs):
        feed_dict, seq_len_list = self.get_feed_dict(seqs, dropout=1.0)

        if self.CRF:
            logits, transition_params = sess.run([self.logits, self.transition_params],
                                                 feed_dict=feed_dict)
            label_list = []
            for logit, seq_len in zip(logits, seq_len_list):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)   # 返回得分最高的标注序列
                label_list.append(viterbi_seq)
            return label_list, seq_len_list

        else:
            label_list = sess.run(self.labels_softmax_, feed_dict=feed_dict)
            return label_list, seq_len_list

    def evaluate(self, label_list, seq_len_list, data, epoch=None):    # 评测，label_list为预测的结果
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label

        model_predict = []
        for label_, (sent, tag) in zip(label_list, data):
            tag_ = [label2tag[label__] for label__ in label_]
            sent_res = []
            for i in range(len(sent)):
                sent_res.append([sent[i], tag[i], tag_[i]])
            model_predict.append(sent_res)
        epoch_num = str(epoch+1) if epoch is not None else 'test'
        label_path = os.path.join(self.result_path, 'label_' + epoch_num)
        accurary, precision, recall, f1 = cal_main(model_predict, label_path)
        self.logger.info('accurary:{}, precision:{}, recall:{}, f1:{}'.format(accurary, precision, recall, f1))
