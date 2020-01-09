# 运行
import tensorflow as tf
import numpy as np
import os
import argparse
import time
from model import BiLSTM_CRF
from utils import str2bool, get_logger, get_entity
from data import read_corpus, read_dictionary, tag2label, random_embedding


def run(sentences):
    # 配置session的参数
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用GPU 0
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 日志级别设置
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory

    # hyperparameters超参数设置
    # 创建一个解析器对象，并告诉它将会有些什么参数
    # 那么当你的程序运行时，该解析器就可以用于处理命令行参数
    parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
    parser.add_argument('--train_data', type=str, default='data_path', help='train data source')
    parser.add_argument('--test_data', type=str, default='data_path', help='test data source')
    parser.add_argument('--batch_size', type=int, default=64, help='#sample of each minibatch')
    # batch :批次大小 在深度学习中，一般采用SGD训练，即每次训练在训练集中取batchsize个样本训练
    # iteration：中文翻译为迭代，1个iteration等于使用batchsize个样本训练一次
    # 一个迭代 = 一个正向通过+一个反向通过
    parser.add_argument('--epoch', type=int, default=40, help='#epoch of training')
    # epoch：迭代次数，1个epoch等于使用训练集中的全部样本训练一次
    # 一个epoch = 所有训练样本的一个正向传递和一个反向传递 举个例子，训练集有1000个样本，batchsize=10，那么： 训练完整个样本集需要： 100次iteration，1次epoch。
    parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
    # 输出向量的维度：300维
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
    # 优化器用的Adam
    parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
    # dropout是指在深度学习网络的训练过程中，对于神经网络单元，按照一定的概率将其暂时从网络中丢弃
    parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
    parser.add_argument('--pretrain_embedding', type=str, default='random', help='use pretrained char embedding or init it randomly')
    parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
    parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')
    parser.add_argument('--mode', type=str, default='demo', help='train/test/demo')
    parser.add_argument('--demo_model', type=str, default='1559398699', help='model for test and demo')
    # 传递参数送入模型中
    args = parser.parse_args()

    # 初始化embedding矩阵，读取词典
    word2id = read_dictionary(os.path.join('.', args.train_data, 'word2id.pkl'))
    # 通过调用random_embedding函数返回一个len(vocab)*embedding_dim=3905*300的矩阵(矩阵元素均在-0.25到0.25之间)作为初始值
    if args.pretrain_embedding == 'random':
        embeddings = random_embedding(word2id, args.embedding_dim)
    else:
        embedding_path = 'pretrain_embedding.npy'
        embeddings = np.array(np.load(embedding_path), dtype='float32')

    # 读取训练集和测试集
    if args.mode != 'demo':
        train_path = os.path.join('.', args.train_data, 'train_data')
        test_path = os.path.join('.', args.test_data, 'test_data')
        train_data = read_corpus(train_path)
        test_data = read_corpus(test_path)
        test_size = len(test_data)

    # 设置路径
    paths = {}
    timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model
    output_path = os.path.join('.', args.train_data+"_save", timestamp)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    summary_path = os.path.join(output_path, "summaries")
    paths['summary_path'] = summary_path
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    model_path = os.path.join(output_path, "checkpoints/")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    ckpt_prefix = os.path.join(model_path, "model")
    paths['model_path'] = ckpt_prefix
    result_path = os.path.join(output_path, "results")
    paths['result_path'] = result_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    log_path = os.path.join(result_path, "log.txt")
    paths['log_path'] = log_path
    get_logger(log_path).info(str(args))    # 将参数写入日志文件

    if args.mode == 'train':         # 训练模型
        model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
        model.build_graph()
        model.train(train=train_data, dev=test_data)

    elif args.mode == 'test':        # 测试模型
        ckpt_file = tf.train.latest_checkpoint(model_path)
        print(ckpt_file)
        paths['model_path'] = ckpt_file
        model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
        model.build_graph()
        print("test data: {}".format(test_size))
        model.test(test_data)

    # demo
    elif args.mode == 'demo':
        location = []
        ckpt_file = tf.train.latest_checkpoint(model_path)
        print("model path: ", ckpt_file)
        paths['model_path'] = ckpt_file         # 设置模型路径
        model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
        model.build_graph()
        saver = tf.train.Saver()
        with tf.Session(config=config) as sess:
            saver.restore(sess, ckpt_file)
            for sentence in sentences:
                demo_sent = sentence
                demo_sent = list(demo_sent.strip())        # 删除空白符
                demo_data = [(demo_sent, ['O'] * len(demo_sent))]
                tag = model.demo_one(sess, demo_data)
                PER, LOC, ORG = get_entity(tag, demo_sent)      # 根据标注序列输出对应的字符
                new_LOC = list(set(LOC))       # 去重
                loc = ' '.join(new_LOC)
                location.append(loc)
            return location


if __name__ == '__main__':
    s = ["沈阳的夏天很凉爽", "我在丹东工作", "大连的风很大"]
    r = run(s)
    print(r)
    for i in r:
        print(i)
