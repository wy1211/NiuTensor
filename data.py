# 数据处理
# pickle是一个将任意复杂的对象转成对象的文本或二进制表示的过程
# 也可以将这些字符串、文件或任何类似于文件的对象 unpickle 成原来的对象

import pickle
import os
import random
import numpy as np

# 标签字典
tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6
             }


def read_corpus(corpus_path):         # 输入train_data文件的路径，读取训练集的语料，输出train_data
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()      # 返回的是一个列表，一行数据一个元素
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            [char, label] = line.strip().split()
            sent_.append(char)     # 字放进sent_
            tag_.append(label)     # tag放进tag_
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []

    return data

# 由train_data来构造一个(统计非重复字)字典{'第一个字':[对应的id,该字出现的次数],'第二个字':[对应的id,该字出现的次数], , ,}
# 去除低频词，生成一个word_id的字典并保存在输入的vocab_path的路径下，
# 保存的方法是pickle模块自带的dump方法，保存后的文件格式是word2id.pkl文件


def vocab_build(vocab_path, corpus_path, min_count):      # min_count设为3
    data = read_corpus(corpus_path)
    word2id = {}
    for sent_, tag_ in data:
        for word in sent_:
            if word.isdigit():    # 字符是数字
                word = '<NUM>'
            elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):   # 字符是字母
                word = '<ENG>'
            if word not in word2id:     # 如果不在字典中，就加入到字典中
                word2id[word] = [len(word2id)+1, 1]
            else:        # 在字典中就次数+1
                word2id[word][1] += 1

    low_freq_words = []   # 低频词
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':   # 统计低频词
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]       # 从字典中删除低频词

    new_id = 1    # 重构字典
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    print(len(word2id))
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)      # 序列化到名字为word2id.pkl文件中


def sentence2id(sent, word2id):       # 输入一句话，生成一个 sentence_id
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:       # 在字典中找不到就用<UNK>表示
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def read_dictionary(vocab_path):       # 通过pickle模块自带的load方法(反序列化方法)加载输出word2id.pkl文件
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id


def random_embedding(vocab, embedding_dim):     # 输入vocab，vocab就是前面得到的word2id，embedding_dim=300
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    # 返回一个len(vocab)*embedding_dim=3905*300的矩阵(每个字投射到300维)作为初始值
    return embedding_mat

# padding,输入一句话，不够标准的样本用pad_mark来补齐


"""输入：seqs的形状为二维矩阵，形状为[[33,12,17,88,50]-第一句话
                                 [52,19,14,48,66,31,89]-第二句话] 
输出：seq_list为seqs经过padding后的序列
      seq_len_list保留了padding之前每条样本的真实长度
      seq_list和seq_len_list用来喂给feed_dict"""


def pad_sequences(sequences, pad_mark=0):
    max_len = max(map(lambda x: len(x), sequences))   # 返回一个序列中长度最长的那条样本的长度
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        # 不够最大长度的样本用0补上放到列表seq_list
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


''' seqs的形状为二维矩阵，形状为[[33,12,17,88,50....]...第一句话
                                [52,19,14,48,66....]...第二句话
                                                    ] 
   labels的形状为二维矩阵，形状为[[0, 0, 3, 4]....第一句话
                                 [0, 0, 3, 4]...第二句话
                                             ]
'''


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):    # 生成batch
    if shuffle:     # 乱序数据
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)      # 返回在字典中的编号
        label_ = [tag2label[tag] for tag in tag_]     # 返回tag的value值

        if len(seqs) == batch_size:
            yield seqs, labels      # yield 是一个类似 return 的关键字，只是这个函数返回的是个生成器
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels
