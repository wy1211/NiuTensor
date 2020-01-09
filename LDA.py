# 运行LDA模型，结果存储到csv类型的文件中

import pymysql.cursors
import lda
import lda.datasets
from sklearn.feature_extraction.text import CountVectorizer
import numpy

conn = pymysql.Connect(    # 连接数据库
    host='localhost',
    port=3306,
    user='root',
    passwd='lijianwei650287',
    db='weibo',
    charset='utf8mb4'         # 设置编码
)
cursor = conn.cursor()     # 获取游标

wt_old, wt_new = [], []  # 词典(未去重),词典(去重)
dt = []
all_du, all_new_du = [], []  # 所有用户的用户微博文档(未去重),(去重)
position_list = []       # 每个用户微博的位置信息


def build_documents():      # 构建微博文档

    sql = "SELECT ID, Text, Position, Time FROM weibo_history_split"
    cursor.execute(sql)
    results = cursor.fetchall()
    ids, one_user_position = [], []
    for ii in results:
        ids.append(ii[0])
    all_id = list(set(ids))
    all_id.sort(key=ids.index)

    for iid in all_id:
        one_user_position.clear()
        du = []     # 用户微博文档(未去重)

        for j in results:
            if j[0] == iid:
                str_position = ''.join(j[2])
                if str_position != '':
                    one_user_position.append(str_position)
                mid = j[1].split(',')
                for mid_i in mid:
                    du.append(mid_i)
                    wt_old.append(mid_i)

        one_user_position_str = ' '.join(one_user_position)
        position_list.append(one_user_position_str)

        all_du.append(du)
        print("\n用户微博文档： ", du)     # 发布的所有微博
        new_du = list(set(du))
        new_du.sort(key=du.index)
        all_new_du.append(new_du)
        str_du = ' '.join(du)
        dt.append(str_du)

    print("\n所有用户的文档语料库: ", dt)      # 所有用户的微博文档
    wt_new.extend(list(set(wt_old)))
    wt_new.sort(key=wt_old.index)
    print("用户微博位置集合: ", position_list)    # 所有用户的位置集合


def lda_text():
    # LDA部分
    dt_part1, dt_part2, dt_part3, dt_part4 = [], [], [], []        # 把每个用户的文档均分为四部分

    for one_user in dt:
        sp = one_user.split(' ')
        one_part_len = int(len(sp)/4)
        part1 = ' '.join(sp[:one_part_len])
        part2 = ' '.join(sp[one_part_len:one_part_len*2])
        part3 = ' ' .join(sp[one_part_len*2:one_part_len*3])
        part4 = ' '.join(sp[one_part_len*3:])
        dt_part1.append(part1)
        dt_part2.append(part2)
        dt_part3.append(part3)
        dt_part4.append(part4)

    # 对第一部分进行LDA运算

    corpus = dt_part1
    # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频,sklen.countVectorizer()类能够把文档词块化
    vectorizer = CountVectorizer()
    # 统计每个行每个单词的词频
    x = vectorizer.fit_transform(corpus)
    # 这里toarray()和todense()结果一样，都是单词根据词典的分布。
    weight = x.toarray()
    model = lda.LDA(n_topics=10, n_iter=500, random_state=1)
    model.fit(numpy.asarray(weight))  # model.fit_transform(X) is also available
    topic_word = model.topic_word_  # model.components_ also works
    # 文档-主题（Document-Topic）分布
    doc_topic = model.doc_topic_
    a = doc_topic
    b = topic_word
    numpy.savetxt('C:/Users/MyPC/Desktop/topicword1.csv', b, delimiter=',')
    numpy.savetxt('C:/Users/MyPC/Desktop/docTopic1.csv', a, delimiter=',')  # 将得到的文档-主题分布保存

    # 对第二部分进行LDA运算

    corpus = dt_part2
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(corpus)
    weight = x.toarray()
    model = lda.LDA(n_topics=10, n_iter=500, random_state=1)
    model.fit(numpy.asarray(weight))  # model.fit_transform(X) is also available
    topic_word = model.topic_word_  # model.components_ also works
    # 文档-主题（Document-Topic）分布
    doc_topic = model.doc_topic_
    a = doc_topic
    b = topic_word
    numpy.savetxt('C:/Users/MyPC/Desktop/topicword2.csv', b, delimiter=',')
    numpy.savetxt('C:/Users/MyPC/Desktop/docTopic2.csv', a, delimiter=',')  # 将得到的文档-主题分布保存

    # 对第三部分进行LDA运算

    corpus = dt_part3
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(corpus)
    weight = x.toarray()
    model = lda.LDA(n_topics=10, n_iter=500, random_state=1)
    model.fit(numpy.asarray(weight))  # model.fit_transform(X) is also available
    topic_word = model.topic_word_  # model.components_ also works
    # 文档-主题（Document-Topic）分布
    doc_topic = model.doc_topic_
    a = doc_topic
    b = topic_word
    numpy.savetxt('C:/Users/MyPC/Desktop/topicword3.csv', b, delimiter=',')
    numpy.savetxt('C:/Users/MyPC/Desktop/docTopic3.csv', a, delimiter=',')  # 将得到的文档-主题分布保存

    # 对第四部分进行LDA运算

    corpus = dt_part4
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(corpus)
    weight = x.toarray()
    model = lda.LDA(n_topics=10, n_iter=500, random_state=1)
    model.fit(numpy.asarray(weight))  # model.fit_transform(X) is also available
    topic_word = model.topic_word_  # model.components_ also works
    # 文档-主题（Document-Topic）分布
    doc_topic = model.doc_topic_
    a = doc_topic
    b = topic_word
    numpy.savetxt('C:/Users/MyPC/Desktop/topicword4.csv', b, delimiter=',')
    numpy.savetxt('C:/Users/MyPC/Desktop/docTopic4.csv', a, delimiter=',')  # 将得到的文档-主题分布保存


def lda_text_all():           # 没有时间窗口划分的LDA模型(用于对比)
    corpus = dt
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(corpus)
    weight = x.toarray()
    model = lda.LDA(n_topics=10, n_iter=500, random_state=1)
    model.fit(numpy.asarray(weight))  # model.fit_transform(X) is also available
    topic_word = model.topic_word_  # model.components_ also works
    # 文档-主题（Document-Topic）分布
    doc_topic = model.doc_topic_
    a = doc_topic
    b = topic_word
    numpy.savetxt('C:/Users/MyPC/Desktop/topicword.csv', b, delimiter=',')
    numpy.savetxt('C:/Users/MyPC/Desktop/docTopic.csv', a, delimiter=',')  # 将得到的文档-主题分布保存


def lda_location():
    corpus = position_list

    # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频,sklen.countVectorizer()类能够把文档词块化
    vectorizer = CountVectorizer()
    # 统计每个行每个单词的词频
    x = vectorizer.fit_transform(corpus)

    # 这里toarray()和todense()结果一样，都是单词根据词典的分布。
    weight = x.toarray()
    model = lda.LDA(n_topics=10, n_iter=500, random_state=1)
    model.fit(numpy.asarray(weight))  # model.fit_transform(X) is also available
    # 文档-主题（Document-Topic）分布
    doc_topic = model.doc_topic_
    a = doc_topic
    numpy.savetxt('C:/Users/MyPC/Desktop/doc_location.csv', a, delimiter=',')  # 将得到的文档-主题分布保存


if __name__ == '__main__':
    build_documents()
    lda_text()
    lda_text_all()
    lda_location()
