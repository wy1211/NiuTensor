# 计算准确率(Accurary) 、 精确率（Precision）、召回率（ Recall）、F


def cal_tp(data, tag):       # 计算TP
    TP = 0
    for line in data:
        if line == '\n':
            continue
        line = line.strip('\n').split(' ')
        actual_value = line[-2]
        predicted_value = line[-1]
        if actual_value == tag and actual_value == predicted_value:
            TP += 1
    return TP


def cal_fn(data, tag):      # 计算FN
    FN = 0
    for line in data:
        if line == '\n':
            continue
        line = line.strip('\n').split(' ')
        actual_value = line[-2]
        predicted_value = line[-1]
        if actual_value == tag and actual_value != predicted_value:
            FN += 1
    return FN


def cal_fp(data, tag):      # 计算FP
    FP = 0
    for line in data:
        if line == '\n':
            continue
        line = line.strip('\n').split(' ')
        actual_value = line[-2]
        predicted_value = line[-1]
        if predicted_value == tag and actual_value != predicted_value:
            FP += 1
    return FP


def cal_tn(data, tag):     # 计算TN
    TN = 0
    for line in data:
        if line == '\n':
            continue
        line = line.strip('\n').split(' ')
        actual_value = line[-2]
        predicted_value = line[-1]
        if predicted_value != tag and actual_value != tag:
            TN += 1
    return TN


def cal_main(predict_results, label_path):
    with open(label_path, "w") as fw:
        line = []
        for sent_result in predict_results:
            for char, tag, tag_ in sent_result:
                tag = '0' if tag == 'O' else tag
                char = char.encode("utf-8")
                line.append("{} {} {}\n".format(char, tag, tag_))
            line.append("\n")
        fw.writelines(line)

    with open(label_path, 'r', encoding='UTF-8') as fw:
        data = fw.readlines()
    tag1 = 'B-LOC'
    tag2 = 'I-LOC'
    TP_1 = cal_tp(data, tag1)
    FN_1 = cal_fn(data, tag1)
    FP_1 = cal_fp(data, tag1)
    TN_1 = cal_tn(data, tag1)
    TP_2 = cal_tp(data, tag2)
    FN_2 = cal_fn(data, tag2)
    FP_2 = cal_fp(data, tag2)
    TN_2 = cal_tn(data, tag2)
    TP = TP_1 + TP_2
    FN = FN_1 + FN_2
    FP = FP_1 + FP_2
    TN = TN_1 + TN_2
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accurary = (TP + TN) / (TP + TN + FN + FP)
    f1 = 2 * precision * recall / (precision + recall)
    return round(accurary, 4), round(precision, 4), round(recall, 4), round(f1, 4)
