# 用于评估情感分类模型的性能，支持两种数据集（SIMS和MOSI）的三分类（负面/中性/正面）情感分析。

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


# 用于计算多分类问题的准确率。它比较预测值和真实值，看看有多少预测是正确的。
# preds：模型预测结果，形状为(N,)
# truths：真实标签，形状为(N,)
def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))
# np.round(preds) 对预测值四舍五入到最接近的整数
# np.round(truths) 对真实值四舍五入到最接近的整数
# 对比后返回一个true/false的布尔值数组
# np.sun() 求和，算有多少个true
# len(truths) 总样本数
# float(...) 确保进行浮点数除法


# 用于评估情感分类模型在SIMS数据集上的性能。它计算了多种不同的评估指标，包括三分类和两种二分类的准确率和F1分数。
def eval_sims_classification(y_pred, y_true):
    """
    {
        "Negative": 0,
        "Neutral": 1,
        "Positive": 2
    }
    """
    y_pred = y_pred.cpu().detach().numpy()
    y_true = y_true.cpu().detach().numpy()
# .cpu(): 将数据从GPU移动到CPU
# .detach(): 断开与计算图的连接（用于PyTorch模型）
# .numpy(): 转换为NumPy数组
# 作用：将PyTorch张量转换为普通的NumPy数组以便计算

    # three classes
    y_pred_3 = np.argmax(y_pred, axis=1)
# y_pred 的形状为(N, 3)
# np.argmax(..., axis=1) 找出每行的最大值的索引列
    Mult_acc_3 = accuracy_score(y_pred_3, y_true)   # accuracy_score来自sklearn.metrics包
# 计算三分类的准确率
    F1_score_3 = f1_score(y_true, y_pred_3, average='weighted')     # f1_score来自sklearn.metrics包
# 计算加权F1分数，考虑每个类别的样本数量
# average='weighted': 关键参数，指定如何计算多类别的平均F1

    # two classes
# 将三分类问题转换为二分类问题
# 0: negative, 1: neutral -> 0: 非正面
# 2：positive -> 1: 正面
    y_pred = np.array([[v[0], v[2]] for v in y_pred])
# 只保留负面和正面的预测分数，忽略中性  例子：原始 [0.1, 0.2, 0.7] → 新 [0.1, 0.7]
    # with 0 (<= 0 or > 0) **NOTE: Different from MOSI!** SIMS: non-positive / positive
    y_pred_2 = np.argmax(y_pred, axis=1)
# y_pred_2 的形状为(N,)
# np.argmax(..., axis=1) 找出每行的最大值的索引列
    y_true_2 = []
    for v in y_true:
        y_true_2.append(0 if v <= 1 else 1)
# .append() 添加一个元素到列表的末尾
# 0: negative, 1: neutral -> 0: 非正面
# 1: neutral -> 1: 正面
    y_true_2 = np.array(y_true_2)
# np.array() 将列表转换为数组
    Has0_acc_2 = accuracy_score(y_pred_2, y_true_2)
    Has0_F1_score = f1_score(y_true_2, y_pred_2, average='weighted')
    # without neutral (exclude Neutral=1): Negative vs Positive
    non_zeros = np.array([i for i, e in enumerate(y_true) if e != 1])

    y_pred_2 = y_pred[non_zeros]  # shape (N_non0, 2) for [neg, pos]
    y_pred_2 = np.argmax(y_pred_2, axis=1)  # {0:neg, 1:pos}

    y_true_2 = y_true[non_zeros]  # {0:neg, 2:pos}
    y_true_2 = (y_true_2 == 2).astype(np.int64)  # map {0,2} -> {0,1}

    Non0_acc_2 = accuracy_score(y_pred_2, y_true_2)
    Non0_F1_score = f1_score(y_true_2, y_pred_2, average='weighted')

    eval_results = {
        # 二分类评估（包含中性）
        "Has0_acc_2":  round(Has0_acc_2, 4),
        "Has0_F1_score": round(Has0_F1_score, 4),
        #  二分类评估（排除中性）
        "Non0_acc_2":  round(Non0_acc_2, 4),
        "Non0_F1_score": round(Non0_F1_score, 4),
        # 三分类评估
        "Acc_3": round(Mult_acc_3, 4),
        "F1_score_3": round(F1_score_3, 4)
# round(..., 4) 四舍五入且保留4位小数
    }
# 创建一个字典，包含各种评估指标

    return eval_results

def eval_mosi_classification(y_pred, y_true):
    """
    {
        "Negative": 0,
        "Neutral": 1,
        "Positive": 2
    }
    """
    y_pred = y_pred.cpu().detach().numpy()
    y_true = y_true.cpu().detach().numpy()

    # three classes
    y_pred_3 = np.argmax(y_pred, axis=1)
    Mult_acc_3 = accuracy_score(y_pred_3, y_true)
    F1_score_3 = f1_score(y_true, y_pred_3, average='weighted')

    # two classes
    y_pred = np.array([[v[0], v[2]] for v in y_pred])
# np.array() 将列表转换为数组
    # with 0 (< 0 or >= 0) **NOTE: Different from SIMS!** MOSI/MOSEI: negative / non-negative
    y_pred_2 = np.argmax(y_pred, axis=1)
    y_true_2 = []
    for v in y_true:
        y_true_2.append(0 if v < 1 else 1)
# 0 表示非正面，1、2 表示正面
    y_true_2 = np.array(y_true_2)
    Has0_acc_2 = accuracy_score(y_pred_2, y_true_2)
    Has0_F1_score = f1_score(y_true_2, y_pred_2, average='weighted')
    # without neutral (exclude Neutral=1): Negative vs Positive
    non_zeros = np.array([i for i, e in enumerate(y_true) if e != 1])

    y_pred_2 = y_pred[non_zeros]
    y_pred_2 = np.argmax(y_pred_2, axis=1)  # {0:neg, 1:pos}

    y_true_2 = y_true[non_zeros]  # {0:neg, 2:pos}
    y_true_2 = (y_true_2 == 2).astype(np.int64)  # map {0,2} -> {0,1}

    Non0_acc_2 = accuracy_score(y_pred_2, y_true_2)
    Non0_F1_score = f1_score(y_true_2, y_pred_2, average='weighted')

    eval_results = {
        "Has0_acc_2": round(Has0_acc_2, 4),
        "Has0_F1_score": round(Has0_F1_score, 4),
        "Non0_acc_2": round(Non0_acc_2, 4),
        "Non0_F1_score": round(Non0_F1_score, 4),
        "Acc_3": round(Mult_acc_3, 4),
        "F1_score_3": round(F1_score_3, 4)
    }
    return eval_results