from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from sklearn.utils.linear_assignment import linear_assignment
import numpy as np
import torch


def evaluate(model, data_loader, device):
    """
    评估聚类模型的性能，包括计算 NMI, ARI 和 ACC。

    参数：
    - model: 已训练的聚类模型。
    - data_loader: 数据加载器，包含待评估的数据。
    - device: 训练设备（CPU 或 GPU）。

    返回：
    - nmi: 归一化互信息（Normalized Mutual Information）。
    - ari: 调整兰德指数（Adjusted Rand Index）。
    - acc: 聚类准确率（Accuracy）。
    """

    model.eval()  # 设置模型为评估模式
    all_labels = []  # 用于存储所有的真实标签
    all_preds = []  # 用于存储所有预测的聚类标签

    with torch.no_grad():  # 在评估时不需要计算梯度
        for batch, (x, y) in enumerate(data_loader):
            x = x.to(device)  # 将输入数据转移到设备
            labels = y.numpy()  # 获取真实标签

            # 获取模型的潜在空间表示（即编码器输出）
            z_enc = model.encoder(x)

            # 使用 GMM 对潜在空间进行聚类预测
            preds = model.gmm.predict(z_enc)

            # 收集所有批次的真实标签和预测标签
            all_labels.extend(labels)
            all_preds.extend(preds.cpu().numpy())  # 转换为 CPU 张量并转为 numpy 数组

    all_labels = np.array(all_labels)  # 将所有真实标签转为 numpy 数组
    all_preds = np.array(all_preds)  # 将所有预测标签转为 numpy 数组

    # 使用匈牙利算法对预测标签与真实标签进行最佳匹配
    # 生成标签映射矩阵
    num_classes = len(np.unique(all_labels))
    cost_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for i in range(num_classes):
        for j in range(num_classes):
            cost_matrix[i, j] = np.sum((all_labels == i) & (all_preds == j))

    # 匈牙利算法计算最优匹配
    matched_indices = linear_assignment(-cost_matrix)  # 使用负的成本矩阵进行最大化匹配

    # 通过匹配结果重新映射聚类标签
    new_preds = np.copy(all_preds)
    for i, j in matched_indices:
        new_preds[all_preds == j] = i

    # 计算 NMI, ARI 和 ACC
    nmi = normalized_mutual_info_score(all_labels, new_preds)  # 归一化互信息
    ari = adjusted_rand_score(all_labels, new_preds)  # 调整兰德指数
    acc = accuracy_score(all_labels, new_preds)  # 聚类准确率

    return nmi, ari, acc

