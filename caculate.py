import tkinter as tk
from tkinter import filedialog
from PIL import Image,ImageTk
import features_extract
import numpy as np
from tkinter import ttk
import torch
import time
import model_retrain
import VGG_Transfer_learning
import os
import shutil
import threading
import torch.nn as nn
import torch.optim as optim
import pickle
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
import matplotlib
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.neighbors import KNeighborsClassifier

#本模块用于计算不同相似性度量算法的效果

# 定义一个函数来计算准确率、召回率、精确度和平均精确度
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    return accuracy, recall, precision

#获得训练集的特征向量和标签
data=np.load('features.npz')
features=data['features']
labels=data['labels']

test_path='D:\python\images\\test'#测试集路径
#target_image_path='D:\python\images\\test\\00\\12522.jpg'#某张测试集图像的路径
model=features_extract.get_model()  #获得可以提取特征向量的模型
#train_data=features_extract.get_train_data('../images/train')#获得预处理后的训练集的数据
#target_image_feature_vector=features_extract.get_target_image_feature_vector(model,target_image_path)#获得目标图像的特征向量

# 加载测试数据并提取特征向量
test_data = features_extract.get_train_data(test_path)
test_features, test_labels = features_extract.get_train_image_feature_vector(model, test_data)


distance_metrics = ['euclidean', 'manhattan', 'chebyshev', 'cosine']#欧式距离、曼哈顿距离、切比雪夫距离、余弦距离

accuracy_list=[]
recall_list=[]
precision_list=[]
for metric in distance_metrics:
    # 定义并训练 KNN 分类器
    knn = KNeighborsClassifier(n_neighbors=10, metric=metric)
    knn.fit(features, labels)

    # 预测测试数据的类别
    y_pred = knn.predict(test_features)

    # 计算准确率、召回率、精确度和平均精确度
    accuracy, recall, precision= calculate_metrics(test_labels, y_pred)

    accuracy_list.append(accuracy)
    recall_list.append(recall)
    precision_list.append(precision)

algorithms = ['欧式距离', '曼哈顿距离', '切比雪夫距离', '余弦距离']

# 设置柱状图的宽度
bar_width = 0.25
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']#设置中文显示
matplotlib.rcParams['axes.unicode_minus'] = False
# 创建x轴的坐标
x = np.arange(len(algorithms))

# 绘制柱状图
plt.bar(x, accuracy_list, width=bar_width, label='准确率')
plt.bar(x + bar_width, recall_list, width=bar_width, label='召回率')
plt.bar(x + 2 * bar_width,precision_list, width=bar_width, label='精确度')


# 设置x轴标签
plt.xticks(x + bar_width, algorithms, rotation=45)

# 设置标题和坐标轴标签
plt.title('不同相似度度量算法对检索效果的影响')
plt.xlabel('相似度度量算法')
plt.ylabel('性能指标')

# 添加图例
plt.legend()

# 显示图表
plt.show()
    
