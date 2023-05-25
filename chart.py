import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import VGG_Transfer_learning
import torch
import features_extract
import torch.nn as nn
from torch.utils.data import DataLoader

#本模块用于绘制模型迭代过程中损失值和准确率的变化曲线和计算系统在各个类别上的准确率

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']#设置中文显示
matplotlib.rcParams['axes.unicode_minus'] = False
'''
#绘制模型迭代过程中损失值和准确率的变化曲线
# 第一阶段训练损失、准确率和测试损失、准确率数据
train_losses_1 = [1.411, 0.674, 0.594, 0.525, 0.501, 0.323, 0.253, 0.241, 0.228, 0.210]
train_accs_1 = [0.635, 0.806, 0.833, 0.841, 0.854, 0.899, 0.921, 0.923, 0.930, 0.933]
test_losses_1 = [0.637, 0.482, 0.527, 0.439, 0.406, 0.326, 0.349, 0.361, 0.341, 0.304]
test_accs_1 = [0.829, 0.857, 0.842, 0.864, 0.887, 0.895, 0.904, 0.905, 0.909, 0.913]

# 第二阶段训练损失、准确率和测试损失、准确率数据
train_losses_2 = [0.545, 0.347, 0.269, 0.258, 0.242, 0.220, 0.218, 0.171, 0.207, 0.189]
train_accs_2 = [0.837, 0.883, 0.910, 0.920, 0.923, 0.933, 0.928, 0.945, 0.938, 0.945]
test_losses_2 = [0.490, 0.428, 0.299, 0.240, 0.311, 0.228, 0.203, 0.172, 0.240, 0.185]
test_accs_2 = [0.859, 0.879, 0.904, 0.921, 0.903, 0.925, 0.918, 0.951, 0.932, 0.950]

epochs = list(range(1, 11))

# 设置画布大小
plt.figure(figsize=(12, 6))

# 绘制训练损失与测试损失折线图
plt.subplot(121)
plt.plot(epochs, train_losses_1, label="1阶段Train_loss", marker="o", linestyle="--")
plt.plot(epochs, test_losses_1, label="1阶段Test_loss", marker="o", linestyle="-")
plt.plot(epochs, train_losses_2, label="2阶段Train_loss", marker="o", linestyle="--")
plt.plot(epochs, test_losses_2, label="2阶段Test_loss", marker="o", linestyle="-")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Testing Loss")

# 绘制训练准确率与测试准确率折线图
plt.subplot(122)
plt.plot(epochs, train_accs_1, label="1阶段Train_acc", marker="o", linestyle="--")
plt.plot(epochs, test_accs_1, label="1阶段Test_acc", marker="o", linestyle="-")
plt.plot(epochs, train_accs_2, label="2阶段Train_acc", marker="o", linestyle="--")
plt.plot(epochs, test_accs_2, label="2阶段Test_acc", marker="o", linestyle="-")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training and Testing Accuracy")

plt.tight_layout()
plt.show()
'''

model,criterion,optimizer,scheduler,device=VGG_Transfer_learning.model_load()  
best_model_param=torch.load('best_model_param.pth')
in_features=model.classifier[3].in_features
out_features=best_model_param['outfeatures']
model.classifier[3]=nn.Linear(in_features,out_features)
model.load_state_dict(best_model_param['best_model'])
model.eval()#设置模型为评估模型，避免dropout等层的影响
model.to(device)#将模型移动到GPU上

test_path='D:\python\images\\test'#测试集路径
test_data = features_extract.get_train_data(test_path)
test_load = DataLoader(test_data, batch_size=4, shuffle=False)

# 准备混淆矩阵
num_classes = len(test_data.classes)
confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
#print(num_classes)
#print(confusion_matrix.shape)


# 在测试数据上评估模型
with torch.no_grad():
    for inputs, labels in test_load:
        inputs = inputs.to(device)  # 将输入数据移动到 GPU
        labels = labels.to(device)  # 将标签数据移动到 GPU

        # 获取预测结果
        logits = model(inputs)
        predictions = torch.argmax(logits, dim=1)

        # 更新混淆矩阵
        for t, p in zip(labels.view(-1), predictions.view(-1)):
            confusion_matrix[t.int(), p.int()] += 1
print(confusion_matrix)

# 计算每个类别的准确率
class_accuracies = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)
print(class_accuracies)
# 输出每个类别的准确率
for i, acc in enumerate(class_accuracies):
    print(f"Class {i}: {acc * 100:.2f}%")

# 绘制每个类别准确率的柱状图
plt.figure(figsize=(10, 5))
plt.bar(range(num_classes), class_accuracies)
plt.xlabel('Class')
plt.ylabel('Accuracy')
plt.title('Accuracy per Class')
plt.show()



