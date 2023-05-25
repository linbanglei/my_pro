import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import copy
import VGG_Transfer_learning
import numpy as np
import cv2
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os 
import tkinter as tk


#获得可以提取特征向量的模型
def get_model():
    model,criterion,optimizer,scheduler,device=VGG_Transfer_learning.model_load()  
    best_model_param=torch.load('best_model_param.pth')
    in_features=model.classifier[3].in_features
    #out_features=len(os.listdir('D:\python\Medical image retrieval\difficult\\train'))
    out_features=best_model_param['outfeatures']
    model.classifier[3]=nn.Linear(in_features,out_features)
    #current_out_features=best_model_param['best_model']['classifier.3.bias'].shape[0]
    model.load_state_dict(best_model_param['best_model'])

    #去除模型的最后的分类层，分类层的输入量即为特征向量
    new_classifier=nn.Sequential(*list(model.classifier.children())[:-1])#加*是对于list而言，就是要把list中的值打散了，传进去就不是list，而是各个值。
    model.classifier=new_classifier
    #print(model)   现在模型的输出不是类别而是特征向量
    model.eval()#设置模型为评估模型，避免dropout等层的影响
    return model


#提取目标图像的特征向量
def get_target_image_feature_vector(model,target_image_path):
    img_transforms=transforms.Compose(
        [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#均值，标准差,
        ]
    )
    img=Image.open(target_image_path).convert('RGB')  #Image.open()读取的图像是单颜色通道的
    img=img_transforms(img) #数据预处理
    img=img.unsqueeze(0)#增加一个维度，变成1*3*224*224

    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img=img.to(device)

    with torch.no_grad():
        output=model(img).cpu().numpy()   #(1,1024)
    #print(output.shape)

    return output


#获得预处理过后的训练集图像数据
def get_train_data(train_path):
    img_transforms=transforms.Compose(
        [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#均值，标准差,
        ]
    )
    train_data=ImageFolder(train_path,img_transforms)
    return train_data



#提取训练集中的图像特征向量
def get_train_image_feature_vector(model,train_data):

    features = []
    labels = []

    with torch.no_grad(): #在评估模式下，仍然会计算和存储梯度
        for input, label in train_data: 
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            input = input.to(device)   
            #print(input.shape,label)   #torch.Size([3, 224, 224]) 0
            input = input.unsqueeze(0)  #torch.Size([1,3, 224, 224])
            output = model(input)
            #print(output.shape)   #torch.Size([1, 1024])
            output=output.tolist()[0]  
            features.append(output)
            labels.append(label)
            #print(features[0].shape,labels)  #原本是(1, 1024),通过reshape变成(1024,) [0]


    #转变为numpy数组
    features=np.array(features)
    labels=np.array(labels)

    #保存特征向量和标签
    np.savez('features.npz',features=features,labels=labels)
    

    return features,labels



#通过计算余弦相似度，找到最相似的图像的索引
def find_similar_image_index(target_image_feature_vector,train_image_feature_vector,top=50):
    #目标图像和训练集的特征向量都是二维数组,分别为(3000,1024)和(1,1024)

    #计算余弦相似度
    cos_sim=cosine_similarity(target_image_feature_vector,train_image_feature_vector)  #使用consine_similarity（）函数计算余弦相似度   (1,3000)
    index=np.argsort(cos_sim[0])[::-1] #索引从大到小排序
    similar_image_index=index[:top]  #获得前top张最相似的图像的索引  (10,)
    return similar_image_index



#在画布上显示图像
def show_image(target_image_path,train_data,similar_image_index):
    similar_image_path=[]
    similar_image_label=[]
    for i in range(len(similar_image_index)):
        path,label=train_data.imgs[similar_image_index[i]] #train_data.imgs由很多包含路径和标签的元组组成
        similar_image_path.append(path)
        similar_image_label.append(label)
    #print(similar_image_path)
    #print(similar_image_label)

    #接下来将所有的图像通过画布展示出来
    fig, axs = plt.subplots(3, 5, figsize=(10, 6))  #创建一个3*5的画布
    target_img=Image.open(target_image_path).convert('RGB')
    axs[0, 2].imshow(target_img)
    axs[0, 2].set_title('Target Image')
    k=0
    for i in range(2):
        for j in range(5):
            img=Image.open(similar_image_path[k]).convert('RGB') 
            axs[i+1,j].imshow(img)
            axs[i+1,j].set_title(similar_image_label[k])
            k+=1

    # 隐藏坐标轴
    for ax in axs.flat:
        ax.axis('off')

    # 调整子图之间的间距
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    # 显示画布
    plt.show()




if __name__ == '__main__':
    model=get_model()#获得可以提取特征向量的模型
    target_image_path='../images/test/09/2525.png'#目标图像的路径
    target_image_feature_vector=get_target_image_feature_vector(model,target_image_path)#获得目标图像的特征向量
    train_data=get_train_data('../images/train')#获得预处理后的训练集的数据
    #features,labels=get_train_image_feature_vector(model,train_data)#提取训练集中的图像特征向量，并保存

    data=np.load('features.npz')
    features=data['features']
    labels=data['labels']

    similar_image_index=find_similar_image_index(target_image_feature_vector,features)
    show_image(target_image_path,train_data,similar_image_index)
















    
