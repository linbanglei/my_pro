import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import copy
import VGG_Transfer_learning
import os



def retrain(): 

    train_path='../images/train'
    test_path='../images/test'
    train_load,test_load=VGG_Transfer_learning.data_load(train_path,test_path)
    model,criterion,optimizer,scheduler,device=VGG_Transfer_learning.model_load()


    #重新训练所有层
    for param in model.parameters():
            param.requires_grad_(True)


    #查看各个层的冻结情况
    #for name,param in model.named_parameters():
    #    print(name,param.requires_grad)

    optimizer=optim.Adam(model.parameters(),lr=0.0001)#要进行微调，优化器学习率改小点

    #导入上次训练中最好的模型和优化器参数
    best_model_param=torch.load('best_model_param.pth')
    best_acc=best_model_param['best_acc']
    #print(model,best_acc)
    model.load_state_dict(best_model_param['best_model'])
    optimizer.load_state_dict(best_model_param['optimizer'])
    #print(best_acc)   #0.964
    #print(best_model_param['outfeatures'])



    return model,train_load,test_load,criterion,optimizer,scheduler,device,best_acc


#重新训练模型
if __name__=='__main__':
    model,train_load,test_load,criterion,optimizer,scheduler,device,best_acc= retrain()
    #VGG_Transfer_learning.train_model(model,train_load,test_load,criterion,optimizer,scheduler,device,best_acc=0.9) 










