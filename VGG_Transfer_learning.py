import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import copy
import UI123
import os



def data_load(train_path,test_path):  #用于数据的加载
    #自定义数据预处理和数据增强
    train_transforms=transforms.Compose(
        [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(0.5),#垂直翻转
        transforms.RandomVerticalFlip(0.5),#水平翻转
        transforms.RandomRotation(45),#随机旋转-45~45
        transforms.RandomGrayscale(0.025),#概率转化成灰度率
        transforms.ColorJitter(0.2,0.1,0.1,0.1),#亮度，对比度，饱和度，色相
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#均值，标准差,
        ]
    )

    test_transforms=transforms.Compose(
        [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#-均值，/标准差,
        ]
    )

    train_data=ImageFolder(train_path,train_transforms) #加载训练集   默认将所有加载的图像转换为RGB格式
    #print(train_data) #可以看成很多dataset 一张图片带一个标签
    train_load=DataLoader(train_data,batch_size=4,shuffle=True)#分批处理，每批4个数据，打乱
    #for i in train_load:
    #    j,k=i
    #    print(j.shape,k)  #torch.Size([4, 3, 224, 224]) tensor([ 0, 13,  2, 11])
    #    print(k.size(0))
    #    break

    test_data=ImageFolder(test_path,test_transforms)
    test_load=DataLoader(test_data,batch_size=4,shuffle=True)

    return train_load,test_load



def model_load(): #用于模型的加载
    #model=torchvision.models.vgg16(pretrained=True)
    model=torchvision.models.vgg13(pretrained=True)
    #model=torchvision.models.vgg11(pretrained=True)
    #model=torchvision.models.mobilenet_v3_large(pretrained=True)
    #print(model.parameters())
    #print(model.features)
    #for i in model.features:
    #    print(i)

    #调整全连接层
    
    model.classifier=nn.Sequential(
        nn.Linear(25088,1024),
        nn.ReLU(True),              #激活函数max(0,x)，更好的学习非线性特征，计算简单;不会出现梯度消失;增加模型的稀疏性,减少模型的复杂程度
        nn.Dropout(),               #减少过拟合，每个神经元的输出被置为0的概率，通常在0.2~0.5，vgg模型较复杂，通常0.5
        nn.Linear(1024,15),
    )
    '''
    
    model.classifier = nn.Sequential(
        #nn.Flatten(),  #展开变成一维，VGG13模型存在全局平均池化层，这是不必要的
        nn.Dropout(),
        nn.Linear(25088, 15) # 将原始的3个全连接层替换为一个全连接层 vgg
        #nn.Linear(960, 15) # 将原始的3个全连接层替换为一个全连接层   mobilenet_v3_large

    )'''

    #经过测试后发现使用多个全连接层的效果更差，可能是由于参数过多产生了过拟合
    #print(model)
    
    #卷积层的冻结
    for param in model.features.parameters():
        param.requires_grad=False                    #冻结卷积层参数，只训练最后一层全连接层
    '''
    for i,param in enumerate(model.features): #根据需要冻结前面的卷积层
        if isinstance(param,nn.Conv2d):
          if i>=22:
            param.requires_grad_(True)
          else:
            param.requires_grad_(False)  
    '''
    #for  name,param in model.named_parameters():  #查看前面的卷积层是否已经被冻结住了
     #   print( name,param.requires_grad)

    #print(model.parameters())
    #print(model.classifier)#全连接层
    #print(model.classifier.parameters())#.parameters()返回一个包含模型所有参数的迭代器，我们可以使用这个迭代器来访问和更新模型的参数。

    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')#如果gpu可用，就将模型放到gpu上去跑
    model.to(device)
    #print(torch.cuda.is_available())

    #定义损失函数和优化器
    criterion=nn.CrossEntropyLoss() #交叉熵损失函数    存在激活函数softmax（）操作
    #print(model.features[17:])
    #optimizer=optim.Adam([{'params':model.classifier.parameters(),'lr':0.1},{'params':model.features[10:].parameters(),'lr':0.01}])
    optimizer=optim.Adam(model.parameters(),lr=0.001)#
    scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.1)#学习率每5个epoch衰减成原来的1/10

    return model,criterion,optimizer,scheduler,device




def train_model(model,train_load,test_load,criterion,optimizer,scheduler,device,best_acc,epoch_num=25,fillname='best_model_param.pth'):
    #训练模型
    for epoch in range(epoch_num):
        for phase in ['train','test']:
            if phase=='train':
                print('Train:')
                model.train() #设置为训练模式
            else:
                print('Test:')
                model.eval()  #设置为评估模式
            
            runing_loss=0.00 #累计损失值
            num_batch=0      #batch的数量
            pre_correct=0    #预测正确数
            if phase=='train':  #判断加载训练集数据还是测试集数据
                data_load=train_load  
            else:
                data_load=test_load
            
            for i,j in enumerate(data_load): 
                inputs,labels=j     
                #print(inputs.shape,labels)    #torch.Size([4, 3, 224, 224]) tensor([8, 7, 3, 3])
                inputs=inputs.to(device)
                labels=labels.to(device)
                optimizer.zero_grad()   #梯度清零 
                outputs=model(inputs)
                loss=criterion(outputs,labels) #通过输出地结果和正确地结果比对计算出损失值 
                _,predicted=torch.max(outputs,1)
                if phase=='train': #只在进行训练的时候进行反向传播和更新梯度
                    loss.backward() #反向传播 
                    optimizer.step()  #更新梯度

                runing_loss+=loss.item()*inputs.size(0)  #item()将张量的值转化为python中的标量的值
                num_batch+=inputs.size(0)
                #print(predicted,labels)
                pre_correct+=(predicted==labels).sum().item()
                #torch.cuda.empty_cache()

            epoch_loss=runing_loss / num_batch
            epoch_acc=pre_correct/num_batch
            #print(pre_correct,num_batch)
            print('epoch %d / %d,  epoch_loss: %.3f,  epoch_acc:%.3f ' %(epoch + 1,epoch_num, epoch_loss,epoch_acc))

            if phase=='test' and epoch_acc>best_acc:
                best_acc=epoch_acc
                best_model=copy.deepcopy(model.state_dict()) #深度复制出一个独立的包含所有参数的模型，复制出的模型的参数不随元模型参数的变化而改变
                state={
                'best_model':best_model,
                'best_acc':best_acc,
                'optimizer':optimizer.state_dict(),
                'outfeatures':model.classifier[3].out_features
                }
                torch.save(state,fillname)

        scheduler.step()  #更新学习率
        #print(scheduler.get_last_lr()[0]) #查看学习率
        print('\n')

        #每个epoch结束后更新学习率
        #scheduler.step()



if __name__=='__main__':
    train_path='../images/train'
    test_path='../images/test'
    train_load,test_load=data_load(train_path,test_path)
    model,criterion,optimizer,scheduler,device=model_load()
    #train_model(model,train_load,test_load,criterion,optimizer,scheduler,device,best_acc=0)
   

































 
#保存模型
#PATH = 'D:\python\Medical image retrieval\model_best.pth'
#torch.save(model.state_dict(), PATH)

'''
            with torch.set_grad_enabled(phase=='train'): #判断内部需不需要进行梯度计算
                #outputs=model(inputs)
                #print(outpus.shape) #torch.Size([4, 15])
                #print(outpus.data) #没有grad_fn属性
                #print(outpus) #有grad_fn属性 
                #loss=criterion(outputs,labels) #通过输出地结果和正确地结果比对计算出损失值  
                #_,predicted=torch.max(outputs,1)

                if phase=='train': #只在进行训练的时候进行反向传播和更新梯度
                    loss.backward() #反向传播  
                    optims.step()  #更新梯度
            '''   












        













    











