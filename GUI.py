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

#本模块用于实现GUI界面

class Application(tk.Frame):
    global epoch_num,out_features,target_image_path,current_text,similar_image_index_all,first_retrieval
    target_image_path=None
    epoch_num=1
    best_model_param=torch.load('best_model_param.pth')
    out_features=best_model_param['outfeatures']
    current_text=''
    similar_image_index_all=None   
    first_retrieval=True 


    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Medical Image Retrieval")
        self.place(x=0, y=0, width=1400, height=800)
        #self.pack(fill=tk.BOTH, expand=True)  
        self.create_widgets()
        self.create_menu()
        self.select_symptom()


    def create_widgets(self):
        #设置背景图片
        image=Image.open('background.jpg').convert('RGB')
        image=image.resize( (1400,800))
        global photo
        photo=ImageTk.PhotoImage(image)

        self.label1=tk.Label(self,image=photo)
        self.label1.place(x=0,y=0,width=1400,height=800)


        self.button1=tk.Button(self,text='目标图像',bg='white',command=self.open_file_dialog,relief='flat',font=('Times',30,'bold'))#relief='flat'表示按钮的边框不可见
        self.button1.place(x=30, y=30, width=330, height=330)
        self.button2=tk.Button(self,text='基于图像的一次检索',command=self.search_image,font=('Times',15,'bold'))
        self.button2.place(x=750, y=180, height=70, width=200)
        self.button3=tk.Button(self,text='基于症状的二次检索',command=self.base_sympotm_retrieval,font=('Times',15,'bold'))
        self.button3.place(x=750, y=270, height=70, width=200)
        self.label2=tk.Label(self,text='基于CT图像和病患症状大数据相似检索的疾病辅助诊断系统',font=('Times',28,'bold'))
        self.label2.place(x=380,y=40)

        #绘制饼状图
        self.fig, self.ax = plt.subplots(figsize=(8,8),dpi=100)
        self.fig.set_facecolor('lightgreen')
        self.ax.set_facecolor('lightgreen')
        #self.ax.axis('off')#隐藏坐标轴
        self.ax.set_position([0.15, 0.32, 0.7, 0.7])#四个参数分别表示距离左边缘的位置，距离底边缘的位置，饼图的宽度，饼图的高度
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)#将绘制的图形显示到tkinter窗口上
        self.canvas.get_tk_widget().place(x=980,y=120,width=420,height=680)
        self.button4=tk.Button(self,text='更新',command=self.pie_chart_show,font=('Times',15,'bold'))
        self.button4.place(x=1130,y=130,width=100,height=40)


        
    
    def create_menu(self):
        self.menubar=tk.Menu(self.master)
        self.filemenu=tk.Menu(self.menubar,tearoff=0)#tearoff=0表示菜单不可独立出来
        self.filemenu.add_command(label='打开',command=self.open_file_dialog,font=('Times',15,'bold'))
        self.filemenu.add_command(label="新增",command=self.add_data,font=('Times',15,'bold'))
        self.filemenu.add_command(label='更新特征向量',command=self.update_features,font=('Times',15,'bold'))
        self.filemenu.add_command(label='删除数据',command=self.delete_data,font=('Times',15,'bold'))
        self.filemenu.add_separator()
        self.filemenu.add_command(label="退出",command=self.master.quit,font=('Times',15,'bold'))
        self.menubar.add_cascade(label="文件",menu=self.filemenu,font=('Times',15,'bold'))

        self.trainmenu=tk.Menu(self.menubar,tearoff=0)
        self.trainmenu.add_command(label='初步训练',command=self.preliminary_training,font=('Times',15,'bold'))
        self.trainmenu.add_command(label='深度训练',command=self.depth_training,font=('Times',15,'bold'))
        self.trainmenu.add_command(label='设置迭代次数',command=self.set_epoch_num,font=('Times',15,'bold'))
        self.menubar.add_cascade(label='训练',menu=self.trainmenu,font=('Times',15,'bold'))

        self.mapping=tk.Menu(self.menubar,tearoff=0)
        self.mapping.add_command(label='查看映射关系',command=self.show_mapping,font=('Times',15,'bold'))
        self.mapping.add_cascade(label='修改映射',command=self.modify_mapping,font=('Times',15,'bold'))
        self.menubar.add_cascade(label='映射',menu=self.mapping,font=('Times',15,'bold'))


        self.master.config(menu=self.menubar)


    def select_symptom(self):
        def combox_select(event):
            global current_text
            select_text=self.combox.get()
            current_text=self.entry.get()
            if current_text:
                current_text=f'{current_text};{select_text}'
            else:
                current_text=select_text
            self.entry.delete(0,tk.END)
            self.entry.insert(0,current_text)



        #创建列表框和文本显示框
        values=[
        '无症状', '乏力', '黄疸', '肝区疼痛', '恶心', '发热', '食欲不振', '正常肝功能', '肝脏肿大', '轻度肝区不适', 
        '肝功能异常', '腹痛', '腹胀', '食欲减退', '消瘦', '头痛', '头晕', '失眠', '胸闷', '心悸', '呼吸困难', 
        '咳嗽', '咳痰', '鼻塞', '喉咙痛', '腹泻', '便秘', '关节疼痛', '肌肉疼痛', '皮疹', '瘙痒', '多尿', 
        '尿频', '尿急', '尿痛', '体重减轻', '体重增加', '乳房胀痛', '恶心与呕吐', '消化不良', '胃痛', '胃胀', 
        '背痛', '口渴', '出汗过多', '贫血', '易疲劳', '活动耐量下降', '腿肿', '心跳过速', '心跳过缓', '口臭', '嗳气'
        ]
        self.label3=tk.Label(self,text='请输入症状:',font=('Times',15,'bold'))
        self.label3.place(x=400,y=190,height=35)
        self.combox=ttk.Combobox(self,values=values,font=('Times',15,'bold'))
        self.combox.place(x=400,y=250,width=140,height=35)
        self.combox.current(0)
        self.combox.bind("<<ComboboxSelected>>",combox_select)

        self.entry=tk.Entry(self,font=('Times',15,'bold'))
        self.entry.place(x=400,y=310,width=300,height=35)
        #var2=tk.StringVar()
        #var2.set(current_text)
        self.entry.insert(0,current_text)
        



    def open_file_dialog(self):
        global target_image_path
        target_image_path = filedialog.askopenfilename(title="Select Image File", filetypes=[ ("All Files", "*.*"),("PNG Files", "*.png"), ("JPG Files", "*.jpg")],initialdir='../images/test')
        if target_image_path:  #是否选中了图像
            #print(target_image_path)
            image = Image.open(target_image_path).resize((330,330))
            global photo1   #设置为全局变量，避免函数执行完后变量被收回导致图像的消失
            photo1 = ImageTk.PhotoImage(image)
            self.button1.config(image=photo1)
        else:
            target_image_path=None



    def search_image(self):
        def image_show():
            global target_image_path,first_retrieval
            first_retrieval=True
            if target_image_path==None:
                tk.messagebox.showinfo('提示','请先选择目标图像')
                return
            model=features_extract.get_model()  #获得可以提取特征向量的模型
            if model==None:return
            train_data=features_extract.get_train_data('../images/train')#获得预处理后的训练集的数据
            target_image_feature_vector=features_extract.get_target_image_feature_vector(model,target_image_path)#获得目标图像的特征向量


            #获得训练集的特征向量和标签
            data=np.load('features.npz')
            features=data['features']
            labels=data['labels']
   
            global similar_image_index_all
            similar_image_index_all=features_extract.find_similar_image_index(target_image_feature_vector,features)  #通过余弦相似度找到最相似的图像的索引,这里返回了50个
            similar_image_index=similar_image_index_all[:10]  #取前10个索引
            #print(similar_image_index_all)

            #根据索引找到相似图像的路径和标签
            similar_image_path=[]
            similar_image_label=[]

            for i in range(len(similar_image_index)):
                path,label=train_data.imgs[similar_image_index[i]] #train_data.imgs由很多包含路径和标签的元组组成
                similar_image_path.append(path)
                similar_image_label.append(label)
            #print(similar_image_path)
            #print(similar_image_label)

            self.image_frame=tk.Frame(self,bg='light blue')
            self.image_frame.place(x=0, y=400, width=980, height=400)
        
            #显示相似图像
            for i in range(len(similar_image_index)):
                image=Image.open(similar_image_path[i]).convert('RGB').resize((170,170))
                global photo2
                photo2 = ImageTk.PhotoImage(image)
                #在画布上显示图像
                label=tk.Label(self.image_frame,image=photo2)
                label.image = photo2  # 保持对图像的引用，避免被垃圾回收
                #将标签映射到对应的具体类别
                # 从文件加载字典
                mapping_file = 'mapping.pkl'
                with open(mapping_file, 'rb') as f:#rb以二进制读取
                    loaded_mapping = pickle.load(f)
                label_class=loaded_mapping.get(str(similar_image_label[i]))
                #print(similar_image_label[i],label_class)
                label_title=tk.Label(self.image_frame,text=label_class,font=('Arial', 10),anchor='center')


                row, col = divmod(i, 5)  # 例如，每行放置5个图像 （商，余数）
                label_title.grid(row=row*2, column=col)
                label.grid(row=row*2+1, column=col,padx=10) 

                time.sleep(0.5)
            self.pie_chart_show()  #显示饼图
            first_retrieval=False
                
        #创建线程
        t1 = threading.Thread(target=image_show)#创建线程,避免GUI界面卡死
        t1.start()


    def add_data(self):
        dir_path=filedialog.askdirectory(title="选择目标文件夹，需要包含train和test两个子文件夹",initialdir='../')
        #print(dir_path)

        if dir_path:#如果选择了文件夹

            #获取目标文件夹中的train和test文件夹的路径
            train_dir_path=os.path.join(dir_path,'train')
            test_dir_path=os.path.join(dir_path,'test')

            for i in ['train','test']:
                if i=='train':
                    for j in os.listdir(train_dir_path):
                        #获取每个类别的文件夹的路径
                        class_dir_path=os.path.join(train_dir_path,j)
                        shutil.copytree(class_dir_path,'../images/train/'+j)
                else:
                    for j in os.listdir(test_dir_path):
                        #获取每个类别的文件夹的路径
                        class_dir_path=os.path.join(test_dir_path,j)
                        shutil.copytree(class_dir_path,'../images/test/'+j)
            print('数据添加完成')
            print('请及时更新特征向量，否则仍无法从训练集中检索出新增的图像')
            print('请更新映射关系')


    def update_features(self):
        def update_features_thread():
            #更新特征向量
   
            print('训练集数据变化，重新计算特征向量和标签')
            model=features_extract.get_model()  #获得可以提取特征向量的模型
            train_data=features_extract.get_train_data('../images/train')#获得预处理后的训练集的数据
            features,labels=features_extract.get_train_image_feature_vector(model,train_data)
            print('计算特征向量和标签完成')
        #创建线程
        t2 = threading.Thread(target=update_features_thread)#创建线程,避免GUI界面卡死
        t2.start()
    

    def delete_data(self):
        # 选择要删除的文件夹
        selected_folder = filedialog.askdirectory(title="选择要删除的文件夹",initialdir='../images/train')

        if selected_folder:
            # 获取文件夹名
            folder_name = os.path.basename(selected_folder)

            # 构造train和test子文件夹中对应文件夹的路径
            train_dir = "../images/train"  
            test_dir = "../images/test"    

            # 删除train和test子文件夹中的同名文件夹
            train_folder_path = os.path.join(train_dir, folder_name)
            test_folder_path = os.path.join(test_dir, folder_name)

            if os.path.exists(train_folder_path):
                shutil.rmtree(train_folder_path)

            if os.path.exists(test_folder_path):
                shutil.rmtree(test_folder_path)

            print('成功删除类别（文件夹名）：', folder_name)
            print('请及时更新特征向量,否则使用未更新的特征向量查找到的图像的索引可能超出当前训练集列表的范围从而产生错误')
            print('请更新映射关系')


    def set_epoch_num(self):
        def save_epoch_num():
            global epoch_num
            if entry.get():
                epoch_num=int(entry.get())
                window.destroy()

        window=tk.Toplevel()
        window.title('设置迭代数')
        window.geometry('400x150+500+300')
        label = tk.Label(window, text="epoch_num:",font=('Arial', 15,'bold'),anchor='center')
        label.place(x=50,y=10,height=50,width=120)

        var = tk.StringVar()
        var.set('1')
        entry = tk.Entry(window,textvariable=var,font=('Arial', 15,'bold'))
        entry.place(x=200,y=10,height=50,width=80)

        button = tk.Button(window, text="Save", command=save_epoch_num,font=('Arial', 15,'bold'),anchor='center')
        button.place(x=130,y=80,height=50,width=100)

        window.grab_set()#阻止用户在弹出窗口之外的地方进行操作
        window.attributes('-topmost', True)#窗口置顶
        window.wait_window()#等待窗口关闭




    def preliminary_training(self):#只训练全连接层

        def confirm_retrain():#在数据集未变化的情况下，确认是否重新训练

            global result
            result=False

            def yes():
                global result
                result=True
                window.destroy()
            def no():
                global result
                result=False
                window.destroy()

            window=tk.Toplevel()
            window.title('确认')
            window.geometry('350x100+600+400')
            label=tk.Label(window,text='数据集可能未变化，是否重新训练？',font=('times',12,'bold'))
            label.place(x=55,y=10)
            button1=tk.Button(window,text='是',command=yes,font='bold')
            button1.place(x=110,y=40,width=40)
            button2=tk.Button(window,text='否',command=no,font='bold')
            button2.place(x=220,y=40,width=40)

            window.grab_set()#阻止用户在弹出窗口之外的地方进行操作
            window.attributes('-topmost', True)#窗口置顶
            window.wait_window()#等待窗口关闭

            return result


        def preliminary_training_thread():
            global out_features,epoch_num
            train_path='../images/train'
            test_path='../images/test'
            if out_features==len(os.listdir(train_path)):
                if not confirm_retrain():
                    return
            print('开始初步训练')
            train_load,test_load=VGG_Transfer_learning.data_load(train_path,test_path)
            model,criterion,optimizer,scheduler,device=VGG_Transfer_learning.model_load()
            #print(model)
            
            in_features=model.classifier[3].in_features
            out_features=len(os.listdir(train_path))
            model.classifier[3]=nn.Linear(in_features,out_features)
            model.to(device)
            #print(model)
            VGG_Transfer_learning.train_model(model,train_load,test_load,criterion,optimizer,scheduler,device,best_acc=0,epoch_num=epoch_num)
            print('初步训练完成,想要获得更好的效果请进行深度训练')
            print('请更新特征向量')
        #创建线程
        t3 = threading.Thread(target=preliminary_training_thread)#创建线程,避免GUI界面卡死
        t3.start()  
        


    def depth_training(self):#训练全连接层和卷积层
        def depth_training_thread():
            train_path='../images/train'
            test_path='../images/test'
            global out_features,epoch_num
            if out_features!=len(os.listdir(train_path)):
                #弹出警告窗口
                tk.messagebox.showerror("警告", "不能在进行深度训练之前改变数据集")
                print('数据集改变训练中止，请重新进行初步训练')
                return

            print('开始深度训练')
            train_load,test_load=VGG_Transfer_learning.data_load(train_path,test_path)
            model,criterion,optimizer,scheduler,device=VGG_Transfer_learning.model_load()
            for param in model.parameters():
                param.requires_grad=True
            in_features=model.classifier[3].in_features
            model.classifier[3]=nn.Linear(in_features,out_features)
            model.to(device)
            optimizer=optim.Adam(model.parameters(),lr=0.0001)#要进行微调，优化器学习率改小点

            best_model_param=torch.load('best_model_param.pth')
            best_acc=best_model_param['best_acc']
            #print(model,best_acc)
            model.load_state_dict(best_model_param['best_model'])
            optimizer.load_state_dict(best_model_param['optimizer'])

            VGG_Transfer_learning.train_model(model,train_load,test_load,criterion,optimizer,scheduler,device,best_acc,epoch_num=epoch_num)
            print('深度训练完成,请更新特征向量')
        
        #创建线程
        t4 = threading.Thread(target=depth_training_thread)#创建线程,避免GUI界面卡死
        t4.start()



    def show_mapping(self):
        window=tk.Toplevel()
        window.title('映射关系')
        window.geometry('300x300+600+300')
        listbox=tk.Listbox(window,font=('微软雅黑',15))
        listbox.place(x=0,y=0,width=300,height=300)

        with open('mapping.pkl','rb') as f:
            mapping=pickle.load(f)

        for i in mapping:
                listbox.insert(i,i+'--->'+mapping.get(i))



    def modify_mapping(self):
        def add_mapping():
            num=entry1.get()
            name=entry2.get()
            if num and name:
                with open('mapping.pkl','rb') as f:
                    mapping=pickle.load(f)
                if mapping.get(num)!=None:
                    if mapping.get(num)=='None':
                        mapping[num]=name
                    else:
                        tk.messagebox.showerror("警告", "该编号已存在")
                        return
                else:
                    mapping[num]=name
                with open('mapping.pkl','wb') as f:
                    pickle.dump(mapping,f)
                print('添加成功')
            else:
                tk.messagebox.showinfo("提示", "请输入映射关系")

        def delete_mapping():
            num=entry1.get()
            name=entry2.get()
            if num:
                with open('mapping.pkl','rb') as f:
                    mapping=pickle.load(f)
                if num==list(mapping.keys())[-1]:
                    del mapping[num]
                else:
                    mapping[num]='None'
                with open('mapping.pkl','wb') as f:
                    pickle.dump(mapping,f)
                print('删除成功')
            else:
                tk.messagebox.showinfo("提示", "请输入映射关系")

        def replace_mapping():
            num=entry1.get()
            name=entry2.get()
            
            if num and name:
                with open('mapping.pkl','rb') as f:
                    mapping=pickle.load(f)
                if sum([x==num for x in mapping.keys()])==1:
                    mapping[num]=name
                else:
                    tk.messagebox.showerror("警告", "该编号不存在")
                    return
                with open('mapping.pkl','wb') as f:
                    pickle.dump(mapping,f)
                print('替换成功')
            else:
                tk.messagebox.showinfo("提示", "请输入映射关系")

        window=tk.Toplevel()
        window.title('修改映射关系')
        window.geometry('500x150+600+400')
        add_button=tk.Button(window,text='添加',command=add_mapping,font=('bold'))
        add_button.place(x=70,y=70,width=70,)
        delete_button=tk.Button(window,text='删除',command=delete_mapping,font=('bold'))
        delete_button.place(x=210,y=70,width=70)
        modify_button=tk.Button(window,text='替换',command=replace_mapping,font=('bold'))
        modify_button.place(x=350,y=70,width=70)
        entry1=tk.Entry(window)
        entry1.place(x=140,y=30,width=70)
        entry2=tk.Entry(window)
        entry2.place(x=280,y=30,width=70)
        label=tk.Label(window,text='->')
        label.place(x=240,y=30)

        window.grab_set()#阻止用户在弹出窗口之外的地方进行操作
        #window.attributes('-topmost', True)#窗口置顶
        #window.wait_window()#等待窗口关闭



    def base_sympotm_retrieval(self):
        if self.entry.get()=='':
            tk.messagebox.showinfo("提示", "请先输入症状信息")
            return
        if similar_image_index_all is not None:
            print('正在通过病人信息检索图像')

            def base_sympotm_retrieval_thread():
                syptom_list=self.entry.get().split(';')
                #print(syptom_list)
                disease_list=[]
                file='symptom_message.json'
                for i in syptom_list:   #['乏力', '黄疸']
                    with open(file,'r',encoding='utf-8') as f:
                        data=json.load(f)
                    for j in data:
                        if i in data[j]:
                            disease_list.append(j)
                disease_list=list(set(disease_list))
                #print(disease_list)#通过先转化成集合再转化成列表，去除重复元素  ['肝炎', '肿瘤癌']

                #根据索引找到相似图像的路径和标签
                similar_image_path=[]
                similar_image_label=[]  #映射之前的标签
                train_data=features_extract.get_train_data('../images/train')#获得预处理后的训练集的数据
                for i in range(len(similar_image_index_all)):
                    path,label=train_data.imgs[similar_image_index_all[i]] #train_data.imgs由很多包含路径和标签的元组组成
                    similar_image_path.append(path)
                    similar_image_label.append(label)
                #print(similar_image_path)
                #print(similar_image_label)

                #获得映射之后的标签
                with open('mapping.pkl','rb') as f:
                    mapping=pickle.load(f)
                similar_image_label=[mapping[str(x)] for x in similar_image_label]#映射之后的标签
                #print(similar_image_label)
                show_image_path=[]
                show_image_label=[]
                for i in range(len(similar_image_label)):
                    if similar_image_label[i] in disease_list:
                        show_image_path.append(similar_image_path[i])
                        show_image_label.append(similar_image_label[i])
            
                #显示图像
                if show_image_path:
                    self.image_frame=tk.Frame(self,bg='light blue')
                    self.image_frame.place(x=0, y=400, width=980, height=400)
        
                    for i in range(len(show_image_path[:10])):
                        image=Image.open(show_image_path[i]).convert('RGB').resize((170,170))
                        global photo3
                        photo3 = ImageTk.PhotoImage(image)
                        #在画布上显示图像
                        label=tk.Label(self.image_frame,image=photo3)
                        label.image = photo3  # 保持对图像的引用，避免被垃圾回收
                        label_title=tk.Label(self.image_frame,text=show_image_label[i],font=('Arial', 10),anchor='center')


                        row, col = divmod(i, 5)  # 例如，每行放置5个图像 （商，余数）
                        label_title.grid(row=row*2, column=col)
                        label.grid(row=row*2+1, column=col,padx=8) 

                        time.sleep(0.5)
                else:
                    self.image_frame=tk.Frame(self,bg='light blue')
                    self.image_frame.place(x=0, y=400, width=980, height=400)
                    tk.messagebox.showinfo("提示", "没有检索到相关图像,可能是由于录入的症状信息与当前目标图像不匹配")
                    #print('没有检索到相关图像,可能是由于录入的症状信息与当前目标图像不匹配')
                print('检索结束')

            #创建线程
            t4=threading.Thread(target=base_sympotm_retrieval_thread)
            t4.start()

        else:
            tk.messagebox.showinfo("提示", "请先检索图像")
            return


    def pie_chart_show(self):
        #部分代码与上面的base_sympotm_retrieval()函数相同
        if similar_image_index_all is not None:
            global first_retrieval
            syptom_list=self.entry.get().split(';')
            #print(syptom_list)
            disease_list=[]
            file='symptom_message.json'
            for i in syptom_list:   #['乏力', '黄疸']
                with open(file,'r',encoding='utf-8') as f:
                    data=json.load(f)
                for j in data:
                    if i in data[j]:
                        disease_list.append(j)
            disease_list=list(set(disease_list))
            #print(disease_list)#通过先转化成集合再转化成列表，去除重复元素  ['肝炎', '肿瘤癌']

            #根据索引找到相似图像的路径和标签
            similar_image_path=[]
            similar_image_label=[]  #映射之前的标签
            train_data=features_extract.get_train_data('../images/train')#获得预处理后的训练集的数据
            for i in range(len(similar_image_index_all)):
                path,label=train_data.imgs[similar_image_index_all[i]] #train_data.imgs由很多包含路径和标签的元组组成
                similar_image_path.append(path)
                similar_image_label.append(label)
            #print(similar_image_path)
            #print(similar_image_label)
        

            #获得映射之后的标签
            with open('mapping.pkl','rb') as f:
                mapping=pickle.load(f)
            similar_image_label=[mapping[str(x)] for x in similar_image_label]#映射之后的标签
            #print(similar_image_label)

            matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']#设置中文显示
            matplotlib.rcParams['axes.unicode_minus'] = False

            values=[]
            if first_retrieval==True:
                categories=list(set(similar_image_label))
                for i in categories:
                    values.append(similar_image_label.count(i))
                #print(values)
                values=values/np.sum(values)*100
                #print(categories,values)
            else:
                if self.entry.get()=='':
                    tk.messagebox.showinfo("提示", "请先输入症状信息并进行基于症状的二次检索")
                    return
                else:
                    categories=disease_list
                    for i in categories:
                        values.append(similar_image_label.count(i))
                    #print(values)
                    if np.sum(values)==0:
                        tk.messagebox.showinfo("提示", "没有检索到相关图像,可能是由于录入的症状信息与当前目标图像不匹配")
                        return
                    else:
                        #去除值为0的元素
                        for i in range(len(values)-1,-1,-1):
                            if values[i]==0:
                                del values[i]
                                del categories[i]
                        values=values/np.sum(values)*100
                    #print(categories,values)

            self.ax.clear()
            #self.ax.pie(values, labels=categories, autopct='%1.1f%%', startangle=90)
            wedges, text, autotexts = self.ax.pie(values, labels=categories, autopct='%1.1f%%', startangle=90)

            self.ax.axis('equal')

            # 添加图例
            legend_texts = [f'{category}: {percent}%' for category, percent in zip(categories, np.round(values / np.sum(values) * 100, 1))]
            self.ax.legend(wedges, legend_texts, title="类别和百分比",  loc="upper center", bbox_to_anchor=(0.5, 0.1),ncol=2)

            self.canvas.draw()
        else:
            tk.messagebox.showinfo("提示", "请先检索图像")
            return

        '''
        show_image_path=[]
        show_image_label=[]
        for i in range(len(similar_image_label)):
            if similar_image_label[i] in disease_list:
                show_image_path.append(similar_image_path[i])
                show_image_label.append(similar_image_label[i])'''



        



        
    
if __name__ == '__main__':
    root=tk.Tk()
    root.geometry('1400x800+200+70')
    app=Application(master=root)
    app.mainloop()
        
        