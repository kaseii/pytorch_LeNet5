
# coding: utf-8

# In[43]:


import numpy as np
from PIL import Image
from scipy.misc import imread
from skimage import transform,data
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import cv2
#设置k值和取样数
k = 7
TEST_RATE = 20
#X_train = np.loadtxt("train_X.txt")
#已经处理好的特征矩阵
colors = np.loadtxt("colors.txt")
LBP_hist = np.loadtxt("LBP_hist.txt")

def LBP(img):#LBP算法，提取图像纹理特征
    col, row = img.shape
    tempSave = []
    dataLBP = []
    for i in range(1,col-1):
        for j in range(1,row-1):
            if img[i][j] > img[i-1][j-1]:#
                tempSave.append(0)
            else:
                tempSave.append(1)
            if img[i][j] > img[i-1][j]:#
                tempSave.append(0)
            else:
                tempSave.append(1)
            if img[i][j] > img[i-1][j+1]:#
                tempSave.append(0)
            else:
                tempSave.append(1)
            if img[i][j] > img[i][j+1]:#
                tempSave.append(0)
            else:
                tempSave.append(1)
            if img[i][j] > img[i+1][j+1]:#
                tempSave.append(0)
            else:
                tempSave.append(1)
            if img[i][j] > img[i+1][j]:#
                tempSave.append(0)
            else:
                tempSave.append(1)
            if img[i][j] > img[i+1][j-1]:#
                tempSave.append(0)
            else:
                tempSave.append(1)
            if img[i][j] > img[i][j-1]:#
                tempSave.append(0)
            else:
                tempSave.append(1)
            temp = [str(k) for k in tempSave]
            p = ''.join(temp)
            dataLBP.append(int(p,2))
            tempSave = []
    a = np.array([dataLBP])
    return a#a为一维行向量

'''
数据处理，利用LBP提取纹理特征后，做成训练矩阵

for i in range(1,201):
    s="%s%s%d%s%s"%('train_data/','data (',i,')','.jpg')#原图片名为data (i).jpg
    im = Image.open(s)
    L = im.convert('L')#转化为灰度图
    s="%s%d%s"%('train/',i,'.jpg')
    L.save(s)
img = imread('train/1.jpg')
train_X = LBP(img)
for i in range(2,201):
    s="%s%d%s"%('train/',i,'.jpg')
    img = imread(s)
    img = LBP(img)
    train_X = np.concatenate((train_X,img), axis = 0)#将每个图片的特征拼接起来，成为（200，22940）的矩阵
np.savetxt("train_X.txt",train_X)
   
 '''

'''
加载训练好的LBP特征，将其作为LBP直方图
x_train = np.loadtxt("train_X.txt")
X_train = np.ones([200,256])
X = []
print(x_train.shape)
for i in range(200):
    for j in range(256):
        for x in map(lambda x:(j<=x and x<j+1), x_train[i,:]):#0-1 1-2 2-3......
            if x == True:
                train_temp.append(x)
        X.append(len(train_temp))
        train_temp = []
    X = np.array(X)
    X_train[i,:] = X
    X = []
    if i%10 == 0:
        print("step:",i)
#最后得到（200，256）的LBP直方图，实现了降维
np.savetxt("LBP_hist.txt",X_train)

#利用opencv提取颜色特征，为了合成特征方便，这里提取的是rgb颜色特征
colors = np.zeros([200,256])
for i in range(1,201):
    s="%s%s%d%s%s"%('train_data/','data (',i,')','.jpg')
    img = cv2.imread(s,cv2.IMREAD_COLOR)
    for j  in range(3):
        histr = cv2.calcHist([img], [0], None, [256], [0, 256])
        colors[i-1,:] = colors[i-1,:] + (histr.T/3)#将rgb三层进行归一化，合成一个矩阵
np.savetxt("colors.txt",colors)

'''
#对训练结果成好标签，储存起来
Y_train = np.ones([200,1])
for label in range(10):
    Y_train[(label*20):(label*20+21),:] = label +1
np.savetxt("train_Y.txt",Y_train)

#change the value to get different results
def distance(X1, X2):
    return np.sqrt(sum((X1-X2)**2))

def knn(train_x, train_y, test_x, K = k):
    x=0
    target = []
    for idx, X1 in enumerate(test_x):#遍历训练集中的向量，并返回向量的序号
        neighbors = []#建立储存他的K个紧邻的列表
        for X2, Y2 in zip(train_x, train_y):#遍历测试集的向量和标签
            dis = distance(X1, X2)#求解测试集向量和训练集每一个向量的距离
            neighbors.append((dis,Y2[0]))#将得到的距离和对应的标签以元组的形式放在紧邻列表中
        neighbors.sort(key=lambda x: x[0])#将元组的第一个元素距离作为键进行排序
        neighbors=neighbors[:K]#得到样本的K个近邻
        votes = {}#进行投票，建立字典
        for dis, label in neighbors:#遍历近列表中的距离和对应标签
            votes.setdefault(label,0)#将标签加入字典中，并设置初始值为0
            votes[label] += 1#每得到一个相同的label就将label的键值+1
        target.append(max(votes.items(), key=lambda x: x[1])[0])#得到最大的键值，即最后的投票结果，决定样本是谁
    return target#返回样本的标签列表

def accuracy(output, orignal):#计算准确率
    acc = (np.count_nonzero(np.equal(output, orignal)) )/ len(output)
    return acc
    
def train_test_split(X, Y, Rate):#将训练集分开，分为测试机和训练集
    test_x = []
    test_y = []
    for i in range(Rate):
        rand = random.randint(0,199-i)#利用random函数，随机抽取训练集中的Rate个作为测试集
        test_x.append(X[rand, :])
        X = np.delete(X, rand,axis = 0)#删除原训练集中作为测试的部分，以免影响准确率
        test_y.append(Y[rand, :])
        Y = np.delete(Y,rand,axis = 0)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    return test_x, test_y, X, Y

def main():
        temp = 0
        print("start")
        #pca = PCA()
        #X = pca.fit_transform(X_train)#将数据降到k
        X_train =LBP_hist + colors
        print (X_train.shape)
        for i in range(20):#训练10次，取平均值
            x_test, y_test, x_train, y_train = train_test_split(X_train, Y_train, TEST_RATE)
            target = knn(x_train, y_train, x_test, K =k)
            temp = temp + accuracy(target, y_test.T)#y_test为正确标签
        print(temp/20)
        print("end")  
main()

