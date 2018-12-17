#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 21:25:08 2018

@author: genggejianyi
"""
import pandas as pd
import numpy as np
import seaborn 
import matplotlib.pyplot as plt
from sklearn import linear_model,metrics
from sklearn.model_selection import train_test_split
from sklearn import neighbors
###数据预处理
# 导入数据
info=pd.read_csv('Weather Station Locations.csv',index_col=0)
data=pd.read_csv('Summary of Weather.csv',index_col=0)
# 取出经纬度
loc=info[['Latitude','Longitude']]
# 根据地区编码索引合并两个dataframe
data=data.join(loc)
# 去掉全为空值的列 
data=data.dropna(how='all',axis=1)
# 查看每列的缺失值情况
data.isnull().sum()
# 去除缺失值较多以及重复的列
data.drop(['WindGustSpd','DR','SPD','SND','PGT','TSHDSBRSGF','MAX','MIN','MEA','SNF','PRCP',
           'YR','MO','DA'],axis=1,inplace=True)
# 填充poorweather以及snowfall两列缺失值
data['PoorWeather']=data['PoorWeather'].fillna(0)
data['Snowfall']=data['Snowfall'].fillna(0)
# 查看各类数据分布情况
list(map(lambda i:data.iloc[:,i].value_counts(),range(1,7)))
# 将poorweather处理为0-1变量
data.loc[data['PoorWeather']!=0,'PoorWeather']=1
# 替换异常数据
data=data.replace('T',0)
data=data.replace('#VALUE!',0)
# 查看各列数据类型                  
data.info()
# 改变precip数据类型
data.Precip=data.Precip.astype(np.float64)
# 改变snowfall数据类型
data.Snowfall=data.Snowfall.astype(np.float64)


##描述性分析
# 描述统计
data.describe()
# 选取某一地区数据
data1=data[data.index==10001]
# 转换为时间类型
time=pd.to_datetime(data1.Date)
# 画出precip,maxtemp,mintemp,meantemp随时间变化图，温度存在季节性
y1=data1.Precip
y2=data1.MaxTemp
y3=data1.MinTemp
y4=data1.MeanTemp
plt.subplots_adjust(hspace = 0.5)  #设置subplot高度间距
plt.subplot(221)
plt.title('Precip')      #图表标题
plt.xticks([])           #隐藏x轴标签
plt.plot(time,y1)
plt.subplot(222)
plt.title('MaxTemp')
plt.xticks([])
plt.plot(time,y2)
plt.subplot(223)
plt.title('MinTemp')
plt.xticks([])
plt.plot(time,y3)
plt.subplot(224)
plt.title('MeanTemp')
plt.xticks([])
plt.plot(time,y4)
# 画出precip,maxtemp,mintemp,meantemp分布直方图
plt.subplots_adjust(hspace = 0.5)  #设置subplot高度间距
plt.subplot(221)
plt.title('Precip')
plt.hist(y1)
plt.subplot(222)
plt.title('MaxTemp')
plt.hist(y2)
plt.subplot(223)
plt.title('MinTemp')
plt.hist(y3)
plt.subplot(224)
plt.title('MeanTemp')
plt.hist(y4)


# 随机选取四个地区，画出poorweather分布饼图
Y1=data1.PoorWeather
data2=data[data.index==12001]
Y2=data2.PoorWeather
data3=data[data.index==10002]
Y3=data3.PoorWeather
data4=data[data.index==61501]
Y4=data4.PoorWeather
plt.subplots_adjust(hspace = 0.1)  #设置subplot高度间距
plt.subplot(221)
plt.pie(Y1.value_counts(),labels=[0,1],autopct='%1.1f%%')
plt.title('poorweather Y1')
plt.subplot(222)
plt.pie(Y2.value_counts(),labels=[0,1],autopct='%1.1f%%')
plt.title('poorweather Y2')
plt.subplot(223)
plt.pie(Y3.value_counts(),labels=[0,1],autopct='%1.1f%%')
plt.title('poorweather Y3')
plt.subplot(224)
plt.pie(Y4.value_counts(),labels=[0,1],autopct='%1.1f%%')
plt.title('poorweather Y4')
# 随意取某一天各地区的数据
data_reg=data[data.Date==list(data.Date)[1000]]
# 画出相关系数图，可以发现maxtemp,mintemp,meantemp共线性，取meantemp即可
data_cor=data_reg[['Precip','MaxTemp','MinTemp','MeanTemp','PoorWeather','Latitude','Longitude']]
corr=data_cor.corr()
seaborn.heatmap(corr)

##转换x,y数据格式并划分训练集测试集
X=np.array(data[['Precip','MeanTemp','Snowfall','PoorWeather']])
Y=np.array(list(data.index))
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=123,train_size=0.8)



##利用逻辑回归进行训练并得到测试集正确率
# 单独取出precip,meantemp,snowfall,poorweather四列，作为特征，标签为163个地区编号，训练逻辑回归模型
lr=linear_model.LogisticRegression(C=10,solver='saga',class_weight='balanced',multi_class='multinomial')
lr.fit(X_train,Y_train)
print('logistic regression train acc:',\
      metrics.accuracy_score(Y_train,lr.predict(X_train)))
print('logistic regression test acc:',\
      metrics.accuracy_score(Y_test,lr.predict(X_test)))



##利用knn进行训练并得到不同n_neighbors情况下测试集准确率
n_nb=[]
ACC=[]
for i in range(1,50):
    # 建立KNN模型
    knn_clf=neighbors.KNeighborsClassifier(i,algorithm='ball_tree',weights='distance')
    knn_clf.fit(X_train,Y_train)
    # 计算测试集准确率
    acc=sum(map(int,knn_clf.predict(X_test)==Y_test))/len(Y_test)
    n_nb.append(i)
    ACC.append(acc)
    # 画出不同n_neighbors（取1-49）情况下测试集准确率变化图
    plt.plot(n_nb,ACC)
plt.xlabel('n_neighbors')
plt.ylabel('ACC')
plt.title('diff n_neighbors acc')
plt.show()




##回归分析
# 取纬度作为y
y_reg=data_reg['Latitude']
# 取三个变量作为x
x_reg=data_reg[['Precip','MeanTemp','PoorWeather']]
reg=linear_model.LinearRegression()
reg.fit(x_reg,y_reg)
print('R方：',\
      reg.score(x_reg, y_reg))  #R方
print('自变量系数：',\
      reg.coef_)                #参数值
print('截距：',\
      reg.intercept_)           #截距项





