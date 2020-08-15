# -*- coding: utf-8 -*-
"""
Created on Sat May 11 17:25:06 2019

@author: HF
"""
#%%載入套件
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras as kr
import sklearn as sk
from time import time
import tensorflow as tf 

#%%讀取資料檔
'''
此資料檔為美玟給我的
我再分別對Gossip和Stat去合併
所以只各讀取一個檔
'''
Gossip = pd.read_excel("D:\Anaconda3\mycode\MLlab\PTT_Gossiping.xlsx",encoding = 'utf-8')
Stat = pd.read_excel("D:\Anaconda3\mycode\MLlab\PTT_Statistics.xlsx",encoding = 'utf-8')
FB_club = pd.read_excel("D:\Anaconda3\mycode\MLlab\FB_105Adv_Post_Comment_coding2.xlsx",encoding = 'utf-8')
Stop_word = pd.read_csv("D:\Anaconda3\mycode\MLlab\stopwords_zh.csv",encoding = 'big5',header = None)
FB_club = FB_club[["ID","message"]]

#%%整理資料檔
'''
先將Gossiping和Statistics轉成0和1，方便之後的運算
然後將兩個資料合併
因為id不重要，丟掉它
長度太小也沒有意義，我設定是:不到20字都丟掉
兩個版的文章各取20000筆變成dataframe
'''
Gossip["board"] = Gossip["board"].str.replace("Gossiping","0")#將單字轉成0比較方便
Stat["board"] = Stat["board"].str.replace("Statistics","1")#將單字轉成1比較方便

Gossip["art_length"] = Gossip["article"].str.len()
Gossip = Gossip[Gossip["art_length"]>=20]
Gossip = Gossip[:20708]

Stat["art_length"] = Stat["article"].str.len()
Stat = Stat[Stat["art_length"]>=20]
Stat = Stat[:20708]

df = Gossip.append(Stat)#將兩dataframe合併
df = df[["article","board"]]#交換第一行和第二行
#df = df[:41488]#確定是1:1了
#%%

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer
X, y = df.article , df.board
stopword = list(Stop_word[0])
'''
cv = CountVectorizer(max_df=0.8, min_df=5,
                                     max_features=4000,
                                     stop_words=stopword,token_pattern='\\b\\w+\\b')
'''
cv = CountVectorizer(stop_words=stopword,token_pattern='\\b\\w+\\b')
X_vec = cv.fit_transform(X)

res = dict(zip(cv.get_feature_names(),
               mutual_info_classif(X_vec, y, discrete_features=True)
               ))
print(res)
#%%


#%%
'''
res_sort = [ v for v in sorted(res.values())]
total = np.sum(res_sort)
for i in range(6700):
    IG_loss = np.sum(res_sort[i:8773])/total
    if IG_loss < 0.80:
        print (i) 
#i=6520 ,所以我要選取編號6519~8773的詞，共是2254個字
'''
#%%文字轉TF-IDF
'''
用美玟給我的斷詞袋做斷詞
然後算出重要的特徵(詞)
超過80%或低於5次的詞都會被忽略掉
只取1000個最重要的特徵(tfidf最大的)
'''
from sklearn.feature_extraction.text import TfidfVectorizer

stopword = list(Stop_word[0])#斷詞們
#總共有513902個字
#下界設為0 ,上界設為80%後還有513902
#下界設為0 ,上界設為50%後還有513902
#下界設為0 ,上界設為10%後還有513898 ，表示最大TF值的字只有佔全部字的10%多一點
#下界設為5 ,上界設為20%後還有8774個字
#下界設為10 ,上界設為20%後還有3986個字
#下界設為15 ,上界設為20%後還有2524個字
#下界設為20 ,上界設為20%後還有1885個字
#下界設為25 ,上界設為20%後還有1464個字
#下界設為30 ,上界設為20%後還有1205個字
#下界設為35 ,上界設為20%後還有1027個字
#下界設為0.01% 後還有8774個字，代表下界設定5次和0.01%會殺掉一樣的詞
#下界設為0.05% 後還有1776個字
#下界設為0.08% 後還有1062個字
#下界設為0.09% 後還有959個字
#下界設為0.1% 後還有880個字
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=5,max_features=4000,use_idf=True,
                                   stop_words=stopword,token_pattern='\\b\\w+\\b')#學長是用max=0.8/min=0.001
Train_text = df["article"]#文章
Train_label = df["board"]#標籤...這邊好像用不到
tfidf = tfidf_vectorizer.fit_transform(Train_text)
print(tfidf.shape)

#%%設定依變項和自變項
from sklearn.model_selection import train_test_split
from keras.utils import np_utils#1-hot encoding
from keras.preprocessing import sequence
'''
90%為訓練，10%為測試資料
把y做1-hot encoding
'''
X = tfidf.A #matrix to ndarray
y = Train_label
X_train , X_test ,y_train, y_test = train_test_split(X,y,test_size=1416,random_state=1,stratify=y)

X_train =X_train.reshape(40000,4000)
X_test =X_test.reshape(1416,4000)

y_train = np_utils.to_categorical(y_train,2)
y_test = np_utils.to_categorical(y_test,2)
#%%
np.savez("DT_tfidf_4000",article_train=X_train,article_test=X_test,board_train=y_train,board_test=y_test)
#%%建立LSTM模型所需的語法
from keras.models import Sequential
from keras.layers import LSTM,GRU,SimpleRNN,Activation,Dense,Embedding,Flatten #flatten是要將矩陣拉平成向量
from keras.layers import noise,BatchNormalization,Dropout,PReLU
from keras.optimizers import SGD,Adam,RMSprop#RMS適合RNN
#%%建立模型
model_LSTM = Sequential()
model_LSTM.add(Embedding(1000,128))
model_LSTM.add(LSTM(
    # 如果後端使用tensorflow，batch_input_shape 的 batch_size 需設為 None.
    # 否則執行 model.evaluate() 會有錯誤產生.
    batch_input_shape=(None,128,1), 
    units=10,
    )) 
model_LSTM.add(Dense(2,activation="sigmoid"))
model_LSTM.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001, rho=0.9),metrics =['accuracy'] )
model_LSTM.summary()
#%%訓練模型
history = model_LSTM.fit(X_train,y_train,batch_size=32,epochs=5,validation_split=0.01)
(err,acc) = model_LSTM.evaluate(X_test,y_test)
print("(err,acc) = ",(err,acc))
#%%
'''
RNN最高79%正確率
而NN居然有90%
'''
#%% NN的表現
model = Sequential()
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001,input_shape = (3000,)))
model.add(Dense(1024,input_dim=3000))
model.add(PReLU(alpha_initializer='zeros'))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))

model.add(Dense(32))
model.add(PReLU(alpha_initializer='zeros'))   # add an advanced activation
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))

model.add(Dense(2))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
model.add(Activation("sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam',metrics =['accuracy'])
model.summary()
#%%
history = model.fit(X_train,y_train,batch_size=128,epochs=100,validation_data=(X_test,y_test))
(err,acc) = model.evaluate(X_test,y_test)

#%%
model.summary()
print("(err,acc) = ",(err,acc))#90%