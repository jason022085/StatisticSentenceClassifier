# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 07:29:52 2019

@author: USER
"""
#%%檢查model的正確率
import numpy as np
import keras as kr
from time import time
DT = np.load('D:\Anaconda3\mycode\MLlab\DT_ns_embed.npz')#取前200字的DT_ns(有加入統計字典的)
X_train = DT["article_train"]
X_test = DT["article_test"]
y_train = DT["board_train"]
y_test = DT["board_test"]

#%%建立一個RNN
def buildRNN():
    from keras.models import Sequential
    from keras.layers import Dense,Embedding,CuDNNGRU,Dropout,BatchNormalization
    
    #volca_size = np.max([np.max(i) for i in articles])
    model = Sequential()
    
    model.add(CuDNNGRU(units= 128,return_sequences=True,input_shape=(None,128),kernel_initializer="glorot_normal"))#首層的unit>下一層的unit
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    
    model.add(CuDNNGRU(units= 128,return_sequences=False))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    
    model.add(Dense(2,activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam",metrics =['accuracy'] )
    print(model.summary())
    return model
#%%load old model
# Load the model
from keras.models import load_model
model1= load_model(r"D:\Anaconda3\mycode\MLlab\model_ns_rnn\40000_21\v1\model_ns_rnn_v1_40000_609_0.998_0.986.h5")#最強
model2= load_model(r"D:\Anaconda3\mycode\MLlab\model_ns_rnn\40000_21\v2\model_ns_rnn_v2_40000_483_0.998_0.985.h5")
model3= load_model(r"D:\Anaconda3\mycode\MLlab\model_ns_rnn\40000_21\v3\model_ns_rnn_v3_40000_525_0.998_0.986.h5")
model4= load_model(r"D:\Anaconda3\mycode\MLlab\model_ns_rnn\40000_21\v4\model_ns_rnn_v4_40000_189_0.998_0.984.h5")
model5= load_model(r"D:\Anaconda3\mycode\MLlab\model_ns_rnn\40000_21\v5\model_ns_rnn_v5_40000_252_0.998_0.983.h5")
model6= load_model(r"D:\Anaconda3\mycode\MLlab\model_ns_rnn\40000_21\v6\model_ns_rnn_v6_40000_672_0.999_0.982.h5")
model7= load_model(r"D:\Anaconda3\mycode\MLlab\model_ns_rnn\40000_21\v7\model_ns_rnn_v7_40000_609_0.998_0.983.h5")
model8= load_model(r"D:\Anaconda3\mycode\MLlab\model_ns_rnn\40000_21\v8\model_ns_rnn_v8_40000_378_0.998_0.984.h5")
model9= load_model(r"D:\Anaconda3\mycode\MLlab\model_ns_rnn\40000_21\v9\model_ns_rnn_v9_40000_189_0.998_0.985.h5")
model10= load_model(r"D:\Anaconda3\mycode\MLlab\model_ns_rnn\40000_21\v10\model_ns_rnn_v10_40000_168_0.998_0.981.h5")
#%%
model1= load_model(r"D:\Anaconda3\mycode\MLlab\model_ns_rnn\40000_21\v1\model_ns_rnn_v1_40000_609_0.998_0.986.h5")#最強
scores=model1.evaluate(X_train, y_train,verbose=1)
print('val_{}: {:.3%}'.format(model1.metrics_names[1], scores[1]))
scores=model1.evaluate(X_test, y_test,verbose=1)
print('val_{}: {:.3%}'.format(model1.metrics_names[1], scores[1]))

#%%以平均權重創造一個新model
models = [model1,model2,model3,model4,model5,model6,model7,model8,model9,model10]
weights = [model.get_weights() for model in models]

new_weights = list()

for weights_list_tuple in zip(*weights):
    new_weights.append(
        [np.array(weights_).mean(axis=0)\
            for weights_ in zip(*weights_list_tuple)])
new_model = buildRNN()
new_model.set_weights(new_weights)
#%%新model的訓練集正確率
scores=new_model.evaluate(X_train, y_train,verbose=0)
print('{}: {:.2%}'.format(new_model.metrics_names[1], scores[1]))#87.38%
#%%新model的驗證集正確率
scores=new_model.evaluate(X_test, y_test,verbose=0)
print('val_{}: {:.2%}'.format(new_model.metrics_names[1], scores[1]))#83.72%
#%%
'''
拿20000的模型攻擊40000筆資料
'''
# Load the model
from keras.models import load_model
model1= load_model(r"D:\Anaconda3\mycode\MLlab\model_ns_rnn\20000_11\v1\model_ns_rnn_v1_20000_176_0.999_0.980.h5")#最弱
model2= load_model(r"D:\Anaconda3\mycode\MLlab\model_ns_rnn\20000_11\v2\model_ns_rnn_v2_20000_341_0.998_0.984.h5")
model3= load_model(r"D:\Anaconda3\mycode\MLlab\model_ns_rnn\20000_11\v3\model_ns_rnn_v3_20000_275_0.998_0.984.h5")
model4= load_model(r"D:\Anaconda3\mycode\MLlab\model_ns_rnn\20000_11\v4\model_ns_rnn_v4_20000_308_0.999_0.983.h5")
model5= load_model(r"D:\Anaconda3\mycode\MLlab\model_ns_rnn\20000_11\v5\model_ns_rnn_v5_20000_143_0.998_0.985.h5")
model6= load_model(r"D:\Anaconda3\mycode\MLlab\model_ns_rnn\20000_11\v6\model_ns_rnn_v6_20000_343_0.998_0.983.h5")
model7= load_model(r"D:\Anaconda3\mycode\MLlab\model_ns_rnn\20000_11\v7\model_ns_rnn_v7_20000_187_0.999_0.984.h5")
model8= load_model(r"D:\Anaconda3\mycode\MLlab\model_ns_rnn\20000_11\v8\model_ns_rnn_v8_20000_231_0.999_0.977.h5")
model9= load_model(r"D:\Anaconda3\mycode\MLlab\model_ns_rnn\20000_11\v9\model_ns_rnn_v9_20000_308_0.999_0.987.h5")#最強
model10= load_model(r"D:\Anaconda3\mycode\MLlab\model_ns_rnn\20000_11\v10\model_ns_rnn_v10_20000_176_0.998_0.983.h5")
#%%取20000中最強的model9
new_weight = model9.get_weights()
mix_model = buildRNN()
mix_model.set_weights(new_weight)
#%%
X_train_1 = X_train[:20000] 
y_train_1 = y_train[:20000] 

scores=mix_model.evaluate(X_train_1, y_train_1,verbose=1)
print('{}: {:.3%}'.format(mix_model.metrics_names[1], scores[1]))#99.790不能自創驗證集,恐與訓練集重覆
#%%
X_train_2 = X_train[10000:30000] 
y_train_2 = y_train[10000:30000] 

scores=mix_model.evaluate(X_train_2, y_train_2,verbose=1)
print('{}: {:.3%}'.format(mix_model.metrics_names[1], scores[1]))#99.778
#%%
X_train_3 = X_train[20000:] 
y_train_3 = y_train[20000:] 

scores=mix_model.evaluate(X_train_3, y_train_3,verbose=1)
print('{}: {:.3%}'.format(mix_model.metrics_names[1], scores[1]))#99.792
#%%
scores=mix_model.evaluate(X_train, y_train,verbose=1)
print('{}: {:.3%}'.format(mix_model.metrics_names[1], scores[1]))#99.791
