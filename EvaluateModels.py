# -*- coding: utf-8 -*-
"""
計算precision recall F1-score
"""
import numpy as np
#%%word2vec的訓練資料
WV = np.load('D:\Anaconda3\mycode\MLlab\DT_ns_embed.npz')#取前200字的DT_ns(有加入統計字典的)
WV_X_train = WV["article_train"]
WV_X_test = WV["article_test"]
y_train = WV["board_train"]
y_test = WV["board_test"]
#WV_X_train = WV_X_train.reshape(len(WV_X_train),200)
#WV_X_test = WV_X_test.reshape(len(WV_X_test),200)
#%%NumberSeq的訓練資料
NS = np.load('D:\Anaconda3\mycode\MLlab\DT_ns.npz')#取前200字的DT_ns(有加入統計字典的)
NS_X_train = NS["article_train"]
NS_X_test = NS["article_test"]
y_train = NS["board_train"]
y_test = NS["board_test"]
#NS_X_train = NS_X_train.reshape(len(NS_X_train),200)
#NS_X_test = NS_X_test.reshape(len(NS_X_test),200)
#%%TF-IDF的訓練資料
TF = np.load('D:\Anaconda3\mycode\MLlab\DT_tfidf_4000.npz')#取前200字的DT_ns(有加入統計字典的)
TF_X_train = TF["article_train"]
TF_X_test = TF["article_test"]
y_train = TF["board_train"]
y_test = TF["board_test"]
#TF_X_train = TF_X_train.reshape(len(TF_X_train),200)
#TF_X_test = TF_X_test.reshape(len(TF_X_test),200)
#%%
import ConfusionMatirx as CM #ConfusionMatirx.py
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import keras as kr
from sklearn.externals import joblib #jbolib模块
#%%算與W2V搭配的指標

#以下兩行為一組
#model = kr.models.load_model('D:\Anaconda3\mycode\MLlab\model_selected_20000\model_w2v_nn_48_0.999_0.983.h5')
#y_pred = model.predict_classes(WV_X_test)
#model_w2v_nn_48_0.999_0.983.h5
#model_w2v_rnn_308_0.999_0.987.h5


#以下四行為一組
model = joblib.load('D:\Anaconda3\mycode\MLlab\model_selected_20000\model_w2v_svm_100.27_0.990_0.955.pkl')
WV_X_train = WV_X_train.reshape(len(WV_X_train),25600)
WV_X_test = WV_X_test.reshape(len(WV_X_test),25600)
y_pred = model.predict(WV_X_test)
#model_w2v_rf_0.998_0.975.pkl
#model_w2v_svm_100.27_0.990_0.955.pkl


y_test_raw = np.argwhere(y_test == 1 )[:,1]
cmrnn = confusion_matrix(y_test_raw, y_pred)
classes = ['Gossiping', 'Statistics']

CM.plot_confusion_matrix(cmrnn,classes)
print(classification_report(y_test_raw, y_pred, target_names=classes))
#%%與NS搭配
#以下兩行為一組
#model = kr.models.load_model('D:\Anaconda3\mycode\MLlab\model_selected_20000\model_ns_rnn_320_0.990_0.855.h5')
#NS_X_train = NS_X_train.reshape(len(NS_X_train),200,1)#RNN要搭配這兩行
#NS_X_test = NS_X_test.reshape(len(NS_X_test),200,1)#RNN要搭配這兩行
#y_pred = model.predict_classes(NS_X_test)
#model_ns_nn_31_0.908_0.713.h5
#model_ns_rnn_320_0.990_0.855.h5


#以下四行為一組
model = joblib.load('D:\Anaconda3\mycode\MLlab\model_selected_20000\model_ns_svm_0.543_0.543.pkl')
NS_X_train = NS_X_train.reshape(len(NS_X_train),200)#RNN要搭配這兩行
NS_X_test = NS_X_test.reshape(len(NS_X_test),200)#RNN要搭配這兩行
y_pred = model.predict(NS_X_test)
#model_ns_rf_14.54_0.987_0.775.pkl
#model_ns_svm_0.543_0.543.pkl


y_test_raw = np.argwhere(y_test == 1 )[:,1]
cmrnn = confusion_matrix(y_test_raw, y_pred)
classes = ['Gossiping', 'Statistics']

CM.plot_confusion_matrix(cmrnn,classes)
print(classification_report(y_test_raw, y_pred, target_names=classes))
#%%
#以下兩行為一組
#model = kr.models.load_model('D:\Anaconda3\mycode\MLlab\model_selected_20000\model_tf_nn_64_0.939_0.892.h5')
#TF_X_train = TF_X_train.reshape(len(TF_X_train),1,4000)#RNN要搭配這兩行
#TF_X_test = TF_X_test.reshape(len(TF_X_test),1,4000)#RNN要搭配這兩行
#y_pred = model.predict_classes(TF_X_test)
#model_tf_nn_64_0.939_0.892.h5
#model_tf_rnn_42_0.935_0.885.h5


#以下四行為一組
model = joblib.load('D:\Anaconda3\mycode\MLlab\model_selected_20000\model_tf_svm_0.671_0.667.pkl')
TF_X_train = TF_X_train.reshape(len(TF_X_train),4000)#RNN要搭配這兩行
TF_X_test = TF_X_test.reshape(len(TF_X_test),4000)#RNN要搭配這兩行
y_pred = model.predict(TF_X_test)
#model_tf_rf_36.94_0.829_0.816.pkl
#model_tf_svm_0.671_0.667.pkl


y_test_raw = np.argwhere(y_test == 1 )[:,1]
cmrnn = confusion_matrix(y_test_raw, y_pred)
classes = ['Gossiping', 'Statistics']

CM.plot_confusion_matrix(cmrnn,classes)
print(classification_report(y_test_raw, y_pred, target_names=classes))
#%%
"""

計算信度

"""
FB = np.load('D:\Anaconda3\mycode\MLlab\FB_text.npz')
FB_text_NS = FB["NS"]
FB_text_WV = FB["WV"]
FB_text_TF = np.load('D:\Anaconda3\mycode\MLlab\FB_text_ML.npy')
#%%
import pandas as pd
FB_club = pd.read_excel(r"D:\Anaconda3\mycode\MLlab\FB_105Adv_Post_Comment_coding_v3.xlsx",encoding = 'utf-8')
#Stop_word = pd.read_csv("D:\Anaconda3\mycode\MLlab\stopwords_zh.csv",encoding = 'big5',header = None)
FB_club = FB_club[["ID","message"]]
#%%DL algorithms 的分類
model = kr.models.load_model('D:\Anaconda3\mycode\MLlab\model_selected_20000\model_tf_rnn_42_0.935_0.885.h5')
#model_ns_nn_31_0.908_0.713.h5
#model_ns_rnn_320_0.990_0.855.h5
#model_w2v_nn_48_0.999_0.983.h5
#model_w2v_rnn_308_0.999_0.987.h5
#model_tf_nn_64_0.939_0.892.h5
#model_tf_rnn_42_0.935_0.885.h5
#RNN 要將資料reshape((376,1,200))
FB_text_NS = FB_text_NS.reshape((376,200,1))
FB_text_TF = FB_text_TF.reshape((376,1,4000))
new_col = model.predict_classes(FB_text_TF)
#FB_text_NS
#FB_text_WV
#FB_text_TF
FB_club["RNN_TF_20000"] = new_col
#%%ML algorithms 的分類
model = joblib.load('D:\Anaconda3\mycode\MLlab\model_selected_20000\model_tf_svm_0.671_0.667.pkl')
#model_ns_rf_14.54_0.987_0.775.pkl
#model_ns_svm_0.543_0.543.pkl
#model_w2v_rf_0.998_0.975.pkl
#model_w2v_svm_100.27_0.990_0.955.pkl
#model_tf_rf_36.94_0.829_0.816.pkl
#model_tf_svm_0.671_0.667.pkl
#W2V要改成
FB_text_WV=FB_text_WV.reshape((376,25600))
new_col = model.predict(FB_text_TF)
#FB_text_NS
#FB_text_WV
#FB_text_TF
FB_club["SVM_TF_20000"] = new_col