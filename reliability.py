# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:10:46 2019

@author: TM
"""

import pandas as pd
FB = pd.read_excel(r"D:\Anaconda3\mycode\MLlab\FB_105Adv_Post_Comment_coding_v3.xlsx",encoding = 'utf-8')
#%%
import numpy as np


def svar(X):
    n = float(len(X))
    svar=(sum([(x-np.mean(X))**2 for x in X]) / n)* n/(n-1.)
    return svar

def CronbachAlpha(itemscores):
    itemvars = [svar(item) for item in itemscores]
    tscores = [0] * len(itemscores[0])
    for item in itemscores:
       for i in range(len(item)):
          tscores[i]+= item[i]
    nitems = len(itemscores)
    #print ("total scores=", tscores, 'number of items=', nitems)

    Calpha=nitems/(nitems-1.) * (1-sum(itemvars)/ svar(tscores))

    return Calpha

#%%phi相關係數
def phi_corr(X,Y):
    length = len(X)
    n11 = 0
    n10 = 0
    n01 = 0
    n00 = 0
    for idx in range(length):
        if (X[idx]==1 and Y[idx]==1):
            n11 = n11 +1
        if (X[idx]==1 and Y[idx]==0):
            n10 = n10 +1
        if (X[idx]==0 and Y[idx]==1):
            n01 = n01 +1
        if (X[idx]==0 and Y[idx]==0):
            n00 = n00 +1
    n0_ = n01+n00
    n_0 = n00+n10
    n1_ = n10+n11
    n_1 = n01+n11         
    phi = (n11*n00-n10*n01)/((n0_ * n_0 * n1_ * n_1)**(1/2))
    print(phi)
    return phi
#%%讀取機器人的評分
'''
SVM_DF = FB.SVMR_DF
SVM_TF = FB.SVM_TF

RF_DF = FB.RF_DF
RF_TF = FB.RF_TF
RF_NS = FB.RF_NS


NN_IG = FB.ANN_IG
NN_TF = FB.NN_TF
NN_NS = FB.NN_NS

RNN_TF = FB.RNN_TF
RNN_NS = FB.RNN_NS
'''
#%%讀取人類的評分
'''
course_rela = FB.course_rela#課程相關程度0~3
is_sta = FB.is_sta#統計相關0、1
bloom = FB.bloom#布魯姆認知階層0~4
is_course = FB.is_course#與課程相關0、1

'''
#%%計算皮爾森相關係數
cols = list(range(16,53))
df = FB.ix[:,cols]
#df = pd.DataFrame([course_rela,is_sta,bloom,is_course,SVM_DF,SVM_TF,RF_DF,RF_TF,RF_NS,NN_IG,NN_TF,NN_NS,RNN_TF,RNN_NS])
#df = df.T
corr_table_pear = df.corr(method = 'pearson')#'pearson', 'spearman', or 'kendall'
corr_table_spea = df.corr(method = 'spearman')#'pearson', 'spearman', or 'kendall'
corr_table_kend = df.corr(method = 'kendall')#'pearson', 'spearman', or 'kendall'

#%%
# =============================================================================
# 以下是信度
# =============================================================================
#%%我與蕭學長的
CronbachAlpha([df['is_sta'],df['is_sta2']])#0.948
#%%最高的
print(CronbachAlpha([df['is_sta2'],df['RNN_TF']]))#0.534
print(CronbachAlpha([df['is_sta2'],df['NN_TF']]))#0.532
print(CronbachAlpha([df['is_sta2'],df['RF_NS']]))#0.465
print(CronbachAlpha([df['is_sta2'],df['RF_TF']]))#0.382
print(CronbachAlpha([df['is_sta2'],df['RNN_NS']]))#0.372
print(CronbachAlpha([df['is_sta2'],df['RF_WV']]))#0.211
print(CronbachAlpha([df['is_sta2'],df['RNN_WV']]))#0.137
print(CronbachAlpha([df['is_sta2'],df['NN_WV']]))#0.068
print(CronbachAlpha([df['is_sta2'],df['NN_NS']]))#0
print(CronbachAlpha([df['is_sta2'],df['SVM_WV']]))#-0.129
print(CronbachAlpha([df['is_sta2'],df['SVMR_DF']]))#0.639
print(CronbachAlpha([df['is_sta2'],df['ANN_IG']]))#0.393
print(CronbachAlpha([df['is_sta2'],df['RF_DF']]))#0.273
#%%最高的
CronbachAlpha([df['is_course'],df['NN_WV']])#0.362
#%%最高的
print(CronbachAlpha([df['course_rela'],df['RF_NS']]))#0.384
print(CronbachAlpha([df['course_rela'],df['RNN_TF']]))#0.342
print(CronbachAlpha([df['course_rela'],df['NN_TF']]))#0.333
print(CronbachAlpha([df['course_rela'],df['RF_TF']]))#0.171
print(CronbachAlpha([df['course_rela'],df['RNN_NS']]))#0.171
print(CronbachAlpha([df['course_rela'],df['RF_WV']]))#0.050
print(CronbachAlpha([df['course_rela'],df['SVM_WV']]))#0.035
print(CronbachAlpha([df['course_rela'],df['NN_NS']]))#0
print(CronbachAlpha([df['course_rela'],df['RNN_WV']]))#-0.0005
print(CronbachAlpha([df['course_rela'],df['NN_WV']]))#-0.084
print(CronbachAlpha([df['course_rela'],df['SVMR_DF']]))#0.495
print(CronbachAlpha([df['course_rela'],df['ANN_IG']]))#0.355
print(CronbachAlpha([df['course_rela'],df['RF_DF']]))#0.316
#%%
CronbachAlpha([df['bloom'],df['bloom2']])#0.949
#%%
print(CronbachAlpha([df['bloom2'],df['RNN_TF']]))#0.404
print(CronbachAlpha([df['bloom2'],df['NN_TF']]))#0.400
print(CronbachAlpha([df['bloom2'],df['RF_NS']]))#0.397
print(CronbachAlpha([df['bloom2'],df['RF_TF']]))#0.264
print(CronbachAlpha([df['bloom2'],df['RNN_NS']]))#0.260
print(CronbachAlpha([df['bloom2'],df['RF_WV']]))#0.143
print(CronbachAlpha([df['bloom2'],df['RNN_WV']]))#0.119
print(CronbachAlpha([df['bloom2'],df['SVM_WV']]))#0.054
print(CronbachAlpha([df['bloom2'],df['NN_WV']]))#0.060
print(CronbachAlpha([df['bloom2'],df['NN_NS']]))#0
print(CronbachAlpha([df['bloom2'],df['SVMR_DF']]))#0.459
print(CronbachAlpha([df['bloom2'],df['ANN_IG']]))#0.278
print(CronbachAlpha([df['bloom2'],df['RF_DF']]))#0.303
#%%未加入ONS系列且與is_sta2最高的3個
con1 = df['RNN_TF']+df['NN_TF']+df['RF_TF']

print(CronbachAlpha([df['course_rela'],con1]))#0.459
print(CronbachAlpha([df['bloom'],con1]))#0.490
print(CronbachAlpha([df['bloom2'],con1]))#0.584
#%%加入ONS系列後以is_sta2最高的3個
con2 = df['RNN_TF']+df['NN_TF']+df['RF_NS']
print(CronbachAlpha([df['is_sta2'],con2]))#0.523
print(CronbachAlpha([df['course_rela'],con2]))#0.576
print(CronbachAlpha([df['bloom2'],con2]))#0.663
#df['con_best']=con2
#%%蕭學長的
Xioa = df['SVMR_DF']+df['ANN_IG']+df["RF_DF"]
print(CronbachAlpha([df['is_sta2'],df['SVMR_DF']]))#0.639
print(CronbachAlpha([df['course_rela'],Xioa]))#0.635
#print(CronbachAlpha([df['bloom'],Xioa]))#0.561
print(CronbachAlpha([df['bloom2'],Xioa]))#0.597
#print(CronbachAlpha([df['course_rela'],df['SVMR_DF']]))#0.495
#print(CronbachAlpha([df['bloom'],df['SVMR_DF']]))#0.484

#%%最高的
'''
以下是20000的分類模型
'''
print(CronbachAlpha([df['course_rela'],df['RF_NS_20000']]))#
print(CronbachAlpha([df['course_rela'],df['RNN_TF_20000']]))#
print(CronbachAlpha([df['course_rela'],df['NN_WV_20000']]))#
print(CronbachAlpha([df['course_rela'],df['RF_TF_20000']]))#
print(CronbachAlpha([df['course_rela'],df['RNN_NS_20000']]))#
print(CronbachAlpha([df['course_rela'],df['RF_WV_20000']]))#
print(CronbachAlpha([df['course_rela'],df['SVM_WV_20000']]))
print(CronbachAlpha([df['course_rela'],df['NN_NS_20000']]))#
print(CronbachAlpha([df['course_rela'],df['RNN_WV_20000']]))#
print(CronbachAlpha([df['course_rela'],df['NN_WV_20000']]))#
print(CronbachAlpha([df['course_rela'],df['SVM_NS_20000']]))#
print(CronbachAlpha([df['course_rela'],df['SVM_TF_20000']]))#
#%%
print(CronbachAlpha([df['is_sta2'],df['RNN_WV_20000']]))#
print(CronbachAlpha([df['is_sta2'],df['RNN_NS_20000']]))#
print(CronbachAlpha([df['is_sta2'],df['RNN_TF_20000']]))#
print(CronbachAlpha([df['is_sta2'],df['NN_WV_20000']]))#
print(CronbachAlpha([df['is_sta2'],df['NN_NS_20000']]))#
print(CronbachAlpha([df['is_sta2'],df['NN_TF_20000']]))#
print(CronbachAlpha([df['is_sta2'],df['RF_WV_20000']]))#
print(CronbachAlpha([df['is_sta2'],df['RF_NS_20000']]))#
print(CronbachAlpha([df['is_sta2'],df['RF_TF_20000']]))#
print(CronbachAlpha([df['is_sta2'],df['SVM_WV_20000']]))#
print(CronbachAlpha([df['is_sta2'],df['SVM_NS_20000']]))#
print(CronbachAlpha([df['is_sta2'],df['SVM_TF_20000']]))#
#%%
print(CronbachAlpha([df['bloom2'],df['RNN_WV_20000']]))#
print(CronbachAlpha([df['bloom2'],df['RNN_NS_20000']]))#
print(CronbachAlpha([df['bloom2'],df['RNN_TF_20000']]))#
print(CronbachAlpha([df['bloom2'],df['NN_WV_20000']]))#
print(CronbachAlpha([df['bloom2'],df['NN_NS_20000']]))#
print(CronbachAlpha([df['bloom2'],df['NN_TF_20000']]))#
print(CronbachAlpha([df['bloom2'],df['RF_WV_20000']]))#
print(CronbachAlpha([df['bloom2'],df['RF_NS_20000']]))#
print(CronbachAlpha([df['bloom2'],df['RF_TF_20000']]))#
print(CronbachAlpha([df['bloom2'],df['SVM_WV_20000']]))#
print(CronbachAlpha([df['bloom2'],df['SVM_NS_20000']]))#
print(CronbachAlpha([df['bloom2'],df['SVM_TF_20000']]))#
#%%
Con1 = df["RNN_TF_20000"]+df["RNN_NS_20000"]+df["NN_TF_20000"]+df["RF_NS_20000"]
print(CronbachAlpha([df['course_rela'],Con1]))#0.588
print(CronbachAlpha([df['bloom2'],Con1]))#0.657
print(CronbachAlpha([df['is_sta2'],Con1]))#0.465