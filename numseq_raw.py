#%%載入套件
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt#畫圖用套件 
import keras as kr #神經網路常用套件(易)
from time import time #算時間的套件 
import jieba as jb#斷詞用套件
import pyprind
#%%
'''
將1-hot encoding的類別資料轉回原本的類別
'''
def re1hot(c):
    length = len(c)
    for i in range(length):
        if c[i]!=0:
            c = np.int(i)
            if c == 0:
                return("Gossiping")
            else:
                return("Statistics")
#%%
def plot_confusion_matrix(confusion_matrix, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots(figsize=(20, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
#%%讀取資料檔
'''
此資料檔為美玟給我的
我再分別對Gossip和Stat去合併
所以只各讀取一個檔
'''
t0 = time()
Gossip = pd.read_excel("D:\Anaconda3\mycode\MLlab\PTT_Gossiping.xlsx",encoding = 'utf-8')
Stat = pd.read_excel("D:\Anaconda3\mycode\MLlab\PTT_Statistics.xlsx",encoding = 'utf-8')
#Stop_word = pd.read_csv("D:\Anaconda3\mycode\MLlab\stopwords_zh.csv",encoding = 'big5',header = None)
print("耗時(秒)=",time()-t0)
#%%整理資料檔
'''
先將Gossiping和Statistics轉成0和1，方便之後的運算
然後將兩個資料合併
因為id不重要，丟掉它
長度太小也沒有意義，我設定是:不到20字都丟掉
兩個版的文章各取20708筆變成dataframe
'''
Gossip["board"] = Gossip["board"].str.replace("Gossiping","0")#將單字轉成0比較方便
Stat["board"] = Stat["board"].str.replace("Statistics","1")#將單字轉成1比較方便

Gossip["art_length"] = Gossip["article"].str.len()
print(np.mean(Gossip["art_length"]))
#%%
Gossip = Gossip[Gossip["art_length"]>=20]
print(np.mean(Gossip["art_length"]))
#%%
Gossip = Gossip[:20708]

Stat["art_length"] = Stat["article"].str.len()
print(np.mean(Stat["art_length"]))
#%%
Stat = Stat[Stat["art_length"]>=20]
Stat = Stat[:20708]

df = Gossip.append(Stat)#將兩dataframe合併
df = df[["article","board"]]#交換第一行和第二行
#df = df[:41488]#確定是1:1了
#%%讀取統計詞庫
#國家教育研究院的 雙語詞彙學術名詞暨辭書資訊網
#http://terms.naer.edu.tw/download/466/
Stat_vocab = pd.read_excel("D:\Anaconda3\mycode\MLlab\stat_vocab.xlsx",encoding = 'utf-8')
vocab_eng = Stat_vocab["英文名稱"]
vocab_ch = Stat_vocab["中文名稱"]
stat_dict = np.append(vocab_ch,vocab_eng)
#%%斷詞
t0 = time()
#所有句子斷詞後再合併存入articles
#words = jb.cut(text)
jb.set_dictionary("D:\Anaconda3\mycode\MLlab\dict.txt.big.txt")#使用繁體+簡體中文詞庫
jb.load_userdict(stat_dict) #加入我的統計詞庫

articles = []
for text in df["article"]:
    articles.append(jb.cut(text,cut_all=False))#直接返回list
    
new_articles = []
for article in articles:
    article = [x for x in article if (x != '-' and x != '\r\n' and x!= ' ')]
    #article = [x for x in article]
    article = np.array(article)
    new_articles.append(article)
df["new_article"] = np.array(new_articles)
print("耗時(秒)=",time()-t0)
#%%
df["order"] = range(41416)
df["new_art_length"] = df["new_article"].str.len()
df2 = df[df["new_art_length"]>200]
len(df2)#4674

#%%計算每個版的文章數
df['board'].value_counts()
#20653個統計文章
#20619個八卦文章
#%%資料預處理
'''
創造一個字典，紀錄每個字詞用的次數
'''
from collections import Counter
counts = Counter()
for new_article in new_articles:
    for word in new_article :
        counts[word]+=1
#這裡的counts是dictionary，紀錄了使用的字詞和次數

'''排序由多到少，然後將每個字詞給一個數字去對應'''

word_counts = sorted(counts, key=counts.get, reverse=True)
print(word_counts[:10])
word_to_int = {word: ii for ii, word in enumerate(word_counts, 1)}
#將所有字對應到一個數字

'''順便創一個可以將數字返回文字的字典'''

int_to_word = dict(zip(word_to_int.values(), word_to_int.keys()) ) 


'''創造一個只由數字組成的文章#後續的X_train由此而來'''

mapped_article = []
pbar = pyprind.ProgBar(len(df['article']),
                       title='Map article to ints')
for article in df['new_article']:
    mapped_article.append([word_to_int[word] for word in article])
    pbar.update()

'''將每個文章限制最大為200個字(後200字)，不足的會在前面補0'''   

from keras.preprocessing import sequence
max_length = 200
mapped_article_200 = sequence.pad_sequences(mapped_article, maxlen=max_length)

#%%
X = mapped_article_200
y = df["board"]
volca_size = np.max([np.max(i) for i in X])#算出總共有多少詞彙157661
#%%分割訓練集和測試集
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.preprocessing import sequence
'''
40000為訓練，1416為測試資料
把y做1-hot encoding
'''

X_train , X_test ,y_train, y_test = train_test_split(X,y,test_size=1416,random_state=1,stratify=y)
X_train =X_train.reshape(len(X_train),200)
X_test =X_test.reshape(len(X_test),200)
y_train = np_utils.to_categorical(y_train,2)
y_test = np_utils.to_categorical(y_test,2)
#%%完美的word2vec

from keras.models import Model
from keras.layers import Flatten, Dense, Embedding, Input


    
#輸出(samples, sequence_length, output_dim)的3D張量
input_layer = Input(shape=(200,)) 
x = Embedding(input_dim= volca_size ,output_dim= 128 )(input_layer)
#單獨做一個embedding模型，利於後面觀察
embedding = Model(input_layer,x)
y = Flatten()(x)
z = Dense( 2 ,activation= 'sigmoid' )(y)
#做一個model去看看embedding的效果
model = Model(input_layer,z)
model.compile(loss= 'binary_crossentropy',optimizer= 'adam' ,metrics=[ 'accuracy' ])
model.summary()
model.fit(X_train,y_train,batch_size=128,epochs= 8,validation_data=(X_test,y_test))
score = model.evaluate(X_test,y_test)
print(score)
#%%
#model.save('model_ns_embed_40000_997_985.h5') #acc=99.90,val_acc=98.03
model = kr.models.load_model('D:\Anaconda3\mycode\MLlab\model_ns_embed_40000_997_985.h5')
from keras.models import Model
embedding_model = Model(inputs=model.input , outputs=model.get_layer('embedding_8').output)#取出詞嵌入層
#embedding.summary()
#embedding.predict(X[: 1 ]).shape
#embedding.predict(X[: 1 ])
X_train_embed = embedding_model.predict(X_train)
X_test_embed = embedding_model.predict(X_test)
#%%儲存資料
#np.savez("DT_ns",article_train=X_train,article_test=X_test,board_train=y_train,board_test=y_test)
#40000 : 1416
#200字
#%%儲存資料
#np.savez("DT_ns_embed",article_train=X_train_embed,article_test=X_test_embed,board_train=y_train,board_test=y_test)
#40000 : 1416
#200字
#%%
'''
以後執行這行以下的部分就好了
'''
#%%檢查model的正確率
DT = np.load('D:\Anaconda3\mycode\MLlab\DT_ns.npz')#取前200字的DT_ns(有加入統計字典的)
X_train = DT["article_train"]
X_test = DT["article_test"]
y_train = DT["board_train"]
y_test = DT["board_test"]
X_train =X_train.reshape(len(X_train),200)
X_test =X_test.reshape(len(X_test),200)
#%%
model = kr.models.load_model('D:\Anaconda3\mycode\MLlab\model_selected\model_nso_rnn_40000_570_0.996_0.950.h5')
model.summary()
(err,acc) = model.evaluate(X_test,y_test)
print("(err,acc) = ",(err,acc))
y_pred = model.predict_classes(X_test)
#%%
'''
轉回去拉拉拉
'''
mapped_int = []
pbar2 = pyprind.ProgBar(1272,
                       title='Map ints to reviews')
X_test =X_test.reshape(len(X_test),200)
X_test_list = list(X_test)
for article in X_test_list:
    mapped_int.append([int_to_word[word] for word in article if word != 0])
    pbar2.update()
#%%
rd = np.random.randint(1272)
rd=821
print(mapped_int[rd],"\n")
print("真實類別：",re1hot(y_test[rd]))
if y_pred[rd] == 0:
    y_pred_class = "Gossiping"
else:
    y_pred_class = "Statistics"
print("預測類別：",y_pred_class,"\n")

#%%混淆矩陣
#把onehot轉回來的第二種方法
index = np.argwhere(y_test == 1 )
#print(index)
print(index[:,1])
y_test_raw = index[:,1]

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
y_pred = model.predict_classes(X_test)
cmrnn = confusion_matrix(y_test_raw, y_pred)
classes = ['Gossiping', 'Statistics']
plot_confusion_matrix(cmrnn,classes)
print('Accuracy:', model.evaluate(X_test,y_test))
print(classification_report(y_test_raw, y_pred, target_names=classes))
#%%
model = kr.models.load_model('D:\Anaconda3\mycode\MLlab\model_selected\model_ns_rnn_v6_40000_672_0.999_0.982.h5')
(err,acc) = model.evaluate(X_test,y_test)
print("(err,acc) = ",(err,acc))
#%%降2維
'''
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train_embed_re = X_train_embed.reshape((40000,25600))
pca.fit(X_train_embed_re)
X_pca = pca.transform(X_train_embed_re)

y_train_raw = np.argwhere(y_train == 1 )
plt.scatter(X_pca[:,0],X_pca[:,1])
'''
