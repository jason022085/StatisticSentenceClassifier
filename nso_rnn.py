# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 15:23:14 2019

@author: USER
"""
#%%檢查model的正確率
import numpy as np
import keras as kr
from time import time
DT = np.load('D:\Anaconda3\mycode\MLlab\DT_ns.npz')#取前200字的DT_ns(有加入統計字典的)
X_train = DT["article_train"]
X_test = DT["article_test"]
y_train = DT["board_train"]
y_test = DT["board_test"]

X_train = X_train.reshape((40000,200,1))
X_test = X_test.reshape((1416,200,1))
#%%
def buildRNN():
    from keras.models import Sequential
    from keras.layers import Dense,Embedding,CuDNNGRU,Dropout,BatchNormalization
    
    #volca_size = np.max([np.max(i) for i in articles])
    model = Sequential()
    model.add(CuDNNGRU(units= 128,return_sequences=True,input_shape=(200,1),kernel_initializer="glorot_normal"))#首層的unit>下一層的unit
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    
    model.add(CuDNNGRU(units= 128,return_sequences=False))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    
    model.add(Dense(2,activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam",metrics =['accuracy'] )
    print(model.summary())
    return model
#%%
def trainModel(model,X_train,y_train,X_test,y_test):
    from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
    reduce_learning_rate_by_acc = ReduceLROnPlateau(monitor='acc', factor=0.1, 
                                                    patience=10, verbose=1, 
                                                    mode='auto', cooldown=0, 
                                                    min_lr=0)
    check_point_by_acc = ModelCheckpoint("D:\Anaconda3\mycode\MLlab\model_nso_rnn\model_ns_rnn_r1_{epoch:02d}_{acc:.3f}_{val_acc:.3f}.h5", 
                                         monitor='acc', verbose=1, 
                                         save_best_only=True, 
                                         save_weights_only=False, 
                                         mode='auto', period=1)
    history1 = model.fit(X_train,y_train,batch_size=128,epochs= 32,
              validation_data=(X_test,y_test),
              callbacks = [reduce_learning_rate_by_acc,check_point_by_acc],
              shuffle=True)
    return history1

#%%訓練集40000筆
for i in range(2,11):
    model = buildRNN()
    history_40000 = trainModel(model,X_train,y_train,X_test,y_test)

    #存下訓練紀錄
    np.savez('history_nso_rnn_v'+str(i)+'_40000', val_loss=history_40000.history['val_loss'], 
             val_acc=history_40000.history['val_acc'],loss=history_40000.history['loss'],
             acc=history_40000.history['acc'],lr=history_40000.history['lr'])
#%%訓練集20000筆
for i in range(1,11):
    pick_2 = np.random.randint(X_train.shape[0], size=20000)
    X_train_2 = X_train[pick_2, :] 
    y_train_2 = y_train[pick_2, :] 

    model = buildRNN()
    history_20000 = trainModel(model,X_train_2,y_train_2,X_test,y_test)

    np.savez('history_nso_rnn_v'+str(i)+'_20000', val_loss=history_20000.history['val_loss'],
             val_acc=history_20000.history['val_acc'],loss=history_20000.history['loss'],
             acc=history_20000.history['acc'],lr=history_20000.history['lr'])
#%%訓練集10000筆
for i in range(1,11):
    pick_1 = np.random.randint(X_train.shape[0], size=10000)
    X_train_1 = X_train[pick_1, :] 
    y_train_1 = y_train[pick_1, :] 

    model = buildRNN()
    history_10000 = trainModel(model,X_train_1,y_train_1,X_test,y_test)

    np.savez('history_nso_rnn_v'+str(i)+'_10000', val_loss=history_10000.history['val_loss'], 
             val_acc=history_10000.history['val_acc'],loss=history_10000.history['loss'],
             acc=history_10000.history['acc'],lr=history_10000.history['lr'])
#%%
import matplotlib.pyplot as plt
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
#%%這裡會隨時改
from keras.models import load_model
model= load_model(r"D:\Anaconda3\mycode\MLlab\model_selected\nso_nn\model_nso_nn_40000_390_0.907_0.722.h5")
import numpy as np
history = np.load(r"D:\Anaconda3\mycode\MLlab\model_selected\nso_nn\history_nso_nn_v1_40000.npz")
val_loss = history['val_loss']
val_acc = history['val_acc']
loss = history['loss']
acc = history['acc']

#%%
#把onehot轉回來的第二種方法
index = np.argwhere(y_test == 1 )
y_test_raw = index[:,1]
#%%
from sklearn.metrics import classification_report, confusion_matrix
classes = ['Gossip','Stat']
y_pred = model.predict_classes(X_test)
cmdt = confusion_matrix(y_test_raw, y_pred)
plot_confusion_matrix(cmdt,classes)

print('Tra_Accuracy:', model.evaluate(X_train, y_train)[1])
print('Val_Accuracy:', model.evaluate(X_test, y_test)[1])
print(classification_report(y_test_raw, y_pred, target_names=classes))
#%%
# 绘制训练 & 验证的准确率值
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='lower right')
plt.show()
#%%
# 绘制训练 & 验证的准确率值
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()
#%%
