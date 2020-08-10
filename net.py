import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Embedding
from keras.layers import GRU
from keras.optimizers import SGD
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Reshape
from keras.utils import to_categorical
from keras.callbacks import TensorBoard

allheartbeats=np.load('D:\\pyproject\\final_heartbeats.npy')
alllabels=np.load('D:\\pyproject\\final_labels.npy')

inputs = Input(shape=(12,129,1))
x = BatchNormalization(axis=2, name='bn0')(inputs)
x=Conv2D(32, (3, 3), activation='elu',padding='same', name='conV1')(inputs)
x=MaxPooling2D(pool_size=(1, 2),padding='same')(x)
x=Dropout(0.6)(x)
x = BatchNormalization(axis=3, mode=0, name='bn1')(x)
x=Conv2D(64,(3, 3),padding='same', activation='elu',name='conV2')(x)
x=MaxPooling2D(pool_size=(2, 3),padding='same')(x)
x = BatchNormalization(axis=3, mode=0, name='bn2')(x)
x=Conv2D(64,(3, 3),padding='same', activation='elu',name='conV3')(x)
x=MaxPooling2D(pool_size=(2, 4),padding='same')(x)
x=Dropout(0.6)(x)

x=Reshape((9,32))(x)
print (x)
x = GRU(4,activation='tanh', return_sequences = True, name = 'gru1')(x)


x=Flatten()(x)#将图片拉平，进入全连接网络
x=Dense(128, activation='relu')(x)
predictions = Dense(4, activation = 'sigmoid', name = 'output')(x)


model = Model(inputs=inputs, outputs=predictions)
sgd = SGD(lr=0.01)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',#损失函数
              metrics=['accuracy'])

def strtoint(m):
    if m=='V':
        return(0)
    if m=='A':
        return(1)
    if m=='N':
        return(2)
    if m=='R':
        return(3)
new_label=list(map(strtoint,alllabels)) 

one_hot_labels = to_categorical(new_label, num_classes=4)
new_samps=np.expand_dims(allheartbeats,axis=3)
print(new_samps.shape)

train_data=new_samps[:7000,:,:,:]
train_label=one_hot_labels[:7000,:]

test_data=new_samps[7000:,:,:,:]
test_label=one_hot_labels[7000:,:]


model.fit(train_data, train_label,validation_split=0.15,epochs=100,callbacks=[TensorBoard(log_dir='E:\\project\\log')])  # starts
score = model.evaluate(test_data,test_label)
print('Test score:', score[0])
print('Test accuracy:', score[1])
